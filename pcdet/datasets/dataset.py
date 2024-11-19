from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils, file_client
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.client = getattr(file_client, self.dataset_cfg.BACKEND.NAME)(
            **self.dataset_cfg.BACKEND.get('KWARGS', {})
        )
        if self.dataset_cfg is None or class_names is None:
            return
        
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self.cur_epoch = 0
        self._merge_all_iters_to_one_epoch = False

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        self.total_epochs = epochs
        if merge:
            self._merge_all_iters_to_one_epoch = True
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            data_dict_for_augmentation = {
                **data_dict,
                'cur_epoch': self.cur_epoch,
                'total_epochs': self.total_epochs
            }
            if data_dict.get('gt_boxes', None) is not None:
                gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
                data_dict_for_augmentation.update({'gt_boxes_mask': gt_boxes_mask})

            data_dict = self.data_augmentor.forward(
                data_dict=data_dict_for_augmentation
            )

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]
                gt_boxes2d = np.concatenate((data_dict['gt_boxes2d'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes2d'] = gt_boxes2d

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and data_dict.get('gt_boxes', None) is not None and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        data_dict.pop('cur_epoch', None)
        data_dict.pop('total_epochs', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                # SP fusion
                elif key in ['target_gt_voxels', 'target_gt_voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)        
                      
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                # SP fusion
                elif key in ['target_gt_points', 'target_gt_voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i) # batch numbering
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)

                
                elif key in ['gt_boxes', 'gt_boxes2d']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes
                elif key in ['image']:
                    # Get largest image size (H, W)
                    max_h = max([img.shape[0] for img in val])
                    max_w = max([img.shape[1] for img in val])
                    batch_image = []
                    for img in val:
                        image_pad = np.pad(
                            img,
                            pad_width=((0, max_h - img.shape[0]), (0, max_w - img.shape[1]), (0, 0)),
                            mode='constant',
                            constant_values=0
                        )
                        
                        batch_image.append(image_pad)
                    batch_image = np.stack(batch_image, axis=0)  # (B, H, W, 3)
                    ret[key] = np.ascontiguousarray(batch_image.transpose(0, 3, 1, 2))  # (B, 3, H, W)
                elif key in ['gt_image']:
                    # Get largest image size (H, W)
                    max_h = max([img.shape[0] for img in val])
                    max_w = max([img.shape[1] for img in val])
                    batch_image = []
                    for img in val:
                        image_pad = np.pad(
                            img,
                            pad_width=((0, max_h - img.shape[0]), (0, max_w - img.shape[1]), (0, 0)),
                            mode='constant',
                            constant_values=0
                        )
                        
                        batch_image.append(image_pad)
                    batch_image = np.stack(batch_image, axis=0)  # (B, H, W, 3)
                    ret[key] = np.ascontiguousarray(batch_image.transpose(0, 3, 1, 2))  # (B, 3, H, W)
                elif key in ['transformation_2d_list', 'transformation_2d_params', 'transformation_3d_list', 'transformation_3d_params']:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret