from .detector3d_template import Detector3DTemplate
from pytorch3d.loss import chamfer_distance
import torch
class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, logger):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            loss_chamfer = chamfer_distance(self.converting_for_chamfer(batch_dict['recon_points'].indices.float()), self.converting_for_chamfer(batch_dict['target_gt_coords'].float()))
            # print(batch_dict['recon_points'].indices)
            # print(batch_dict['target_gt_coords'])
            
            # loss_chamfer = chamfer_distance(batch_dict['recon_points'].indices.float(), batch_dict['target_gt_coords'].float())
            loss_1 = loss_chamfer[0] * 0.0001
            loss += loss_1*1
            
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    # def converting_for_chamfer(self, coords):
    #     # 주어진 텐서
    #     # 배치 인덱스와 좌표 분리
    #     batch_indices = coords[:, 0]
    #     xyz_coords = coords[:, 1:]

    #     # 배치 인덱스를 기준으로 텐서 재구성
    #     batch_size = int(batch_indices.max().item() + 1)
    #     point_clouds = [xyz_coords[batch_indices == i] for i in range(batch_size)]
    #     point_clouds = [cloud.unsqueeze(0) for cloud in point_clouds]  # 각 포인트 클라우드에 배치 차원 추가
    #     # for i in range(len(point_clouds)):
    #     #     print(f"{i}@@@@@@@@@@@",point_clouds[i].shape)
    #     point_clouds_tensor = torch.cat(point_clouds, dim=0)  # 모든 배치를 하나의 텐서로 합침
    #     return point_clouds_tensor
    def converting_for_chamfer(self, coords):
        # 주어진 텐서
        # 배치 인덱스와 좌표 분리
        batch_indices = coords[:, 0]
        xyz_coords = coords[:, 1:]

        # 배치 인덱스를 기준으로 텐서 재구성
        batch_size = int(batch_indices.max().item() + 1)
        point_clouds = [xyz_coords[batch_indices == i] for i in range(batch_size)]
        
        # 모든 포인트 클라우드의 최대 크기 찾기
        max_size = max([cloud.size(0) for cloud in point_clouds])
        
        # 각 포인트 클라우드를 최대 크기에 맞게 패딩
        padded_point_clouds = []
        for cloud in point_clouds:
            padding_size = max_size - cloud.size(0)
            padded_cloud = torch.cat([cloud, torch.zeros(padding_size, 3).to(cloud.device)], dim=0)
            padded_point_clouds.append(padded_cloud.unsqueeze(0))
        
        point_clouds_tensor = torch.cat(padded_point_clouds, dim=0)  # 모든 배치를 하나의 텐서로 합침
        return point_clouds_tensor


    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
