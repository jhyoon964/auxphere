from .detector3d_template import Detector3DTemplate
import torch
from torchvision.transforms import Resize
from .perceptual_loss import VGGPerceptualLoss as p_loss

class GraphRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, logger):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger)
        self.module_list = self.build_networks()
        if self.model_cfg.get('FREEZE_LAYERS', None) is not None:
            self.freeze(self.model_cfg.FREEZE_LAYERS)
        self.loss_recon = torch.nn.L1Loss()
        self.p_loss = p_loss().cuda()
        
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            h, w = int(batch_dict['gt_image'].shape[2]/4), int(batch_dict['gt_image'].shape[3]/4)
            torch_resize = Resize((h,w))    
            loss_1 = self.loss_recon(batch_dict['recon_image'],torch_resize(batch_dict['gt_image'])) * 0.1
            loss_1 += self.p_loss(batch_dict['recon_image'],torch_resize(batch_dict['gt_image'])) * 0.05
            loss += loss_1*1
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict, tb_dict = {}, {}
        loss = 0
        if self.model_cfg.get('FREEZE_LAYERS', None) is None:
            if self.dense_head is not None:
                loss_rpn, tb_dict = self.dense_head.get_loss(tb_dict)
            else:
                loss_rpn, tb_dict = self.point_head.get_loss(tb_dict)
            loss += loss_rpn

        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss += loss_rcnn
        
        return loss, tb_dict, disp_dict

# ############################################
# from .detector3d_template import Detector3DTemplate
# # from chamferdist import ChamferDistance
# import torch
# from pytorch3d.loss import chamfer_distance
# class GraphRCNN(Detector3DTemplate):
#     def __init__(self, model_cfg, num_class, dataset, logger):
#         super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger)
#         self.module_list = self.build_networks()
#         if self.model_cfg.get('FREEZE_LAYERS', None) is not None:
#             self.freeze(self.model_cfg.FREEZE_LAYERS)
#         # self.chamfer_distance = ChamferDistance()
#     def forward(self, batch_dict):
#         for cur_module in self.module_list:
#             batch_dict = cur_module(batch_dict)

#         if self.training:
#             loss, tb_dict, disp_dict = self.get_training_loss()
            
#             loss_chamfer = chamfer_distance(self.converting_for_chamfer(batch_dict['recon_points'].indices.float()), self.converting_for_chamfer(batch_dict['target_gt_coords'].float()))
#             # loss_chamfer = chamfer_distance(batch_dict['recon_points'].indices, batch_dict['target_gt_coords'])
#             # print(loss_chamfer)[0]

#             loss += loss_chamfer[0] * 0.0001
            
#             ret_dict = {
#                 'loss': loss
#             }
#             return ret_dict, tb_dict, disp_dict
#         else:
#             pred_dicts, recall_dicts = self.post_processing(batch_dict)
#             return pred_dicts, recall_dicts

#     # def chamfer_distance(self, p1, p2):
        
#     #     p1 = torch.tensor(p1, dtype=torch.float, requires_grad=True)
#     #     p2 = torch.tensor(p2, dtype=torch.float)
        
#     #     norm = torch.abs((torch.norm(p1-p2)))
        
#     #     norm_tensor = torch.tensor(norm, requires_grad=True).to('cuda')
        
#     #     return norm_tensor

#     def converting_for_chamfer(self, coords):
#         # 주어진 텐서
#         # 배치 인덱스와 좌표 분리
#         batch_indices = coords[:, 0]
#         xyz_coords = coords[:, 1:]

#         # 배치 인덱스를 기준으로 텐서 재구성
#         batch_size = int(batch_indices.max().item() + 1)
#         point_clouds = [xyz_coords[batch_indices == i] for i in range(batch_size)]
#         point_clouds = [cloud.unsqueeze(0) for cloud in point_clouds]  # 각 포인트 클라우드에 배치 차원 추가

        
#         point_clouds_tensor = torch.cat(point_clouds, dim=0)  # 모든 배치를 하나의 텐서로 합침
#         return point_clouds_tensor


#     def get_training_loss(self):
#         disp_dict, tb_dict = {}, {}
#         loss = 0
#         if self.model_cfg.get('FREEZE_LAYERS', None) is None:
#             if self.dense_head is not None:
#                 loss_rpn, tb_dict = self.dense_head.get_loss(tb_dict)
#             else:
#                 loss_rpn, tb_dict = self.point_head.get_loss(tb_dict)
#             loss += loss_rpn

#         loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
#         loss += loss_rcnn
        
#         return loss, tb_dict, disp_dict
