from functools import partial

import torch.nn as nn
import torch
import numpy as np
from ...utils.spconv_utils import replace_feature, spconv, post_act_block, SparseBasicBlock
from ...datasets.kitti.point_upsampler.model.Network import MyNet
import MinkowskiEngine as ME

# class VoxelBackBone8x(nn.Module):
#     def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
#         super().__init__()
#         self.model_cfg = model_cfg
#         norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

#         self.sparse_shape = grid_size[::-1] + [1, 0, 0]

#         self.conv_input = spconv.SparseSequential(
#             spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
#             norm_fn(16),
#             nn.ReLU(),
#         )
#         self.conv_input2 = spconv.SparseSequential(
#             spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
#             norm_fn(16),
#             nn.ReLU(),
#         )
#         block = post_act_block

#         self.conv1 = spconv.SparseSequential(
#             block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
#         )

#         self.conv2 = spconv.SparseSequential(
#             # [1600, 1408, 41] <- [800, 704, 21]
#             block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
#             block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
#             block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
#         )

#         # SP fusion
#         self.r_conv2 = spconv.SparseSequential(
#             # [1600, 1408, 41] <- [800, 704, 21]
#             block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv_r2', conv_type='spconv'),
#             block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm_r2'),
#             block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm_r2'),
#         )


#         self.conv3 = spconv.SparseSequential(
#             # [800, 704, 21] <- [400, 352, 11]
#             block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
#         )

#         # SP fusion
#         self.r_conv3 = spconv.SparseSequential(
#             # [1600, 1408, 41] <- [800, 704, 21]
#             block(64, 64, 3, norm_fn=norm_fn, stride=4, padding=2, indice_key='spconv_r3', conv_type='spconv'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm_r3'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm_r3'),
#         )

#         self.conv4 = spconv.SparseSequential(
#             # [400, 352, 11] <- [200, 176, 5]
#             block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
#         )

#         last_pad = 0
#         last_pad = self.model_cfg.get('last_pad', last_pad)
#         self.conv_out = spconv.SparseSequential(
#             # [200, 150, 5] -> [200, 150, 2]
#             spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
#                                 bias=False, indice_key='spconv_down2'),
#             norm_fn(128),
#             nn.ReLU(),
#         )
#         self.num_point_features = 128
#         self.backbone_channels = {
#             'x_conv1': 16,
#             'x_conv2': 32,
#             'x_conv3': 64,
#             'x_conv4': 64
#         }
        
#         self.spi_r = MyNet().cuda()
        
#     def sp_concat(self, sparse1, sparse2):
#         feature = torch.cat((sparse1.features,sparse2.features),dim=0)
#         indice = torch.cat((sparse1.indices,sparse2.indices),dim=0)
#         sp_tensor = spconv.SparseConvTensor(
#             features=feature,
#             indices=indice.int(),
#             spatial_shape=sparse1.spatial_shape,
#             # spatial_shape=self.sparse_shape,
#             batch_size=sparse1.batch_size            
#         )
#         return sp_tensor
    
#     def to_sparse_tensor(self, feature, voxel_coords, batch_size):
#         to_sp_tensor = spconv.SparseConvTensor(
#             features=feature,
#             indices=voxel_coords.int(),
#             spatial_shape=self.sparse_shape,
#             batch_size=batch_size            
#         )
#         return to_sp_tensor
    
#     def forward(self, batch_dict):
#         """
#         Args:
#             batch_dict:
#                 batch_size: int
#                 vfe_features: (num_voxels, C)
#                 voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
#         Returns:
#             batch_dict:
#                 encoded_spconv_tensor: sparse tensor
#         """
#         voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
#         batch_size = batch_dict['batch_size']
#         input_sp_tensor = spconv.SparseConvTensor(
#             features=voxel_features,
#             indices=voxel_coords.int(),
#             spatial_shape=self.sparse_shape,
#             batch_size=batch_size
#         )
        
        
#         # sphere_voxel_features = voxel_coords * 50
#         test_feats = torch.from_numpy(np.vstack(np.expand_dims(np.ones(voxel_coords.shape[0]), 1))).float().cuda()
#         # sphere_voxel_features = ME.utils.batched_coordinates([sphere_voxel_features[:,:3]]).cuda()
#         me_voxel_coords = torch.cat([voxel_coords[:,0:1],voxel_coords.flip(dims=[-1])[:,:3]],dim=-1).int()

#         sphere_x = ME.SparseTensor(features=test_feats, coordinates=me_voxel_coords, device=torch.device('cuda'))
#         # sphere_x = ME.SparseTensor(features=test_feats, coordinates=voxel_features.int()[:,1:], device=torch.device('cuda'))
#         s0, s1, s2 = self.spi_r(sphere_x, coords_T=me_voxel_coords, device=torch.device('cuda'), prune=False)        
#         #, _, _, _ 
#         s0_C = torch.cat([s0.C[:, 0:1].long(), torch.flip(s0.C[:,1:4], dims=[-1])], dim=-1)
#         s1_C = torch.cat([s1.C[:, 0:1].long(), torch.flip(s1.C[:,1:4], dims=[-1])], dim=-1)
#         s2_C = torch.cat([s2.C[:, 0:1].long(), torch.flip(s2.C[:,1:4], dims=[-1])], dim=-1)        
#         s0 = self.to_sparse_tensor(s0.F,s0_C,batch_size)#32 # 4
#         s1 = self.to_sparse_tensor(s1.F,s1_C,batch_size)#64 # 32
#         s2 = self.to_sparse_tensor(s2.F,s2_C,batch_size)    # 32
#         # s2 = self.to_sparse_tensor(s2.F,voxel_coords,batch_size)#128   s2.F/50
        
#         # SP fusion
#         batch_dict['recon_points'] = s0
        
        
#         x = self.conv_input(input_sp_tensor)
#         # print(s0.features.shape)
#         # print(input_sp_tensor.features.shape)

#         x_conv1 = self.conv1(x)
#         s0 = self.conv_input2(s0)
#         x_conv1 = self.sp_concat(x_conv1, s0)
        
#         x_conv2 = self.conv2(x_conv1)
#         # SP fusion
#         s1 = self.r_conv2(s1)
#         x_conv2 = self.sp_concat(x_conv2,s1) #addition
#         x_conv3 = self.conv3(x_conv2)
#         x_conv4 = self.conv4(x_conv3)
#         # SP fusion
#         s2 = self.r_conv3(s2)
#         x_conv4 = self.sp_concat(x_conv4,s2) #addition
#         # for detection head
#         # [200, 176, 5] -> [200, 176, 2]
#         out = self.conv_out(x_conv4)

#         batch_dict.update({
#             'encoded_spconv_tensor': out,
#             'encoded_spconv_tensor_stride': 8
#         })
#         batch_dict.update({
#             'multi_scale_3d_features': {
#                 'x_conv1': x_conv1,
#                 'x_conv2': x_conv2,
#                 'x_conv3': x_conv3,
#                 'x_conv4': x_conv4,
#             }
#         })
#         batch_dict.update({
#             'multi_scale_3d_strides': {
#                 'x_conv1': 1,
#                 'x_conv2': 2,
#                 'x_conv3': 4,
#                 'x_conv4': 8,
#             }
#         })

#         return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict






class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        self.conv_input2 = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.r_conv1 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(64, 32, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv_r2', conv_type='spconv'),
            block(32, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm_r2'),
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm_r2'),
        )
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )
        # SP fusion
        self.r_conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv_r2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm_r2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm_r2'),
        )
        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        # SP fusion
        self.r_conv3 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(64, 64, 3, norm_fn=norm_fn, stride=4, padding=2, indice_key='spconv_r3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm_r3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm_r3'),
        )
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        self.spi_r = MyNet().cuda()
    def sp_concat(self, sparse1, sparse2):
        feature = torch.cat((sparse1.features,sparse2.features),dim=0)
        indice = torch.cat((sparse1.indices,sparse2.indices),dim=0)
        sp_tensor = spconv.SparseConvTensor(
            features=feature,
            indices=indice.int(),
            spatial_shape=sparse1.spatial_shape,
            # spatial_shape=self.sparse_shape,
            batch_size=sparse1.batch_size
        )
        return sp_tensor
    def to_sparse_tensor(self, feature, voxel_coords, batch_size):
        to_sp_tensor = spconv.SparseConvTensor(
            features=feature,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        return to_sp_tensor
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        # sphere_voxel_features = voxel_coords * 50
        test_feats = torch.from_numpy(np.vstack(np.expand_dims(np.ones(voxel_coords.shape[0]), 1))).float().cuda()
        # sphere_voxel_features = ME.utils.batched_coordinates([sphere_voxel_features[:,:3]]).cuda()
        me_voxel_coords = torch.cat([voxel_coords[:,0:1],voxel_coords.flip(dims=[-1])[:,:3]],dim=-1).int()
        sphere_x = ME.SparseTensor(features=test_feats, coordinates=me_voxel_coords, device=torch.device('cuda'))
        # sphere_x = ME.SparseTensor(features=test_feats, coordinates=voxel_features.int()[:,1:], device=torch.device('cuda'))
        s0, s1, s2 = self.spi_r(sphere_x, coords_T=me_voxel_coords, device=torch.device('cuda'), prune=False)
        #, _, _, _
        s0_C = torch.cat([s0.C[:, 0:1].long(), torch.flip(s0.C[:,1:4], dims=[-1])], dim=-1)
        s1_C = torch.cat([s1.C[:, 0:1].long(), torch.flip(s1.C[:,1:4], dims=[-1])], dim=-1)
        s2_C = torch.cat([s2.C[:, 0:1].long(), torch.flip(s2.C[:,1:4], dims=[-1])], dim=-1)
        s0 = self.to_sparse_tensor(s0.F,s0_C,batch_size)#32 # 1
        s1 = self.to_sparse_tensor(s1.F,s1_C,batch_size)#64 # 32
        s2 = self.to_sparse_tensor(s2.F,s2_C,batch_size)#64 # 32
        # s2 = self.to_sparse_tensor(s2.F,voxel_coords,batch_size)#128   s2.F/50
        # SP fusion
        batch_dict['recon_points'] = s0
        x = self.conv_input(input_sp_tensor)
        s0 = self.conv_input2(s0)
        x_conv1 = self.conv1(x)
        x_conv1 = self.sp_concat(x_conv1,s0)
        x_conv2 = self.conv2(x_conv1)
        s1 = self.r_conv2(s1)
        x_conv2 = self.sp_concat(x_conv2,s1) #addition
        x_conv3 = self.conv3(x_conv2)
        s2 = self.r_conv3(s2)
        x_conv3 = self.sp_concat(x_conv3,s2) #addition
        x_conv4 = self.conv4(x_conv3)
        # SP fusion
        # s1 = self.r_conv3(s1)
        # x_conv4 = self.sp_concat(x_conv4,s1) #addition
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # test = self.sp_concat(x_conv1, x_conv4)
        out = self.conv_out(x_conv4)
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        return batch_dict
