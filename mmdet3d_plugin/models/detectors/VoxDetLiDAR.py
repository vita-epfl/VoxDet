import torch
import torch.nn as nn
import os
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS, HEADS
from mmdet3d.models import builder
import torch.nn.functional as F

def run_length_positive(t, dim):
    shape = t.shape
    L = shape[dim]
    out = torch.empty_like(t, dtype=torch.int32)
    
    idx_last = [slice(None)] * len(shape)
    idx_last[dim] = -1
    out[tuple(idx_last)] = torch.tensor(1, dtype=torch.int32, device=t.device)
    
    for i in range(L - 2, -1, -1):
        idx = [slice(None)] * len(shape)
        idx[dim] = i
        idx_next = [slice(None)] * len(shape)
        idx_next[dim] = i + 1
        
        current = t[tuple(idx)]
        nxt = t[tuple(idx_next)]
        
        cond = (current == nxt)
        out_next = out[tuple(idx_next)]
        
        val = torch.where(cond, out_next + 1, torch.tensor(1, dtype=torch.int32, device=t.device))
        out[tuple(idx)] = val
        
    return out

def run_length_along_dim(t, dim, direction):
    if direction == 'positive':
        return run_length_positive(t, dim)
    else:
        t_flip = torch.flip(t, dims=(dim,))
        out_flip = run_length_positive(t_flip, dim)
        return torch.flip(out_flip, dims=(dim,))

def compute_all_direction_distances(gt_occ):
    B, X, Y, Z = gt_occ.shape

    dist_x_pos = run_length_along_dim(gt_occ, 1, 'positive')
    dist_x_neg = run_length_along_dim(gt_occ, 1, 'negative')
    dist_y_pos = run_length_along_dim(gt_occ, 2, 'positive')
    dist_y_neg = run_length_along_dim(gt_occ, 2, 'negative')
    dist_z_pos = run_length_along_dim(gt_occ, 3, 'positive')
    dist_z_neg = run_length_along_dim(gt_occ, 3, 'negative')
    
    distances = torch.stack([dist_x_pos, dist_x_neg, dist_y_pos, dist_y_neg, dist_z_pos, dist_z_neg], dim=1)
    return distances


@DETECTORS.register_module()
class VoxDetLiDAR(BaseModule):

    def __init__(
        self,
        lidar_tokenizer=None,
        lidar_backbone=None,
        lidar_neck=None,
        tpv_generator=None,
        tpv_aggregator=None,
        pts_bbox_head=None,
        tpv_aggregator_reg=None,
        pts_bbox_head_aux=None,

        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        use_gt_refine=False,
        car_scale_filter_max=None,
        car_scale_filter_min=None,
        global_scale_filter_min=None,
        **kwargs,
    ):

        super().__init__()
        self.use_gt_refine = use_gt_refine
        self.car_scale_filter_max = car_scale_filter_max
        self.car_scale_filter_min = car_scale_filter_min
        self.global_scale_filter_min = global_scale_filter_min
        self.lidar_tokenizer = builder.build_backbone(lidar_tokenizer)
        self.voxel_backbone = builder.build_backbone(lidar_backbone)
        self.voxel_neck = builder.build_neck(lidar_neck)

        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        

        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if pts_bbox_head_aux is not None:
            self.pts_bbox_head_aux = builder.build_head(pts_bbox_head_aux)
        self.fp16_enabled = False

    def extract_lidar_feat(self, points, grid_ind):
        """Extract features of points."""
        x_3d = self.lidar_tokenizer(points, grid_ind)
        x_list = self.voxel_backbone(x_3d)
        output = self.voxel_neck(x_list)
        output = output[0]

        return output
    
    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)

        return x

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)

    def forward_train(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']



        with torch.no_grad():
            gt_occ_ = gt_occ.clone()  # TODO no need to clone
            gt_offset = compute_all_direction_distances(gt_occ_)


        if self.use_gt_refine:

            len_x = gt_offset[:,0,:,:,:] + gt_offset[:,1,:,:,:] # X axis b x y z
            len_y = gt_offset[:,2,:,:,:] + gt_offset[:,3,:,:,:]
            len_z = gt_offset[:,4,:,:,:] + gt_offset[:,5,:,:,:]
            if self.car_scale_filter_max is not None:

                x_mask_max = len_x > self.car_scale_filter_max[0]
                y_mask_max = len_y > self.car_scale_filter_max[1]
                z_mask_max = len_z > self.car_scale_filter_max[2] 

                cls_mask = (gt_occ_ == 1) # for car
                car_mask_max =(x_mask_max | y_mask_max | z_mask_max).bool() & cls_mask
                gt_occ_[car_mask_max] = 255
                

            if self.car_scale_filter_min is not None:
                x_mask_min = len_x < self.car_scale_filter_min[0]
                y_mask_min = len_y < self.car_scale_filter_min[1]
                z_mask_min = len_z < self.car_scale_filter_min[2]
                
                car_mask_min = (x_mask_min & y_mask_min & z_mask_min).bool()  & cls_mask # all thre axes are one
                gt_occ_[car_mask_min] = 255


            if self.global_scale_filter_min is not None:

                x_mask_min_g = len_x < self.global_scale_filter_min[0]
                y_mask_min_g  = len_x < self.global_scale_filter_min[1]
                z_mask_min_g  = len_x < self.global_scale_filter_min[2]

                mask_min_g = x_mask_min_g & y_mask_min_g & z_mask_min_g # isolated points
                gt_occ_[mask_min_g] = 255

            gt_occ = gt_occ_
        # lidar encoder
        lidar_voxel_feats = self.extract_lidar_feat(points=points, grid_ind=grid_ind)
        voxel_feats_enc = self.occ_encoder(lidar_voxel_feats)

        if type(voxel_feats_enc) is tuple:
            voxel_feats_enc = list(voxel_feats_enc)

        output = self.pts_bbox_head(voxel_feats=[[voxel_feats_enc[0][0]], [voxel_feats_enc[1][0]]], img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # loss
        losses = dict()
        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
            output_bbox=output['output_bbox'],
            img_metas=img_metas,
            gt_offset=gt_offset
        )

        losses.update(losses_occupancy)
        if hasattr(self, 'pts_bbox_head_aux'):
            # auxiliary head
            output_aux = self.pts_bbox_head_aux(voxel_feats=[lidar_voxel_feats], img_metas=img_metas, img_feats=None, gt_occ=gt_occ)
            losses_occupancy_aux = self.pts_bbox_head_aux.loss(
                            output_voxels=output_aux['output_voxels'],
                            target_voxels=gt_occ,
                        )
            loss_dict = {}
            for key in losses_occupancy_aux.keys():
                loss_dict[key.replace('loss', 'loss_aux')] = losses_occupancy_aux[key]
            losses.update(loss_dict)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {'losses': losses, 'pred': pred, 'gt_occ': gt_occ}
        return train_output

    def forward_test(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None

        # lidar encoder
        lidar_voxel_feats = self.extract_lidar_feat(points=points, grid_ind=grid_ind)

        voxel_feats_enc = self.occ_encoder(lidar_voxel_feats)

        if type(voxel_feats_enc) is tuple:
            voxel_feats_enc = list(voxel_feats_enc)

        output = self.pts_bbox_head(voxel_feats=[[voxel_feats_enc[0][0]], [voxel_feats_enc[1][0]]], img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output
