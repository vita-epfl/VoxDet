import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
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
class VoxDet(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_view_transformer,
        proposal_layer,
        VoxFormer_head,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        pts_bbox_head=None,
        pts_bbox_head_aux=None,
        depth_loss=False,
        train_cfg=None,
        test_cfg=None,
        use_gt_refine=False,
        car_scale_filter_max=None,
        car_scale_filter_min=None,
        global_scale_filter_min=None,
    ):
        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.global_scale_filter_min = global_scale_filter_min
        self.depth_net = builder.build_neck(depth_net)
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)
        self.use_gt_refine = use_gt_refine
        self.car_scale_filter_max = car_scale_filter_max
        self.car_scale_filter_min = car_scale_filter_min

        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        
        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if pts_bbox_head_aux is not None:
            self.pts_bbox_head_aux = builder.build_head(pts_bbox_head_aux)
            
        self.depth_loss = depth_loss

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    def extract_img_feat(self, img_inputs, img_metas):
        img_enc_feats = self.image_encoder(img_inputs[0]) # torch.Size([1, 1, 640, 48, 160])
        B,N,C,H,W =img_inputs[0].size()

        mlp_input = self.depth_net.get_mlp_input(*img_inputs[1:7])
        context, depth = self.depth_net([img_enc_feats] + img_inputs[1:7] + [mlp_input], img_metas)
        #1, 1, 128, 48, 160
        if hasattr(self, 'img_view_transformer'):
            coarse_queries = self.img_view_transformer(context, depth, img_inputs[1:7]) # V_QA
        else:
            coarse_queries = None

        proposal = self.proposal_layer(img_inputs[1:7], img_metas)
        # torch.Size([1, 1, 128, 128, 16])
        # torch.Size([1, 1, 128, 48, 160])
        if B > 1:
            x_list = []
            for i in range(B):
                camera_paras = img_inputs[1:7]
                camera_paras_batch = []
                
                for j in range(6):
                    camera_paras_batch.append(camera_paras[j][i:i+1])

                x = self.VoxFormer_head(
                    [context[i:i+1]],
                    proposal[i:i+1],
                    cam_params=camera_paras_batch,
                    lss_volume=coarse_queries[i:i+1],
                    img_metas=img_metas,
                    mlvl_dpt_dists=[depth[i:i+1].unsqueeze(1)]
                )
                x_list.append(x)
            x = torch.cat(x_list, dim=0)
        else:
            x = self.VoxFormer_head(
                [context],
                proposal,
                cam_params=img_inputs[1:7],
                lss_volume=coarse_queries,
                img_metas=img_metas,
                mlvl_dpt_dists=[depth.unsqueeze(1)]
            )

        # ([1, 1, 128, 128, 16])
        # print(x.shape)
        # torch.Size([1, 128, 128, 128, 16])
        # torch.Size([1, 112, 48, 160])
        return x, depth, proposal
    
    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)

        return x

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, depth, proposal = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        # if len(voxel_feats_enc) > 1:
        #     voxel_feats_enc = [voxel_feats_enc[0]]
        if type(voxel_feats_enc) is tuple:
            voxel_feats_enc = list(voxel_feats_enc)

        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        with torch.no_grad():
            gt_occ_ = gt_occ.clone() 
            gt_offset = compute_all_direction_distances(gt_occ_)


        if self.use_gt_refine:
            # Remove super long cars as mentioned in the Appendix
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

                # gt_occ_[] = 255
                # gt_occ_[len_y < self.global_scale_filter_min[1]] = 255
                # gt_occ_[len_z < self.global_scale_filter_min[2]] = 255

            gt_occ = gt_occ_


        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ,
            gt_offset=gt_offset,
        )

        losses = dict()
        if hasattr(self, 'pts_bbox_head_aux'):

            if type(img_voxel_feats) is not list:
                img_voxel_feats = [img_voxel_feats]
            output_aux = self.pts_bbox_head_aux(
                voxel_feats=img_voxel_feats,
                img_metas=img_metas,
                img_feats=None,
                gt_occ=gt_occ
            )
            if 'output_bbox' in output_aux.keys():
                losses_occupancy_aux = self.pts_bbox_head_aux.loss(
                    output_voxels=output_aux['output_voxels'],
                    target_voxels=gt_occ,
                    output_bbox=output_aux['output_bbox'],
                    )
            else:
                losses_occupancy_aux = self.pts_bbox_head_aux.loss(
                    output_voxels=output_aux['output_voxels'],
                    target_voxels=gt_occ,
                )
            
            loss_dict = {}
            for key in losses_occupancy_aux.keys():
                loss_dict[key.replace('loss', 'loss_aux')] = losses_occupancy_aux[key]
            losses.update(loss_dict)


        if self.depth_loss and depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(data_dict['img_metas']['gt_depths'], depth)

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
            output_bbox=output['output_bbox'],
            img_metas=img_metas,
            gt_offset=gt_offset,
        )

        losses.update(losses_occupancy)
        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ
        }

        return train_output
    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ'] if 'gt_occ' in data_dict.keys() else None

        img_voxel_feats, depth, proposal = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        if type(voxel_feats_enc) is tuple:
            voxel_feats_enc = list(voxel_feats_enc)

        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
                
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )
        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ
        }

        return test_output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)
        


