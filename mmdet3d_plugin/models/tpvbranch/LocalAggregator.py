from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F

@BACKBONES.register_module()
class LocalAggregator(BaseModule):
    def __init__(
        self,
        local_encoder_backbone=None,
        local_encoder_neck=None,
        return_multilevel_feature=False,
    ):
        super().__init__()
        self.local_encoder_backbone = builder.build_backbone(local_encoder_backbone)
        self.local_encoder_neck = builder.build_neck(local_encoder_neck)
        self.return_multilevel_feature = return_multilevel_feature
    
    def forward(self, x):
        x_list = self.local_encoder_backbone(x)
        output = self.local_encoder_neck(x_list)
        if not self.return_multilevel_feature:
            output = output[0]

        return output
    


@BACKBONES.register_module()
class VoxelAggregatorDual(BaseModule):
    def __init__(
        self,
        local_encoder_backbone=None,
        local_encoder_neck=None,
        return_multilevel_feature=False,
    ):
        super().__init__()
        self.local_encoder_backbone = builder.build_backbone(local_encoder_backbone)
        self.local_encoder_neck = builder.build_neck(local_encoder_neck)
        self.return_multilevel_feature = return_multilevel_feature
    
    def forward(self, x):
        x_list = self.local_encoder_backbone(x)
        output = self.local_encoder_neck(x_list)

        return output