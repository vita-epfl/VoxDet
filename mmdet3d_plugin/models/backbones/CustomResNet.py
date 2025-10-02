import timm
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import BACKBONES
import torch.utils.model_zoo as model_zoo
from torch.nn.modules.batchnorm import _BatchNorm

@BACKBONES.register_module()
class CustomResNet(BaseModule):
    def __init__(
        self,
        arch='resnet50d.a1_in1k',
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        drop_path_rate=0.5,
        track_running_stats=True,
        init_cfg=None,
        **kwargs
    ):
        super().__init__()
        if pretrained is not None:
            model = timm.create_model(arch, pretrained=True, drop_path_rate=drop_path_rate)
        else:
            model = timm.create_model(arch, pretrained=False)
       
        for m in model.modules():
            if isinstance(m, _BatchNorm):
                m.track_running_stats = track_running_stats

        self.conv1 = model.conv1
        self.norm1 = model.bn1
        self.relu = model.act1

        self.maxpool = model.maxpool

        assert max(out_indices) < num_stages
        self.out_indices = out_indices
        self.res_layers = nn.ModuleList()

        self.res_layers.append(model.layer1)
        self.res_layers.append(model.layer2)
        self.res_layers.append(model.layer3)
        self.res_layers.append(model.layer4)

        self.res_layers = self.res_layers[:num_stages]

        del model
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class CustomResNetGN(BaseModule):
    def __init__(
        self,
        arch='resnet50d.a1_in1k',
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        drop_path_rate=0.5,
        init_cfg=None,
        **kwargs
    ):
        super().__init__()
        if pretrained is not None:
            model = timm.create_model(
                arch, 
                pretrained=True, 
                pretrained_cfg_overlay=dict(file=pretrained), 
                drop_path_rate=drop_path_rate
            )
        else:
            model = timm.create_model(arch, pretrained=False)

        self.replace_bn_with_gn(model, num_groups=32)
        self.conv1 = model.conv1
        self.norm1 = model.bn1 
        self.relu = model.act1
        self.maxpool = model.maxpool

        assert max(out_indices) < num_stages
        self.out_indices = out_indices
        self.res_layers = nn.ModuleList()

        self.res_layers.append(model.layer1)
        self.res_layers.append(model.layer2)
        self.res_layers.append(model.layer3)
        self.res_layers.append(model.layer4)

        self.res_layers = self.res_layers[:num_stages]
        del model

    def replace_bn_with_gn(self, module, num_groups=32):

        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                setattr(module, name, nn.GroupNorm(num_groups, num_channels))
            else:
                self.replace_bn_with_gn(child, num_groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)