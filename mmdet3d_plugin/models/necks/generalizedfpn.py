import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.models import NECKS
from timm.layers import DropPath

@NECKS.register_module()
class GeneralizedLSSFPN(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN2d"),
        act_cfg=dict(type="ReLU"),
        use_resblock=False,
        drop_path_rate=0.3,
        upsample_cfg=dict(mode="bilinear", align_corners=True),
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i]
                + (
                    in_channels[i + 1]
                    if i == self.backbone_end_level - 1
                    else out_channels
                ),
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            if use_resblock:
                fpn_conv = BasicBlock3D(
                    out_channels,
                    out_channels,
                    stride=1,
                    downsample=ConvModule(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    drop_path_rate=drop_path_rate)
            else:
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    # @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)
    
@NECKS.register_module()
class GeneralizedLSSFPNDual(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN2d"),
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(mode="bilinear", align_corners=True),
        drop_path_rate=0.3,
        use_resblock=False, 
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list), "in_channels must be a list"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.drop_path_rate = drop_path_rate

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs_branch1 = nn.ModuleList()
        self.fpn_convs_cls = nn.ModuleList()
        self.lateral_convs_branch2 = nn.ModuleList()
        self.fpn_convs_bbox = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            in_c = in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels)

            lateral_conv1 = ConvModule(
                in_c,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            if use_resblock:
                fpn_conv1 = BasicBlock3D(
                    out_channels,
                    out_channels,
                    stride=1,
                    downsample=ConvModule(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    drop_path_rate=drop_path_rate)
            else:
                fpn_conv1 = ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
            self.lateral_convs_branch1.append(lateral_conv1)
            self.fpn_convs_cls.append(fpn_conv1)

            lateral_conv2 = ConvModule(
                in_c,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            if use_resblock:
                fpn_conv2 = BasicBlock3D(
                    out_channels,
                    out_channels,
                    stride=1,
                    downsample=ConvModule(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    drop_path_rate=drop_path_rate)
            else:
                fpn_conv2 = ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
            self.lateral_convs_branch2.append(lateral_conv2)
            self.fpn_convs_bbox.append(fpn_conv2)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals_cls = [inputs[i] for i in range(self.start_level, self.backbone_end_level + 1)]
        laterals_box = [inputs[i] for i in range(self.start_level, self.backbone_end_level + 1)]

        used_backbone_levels = len(laterals_cls) - 1

        for i in range(used_backbone_levels - 1, -1, -1):
            x1 = F.interpolate(
                laterals_cls[i + 1],
                size=laterals_cls[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals_cls[i] = torch.cat([laterals_cls[i], x1], dim=1)
            laterals_cls[i] = self.lateral_convs_branch1[i](laterals_cls[i])
            laterals_cls[i] = self.fpn_convs_cls[i](laterals_cls[i])

        outs_branch1 = [laterals_cls[i] for i in range(used_backbone_levels)]
        
        for i in range(used_backbone_levels - 1, -1, -1):
            x2 = F.interpolate(
                laterals_box[i + 1],
                size=laterals_box[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals_box[i] = torch.cat([laterals_box[i], x2], dim=1)
            laterals_box[i] = self.lateral_convs_branch2[i](laterals_box[i])
            laterals_box[i] = self.fpn_convs_bbox[i](laterals_box[i])

        outs_branch2 = [laterals_box[i] for i in range(used_backbone_levels)]
        return (outs_branch1, outs_branch2)
    

class BasicBlock3D(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None, drop_path_rate=0.0):
        super(BasicBlock3D, self).__init__()
        self.conv1 = ConvModule(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.conv2 = ConvModule(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop_path(out)
        out = out + identity
        return self.relu(out)

@NECKS.register_module()
class GeneralizedLSSFPNDualFuse(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN2d"),
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(mode="bilinear", align_corners=True),
        drop_path_rate=0.3,
        use_reg_enf_cls=False,
        enhance_pos=1,
        use_resblock=False,
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list), "in_channels must be a list"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.drop_path_rate = drop_path_rate
        self.use_reg_enf_cls = use_reg_enf_cls
        self.enhance_pos = enhance_pos

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs_branch1 = nn.ModuleList()
        self.fpn_convs_cls = nn.ModuleList()
        self.lateral_convs_branch2 = nn.ModuleList()
        self.fpn_convs_bbox = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            in_c = in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels)
            
            lateral_conv1 = ConvModule(
                in_c,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            if use_resblock:
                fpn_conv1 = BasicBlock3D(
                    out_channels,
                    out_channels,
                    stride=1,
                    downsample=ConvModule(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    drop_path_rate=drop_path_rate)
            else:
                fpn_conv1 = ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
            self.lateral_convs_branch1.append(lateral_conv1)
            self.fpn_convs_cls.append(fpn_conv1)

            lateral_conv2 = ConvModule(
                in_c,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            if use_resblock:
                fpn_conv2 = BasicBlock3D(
                    out_channels,
                    out_channels,
                    stride=1,
                    downsample=ConvModule(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    drop_path_rate=drop_path_rate)
            else:
                fpn_conv2 = ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
            self.lateral_convs_branch2.append(lateral_conv2)
            self.fpn_convs_bbox.append(fpn_conv2)

        if self.use_reg_enf_cls:
            self.mapping_convs = nn.ModuleList()
            for i in range(self.start_level, self.backbone_end_level):
                mapping_conv = ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
                self.mapping_convs.append(mapping_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals_cls = [inputs[i] for i in range(self.start_level, self.backbone_end_level + 1)]
        laterals_box = [inputs[i] for i in range(self.start_level, self.backbone_end_level + 1)]
        used_backbone_levels = len(laterals_cls) - 1

        for i in range(used_backbone_levels - 1, -1, -1):
            x1 = F.interpolate(
                laterals_cls[i + 1],
                size=laterals_cls[i].shape[2:],
                **self.upsample_cfg,
            )
            x2 = F.interpolate(
                laterals_box[i + 1],
                size=laterals_box[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals_box[i] = torch.cat([laterals_box[i], x2], dim=1)
            laterals_box[i] = self.lateral_convs_branch2[i](laterals_box[i])
            laterals_box[i] = self.fpn_convs_bbox[i](laterals_box[i])

            if self.use_reg_enf_cls & (self.enhance_pos == 0):
                laterals_cls[i] = laterals_cls[i] + self.mapping_convs[i](laterals_box[i])

            laterals_cls[i] = torch.cat([laterals_cls[i], x1], dim=1)
            laterals_cls[i] = self.lateral_convs_branch1[i](laterals_cls[i])

            if self.use_reg_enf_cls & (self.enhance_pos == 1):
                laterals_cls[i] = laterals_cls[i] + self.mapping_convs[i](laterals_box[i])
            laterals_cls[i] = self.fpn_convs_cls[i](laterals_cls[i])

        outs_branch1 = [laterals_cls[i] for i in range(used_backbone_levels)]
        outs_branch2 = [laterals_box[i] for i in range(used_backbone_levels)]
        return (outs_branch1, outs_branch2)
    






@NECKS.register_module()
class GeneralizedLSSFPNDualFuseDilated(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN2d"),
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(mode="bilinear", align_corners=True),
        drop_path_rate=0.3,
        use_reg_enf_cls=False,
        enhance_pos=0,
        use_resblock=False,
        dilation=1,

    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list), "in_channels must be a list"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.drop_path_rate = drop_path_rate
        self.use_reg_enf_cls = use_reg_enf_cls
        self.enhance_pos = enhance_pos

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs_branch1 = nn.ModuleList()
        self.fpn_convs_cls = nn.ModuleList()
        self.lateral_convs_branch2 = nn.ModuleList()
        self.fpn_convs_bbox = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            in_c = in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels)
            
            lateral_conv1 = ConvModule(
                in_c,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )

            fpn_conv1 = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )
            self.lateral_convs_branch1.append(lateral_conv1)
            self.fpn_convs_cls.append(fpn_conv1)

            kernel_size = 1
            stride=1
            lateral_conv2 = ConvModule(
                in_c,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
                dilation = dilation,
                padding = ((kernel_size - 1) * dilation + 1 - stride) // 2,
            )
            kernel_size = 3
            stride=1
            fpn_conv2 = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=dilation if dilation > 1 else 1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
                dilation = ((kernel_size - 1) * dilation + 1 - stride) // 2,
            )
            self.lateral_convs_branch2.append(lateral_conv2)
            self.fpn_convs_bbox.append(fpn_conv2)

        if self.use_reg_enf_cls:
            self.mapping_convs = nn.ModuleList()
            for i in range(self.start_level, self.backbone_end_level):
                mapping_conv = ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
                self.mapping_convs.append(mapping_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals_cls = [inputs[i] for i in range(self.start_level, self.backbone_end_level + 1)]
        laterals_box = [inputs[i] for i in range(self.start_level, self.backbone_end_level + 1)]
        used_backbone_levels = len(laterals_cls) - 1

        for i in range(used_backbone_levels - 1, -1, -1):
            x1 = F.interpolate(
                laterals_cls[i + 1],
                size=laterals_cls[i].shape[2:],
                **self.upsample_cfg,
            )
            x2 = F.interpolate(
                laterals_box[i + 1],
                size=laterals_box[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals_box[i] = torch.cat([laterals_box[i], x2], dim=1)
            laterals_box[i] = self.lateral_convs_branch2[i](laterals_box[i]) # reduce the channel
            laterals_box[i] = self.fpn_convs_bbox[i](laterals_box[i])

            if self.use_reg_enf_cls and self.enhance_pos == 0:
                laterals_cls[i] = laterals_cls[i] + self.mapping_convs[i](laterals_box[i])

            laterals_cls[i] = torch.cat([laterals_cls[i], x1], dim=1)
            laterals_cls[i] = self.lateral_convs_branch1[i](laterals_cls[i])


            if self.use_reg_enf_cls and self.enhance_pos == 1:
                laterals_cls[i] = laterals_cls[i] + self.mapping_convs[i](laterals_box[i])

            laterals_cls[i] = self.fpn_convs_cls[i](laterals_cls[i])

        outs_branch1 = [laterals_cls[i] for i in range(used_backbone_levels)]
        outs_branch2 = [laterals_box[i] for i in range(used_backbone_levels)]
        return (outs_branch1, outs_branch2)
    



@NECKS.register_module()
class GeneralizedLSSFPNDualFuseTPV(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN2d"),
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(mode="bilinear", align_corners=True),
        drop_path_rate=0.3,
        use_reg_enf_cls=False,
        enhance_pos=0,
        use_resblock=False,
        dilation=1,
        with_residual=False,
        activation='ReLU',

    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list), "in_channels must be a list"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.drop_path_rate = drop_path_rate
        self.use_reg_enf_cls = use_reg_enf_cls
        self.enhance_pos = enhance_pos

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs_branch1 = nn.ModuleList()
        self.fpn_convs_cls = nn.ModuleList()
        self.lateral_convs_branch2 = nn.ModuleList()
        self.fpn_convs_bbox = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            in_c = in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels)
            
            lateral_conv1 = ConvModule(
                in_c,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )

            fpn_conv1 = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,    
                inplace=False,
            )
            self.lateral_convs_branch1.append(lateral_conv1)
            self.fpn_convs_cls.append(fpn_conv1)

            lateral_conv2 = ConvModule(
                in_c,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
                dilation = dilation,
            )
            kernel_size= 3
            stride=1
            fpn_conv2 = TPVConv(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation, 
                num_groups=32,
                activation=activation,
                with_residual=with_residual,
                padding = ((kernel_size - 1) * dilation + 1 - stride) // 2,
            )
            self.lateral_convs_branch2.append(lateral_conv2)
            self.fpn_convs_bbox.append(fpn_conv2)

        if self.use_reg_enf_cls:
            self.mapping_convs = nn.ModuleList()
            for i in range(self.start_level, self.backbone_end_level):
                mapping_conv = ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
                self.mapping_convs.append(mapping_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals_cls = [inputs[i] for i in range(self.start_level, self.backbone_end_level + 1)]
        laterals_box = [inputs[i] for i in range(self.start_level, self.backbone_end_level + 1)]
        used_backbone_levels = len(laterals_cls) - 1

        for i in range(used_backbone_levels - 1, -1, -1):
            x1 = F.interpolate(
                laterals_cls[i + 1],
                size=laterals_cls[i].shape[2:],
                **self.upsample_cfg,
            )
            x2 = F.interpolate(
                laterals_box[i + 1],
                size=laterals_box[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals_box[i] = torch.cat([laterals_box[i], x2], dim=1)
            laterals_box[i] = self.lateral_convs_branch2[i](laterals_box[i]) # reduce the channel
            laterals_box[i] = self.fpn_convs_bbox[i](laterals_box[i])

            if self.use_reg_enf_cls and (self.enhance_pos == 0):
                # laterals_cls[i] = laterals_cls[i] + laterals_box[i]
                laterals_cls[i] = laterals_cls[i] + self.mapping_convs[i](laterals_box[i])

            laterals_cls[i] = torch.cat([laterals_cls[i], x1], dim=1)
            laterals_cls[i] = self.lateral_convs_branch1[i](laterals_cls[i])


            if self.use_reg_enf_cls and (self.enhance_pos == 1):
                laterals_cls[i] = laterals_cls[i] + self.mapping_convs[i](laterals_box[i])
                # laterals_cls[i] = laterals_cls[i] + laterals_box[i]

            laterals_cls[i] = self.fpn_convs_cls[i](laterals_cls[i])

        outs_branch1 = [laterals_cls[i] for i in range(used_backbone_levels)]
        outs_branch2 = [laterals_box[i] for i in range(used_backbone_levels)]
        return (outs_branch1, outs_branch2)

class TPVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, num_groups=32, with_residual=False, activation='ReLU'):

        super(TPVConv, self).__init__()
        # Branch 1: 2D convolution on the XY plane (for each depth slice)
        self.conv_xy = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1),
                                 stride=stride, padding=(padding, padding, 0), dilation=(dilation, dilation, 1), bias=False)
        self.gn_xy = nn.GroupNorm(num_groups, out_channels)
        
        # Branch 2: 2D convolution on the XZ plane (for each height slice)
        self.conv_xz = nn.Conv3d(in_channels, out_channels, (kernel_size, 1, kernel_size),
                                 stride=stride, padding=(padding,0,padding), dilation=(dilation, 1, dilation), bias=False)
        self.gn_xz = nn.GroupNorm(num_groups, out_channels)
        
        # Branch 3: 2D convolution on the YZ plane (for each width slice)
        self.conv_yz = nn.Conv3d(in_channels, out_channels,  (1, kernel_size, kernel_size),
                                 stride=stride, padding=(0, padding, padding), dilation=(1, dilation, dilation), bias=False)
        self.gn_yz = nn.GroupNorm(num_groups, out_channels)
        
        # Fusion layer: 1x1x1 convolution, followed by GroupNorm and activation.
        # Input channels for fusion: 3 * out_channels (concatenation of three branches)
        self.fuse = nn.Conv3d(3 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn_fuse = nn.GroupNorm(num_groups, out_channels)
        
        # Activation function: ReLU
        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif activation == 'GELU':
            self.relu = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        # self.relu = nn.ReLU(inplace=True)
        self.with_residual = with_residual

    def forward(self, x):

        B, C, D, H, W = x.shape
        out_xy = self.relu(self.gn_xy(self.conv_xy(x)))
        out_xz = self.relu(self.gn_xz(self.conv_xz(x)))
        out_yz = self.relu(self.gn_yz(self.conv_yz(x)))
        fused_input = torch.cat([out_xy, out_xz, out_yz], dim=1)
        # out = out_xy + out_xz + out_yz
        out = self.fuse(fused_input)  # Shape: [B, out_channels, D, H, W]
        out = self.relu(self.gn_fuse(out))
        if self.with_residual:
            out = out + x
        return out
    
