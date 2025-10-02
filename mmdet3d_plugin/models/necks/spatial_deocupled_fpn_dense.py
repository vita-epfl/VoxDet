import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.models import NECKS
from timm.layers import DropPath
from torchvision.ops import DeformConv2d



class DensePooling(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, weights):
        # x : (B, C, X, Y, Z)
        # weights : (B, 3, X, Y, Z)
        if self.dim == 'xy':
            w = F.softmax(weights[:, 0:1], dim=-1)    
            feat = (x * w).sum(dim=4)                  
        elif self.dim == 'yz':
            w = F.softmax(weights[:, 1:2], dim=-3)    
            feat = (x * w).sum(dim=2)                   
        elif self.dim == 'zx':
            w = F.softmax(weights[:, 2:3], dim=-2)      
            feat = (x * w).sum(dim=3)                   
        else:
            raise ValueError(f"Unknown dim {self.dim}")
        return feat

@NECKS.register_module()
class SpatiallyDecoupledFPN(BaseModule):
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
        num_groups=32, 
        kernel_size=1,
        tpv_down_ratio=1.0,
        use_bias=False,
        share_fpn=False,
        adaptive_pooling=True,

    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list), "in_channels must be a list"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adaptive_pooling = adaptive_pooling
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.share_fpn = share_fpn
        self.upsample_cfg = upsample_cfg.copy()


        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs_cls = nn.ModuleList()
        self.fpn_convs_cls = nn.ModuleList()
        self.lateral_convs_reg = nn.ModuleList()
        self.fpn_convs_reg = nn.ModuleList()
        self.sp_tpvs_cls = nn.ModuleList()
        self.sp_tpvs_reg = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            in_c = in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels)

            lateral_conv_cls = ConvModule(
                in_c,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )

            fpn_conv_cls = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )
            sp_tpv_cls = TPVDecoupleModule(
                in_channels=out_channels,
                mid_channels=int(out_channels // tpv_down_ratio),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                gn_groups=num_groups,
                use_bias=use_bias,
                adaptive_pooling=adaptive_pooling,
            )

            self.lateral_convs_cls.append(lateral_conv_cls)
            self.fpn_convs_cls.append(fpn_conv_cls)
            self.sp_tpvs_cls.append(sp_tpv_cls)


            sp_tpv_reg = TPVDecoupleModule(
                in_channels=out_channels,
                mid_channels=int(out_channels // tpv_down_ratio),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                gn_groups=num_groups,
                use_bias=use_bias,
                adaptive_pooling=adaptive_pooling,
            )

            if self.share_fpn:
                lateral_conv_reg = nn.Identity()
                fpn_conv_reg = nn.Identity()
            else:
                lateral_conv_reg = ConvModule(
                    in_c,
                    out_channels,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False,
                )
                fpn_conv_reg = ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
            self.lateral_convs_reg.append(lateral_conv_reg)
            self.fpn_convs_reg.append(fpn_conv_reg)
            self.sp_tpvs_reg.append(sp_tpv_reg)

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
            laterals_cls[i] = self.sp_tpvs_cls[i](laterals_cls[i])
            laterals_cls[i] = torch.cat([laterals_cls[i], x1], dim=1)
            laterals_cls[i] = self.lateral_convs_cls[i](laterals_cls[i])
            laterals_cls[i] = self.fpn_convs_cls[i](laterals_cls[i])

        cls_feat_out = [laterals_cls[i] for i in range(used_backbone_levels)]
        
        for i in range(used_backbone_levels - 1, -1, -1):
            x2 = F.interpolate(
                laterals_box[i + 1],
                size=laterals_box[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals_box[i] = self.sp_tpvs_reg[i](laterals_box[i])
            laterals_box[i] = torch.cat([laterals_box[i], x2], dim=1)

            lateral_conv_reg =  self.lateral_convs_cls[i] if self.share_fpn else self.lateral_convs_reg[i]
            fpn_conv_reg =  self.fpn_convs_cls[i] if self.share_fpn else self.fpn_convs_reg[i]

            laterals_box[i] = lateral_conv_reg(laterals_box[i])
            laterals_box[i] = fpn_conv_reg(laterals_box[i])

        reg_feat_out = [laterals_box[i] for i in range(used_backbone_levels)]
        return (cls_feat_out, reg_feat_out)
    
class TPVDecoupleModule(nn.Module):

    def __init__(self,
                 in_channels: int,
                 mid_channels: int = None,
                 kernel_size: int = 1,
                 padding: int = 1,
                 gn_groups: int = 8,
                 use_bias: bool = False,
                 adaptive_pooling: bool = True,
                 ):
        super().__init__()
        mid_channels = mid_channels or in_channels
        self.padding = padding
        self.adaptive_pooling = adaptive_pooling

        
        if adaptive_pooling:
            self.weights_conv = nn.Conv3d(in_channels, 3, kernel_size=1, bias=use_bias)
            self.pool_xy = DensePooling(dim='xy')
            self.pool_yz = DensePooling(dim='yz')
            self.pool_zx = DensePooling(dim='zx')

        self.conv1x1_xy = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.conv1x1_yz = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.conv1x1_zx = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding)

        offset_ch = 2 * 3 * 3
        self.offset_xy = nn.Conv2d(mid_channels, offset_ch, 3, padding=3//2)
        self.offset_yz = nn.Conv2d(mid_channels, offset_ch, 3, padding=3//2)
        self.offset_zx = nn.Conv2d(mid_channels, offset_ch, 3, padding=3//2)

        self.defconv_xy = DeformConv2d(mid_channels, mid_channels, 3, padding=3//2, bias=False)
        self.defconv_yz = DeformConv2d(mid_channels, mid_channels, 3, padding=3//2, bias=False)
        self.defconv_zx = DeformConv2d(mid_channels, mid_channels, 3, padding=3//2, bias=False)

        self.gn_xy = nn.GroupNorm(gn_groups, mid_channels)
        self.gn_yz = nn.GroupNorm(gn_groups, mid_channels)
        self.gn_zx = nn.GroupNorm(gn_groups, mid_channels)
        self.act = nn.ReLU(inplace=True)

        self.fuse_conv = nn.Conv3d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.fuse_gn   = nn.GroupNorm(gn_groups, in_channels)
        self.fuse_act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptive_pooling:
            weights = self.weights_conv(x)  # (B, 3, X, Y, Z)
            feat_xy = self.pool_xy(x, weights)  # (B, C, X, Y)
            feat_yz = self.pool_yz(x, weights)  # (B, C, Y, Z)
            feat_zx = self.pool_zx(x, weights)  # (B, C, X, Z)
        else:
            feat_xy = x.mean(dim=4)        # (B, C, X, Y)
            feat_zx = x.mean(dim=3)        # (B, C, X, Z)
            feat_yz = x.mean(dim=2)        # (B, C, Y, Z)

        # 2) per-view: 1×1 conv → offset → deformable conv → GN → ReLU
        def _proc2d(feat, conv1x1, off_conv, defconv, gn):
            f = self.act(conv1x1(feat))
            off = off_conv(f)
            f_def = defconv(f, off)
            f_def = gn(f_def)
            return self.act(f_def)

        def_xy = _proc2d(feat_xy, self.conv1x1_xy, self.offset_xy, self.defconv_xy, self.gn_xy)
        def_yz = _proc2d(feat_yz, self.conv1x1_yz, self.offset_yz, self.defconv_yz, self.gn_yz)
        def_zx = _proc2d(feat_zx, self.conv1x1_zx, self.offset_zx, self.defconv_zx, self.gn_zx)

        B, _, X, Y, Z = x.shape
        def_xy_3d = def_xy.unsqueeze(-1).expand(-1, -1, -1, -1, Z)
        def_yz_3d = def_yz.unsqueeze(2).expand(-1, -1, X, -1, -1)
        def_zx_3d = def_zx.unsqueeze(3).expand(-1, -1, -1, Y, -1)
        fused_3d = def_xy_3d + def_yz_3d + def_zx_3d  # (B, mid_channels, X, Y, Z)

        out = self.fuse_conv(fused_3d)
        out = self.fuse_gn(out)
        out = self.fuse_act(out)

        return x + out