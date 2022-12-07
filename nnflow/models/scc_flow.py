import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from ezflow.models import MODEL_REGISTRY
from ezflow.modules import BaseModule
from ezflow.encoder import build_encoder, ENCODER_REGISTRY
from ezflow.config import configurable

from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

from timm.models.layers import trunc_normal_, DropPath
from natten.functional import natten2dqkrpb, natten2dav

from nnflow.models.gmflow.transformer import FeatureFlowAttention
from nnflow.models.gmflow.matching import global_correlation_softmax
from nnflow.models.gmflow.position import PositionEmbeddingSine
from nnflow.models.gmflow.nat import ConvTokenizer, ConvDownsampler, Mlp, NAT


class ModifiedNeighborhoodAttention(nn.Module):
    """
    Modified neighborhood attention

    original code: https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/classification/nat.py
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 dilation=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        if type(dilation) is str:
            self.dilation = None
            self.window_size = None
        else:
            assert dilation is None or dilation >= 1, \
                f"Dilation must be greater than or equal to 1, got {dilation}."
            self.dilation = dilation or 1
            self.window_size = self.kernel_size * self.dilation

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        assert q.shape == k.shape == k.shape
        B, Hp, Wp, C = q.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        dilation = self.dilation
        window_size = self.window_size
        if window_size is None:
            dilation = max(min(H, W) // self.kernel_size, 1)
            window_size = dilation * self.kernel_size
        if H < window_size or W < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - W)
            pad_b = max(0, window_size - H)
            q = pad(q, (0, 0, pad_l, pad_r, pad_t, pad_b))
            k = pad(k, (0, 0, pad_l, pad_r, pad_t, pad_b))
            v = pad(v, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = q.shape
        # qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_proj(q).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        k = self.k_proj(k).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = self.v_proj(v).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        # print(q.shape, k.shape, v.shape)
        q = q * self.scale
        attn = natten2dqkrpb(q, k, self.rpb, dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return f'kernel_size={self.kernel_size}, dilation={self.dilation}, head_dim={self.head_dim}, num_heads={self.num_heads}'


class ModifiedNATLayer(nn.Module):
    """
    Modified Neighborhood Attention Layer to support Cross Attention

    original code: https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/classification/nat.py

    """
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        no_ffn=False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.no_ffn = no_ffn

        self.norm1 = norm_layer(dim)
        self.attn = ModifiedNeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=drop,
            )


    def forward(self, source, target):
        shortcut = source

        x = torch.cat([source, target], dim=0)
        x = self.norm1(x)

        source, target = x.chunk(chunks=2, dim=0)   
        query, key, value = source, target, target

        x = self.attn(query, key, value)
        x = shortcut + self.drop_path(x)
        
        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            
        return x
    

class ModifiedNATBlock(nn.Module):
    """
    Modified Neighborhood Attention Block to support Self -> Cross Attention

    original code: https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/classification/nat.py

    """
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        self_no_ffn=True,
        cross_no_ffn=False,
        use_cross_attn=True
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_cross_attn = use_cross_attn

        # Use Feed Forward Layer for Self Attention 
        # if Cross Attention is disabled
        if not use_cross_attn:
            self_no_ffn=False


        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Self Attention Block
            self.blocks.append(
                 ModifiedNATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    no_ffn=self_no_ffn
                )
            )
            
            # Cross Attention Block
            if self.use_cross_attn:
                self.blocks.append(
                    ModifiedNATLayer(
                        dim=dim,
                        num_heads=num_heads,
                        kernel_size=kernel_size,
                        dilation=None if dilations is None else dilations[i],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i]
                        if isinstance(drop_path, list)
                        else drop_path,
                        norm_layer=norm_layer,
                        no_ffn=cross_no_ffn
                    )
                )
            
        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, source, target):

        if self.use_cross_attn:
            for i in range(len(self.blocks)):
                
                # self attention only with default config
                if i % 2 == 0:
                    source = self.blocks[i](source, source)
                    
                # cross attention and feed forward
                else:
                    source = self.blocks[i](source, target)
        else:
            for i in range(len(self.blocks)):
                source = self.blocks[i](source, source)

        if self.downsample is None:
            return source
        
        return self.downsample(source)


@ENCODER_REGISTRY.register()
class ModifiedNAT(nn.Module):
    """
    Modified Neighborhood Attention Transformer Encoder to support Self -> Cross Attention

    original code: https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/classification/nat.py

    """
    @configurable
    def __init__(
        self,
        embed_dim,
        mlp_ratio,
        depths,
        num_heads,
        use_cross_attn=True,
        drop_path_rate=0.2,
        in_chans=3,
        kernel_size=7,
        dilations=None,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        self_no_ffn=True,
        cross_no_ffn=False,
        use_sine_pos_embed=True,
        **kwargs
    ):
        super().__init__()

        
        if isinstance(dilations, str) and dilations == 'None':
            dilations = None 

        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.mlp_ratio = mlp_ratio
        self.use_sine_pos_embed = use_sine_pos_embed
        self.use_cross_attn = use_cross_attn

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = ModifiedNATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                self_no_ffn=self_no_ffn,
                cross_no_ffn=cross_no_ffn,
                use_cross_attn=use_cross_attn
            )
            self.levels.append(level)

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    @classmethod
    def from_config(self, cfg):
        return {
            "in_chans" : cfg.IN_CHANNELS,
            "depths" : cfg.DEPTHS,
            "num_heads" : cfg.NUM_HEADS,
            "embed_dim" : cfg.EMBED_DIMS,
            "mlp_ratio" : cfg.MLP_RATIO,
            "drop_path_rate" : cfg.DROP_PATH_RATE,
            "kernel_size" : cfg.KERNEL_SIZE,
            "dilations" : cfg.DILATIONS,
            "use_cross_attn": cfg.USE_CROSS_ATTN
        }

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rpb"}
    
    def _add_position_embed(self, img1, img2, feature_channels):
        pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)
        position = pos_enc(img1.permute(0,3,1,2))
        
        position = position.permute(0,2,3,1)
        
        img1 = img1 + position
        img2 = img2 + position
        return img1, img2
        
    def forward_features(self, img1, img2):
        b, c, h, w =  img1.shape
        
        x = self.patch_embed(torch.cat([img1,img2], dim=0))  
        x = self.pos_drop(x)
        
        img1, img2 = x.chunk(chunks=2, dim=0)
        
        if self.use_sine_pos_embed:
            img1, img2 = self._add_position_embed(img1, img2, feature_channels=self.embed_dim)
        
        if self.use_cross_attn:
            # Concat img1 and img2 in batch dimension to compute in parallel
            concat1 = torch.cat([img1, img2], dim=0) # 2B, H, W, C 
            concat2 = torch.cat([img2, img1], dim=0) # 2B, H, W, C
            
            for level in self.levels:
                concat1 = level(concat1, concat2)
                
                # update feature2
                concat2 = torch.cat(concat1.chunk(chunks=2,dim=0)[::-1], dim=0)
                
            concat1 = self.norm(concat1)
            
            feature1, feature2 = concat1.chunk(chunks=2,dim=0) # B, H, W, C      
            return feature1, feature2
        else:
            feature1, feature2 = img1, img2
            for level in self.levels:
                feature1 = level(feature1, feature1)
                feature2 = level(feature2, feature2)
            
            return feature1, feature2

    def forward(self, img1, img2):
        feature1, feature2 = self.forward_features(img1, img2)
        
        feature1 = feature1.permute(0,3,1,2)
        feature2 = feature2.permute(0,3,1,2)

        return feature1, feature2


@MODEL_REGISTRY.register()
class SCCFlow(BaseModule):
    """
    Self Cross Correspondence Optical Flow Model.

    An end to end Transformer based Optical Flow model with alternating layers
    of Self Attention and Cross Attention.

    """

    def __init__(self, cfg):
        
        super(SCCFlow, self).__init__()

        self.cfg = cfg

        self.num_scales = cfg.MODEL.NUM_SCALES
        self.feature_channels = cfg.MODEL.FEATURE_CHANNELS
        self.upsample_factor = cfg.MODEL.UPSAMPLE_FACTOR
        self.num_head = cfg.MODEL.NUM_HEADS
        self.attention_type = cfg.MODEL.ATTENTION_TYPE
        self.ffn_dim_expansion = cfg.MODEL.FFN_DIM_EXPANSION
        self.num_transformer_layers = cfg.MODEL.NUM_TRANSFORMER_LAYERS

        self.attn_splits_list=cfg.MODEL.ATTN_SPLITS_LIST
        self.corr_radius_list=cfg.MODEL.CORR_RADIUS_LIST
        self.prop_radius_list=cfg.MODEL.PROP_RADIUS_LIST
        self.pred_bidir_flow=cfg.MODEL.PRED_BIDIR_FLOW
        

        # Transformer Backbone with alternating self attention and cross attention
        self.backbone = build_encoder(cfg.ENCODER)

        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=self.feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + self.feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, self.upsample_factor ** 2 * 9, 1, 1, 0))
    
    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                      self.upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow

    def forward(self, img0, img1):

        results_dict = {}
        flow_preds = []

        # extract features
        feature0, feature1 = self.backbone(img0, img1)          

        assert len(self.attn_splits_list) == len(self.corr_radius_list) == len(self.prop_radius_list) == self.num_scales


        upsample_factor = self.upsample_factor

        attn_splits = self.attn_splits_list[0]
        corr_radius = self.corr_radius_list[0]
        prop_radius = self.prop_radius_list[0]

            
        # Global matching correlation and softmax
        # when predicting bidirectional flow, flow is the 
        # concatenation of forward flow and backward flow in batch dim [2*B,2,H,W]
        flow = global_correlation_softmax(feature0, feature1, self.pred_bidir_flow)[0]
        

        # upsample to the original resolution for supervison
        if self.training:  # only need to upsample intermediate flow predictions at training time
            flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
            flow_preds.append(flow_bilinear)

        # flow propagation with self-attn
        if self.pred_bidir_flow:
            feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation
            
        flow = self.feature_flow_attn(feature0, flow.detach(),
                                      local_window_attn=prop_radius > 0,
                                      local_window_radius=prop_radius)


        flow_up = self.upsample_flow(flow, feature0)
        flow_preds.append(flow_up)

        results_dict.update({'flow_preds': flow_preds})

        if not self.training:
            results_dict["flow_upsampled"] = results_dict["flow_preds"][0]

        return results_dict