import torch
import torch.nn as nn
import torch.nn.functional as F
from ezflow.encoder import ENCODER_REGISTRY, build_encoder
from ezflow.models import MODEL_REGISTRY
from ezflow.modules import BaseModule

from .backbone import CNNEncoder
from .dino_vit import DinoVITS8
from .geometry import flow_warp
from .matching import global_correlation_softmax, local_correlation_softmax
from .swin_hf import HuggingFaceSwinEncoderV2
from .swin_v2 import SwinEncoderV2
from .transformer import FeatureFlowAttention, FeatureTransformer
from .utils import feature_add_position, normalize_img
from .vit import ViTEncoder


@MODEL_REGISTRY.register()
class GMFlowV2(BaseModule):
    def __init__(self, cfg):

        super(GMFlowV2, self).__init__()

        self.cfg = cfg

        self.num_scales = cfg.FLOW_ATTENTION.NUM_SCALES
        self.feature_channels = cfg.FLOW_ATTENTION.FEATURE_CHANNELS
        self.upsample_factor = cfg.FLOW_ATTENTION.UPSAMPLE_FACTOR
        self.num_head = cfg.FLOW_ATTENTION.NUM_HEADS
        self.attention_type = cfg.FLOW_ATTENTION.ATTENTION_TYPE
        self.ffn_dim_expansion = cfg.FLOW_ATTENTION.FFN_DIM_EXPANSION
        self.num_transformer_layers = cfg.FLOW_ATTENTION.NUM_TRANSFORMER_LAYERS

        self.attn_splits_list = cfg.FLOW_ATTENTION.ATTN_SPLITS_LIST
        self.corr_radius_list = cfg.FLOW_ATTENTION.CORR_RADIUS_LIST
        self.prop_radius_list = cfg.FLOW_ATTENTION.PROP_RADIUS_LIST
        self.pred_bidir_flow = cfg.FLOW_ATTENTION.PRED_BIDIR_FLOW

        self.use_sine_pos_embed = cfg.FLOW_ATTENTION.USE_SINE_POS_EMBED

        # Backbone
        self.backbone = build_encoder(cfg.ENCODER)

        # Transformer Feature Enhancement with alternating self attention and cross attention
        self.transformer = FeatureTransformer(
            num_layers=self.num_transformer_layers,
            d_model=self.feature_channels,
            nhead=self.num_head,
            attention_type=self.attention_type,
            ffn_dim_expansion=self.ffn_dim_expansion,
        )

        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=self.feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(
            nn.Conv2d(2 + self.feature_channels, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.upsample_factor**2 * 9, 1, 1, 0),
        )

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(
            concat
        )  # list of [2B, C, H, W], resolution from high to low

        feature0, feature1 = [], []

        chunks = torch.chunk(features, 2, 0)  # tuple
        feature0.append(chunks[0])
        feature1.append(chunks[1])

        return feature0, feature1

        # reverse: resolution from low to high
        # features = features[::-1]

        # feature0, feature1 = [], []

        # for i in range(len(features)):
        #     feature = features[i]
        #     chunks = torch.chunk(feature, 2, 0)  # tuple
        #     feature0.append(chunks[0])
        #     feature1.append(chunks[1])

        # return feature0, feature1

    def upsample_flow(
        self,
        flow,
        feature,
        bilinear=False,
        upsample_factor=8,
    ):
        if bilinear:
            up_flow = (
                F.interpolate(
                    flow,
                    scale_factor=upsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                * upsample_factor
            )

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(
                b, 1, 9, self.upsample_factor, self.upsample_factor, h, w
            )  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(
                b, flow_channel, 9, 1, 1, h, w
            )  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(
                b, flow_channel, self.upsample_factor * h, self.upsample_factor * w
            )  # [B, 2, K*H, K*W]

        return up_flow

    def forward(self, img0, img1):

        results_dict = {}
        flow_preds = []

        # img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # resolution low to high
        feature0_list, feature1_list = self.extract_feature(
            img0, img1
        )  # list of features
        flow = None

        assert (
            len(self.attn_splits_list)
            == len(self.corr_radius_list)
            == len(self.prop_radius_list)
            == self.num_scales
        )

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if self.pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat(
                    (feature1, feature0), dim=0
                )

            upsample_factor = self.upsample_factor * (
                2 ** (self.num_scales - 1 - scale_idx)
            )

            if scale_idx > 0:
                flow = (
                    F.interpolate(
                        flow, scale_factor=2, mode="bilinear", align_corners=True
                    )
                    * 2
                )

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = self.attn_splits_list[scale_idx]
            corr_radius = self.corr_radius_list[scale_idx]
            prop_radius = self.prop_radius_list[scale_idx]

            # add position to features
            if self.use_sine_pos_embed:
                feature0, feature1 = feature_add_position(
                    feature0, feature1, attn_splits, self.feature_channels
                )

            # Transformer
            feature0, feature1 = self.transformer(
                feature0, feature1, attn_num_splits=attn_splits
            )

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(
                    feature0, feature1, self.pred_bidir_flow
                )[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[
                    0
                ]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison
            if (
                self.training
            ):  # only need to upsample intermediate flow predictions at training time
                flow_bilinear = self.upsample_flow(
                    flow, None, bilinear=True, upsample_factor=upsample_factor
                )
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if self.pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat(
                    (feature0, feature1), dim=0
                )  # [2*B, C, H, W] for propagation
            flow = self.feature_flow_attn(
                feature0,
                flow.detach(),
                local_window_attn=prop_radius > 0,
                local_window_radius=prop_radius,
            )

            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(
                    flow, feature0, bilinear=True, upsample_factor=upsample_factor
                )
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_up)

        results_dict.update({"flow_preds": flow_preds})

        if not self.training:
            results_dict["flow_upsampled"] = results_dict["flow_preds"][0]

        return results_dict
