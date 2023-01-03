"""
Modification of the original FlowNetC model to support RAFT's Feature Extractor

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, kaiming_normal_

from ezflow.decoder import build_decoder
from ezflow.encoder import BasicConvEncoder, build_encoder
from ezflow.modules import BaseModule, conv
from ezflow.similarity import IterSpatialCorrelationSampler as SpatialCorrelationSampler
from ezflow.models import MODEL_REGISTRY

from nnflow.residual_encoder import BasicEncoderV2

@MODEL_REGISTRY.register()
class FlowNetC_V2(BaseModule):
    """
    Implementation of **FlowNetCorrelation** from the paper
    `FlowNet: Learning Optical Flow with Convolutional Networks <https://arxiv.org/abs/1504.06852>`_

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Configuration for the model
    """

    def __init__(self, cfg):
        super(FlowNetC_V2, self).__init__()

        self.cfg = cfg

        channels = cfg.ENCODER.CONFIG
        cfg.ENCODER.CONFIG = channels[:3]

        self.feature_encoder = build_encoder(cfg.ENCODER)

        self.correlation_layer = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=2 * cfg.SIMILARITY.MAX_DISPLACEMENT + 1,
            padding=cfg.SIMILARITY.PAD_SIZE,
            dilation_patch=2,
        )

        self.corr_activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_redirect = conv(
            in_channels=cfg.ENCODER.CONFIG[-1], out_channels=32, norm=cfg.ENCODER.NORM
        )

        self.combine_corr = conv(
                    473,
                    256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm=cfg.ENCODER.NORM,
                )

        self.corr_encoder = BasicEncoderV2(
            in_channels=256, 
            out_channels=cfg.ENCODER.CORR_CONFIG[-1], 
            layer_config=cfg.ENCODER.CORR_CONFIG, 
            norm=cfg.ENCODER.NORM,
            intermediate_features=True
        )

        self.decoder = build_decoder(cfg.DECODER)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, img1, img2):
        """
        Performs forward pass of the network

        Parameters
        ----------
        img1 : torch.Tensor
            Image to predict flow from
        img2 : torch.Tensor
            Image to predict flow to

        Returns
        -------
        torch.Tensor
            Flow from img1 to img2
        """

        H, W = img1.shape[-2:]

        conv_outputs1 = self.feature_encoder(img1)
        conv_outputs2 = self.feature_encoder(img2)

        corr_output = self.correlation_layer(conv_outputs1[-1], conv_outputs2[-1])
        corr_output = corr_output.view(
            corr_output.shape[0], -1, corr_output.shape[3], corr_output.shape[4]
        )
        corr_output = self.corr_activation(corr_output)

        # Redirect final feature output of img1
        conv_redirect_output = self.conv_redirect(conv_outputs1[-1])

        x = torch.cat([conv_redirect_output, corr_output], dim=1)
        
        x = self.combine_corr(x)
        
        conv_outputs = self.corr_encoder(x)

        # Add first two convolution output from img1
        conv_outputs = [conv_outputs1[0], conv_outputs1[1], x] + conv_outputs


        flow_preds = self.decoder(conv_outputs)

        output = {"flow_preds": flow_preds}

        if self.training:
            return output

        flow_up = flow_preds[-1]

        flow_up = F.interpolate(
            flow_up, size=(H, W), mode="bilinear", align_corners=False
        )

        output["flow_upsampled"] = flow_up

        return output
