import torch
import torch.nn as nn
import torch.nn.functional as F

from ezflow.decoder import ConvDecoder, DECODER_REGISTRY, build_decoder
from ezflow.encoder import build_encoder
from ezflow.modules import conv, deconv, BaseModule
from ezflow.similarity import IterSpatialCorrelationSampler as SpatialCorrelationSampler
from ezflow.utils import warp
from ezflow.models import MODEL_REGISTRY
from ezflow.config import configurable

@DECODER_REGISTRY.register()
class PyramidDecoder(BaseModule):

    @configurable
    def __init__(
        self, 
        config=[128, 128, 96, 64, 32], 
        to_flow=True, 
        max_displacement=4, 
        pad_size=0, 
        flow_scale_factor=20.0
    ):
        super(PyramidDecoder, self).__init__()
        self.config = config
        self.flow_scale_factor=flow_scale_factor

        self.correlation_layer = SpatialCorrelationSampler(
            kernel_size=1, 
            patch_size=2*max_displacement+1, 
            padding=pad_size
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        search_range = (2 * max_displacement + 1) ** 2

        self.decoder_layers = nn.ModuleList()

        self.up_feature_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()

        for i in range(len(config)):

            if i == 0:
                concat_channels = search_range
            else:
                concat_channels = (
                    search_range + config[i] + max_displacement
                )

            self.decoder_layers.append(
                ConvDecoder(
                    config=config,
                    to_flow=to_flow,
                    concat_channels=concat_channels,
                )
            )

            if i < len(config) - 1:
                self.deconv_layers.append(
                    deconv(2, 2, kernel_size=4, stride=2, padding=1)
                )

                self.up_feature_layers.append(
                    deconv(
                        concat_channels + sum(config),
                        2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )

    @classmethod
    def from_config(self, cfg):
        return {
            "config": cfg.CONFIG,
            "to_flow": cfg.TO_FLOW,
            "max_displacement": cfg.SIMILARITY.MAX_DISPLACEMENT,
            "pad_size": cfg.SIMILARITY.PAD_SIZE,
            "flow_scale_factor": cfg.FLOW_SCALE_FACTOR
        }

    def _corr_relu(self, features1, features2):

        corr = self.correlation_layer(features1, features2)
        corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
        return self.leaky_relu(corr)

    def forward(self, feature_pyramid1, feature_pyramid2):
        up_flow, up_features = None, None
        up_flow_scale = self.flow_scale_factor * 2 ** (-(len(self.config)))

        flow_preds = []

        for i in range(len(self.decoder_layers)):

            if i == 0:
                corr = self._corr_relu(feature_pyramid1[i], feature_pyramid2[i])
                concatenated_features = corr

            else:

                warped_features = warp(feature_pyramid2[i], up_flow * up_flow_scale)
                up_flow_scale *= 2

                corr = self._corr_relu(feature_pyramid1[i], warped_features)

                concatenated_features = torch.cat(
                    [corr, feature_pyramid1[i], up_flow, up_features], dim=1
                )

            flow, features = self.decoder_layers[i](concatenated_features)
            flow_preds.append(flow)

            if i < len(self.decoder_layers) - 1:
                up_flow = self.deconv_layers[i](flow)
                up_features = self.up_feature_layers[i](features)

        return flow_preds, features


class ContextNetwork(BaseModule):

    def __init__(self, in_channels=565, config=[128, 128, 96, 64, 32]):
        super(ContextNetwork, self).__init__()

        self.context_net = nn.ModuleList(
            [
                conv(
                    in_channels,  # 565
                    config[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1
                ),
            ]
        )
        self.context_net.append(
            conv(
                config[0],  # 128
                config[0],  # 128
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2
            )
        )
        self.context_net.append(
            conv(
                config[0],  # 128
                config[1],  # 128
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4
            )
        )
        self.context_net.append(
            conv(
                config[1],  # 128
                config[2],  # 96
                kernel_size=3,
                stride=1,
                padding=8,
                dilation=8
            )
        )
        self.context_net.append(
            conv(
                config[2],  # 96
                config[3],  # 64
                kernel_size=3,
                stride=1,
                padding=16,
                dilation=16
            )
        )
        self.context_net.append(
            conv(
                config[3],  # 64
                config[4],  # 32
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1
            )
        )
        self.context_net.append(
            nn.Conv2d(config[4], 2, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.context_net = nn.Sequential(*self.context_net)

    def forward(self, x):
        return self.context_net(x)

    

@MODEL_REGISTRY.register()
class PWCNetV3(BaseModule):
    """
    Implementation of the paper
    `PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume <https://arxiv.org/abs/1709.02371>`_

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Configuration for the model
    """

    def __init__(self, cfg):
        super(PWCNetV3, self).__init__()

        self.cfg = cfg
        self.encoder = build_encoder(cfg.ENCODER)

        self.decoder = build_decoder(cfg.DECODER)

        search_range = (2 * cfg.DECODER.SIMILARITY.MAX_DISPLACEMENT + 1) ** 2
        self.context_net = ContextNetwork(
            in_channels=search_range
                    + cfg.DECODER.SIMILARITY.MAX_DISPLACEMENT
                    + cfg.DECODER.CONFIG[-1]
                    + sum(cfg.DECODER.CONFIG),
            config=cfg.DECODER.CONFIG
        )

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.zero_()

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

        # normalize
        img1 = 1.0 * (img1 / 255.0)
        img2 = 1.0 * (img2 / 255.0)

        feature_pyramid1 = self.encoder(img1)
        feature_pyramid2 = self.encoder(img2)

        flow_preds, features = self.decoder(feature_pyramid1, feature_pyramid2)

        flow_preds[-1] += self.context_net(features)

        if self.training:
            return flow_preds

        else:

            flow = flow_preds[-1]

            flow = F.interpolate(
                    flow, 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                )

            flow *= self.cfg.DECODER.FLOW_SCALE_FACTOR

            return flow, flow_preds