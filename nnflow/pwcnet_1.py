import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from abc import abstractmethod

from ezflow.models import MODEL_REGISTRY
from ezflow.modules import BaseModule
from ezflow.similarity import CorrelationLayer

from ptlflow.utils.correlation import IterSpatialCorrelationSampler as SpatialCorrelationSampler

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule as mmcvBaseModule

from torch import Tensor
from typing import Dict, Optional, Sequence, Tuple, Union

"""
    Pytorch Lightning PWCNet

"""

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


@MODEL_REGISTRY.register()
class PWCNetV1(BaseModule):

    def __init__(self, cfg):
        super(PWCNetV1, self).__init__()
        
        self.div_flow = cfg.FLOW_SCALE_FACTOR
        
        self.md = cfg.SIMILARITY.MAX_DISPLACEMENT
        self.padding = cfg.SIMILARITY.PAD_SIZE

        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128,128, kernel_size=3, stride=1)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1)
        self.conv6aa = conv(128,196, kernel_size=3, stride=2)
        self.conv6a  = conv(196,196, kernel_size=3, stride=1)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        self.leakyRELU = nn.LeakyReLU(0.1)

        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*self.md+1, padding=self.padding)
        
        nd = (2*self.md+1)**2
        dd = np.cumsum([128,128,96,64,32])

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(od+dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od+dd[4]) 
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4]) 
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4]) 
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4]) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.to(dtype=x.dtype, device=x.device)
        vgrid = grid + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.ones(x.size()).to(dtype=x.dtype, device=x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask


    def forward(self, im1, im2):
        
        H, W = im1.shape[-2:]
        # im1 = inputs['images'][:, 0]
        # im2 = inputs['images'][:, 1]

        im1 = 1.0 * (im1 / 255.0)
        im2 = 1.0 * (im2 / 255.0)

        im1 = im1.contiguous()
        im1 = im1.contiguous()

        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))


        corr6 = self.corr(c16, c26)
        corr6 = corr6.view(corr6.shape[0], -1, corr6.shape[3], corr6.shape[4])
        corr6 = corr6 / c16.shape[1]
        corr6 = self.leakyRELU(corr6)


        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        
        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = corr5.view(corr5.shape[0], -1, corr5.shape[3], corr5.shape[4])
        corr5 = corr5 / c15.shape[1]
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

       
        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = corr4.view(corr4.shape[0], -1, corr4.shape[3], corr4.shape[4])
        corr4 = corr4 / c14.shape[1]
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)


        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = corr3.view(corr3.shape[0], -1, corr3.shape[3], corr3.shape[4])
        corr3 = corr3 / c13.shape[1]
        corr3 = self.leakyRELU(corr3)
        

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)


        warp2 = self.warp(c22, up_flow3*5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = corr2.view(corr2.shape[0], -1, corr2.shape[3], corr2.shape[4])
        corr2 = corr2 / c12.shape[1]
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)

        flow_up = self.upsample1(flow2*self.div_flow)
        
        flow_preds = [flow2, flow3, flow4, flow5, flow6]

        if self.training:
            return flow_preds

        else:
            return flow_up, flow_preds



"""
    MMFlow Lightning PWCNet Encoder

"""


class BasicConvBlock(nn.Module):
    """Basic convolution block for PWC-Net.
    This module consists of several plain convolution layers.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolution layers. Default: 3.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolution layer to downsample the input feature
            map. Options are 1 or 2. Default: 2.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolution layer and
            the dilation rate of the first convolution layer is always 1.
            Default: 1.
        kernel_size (int): Kernel size of each feature level. Default: 3.
        conv_cfg (dict , optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_convs: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 kernel_size: int = 3,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None) -> None:
        super(BasicConvBlock, self).__init__()

        convs = []
        in_channels = in_channels
        for i in range(num_convs):
            k = kernel_size[i] if isinstance(kernel_size,
                                             (tuple, list)) else kernel_size
            out_ch = out_channels[i] if isinstance(out_channels,
                                                   (tuple,
                                                    list)) else out_channels

            convs.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=k // 2 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channels = out_ch

        self.layers = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""

        out = self.layers(x)
        return out


class BasicEncoder(nn.Module):
    """A basic pyramid feature extraction sub-network.
    Args:
        in_channels (int): Number of input channels.
        pyramid_levels (Sequence[str]): The list of feature pyramid that are
            the keys for output dict.
        num_convs (Sequence[int]): Numbers of conv layers for each
            pyramid level. Default: (3, 3, 3, 3, 3, 3).
        out_channels (Sequence[int]): List of numbers of output
            channels of each pyramid level.
            Default: (16, 32, 64, 96, 128, 196).
        strides (Sequence[int]): List of strides of each pyramid level.
            Default: (2, 2, 2, 2, 2, 2).
        dilations (Sequence[int]): List of dilation of each pyramid level.
            Default: (1, 1, 1, 1, 1, 1).
        kernel_size (Sequence, int): Kernel size of each feature
            level. Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for each normalization layer.
            Default: None.
        act_cfg (dict): Config dict for each activation layer in ConvModule.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict, list, optional): Config for module initialization.
    """

    def __init__(self,
                 in_channels: int,
                 pyramid_levels: Sequence[str],
                 num_convs: Sequence[int] = (3, 3, 3, 3, 3, 3),
                 out_channels: Sequence[int] = (16, 32, 64, 96, 128, 196),
                 strides: Sequence[int] = (2, 2, 2, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1, 1, 1),
                 kernel_size: Union[Sequence, int] = 3,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1)) -> None:
        super(BasicEncoder, self).__init__()

        assert len(out_channels) == len(num_convs) == len(strides) == len(
            dilations) == len(pyramid_levels)
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        self.out_channels = out_channels
        self.num_convs = num_convs
        self.strides = strides
        self.dilations = dilations

        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        convs = []
        for i in range(len(out_channels)):
            if isinstance(self.kernel_size, (list, tuple)) and len(
                    self.kernel_size) == len(out_channels):
                kernel_size_ = self.kernel_size[i]
            elif isinstance(self.kernel_size, int):
                kernel_size_ = self.kernel_size
            else:
                TypeError('kernel_size must be list, tuple or int, '
                          f'but got {type(kernel_size)}')

            convs.append(
                self._make_layer(
                    in_channels,
                    out_channels[i],
                    num_convs[i],
                    strides[i],
                    dilations[i],
                    kernel_size=kernel_size_))
            in_channels = out_channels[i][-1] if isinstance(
                out_channels[i], (tuple, list)) else out_channels[i]

        self.layers = nn.Sequential(*convs)

    def _make_layer(self,
                    in_channels: int,
                    out_channel: int,
                    num_convs: int,
                    stride: int,
                    dilation: int,
                    kernel_size: int = 3) -> torch.nn.Module:
        return BasicConvBlock(
            in_channels=in_channels,
            out_channels=out_channel,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_size,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, x: torch.Tensor) -> dict:
        """Forward function for BasicEncoder.
        Args:
            x (Tensor): The input data.
        Returns:
            dict: The feature pyramid extracted from input data.
        """
        outs = dict()
        for i, convs_layer in enumerate(self.layers):
            x = convs_layer(x)
            if 'level' + str(i + 1) in self.pyramid_levels:
                outs['level' + str(i + 1)] = x

        return outs


class PWCNetEncoder(BasicEncoder):
    """The feature extraction sub-module in PWC-Net.
    Args:
        in_channels (int): Number of input channels.
        pyramid_levels (Sequence[str]): The list of feature pyramid that are
            the keys for output dict.
        net_type (str): The type of this sub-module, if net_type is Basic, the
            the number of convolution layers of each level is 3, if net_type is
            Small, the the number of convolution layers of each level is 2.
        out_channels (Sequence[int]): List of numbers of output
            channels of each pyramid level.
            Default: (16, 32, 64, 96, 128, 196).
        strides (Sequence[int]): List of strides of each pyramid level.
            Default: (2, 2, 2, 2, 2, 2).
        dilations (Sequence[int]): List of dilation of each pyramid level.
            Default: (1, 1, 1, 1, 1, 1).
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for each normalization layer.
            Default: None.
        act_cfg (dict): Config dict for each activation layer in ConvModule.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict, optional): Config of weights initialization. Default:
            None.
    """
    _arch_settings = {'Basic': (3, 3, 3, 3, 3, 3), 'Small': (2, 2, 2, 2, 2, 2)}

    def __init__(self,
                 cfg, 
                 in_channels: int,
                 pyramid_levels: Sequence[str],
                 net_type: str = 'Basic',
                 out_channels: Sequence[int] = (16, 32, 64, 96, 128, 196),
                 strides: Sequence[int] = (2, 2, 2, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1)) -> None:

        if net_type not in self._arch_settings:
            raise KeyError(f'invalid net type {net_type} for PWC-Net')

        num_convs = self._arch_settings[net_type]

        super(PWCNetEncoder, self).__init__(
            in_channels=in_channels,
            pyramid_levels=pyramid_levels,
            num_convs=num_convs,
            out_channels=out_channels,
            strides=strides,
            dilations=dilations,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.ezflow_cfg = cfg



"""
    MMFlow Lightning PWCNet Decoder

"""


class DenseLayer(nn.Module):
    """Densely connected layer.
    Args:
        in_channels (int): Input channels of convolution module.
        feat_channels (int): Output channel of convolution module.
        conv_cfg (dict, optional): Config of convolution layer in module.
            Default: None.
        norm_cfg (dict, optional): Config of norm layer in module.
            Default: None.
        act_cfg (dict): Config of activation layer in module.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict, list, optional): Config dict of initialization of
            BaseModule. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1)) -> None:
        super(DenseLayer, self).__init__()
        self.layers = ConvModule(
            in_channels=in_channels,
            out_channels=feat_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for DenseLayer.
        Args:
            x (Tensor): The input feature.
        Returns:
            Tensor: The output feature of DenseLayer.
        """
        out = self.layers(x)
        return torch.cat((out, x), dim=1)


class BasicDenseBlock(nn.Module):
    """Basic Dense Block.
    A basic block which consists of several dense layers.
    Args:
        in_channels (int): Input channels of the block.
        feat_channels (Sequence[int]): Output channels of convolution module
            in dense layers. Default: (128, 128, 96, 64, 32).
        conv_cfg (dict, optional): Config of convolution layer in dense layers.
            Default: None.
        norm_cfg (dict, optional): Config of norm layer in dense layers.
            Default: None.
        act_cfg (dict, optional): Config of activation layer in dense layers.
            Default: None.
        init_cfg (dict, list, optional): Config for module initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: Sequence[int] = (128, 128, 96, 64, 32),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None):
        super(BasicDenseBlock, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        layers = []
        for _feat_channels in feat_channels:
            layers.append(
                DenseLayer(in_channels, _feat_channels, conv_cfg, norm_cfg,
                           act_cfg))
            in_channels += _feat_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for BasicDenseBlock.
        Args:
            x (Tensor): The input feature.
        Returns:
            Tensor: The output feature of BasicDenseBlock.
        """
        return self.layers(x)


class PWCModule(nn.Module):
    """Basic module of PWC-Net decoder.
    Args:
        in_channels (int): Input channels of basic dense block.
        up_flow (bool, optional): Whether to calculate upsampling flow and
            feature or not. Default: True.
        densefeat_channels (Sequence[int]): Number of output channels for
            dense layers. Default: (128, 128, 96, 64, 32).
        conv_cfg (dict, optional): Config dict of convolution layer in module.
            Default: None.
        norm_cfg (dict, optional): Config dict of norm layer in module.
            Default: None.
        act_cfg (dict, optional): Config dict of activation layer in module.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict, optional): Config dict of initialization of BaseModule.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 up_flow: bool = True,
                 densefeat_channels: Sequence[int] = (128, 128, 96, 64, 32),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1)) -> None:
        
        super(PWCModule, self).__init__()

        self.up_flow = up_flow
        self.dense_net = BasicDenseBlock(in_channels, densefeat_channels,
                                         conv_cfg, norm_cfg, act_cfg)
        self.last_channels = in_channels + sum(densefeat_channels)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self._make_predict_layer()
        self._make_upsample_layer()

    def _make_predict_layer(self) -> torch.nn.Module:
        self.predict_layer = nn.Conv2d(
            self.last_channels, 2, kernel_size=3, padding=1)

    def _make_upsample_layer(self) -> torch.nn.Module:

        if self.up_flow:
            self.upflow_layer = nn.ConvTranspose2d(
                2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat_layer = nn.ConvTranspose2d(
                self.last_channels, 2, kernel_size=4, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function for PWCModule.
        Args:
            x (Tensor): The input feature.
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The predicted optical flow,
                the feature to predict flow, the upsampled flow from the last
                level, and the upsampled feature.
        """
        feat = self.dense_net(x)
        flow = self.predict_layer(feat)
        upflow = None
        upfeat = None
        if self.up_flow:
            upflow = self.upflow_layer(flow)
            upfeat = self.upfeat_layer(feat)
        return flow, feat, upflow, upfeat


class ContextNet(nn.Module):
    """The Context network to exploit contextual information for PWC to refine
    the optical flow.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels. Default: 2.
        feat_channels (Sequence[int]): List of numbers of outputs feature
            channels. Default: (128, 128, 128, 96, 64, 32).
        dilation (Sequence[int]): List of dilation of each layer. Default:
            (1, 2, 4, 8, 16, 1).
        conv_cfg (dict , optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='LeakyReLU').
    """

    def __init__(self,
                 in_channels,
                 out_channels: int = 2,
                 feat_channels: Sequence[int] = (128, 128, 128, 96, 64, 32),
                 dilations: Sequence[int] = (1, 2, 4, 8, 16, 1),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1)) -> None:

        super(ContextNet, self).__init__()

        layers = []
        for _feat_channels, _dilation in zip(feat_channels, dilations):
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=_feat_channels,
                    kernel_size=3,
                    stride=1,
                    dilation=_dilation,
                    padding=_dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channels = _feat_channels

        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True))
        self.out_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for Context network.
        Args:
            x (Tensor): Input feature.
        Returns:
            Tensor: The predicted result.
        """
        return self.layers(x)


def coords_grid(flow: Tensor) -> Tensor:
    """Generate shifted coordinate grid based based input flow.
    Args:
        flow (Tensor): Estimated optical flow.
    Returns:
        Tensor: The coordinate that shifted by input flow and scale in the
            range [-1, 1].
    """
    B, _, H, W = flow.shape
    xx = torch.arange(0, W, device=flow.device, requires_grad=False)
    yy = torch.arange(0, H, device=flow.device, requires_grad=False)
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()
    grid = coords[None].repeat(B, 1, 1, 1) + flow
    grid[:, 0, ...] = grid[:, 0, ...] * 2. / max(W - 1, 1) - 1.
    grid[:, 1, ...] = grid[:, 1, ...] * 2. / max(H - 1, 1) - 1.
    grid = grid.permute(0, 2, 3, 1)
    return grid


class Warp(nn.Module):
    """Warping layer to warp feature using optical flow.
    Args:
        mode (str): interpolation mode to calculate output values. Options are
            'bilinear' and 'nearest'. Defaults to 'bilinear'.
        padding_mode (str): padding mode for outside grid values. Options are
            'zero', 'border' and 'reflection'. Defaults to 'zeros'.
        align_corners (bool): If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s corner
            pixels. If set to False, they are instead considered as referring
            to the corner points of the input’s corner pixels, making the
            sampling more resolution agnostic. Default to False.
    """

    def __init__(self,
                 mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 use_mask: bool = True) -> None:

        super(Warp, self).__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.use_mask = use_mask

    def forward(self, feat: Tensor, flow: Tensor) -> Tensor:
        """Forward function for warp.
        Args:
            feat (Tensor): Input feature
            flow (Tensor): Input optical flow.
        Returns:
            Tensor: The output feature that was generated by warping input
                feature based input flow.
        """

        grid = coords_grid(flow)
        out = F.grid_sample(
            feat,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners)

        mask = torch.ones(feat.size(), device=feat.device, requires_grad=False)
        if self.use_mask:
            mask = F.grid_sample(
                mask,
                grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners)
            mask = (mask > 0.9999).float()
        return out * mask


class CustomCorrelationSampler(nn.Module):

    def __init__(self, padding=0, max_displacement=4):
        super(CustomCorrelationSampler, self).__init__()
        self.corr_block = SpatialCorrelationSampler(
            kernel_size=1, 
            patch_size=2*max_displacement+1, 
            padding=padding
        )
        self.leakyRELU = nn.LeakyReLU(0.1)

    def forward(self, feat1, feat2):
        corr = self.corr_block(feat1, feat2)
        corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
        corr = self.leakyRELU(corr)
        return corr


class PWCNetDecoder(nn.Module):
    """The Decoder of PWC-Net.
    The decoder of PWC-Net which outputs flow predictions and features.
    Args:
        in_channels (dict): Dict of number of input channels for each level.
        densefeat_channels (Sequence[int]): Number of output channels for
            dense layers. Default: (128, 128, 96, 64, 32).
        flow_div (float): The divisor works for scaling the ground truth.
            Default: 20.
        corr_cfg (dict): Config for correlation layer.
            Defaults to dict(type='Correlation', max_displacement=4).
        scaled (bool): Whether to use scaled correlation by the number of
            elements involved to calculate correlation or not.
            Defaults to False.
        warp_cfg (dict): Config for warp operation. Defaults to
            dict(type='Warp', align_corners=True).
        conv_cfg (dict, optional): Config of convolution layer in module.
            Default: None.
        norm_cfg (dict, optional): Config of norm layer in module.
            Default: None.
        act_cfg (dict, optional): Config of activation layer in module.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        post_processor (dict, optional): Config of flow post process module.
            Default: None
        flow_loss: Config of loss function of optical flow. Default: None.
        init_cfg (dict, list, optional): Config of dict weights initialization.
            Default: None.
    """

    def __init__(self,
                 cfg,
                 in_channels: Dict[str, int],
                 densefeat_channels: Sequence[int] = (128, 128, 96, 64, 32),
                 flow_div: float = 20.,
                 corr_cfg: dict = dict(type='Correlation', max_displacement=4),
                 scaled: bool = False,
                 warp_cfg: dict = dict(type='Warp', align_corners=True),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1)):

        assert isinstance(in_channels, dict)

        super(PWCNetDecoder, self).__init__()

        self.ezflow_cfg = cfg
        self.in_channels = in_channels
        self.densefeat_channels = densefeat_channels
        self.flow_div = flow_div

        self.flow_levels = list(in_channels.keys())
        self.flow_levels.sort()
        self.start_level = self.flow_levels[-1]
        self.end_level = self.flow_levels[0]

        self.corr_cfg = corr_cfg
        self.scaled = scaled
        self.warp_cfg = warp_cfg

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self._make_corr_block(self.corr_cfg, self.act_cfg, self.scaled)

        if warp_cfg is not None:
            self._make_warp(self.warp_cfg)

        self.multiplier = dict()
        for level in self.flow_levels:
            self.multiplier[level] = self.flow_div * 2**(-int(level[-1]))

        self.post_processor = ContextNet(in_channels=565)

        self._make_layers()

    def _make_layers(self) -> None:
        """Build sub-modules of this decoder."""
        layers = []

        for level in self.flow_levels:
            up_flow = (level != self.end_level)
            layers.append([
                level,
                self._make_layer(self.in_channels[level], up_flow=up_flow)
            ])
        self.decoders = nn.ModuleDict(layers)

    def _make_layer(self,
                    in_channels: int,
                    up_flow: bool = True) -> torch.nn.Module:
        """Build module at each level of this decoder.
        Args:
            in_channels (int): The channels of input feature
            up_sample (bool): Whether upsample flow for the next level.
                Defaults to True.
        Returns:
            torch.nn.Module: The sub-module for this decoder.
        """

        return PWCModule(
            in_channels,
            up_flow,
            self.densefeat_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _make_corr_block(self, corr_cfg: dict, act_cfg: dict,
                         scaled: bool) -> None:
        """Make correlation.
        Args:
            corr_cfg (dict): Config for correlation layer.
            act_cfg (dict): Config of activation layer in module.
            scaled (bool): Whether to use scaled correlation by the number of
                elements involved to calculate correlation or not.
        """
        md = corr_cfg['max_displacement']
        pad_size = corr_cfg['padding']
        if self.ezflow_cfg.SIMILARITY.TYPE == 'CorrelationLayer':
            self.corr_block = CorrelationLayer(pad_size=4, max_displacement=md)
        elif self.ezflow_cfg.SIMILARITY.TYPE == 'SpatialCorrelation':    
            self.corr_block = CustomCorrelationSampler(padding=pad_size, max_displacement=md)
        else:
            print("invalid correlation layer type")
        # self.corr_block = CorrBlock(
        #     corr_cfg=corr_cfg, act_cfg=act_cfg, scaled=scaled)

    def _make_warp(self, warp_cfg: dict) -> None:
        """Build warp operator.
        Args:
            warp_cfg (dict): Config for warp operation.
        """
        self.warp = Warp(
                    use_mask=warp_cfg['use_mask'],
                    align_corners=warp_cfg['align_corners']
                )

    def forward(self, feat1, feat2):
        """Forward function for PWCNetDecoder.
        Args:
            feat1 (Dict[str, Tensor]): The feature pyramid from the first
                image.
            feat2 (Dict[str, Tensor]): The feature pyramid from the second
                image.
        Returns:
            Dict[str, Tensor]: The predicted multi-levels optical flow.
        """

        flow_pred = dict()
        flow = None
        upflow = None
        upfeat = None

        for level in self.flow_levels[::-1]:
            _feat1, _feat2 = feat1[level], feat2[level]

            if level == self.start_level:
                corr_feat = self.corr_block(_feat1, _feat2)
            else:
                warp_feat = self.warp(_feat2, upflow * self.multiplier[level])
                corr_feat_ = self.corr_block(_feat1, warp_feat)
                corr_feat = torch.cat((corr_feat_, _feat1, upflow, upfeat),
                                      dim=1)

            flow, feat, upflow, upfeat = self.decoders[level](corr_feat)

            flow_pred[level] = flow

        if self.post_processor is not None:
            post_flow = self.post_processor(feat)
            flow_pred[self.end_level] = flow_pred[self.end_level] + post_flow

        return flow_pred, self.end_level


@MODEL_REGISTRY.register()
class PWCNetV2(BaseModule):
    """PWC-Net model.
    Args:
        encoder (dict): The config of encoder.
        decoder (dict): The config of decoder.
        init_cfg (list, dict, optional): Config of dict weights initialization.
            Default: None.
    """

    def __init__(self, cfg):

        super(PWCNetV2, self).__init__()
        self.ezflow_cfg = cfg

        self.encoder = PWCNetEncoder(
            cfg,
            in_channels=3,
            pyramid_levels=[
                'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
            ],
            net_type='Basic',
            out_channels=(16, 32, 64, 96, 128, 196),
            strides=(2, 2, 2, 2, 2, 2),
            dilations=(1, 1, 1, 1, 1, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
        )

        self.decoder = PWCNetDecoder(
            cfg, 
            in_channels=dict(
                level6=81, level5=213, level4=181, level3=149, level2=117
            ),
            flow_div=20.0,
            corr_cfg=dict(type='Correlation', max_displacement=4, padding=0),
            warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            scaled=False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def extract_feat(self, img1, img2):
        """Extract features from images.
        Args:
            imgs (Tensor): The concatenated input images.
        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]: The feature pyramid of
                the first input image and the feature pyramid of secode input
                image.
        """

        img1 = 1.0 * (img1 / 255.0)
        img2 = 1.0 * (img2 / 255.0)

        return self.encoder(img1), self.encoder(img2)

    def forward(self, img1, img2):
        """Forward function for PWCNet when model training.
        Args:
            imgs (Tensor): The concatenated input images.
            flow_gt (Tensor): The ground truth of optical flow.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.
        Returns:
            Dict[str, Tensor]: The losses of output.
        """

        H, W = img1.shape[-2:]
        feat1, feat2 = self.extract_feat(img1, img2)

        flow_pred, end_level = self.decoder(feat1=feat1, feat2=feat2)

        if self.training:
            return flow_pred
        else:
            flow_result = flow_pred[end_level]
            # resize flow to the size of images after augmentation.
            flow_result = F.interpolate(
                    flow_result, 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                )

            flow_result = flow_result * 20.0

            return flow_result, flow_pred

