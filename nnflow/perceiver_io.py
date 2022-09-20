import torch
import torch.nn as nn
import torch.nn.functional as F

from ezflow.models import MODEL_REGISTRY
from ezflow.config import configurable
from ezflow.modules import BaseModule

from transformers import PerceiverModel, PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import PerceiverImagePreprocessor, PerceiverOpticalFlowDecoder

config_dict = {
      "_name_or_path": "deepmind/optical-flow-perceiver",
      "architectures": [
        "PerceiverForOpticalFlow"
      ],
      "attention_probs_dropout_prob": 0.1,
      "audio_samples_per_frame": 1920,
      "cross_attention_shape_for_attention": "kv",
      "cross_attention_widening_factor": 1,
      "d_latents": 512,
      "d_model": 322,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "image_size": 56,
      "initializer_range": 0.02,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 2048,
      "model_type": "perceiver",
      "num_blocks": 1,
      "num_cross_attention_heads": 1,
      "num_frames": 16,
      "num_latents": 2048,
      "num_self_attends_per_block": 24,
      "num_self_attention_heads": 16,
      "output_shape": [
        1,
        16,
        224,
        224
      ],
      "qk_channels": None,
      "samples_per_patch": 16,
      "self_attention_widening_factor": 1,
      "seq_len": 2048,
      "torch_dtype": "float32",
      "train_size": [
        368,
        496
      ],
      "transformers_version": "4.21.3",
      "use_query_residual": True,
      "v_channels": None,
      "vocab_size": 262
}

@MODEL_REGISTRY.register()
class Perceiver(BaseModule):
    """
    Implementation of PerceiverIO Optical Flow
    https://www.deepmind.com/open-source/perceiver-io
    https://huggingface.co/docs/transformers/v4.21.3/en/model_doc/perceiver#transformers.PerceiverForOpticalFlow
    https://github.com/huggingface/transformers/blob/v4.21.3/src/transformers/models/perceiver/modeling_perceiver.py#L1612
    
    example: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Perceiver/Perceiver_for_Optical_Flow.ipynb

    Parameters
    ----------
    cfg : :class:`CfgNode`
        Configuration for the model
    """

    def __init__(self, cfg):
        super(Perceiver, self).__init__()
        self.cfg = config_dict
        
        config = PerceiverConfig(**config_dict)
        
        fourier_position_encoding_kwargs_preprocessor = dict(
            num_bands=64,
            max_resolution=config.train_size,
            sine_only=False,
            concat_pos=True,
        )
        fourier_position_encoding_kwargs_decoder = dict(
            concat_pos=True, max_resolution=config.train_size, num_bands=64, sine_only=False
        )
        
        image_preprocessor = PerceiverImagePreprocessor(
            config,
            prep_type="patches",
            spatial_downsample=1,
            conv_after_patching=True,
            conv_after_patching_in_channels=54,
            temporal_downsample=2,
            position_encoding_type="fourier",
            # position_encoding_kwargs
            fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
        )
        
        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=image_preprocessor,
            decoder=PerceiverOpticalFlowDecoder(
                config,
                num_channels=image_preprocessor.num_channels,
                output_image_shape=config.train_size,
                rescale_factor=100.0,
                use_query_residual=False,
                output_num_channels=2,
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_decoder,
            ),
        )
        
    
    # source: https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/9
    def _extract_image_patches(self, x, kernel=3, stride=1, dilation=1):
        # Do TF 'SAME' Padding
        b,c,h,w = x.shape
        h2 = math.ceil(h / stride)
        w2 = math.ceil(w / stride)
        pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
        pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
        x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))

        # Extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.permute(0,4,5,1,2,3).contiguous()
        
        return patches.view(b,-1,patches.shape[-2], patches.shape[-1])
        
    def forward(self, img1, img2):
        
        B, C, H, W = img1.shape
        
        patches = self._extract_image_patches(torch.concat([img1, img2], dim=0))
        _, C, H, W = patches.shape
        patches = patches.view(B, -1, C, H, W)
        
        flow = self.perceiver(
            inputs=patches,
            return_dict=False
        )[0]
        
        flow = flow.permute(0, 3, 1, 2)
        
        output = {"flow_preds": flow}
        
        if self.training:
            return output
        
        output["flow_upsampled"] = flow
        
        return output
        