import torch
import torch.nn as nn
import torch.nn.functional as F

from ezflow.encoder import ENCODER_REGISTRY,build_encoder
from ezflow.modules import BaseModule
from ezflow.config import configurable

from transformers import ViTModel, ViTConfig

@ENCODER_REGISTRY.register()
class ViTEncoder(nn.Module):
    """
    This class implements the Swin Transformer without classification head.
    """

    @configurable
    def __init__(
        self,
        in_channels=3,
        embedding_channels=96,
        depths=(2, 2),
        input_resolution=(256, 256),
        number_of_heads=(3, 6, 12, 24),
        intermediate_size: int = 768,
        patch_size: int = 4,
        ff_feature_ratio: int = 4,
        dropout: float = 0.0,
        dropout_attention: float = 0.0,
        dropout_path: float = 0.2,
        use_checkpoint: bool = False,
        sequential_self_attention: bool = False,
    ) -> None:
        """
        """
        # Call super constructor
        super(ViTEncoder, self).__init__()

        configuration = ViTConfig()
        
        configuration.patch_size=patch_size
        configuration.hidden_size=embedding_channels
        configuration.num_hidden_layers=depths
        configuration.num_attention_heads=number_of_heads
        configuration.intermediate_size=intermediate_size
        

        self.vit_feature_extractor = ViTModel(configuration, add_pooling_layer=False)

    @classmethod
    def from_config(self, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "embedding_channels": cfg.EMBEDDING_CHANNELS,
            "depths": cfg.DEPTHS,
            "input_resolution": cfg.INPUT_RESOLUTION,
            "number_of_heads": cfg.NUMBER_OF_HEADS,
            "intermediate_size": cfg.INTERMEDIATE_SIZE,
            "patch_size": cfg.PATCH_SIZE,
            "ff_feature_ratio": cfg.FF_FEATURE_RATIO,
            "dropout": cfg.DROPOUT,
            "dropout_attention": cfg.DROPOUT_ATTENTION,
            "dropout_path": cfg.DROPOUT_PATH,
            "use_checkpoint": cfg.USE_CHECKPOINT,
            "sequential_self_attention": cfg.SEQUENTIAL_SELF_ATTENTION,
        }

    def forward(self, input):
        """
        Forward pass
        
        """
        _,c,h,w = input.shape

        output = self.vit_feature_extractor(input, interpolate_pos_encoding=True).last_hidden_state
        
        # remove cls token
        output = output[:,1:,:]
        output = output.permute(0,2,1)

        h, w = int(h/8), int(w/8)
        b, c, _ = output.shape 

        output = output.reshape(b, c, h, w)

        return output