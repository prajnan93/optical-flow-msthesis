import torch
import torch.nn as nn
import torch.nn.functional as F
from ezflow.config import configurable
from ezflow.encoder import ENCODER_REGISTRY, build_encoder
from ezflow.modules import BaseModule
from transformers import Swinv2Config, Swinv2Model


@ENCODER_REGISTRY.register()
class HuggingFaceSwinEncoderV2(nn.Module):
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
        window_size: int = 8,
        patch_size: int = 4,
        ff_feature_ratio: int = 4,
        dropout: float = 0.0,
        dropout_attention: float = 0.0,
        dropout_path: float = 0.2,
        use_checkpoint: bool = False,
        sequential_self_attention: bool = False,
    ) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param depth: (int) Depth of the stage (number of layers)
        :param downscale: (bool) If true input is downsampled (see Fig. 3 or V1 paper)
        :param input_resolution: (Tuple[int, int]) Input resolution
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in input mapping
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param use_checkpoint: (bool) If true checkpointing is utilized
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        """
        # Call super constructor
        super(HuggingFaceSwinEncoderV2, self).__init__()

        configuration = Swinv2Config()
        configuration.depths = depths
        configuration.embed_dim = embedding_channels
        configuration.num_heads = number_of_heads
        configuration.window_size = window_size

        self.swin_feature_extractor = Swinv2Model(
            configuration, add_pooling_layer=False
        )

    @classmethod
    def from_config(self, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "embedding_channels": cfg.EMBEDDING_CHANNELS,
            "depths": cfg.DEPTHS,
            "input_resolution": cfg.INPUT_RESOLUTION,
            "number_of_heads": cfg.NUMBER_OF_HEADS,
            "window_size": cfg.WINDOW_SIZE,
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
        _, c, h, w = input.shape

        output = self.swin_feature_extractor(input)[0]
        output = output.permute(0, 2, 1)

        h, w = int(h / 8), int(w / 8)
        b, c, _ = output.shape

        output = output.reshape(b, c, h, w)

        return output
