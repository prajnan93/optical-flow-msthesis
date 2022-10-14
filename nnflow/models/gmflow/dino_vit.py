import torch
import torch.nn as nn
import torch.nn.functional as F

from ezflow.encoder import ENCODER_REGISTRY
from ezflow.config import configurable

_DINO_VIT = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

@ENCODER_REGISTRY.register()
class DinoVITS8(nn.Module):
    """
    This class is a wrapper for the DinoViT model without the classification head
    
    """

    @configurable
    def __init__(
        self,
        freeze=True,
        pretrained_ckpt_path=None
    ):
        
        super(DinoVITS8, self).__init__()
        
        self.freeze = freeze
        self.feature_extractor = _DINO_VIT
        
        if pretrained_ckpt_path is not None:
            self.feature_extractor.load_state_dict(
                torch.load(pretrained_ckpt_path)
            )
            print(f"Loaded Dino ViT S/8 pretrained checkpoint from {pretrained_ckpt_path}\n")

    @classmethod
    def from_config(self, cfg):
        return {
            "freeze": cfg.FREEZE,
            "pretrained_ckpt_path": cfg.PRETRAINED_CKPT_PATH
        }

    def forward(self, input):
        """
        Forward pass
        
        """
        _,c,h,w = input.shape

        if self.freeze:
            self.eval()
            self.feature_extractor.eval()
        
        output = self.feature_extractor.get_intermediate_layers(input, n=1)[0]
        
        # remove cls token
        output = output[:,1:,:]
        output = output.permute(0,2,1)

        h, w = int(h/8), int(w/8)
        b, c, _ = output.shape 

        output = output.reshape(b, c, h, w)

        return output