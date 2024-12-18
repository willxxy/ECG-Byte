import torch
import torch.nn as nn
from collections import namedtuple

CombinedOutput = namedtuple('CombinedOutput', ['loss', 'contrastive_loss', 'mlm_loss', 'clip_output', 'vit_output'])

class CLIPVIT(nn.Module):
    def __init__(self, clip_model, vit_model, args):
        super(CLIPVIT, self).__init__()
        self.clip = clip_model
        self.vit = vit_model
        self.contrastive_weight = 1
        self.mlm_weight = 1
        
    def forward(self, batch):
        # Get outputs from both models
        clip_output = self.clip(batch)
        vit_output = self.vit(batch)
        
        # Calculate combined loss
        contrastive_loss = clip_output.loss
        mlm_loss = vit_output.loss
        
        combined_loss = (self.contrastive_weight * contrastive_loss + 
                        self.mlm_weight * mlm_loss)
        
        return CombinedOutput(
            loss=combined_loss,
            contrastive_loss=contrastive_loss,
            mlm_loss=mlm_loss,
            clip_output=clip_output,
            vit_output=vit_output
        )