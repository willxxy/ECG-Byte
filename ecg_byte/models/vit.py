import torch
import torch.nn as nn

class VIT(nn.Module):
    def __init__(self, vit, args, device = None):
        super(VIT, self).__init__()
        self.vit = vit
        if device == None:
            self.device = self.vit.device
        else:
            self.device = device
            
    def forward(self, batch):
        out = self.vit( pixel_values = batch['vit_pixel'].to(self.device),
                        bool_masked_pos = batch['mask'].to(self.device),
                        output_hidden_states = True)
        return out

