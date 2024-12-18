import torch
import torch.nn as nn

class CLIP(nn.Module):
    def __init__(self, clip, args, device = None):
        super(CLIP, self).__init__()
        self.clip = clip
        if device == None:
            self.device = self.clip.device
        else:
            self.device = device
        
    def forward(self, batch):
        out = self.clip(input_ids = batch['clip_input_ids'].to(self.device),
                        attention_mask = batch['clip_att_mask'].to(self.device),
                        pixel_values = batch['clip_pixel'].to(self.device),
                        return_loss = True)
        return out

