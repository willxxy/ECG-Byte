import torch
import torch.nn as nn

from ecg_byte.utils.model_utils import adapt_sequence

### LLM FOR ECG-Byte
class LLM(nn.Module):
    def __init__(self, llm, args):
        super(LLM, self).__init__()
        self.args = args
        self.llm = llm
        if self.args.interpret:
            self.output_attentions = True
        else:
            self.output_attentions = False
            
    def forward(self, batch):
        out = self.llm(input_ids = batch['tokenized_signal'].to(self.llm.device),
                    attention_mask = batch['attn_mask'].to(self.llm.device),
                    labels = batch['quantized_signal_ids_input'].to(self.llm.device),
                    position_ids = batch['position_ids'].to(self.llm.device),
                    output_attentions = self.output_attentions # this causes OOM during training so set it as False
                    )
        return out
    
    def generate(self, batch, tokenizer):
        input_len = batch['tokenized_signal'].shape[1]
        generated_ids = self.llm.generate(
                input_ids=batch['tokenized_signal'].to(self.llm.device),
                attention_mask=batch['attn_mask'].to(self.llm.device),
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        decoded_text = tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return decoded_text 
    

### LLMs FOR BASELINES

class CLIP_LLM(nn.Module):
    def __init__(self, llm, clip, args):
        super(CLIP_LLM, self).__init__()
        self.args = args
        self.llm = llm
        self.clip = clip
        for param in self.clip.parameters():
            param.requires_grad = False
        self.get_input_embeddings = self.llm.get_input_embeddings
        self.image_projection = nn.Linear(512, self.llm.config.hidden_size)
        self.image_projection = self.image_projection.to(dtype=self.llm.dtype)
        
    def forward(self, batch):
        
        self.clip.eval()
        with torch.no_grad():
            out = self.clip(
                input_ids=batch['clip_input_ids'].to(self.clip.device),
                attention_mask=batch['clip_att_mask'].to(self.clip.device),
                pixel_values=batch['clip_pixel'].to(self.clip.device),
                return_loss=False
            )
        image_embeds = out.image_embeds.to(dtype=self.llm.dtype)
        projected_image = self.image_projection(image_embeds)  # [b, hidden_size]
        token_ids = batch['tokenized_signal'].to(self.llm.device)
        
        adapted_sequences = adapt_sequence(projected_image.unsqueeze(1), self.get_input_embeddings()(token_ids), 
                                           token_ids, batch['attn_mask'].to(self.llm.device), 
                                           batch['quantized_signal_ids_input'].to(self.llm.device), batch['position_ids'].to(self.llm.device))
        out = self.llm(
            inputs_embeds=adapted_sequences['combined_embeds'],
            attention_mask=adapted_sequences['attn_mask'],
            labels=adapted_sequences['labels'],
            position_ids=adapted_sequences['position_ids'],
        )
        return out
    
    def generate(self, batch, tokenizer):
        self.clip.eval()
        with torch.no_grad():
            out = self.clip(
                input_ids=batch['clip_input_ids'].to(self.clip.device),
                attention_mask=batch['clip_att_mask'].to(self.clip.device),
                pixel_values=batch['clip_pixel'].to(self.clip.device),
                return_loss=False
            )
        image_embeds = out.image_embeds.to(dtype=self.llm.dtype)
        projected_image = self.image_projection(image_embeds)  # [b, hidden_size]
        token_ids = batch['tokenized_signal'].to(self.llm.device)
        
        adapted_sequences = adapt_sequence(projected_image.unsqueeze(1), self.get_input_embeddings()(token_ids), 
                                           token_ids, batch['attn_mask'].to(self.llm.device))

        generated_ids = self.llm.generate(
            input_ids = batch['tokenized_signal2'].to(self.llm.device),
            inputs_embeds=adapted_sequences['combined_embeds'],
            attention_mask=adapted_sequences['attn_mask'],
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        # Compute the length of the input to slice the generated tokens correctly
        input_len = combined_embeds.shape[1]
        decoded_text = tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return decoded_text
    
    
    
class VIT_LLM(nn.Module):
    def __init__(self, llm, vit, args):
        super(VIT_LLM, self).__init__()
        self.args = args
        self.llm = llm
        self.vit = vit
        for param in self.vit.parameters():
            param.requires_grad = False
        self.get_input_embeddings = self.llm.get_input_embeddings
        self.image_projection = nn.Linear(768, self.llm.config.hidden_size)
        self.image_projection = self.image_projection.to(dtype=self.llm.dtype)
        
    def forward(self, batch):
        self.vit.eval()
        with torch.no_grad():
            out = self.vit(
                pixel_values=batch['vit_pixel'].to(self.vit.device),
                bool_masked_pos=batch['mask'].to(self.vit.device),
                output_hidden_states=True
            )
        all_hidden_states = torch.stack(out.hidden_states)    
        averaged_layers = torch.mean(all_hidden_states, dim=0)
        image_embeds = torch.mean(averaged_layers, dim=1).to(dtype=self.llm.dtype)    
        projected_image = self.image_projection(image_embeds)  # [b, hidden_size]
        token_ids = batch['tokenized_signal'].to(self.llm.device)
    
        adapted_sequences = adapt_sequence(projected_image.unsqueeze(1), self.get_input_embeddings()(token_ids), 
                                           token_ids, batch['attn_mask'].to(self.llm.device), 
                                           batch['quantized_signal_ids_input'].to(self.llm.device), batch['position_ids'].to(self.llm.device))

        # Run through the LLM
        out = self.llm(
            inputs_embeds=adapted_sequences['combined_embeds'],
            attention_mask=adapted_sequences['attn_mask'],
            labels=adapted_sequences['labels'],
            position_ids=adapted_sequences['position_ids'],
        )
        return out
    
    def generate(self, batch, tokenizer):
        self.vit.eval()
        with torch.no_grad():
            out = self.vit(
                pixel_values=batch['vit_pixel'].to(self.vit.device),
                bool_masked_pos=batch['mask'].to(self.vit.device),
                output_hidden_states = True
            )
        all_hidden_states = torch.stack(out.hidden_states)    
        averaged_layers = torch.mean(all_hidden_states, dim=0)
        image_embeds = torch.mean(averaged_layers, dim=1).to(dtype=self.llm.dtype)
        projected_image = self.image_projection(image_embeds)  # [b, hidden_size]
        token_ids = batch['tokenized_signal'].to(self.llm.device)
        
        adapted_sequences = adapt_sequence(projected_image.unsqueeze(1), self.get_input_embeddings()(token_ids), 
                                           token_ids, batch['attn_mask'].to(self.llm.device))

        generated_ids = self.llm.generate(
            input_ids = batch['tokenized_signal2'].to(self.llm.device), # we pass in input_ids since we are using inputs_embeds // try with only inputs_embeds
            inputs_embeds=adapted_sequences['combined_embeds'],
            attention_mask=adapted_sequences['attn_mask'],
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        # Compute the length of the input to slice the generated tokens correctly
        input_len = combined_embeds.shape[1]
        decoded_text = tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return decoded_text
    
    
    
class CLIP_VIT_LLM(nn.Module):
    def __init__(self, llm, clip, vit, device, args):
        super(CLIP_VIT_LLM, self).__init__()
        self.args = args
        self.llm = llm
        self.vit = vit
        self.clip = clip
        self.device = device
        
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.clip.parameters():
            param.requires_grad = False
            
        self.get_input_embeddings = self.llm.get_input_embeddings
        
        self.clip_projection = nn.Linear(512, self.llm.config.hidden_size)
        self.vit_projection = nn.Linear(768, self.llm.config.hidden_size)
        
        self.visual_fusion = nn.Sequential(
            nn.Linear(2 * self.llm.config.hidden_size, self.llm.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        )
        
        self.clip_projection = self.clip_projection.to(dtype=self.llm.dtype)
        self.vit_projection = self.vit_projection.to(dtype=self.llm.dtype)
        self.visual_fusion = self.visual_fusion.to(dtype=self.llm.dtype)

    def process_visual_features(self, batch):
        """Process and combine visual features from both models"""
        self.clip.eval()
        self.vit.eval()
        
        with torch.no_grad():
            clip_out = self.clip(batch)
            clip_embeds = clip_out.image_embeds.to(dtype=self.llm.dtype)
            projected_clip = self.clip_projection(clip_embeds)
            
            vit_out = self.vit(batch)
            all_hidden_states = torch.stack(vit_out.hidden_states)    
            averaged_layers = torch.mean(all_hidden_states, dim=0)
            vit_embeds = torch.mean(averaged_layers, dim=1).to(dtype=self.llm.dtype)
            projected_vit = self.vit_projection(vit_embeds)
            
            combined_visual = torch.cat([projected_clip, projected_vit], dim=-1)
            fused_visual = self.visual_fusion(combined_visual)
            
        return fused_visual.unsqueeze(1)

    def forward(self, batch):
        fused_visual = self.process_visual_features(batch)
        token_ids = batch['tokenized_signal'].to(self.llm.device)
        adapted_sequences = adapt_sequence(fused_visual, self.get_input_embeddings()(token_ids), 
                                           token_ids, batch['attn_mask'].to(self.llm.device), 
                                           batch['quantized_signal_ids_input'].to(self.llm.device), batch['position_ids'].to(self.llm.device))
        
        out = self.llm(
            inputs_embeds=adapted_sequences['combined_embeds'],
            attention_mask=adapted_sequences['attn_mask'],
            labels=adapted_sequences['labels'],
            position_ids=adapted_sequences['position_ids'],
        )
        return out

    def generate(self, batch, tokenizer):
        fused_visual = self.process_visual_features(batch)
        token_ids = batch['tokenized_signal'].to(self.llm.device)
        adapted_sequences = adapt_sequence(fused_visual, self.get_input_embeddings()(token_ids), 
                                           token_ids, batch['attn_mask'].to(self.llm.device))
        
        # Generate text
        generated_ids = self.llm.generate(
            input_ids=batch['tokenized_signal2'].to(self.device),
            inputs_embeds=adapted_sequences['combined_embeds'],
            attention_mask=adapted_sequences['attn_mask'],
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        
        # Decode generated text
        input_len = combined_embeds.shape[1]
        decoded_text = tokenizer.batch_decode(generated_ids[:, input_len:], 
                                            skip_special_tokens=True, 
                                            clean_up_tokenization_spaces=False)[0]
        return decoded_text
    
    
    
class ResNet_LLM(nn.Module):
    def __init__(self, llm, resnet, args):
        super(ResNet_LLM, self).__init__()
        self.args = args
        self.llm = llm
        self.resnet = resnet.to(dtype=torch.float32)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.get_input_embeddings = self.llm.get_input_embeddings
        self.image_projection = nn.Linear(2048, self.llm.config.hidden_size)
        self.image_projection = self.image_projection.to(dtype=self.llm.dtype)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        
    def forward(self, batch):
        
        self.resnet.eval()
        with torch.no_grad():
            out = self.resnet(batch)
        image_embeds = out.out.to(dtype=self.llm.dtype)
        image_embeds = self.avgpool(image_embeds)
        image_embeds = image_embeds.squeeze(2)
        
        projected_image = self.image_projection(image_embeds)  # [b, hidden_size]
        token_ids = batch['tokenized_signal'].to(self.llm.device)
        adapted_sequences = adapt_sequence(projected_image.unsqueeze(1), self.get_input_embeddings()(token_ids), 
                                           token_ids, batch['attn_mask'].to(self.llm.device), 
                                           batch['quantized_signal_ids_input'].to(self.llm.device), batch['position_ids'].to(self.llm.device))
        
        out = self.llm(
            inputs_embeds=adapted_sequences['combined_embeds'],
            attention_mask=adapted_sequences['attn_mask'],
            labels=adapted_sequences['labels'],
            position_ids=adapted_sequences['position_ids'],
        )
        return out
    
    def generate(self, batch, tokenizer):
        self.resnet.eval()
        with torch.no_grad():
            out = self.resnet(batch)
        image_embeds = out.out.to(dtype=self.llm.dtype)
        image_embeds = self.avgpool(image_embeds)
        image_embeds = image_embeds.squeeze(2)
        projected_image = self.image_projection(image_embeds)  # [b, hidden_size]
        token_ids = batch['tokenized_signal'].to(self.llm.device)
        
        adapted_sequences = adapt_sequence(projected_image.unsqueeze(1), self.get_input_embeddings()(token_ids), 
                                           token_ids, batch['attn_mask'].to(self.llm.device))
        generated_ids = self.llm.generate(
            input_ids = batch['tokenized_signal2'].to(self.llm.device),
            inputs_embeds=adapted_sequences['combined_embeds'],
            attention_mask=adapted_sequences['attn_mask'],
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        # Compute the length of the input to slice the generated tokens correctly
        input_len = combined_embeds.shape[1]
        decoded_text = tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return decoded_text
