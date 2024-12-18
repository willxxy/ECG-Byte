import nltk
from scipy import stats
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
nltk.download('wordnet')
import torch
import torch.nn as nn
import numpy as np
from evaluate import load
from transformers import logging
logging.set_verbosity_error()

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def early_stopping(validation_losses, patience=5, delta=0):
    if len(validation_losses) < patience + 1:
        return False
    
    best_loss = min(validation_losses[:-patience])
    current_loss = validation_losses[-1]
    
    if current_loss > best_loss + delta:
        return True
    
    return False

def calculate_bleu(references, hypotheses):
    smoother = SmoothingFunction()
    return corpus_bleu([[r.split()] for r in references], [h.split() for h in hypotheses], smoothing_function = smoother.method1)

def calculate_meteor(references, hypotheses):
    return np.mean([meteor_score([r.split()], h.split()) for r, h in zip(references, hypotheses)])

def calculate_rouge(references, hypotheses):
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f']
    }
    
    
def calculate_bertscore(references, hypotheses, device):
    bertscore = load('bertscore')
    results = bertscore.compute(predictions = hypotheses,
                        references = references, lang = 'en', device = device)
    return {
        'hf-prec': results['precision'],
        'hf-rec': results['recall'],
        'hf-f1': results['f1']
    }

def evaluate_strings(references, hypotheses, device):
    if len(references) != len(hypotheses):
        raise ValueError("The number of references and hypotheses must be the same.")
    return {
        'BLEU': calculate_bleu(references, hypotheses),
        'METEOR': calculate_meteor(references, hypotheses),
        'ROUGE': calculate_rouge(references, hypotheses),
        'BERTSCORE': calculate_bertscore(references, hypotheses, device)
    }
    
    

def run_statistical_analysis(all_seeds_results):
    metrics = list(all_seeds_results[0]['metrics'].keys())
    statistical_results = {}
    
    for metric in metrics:
        values = [result['metrics'][metric] * 100 for result in all_seeds_results]
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # ddof=1 for sample standard deviation
        
        confidence = 0.95
        degrees_of_freedom = len(values) - 1
        t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
        margin_of_error = t_value * (std / np.sqrt(len(values)))
        
        conf_interval = (mean - margin_of_error, mean + margin_of_error)
        
        statistical_results[metric] = {
            'mean': mean,
            'std': std,
            'conf_interval': conf_interval,
            'raw_values': values
        }
    
    return statistical_results



def adapt_sequence(sig_embed, text_embed, token_ids, attn_mask, labels=None, position_ids=None, sig_id = 131757, ignore_index = -100):
    
    sig_positions = (token_ids == sig_id).nonzero(as_tuple=True)
    batch_indices = sig_positions[0]
    sig_indices = sig_positions[1]    
    combined_embeds_list = []
    labels_list = []
    position_ids_list = []
    attn_mask_list = []
    
    if labels != None:
        add_idx = 2
    else:
        add_idx = 1
    
    # a bit tricky to vectorize due to dynamic idx of sig token
    for batch_idx, sig_idx in zip(batch_indices, sig_indices):
        combined = torch.cat([
                text_embed[batch_idx:batch_idx+1, :sig_idx+1],  
                sig_embed[batch_idx:batch_idx+1],
                text_embed[batch_idx:batch_idx+1, sig_idx+add_idx:] 
            ], dim=1)

        original_mask = attn_mask[batch_idx]
        new_mask = torch.cat([
            original_mask[:sig_idx+1],
            torch.ones(1, dtype=original_mask.dtype, device=original_mask.device),
            original_mask[sig_idx+add_idx:]
        ], dim=0)

        # Insert ignore_index at the same position as the image token
        if labels != None:
            original_labels = labels[batch_idx]
            new_labels = torch.cat([
                original_labels[:sig_idx+1],
                torch.full((1,), ignore_index, dtype=original_labels.dtype, device=original_labels.device),
                original_labels[sig_idx+add_idx:]
            ], dim=0)

            # Insert a position_id after sig and increment subsequent positions by 1
            original_pos = position_ids[batch_idx]
            before = original_pos[:sig_idx+1]
            new_token_pos_id = before[-1] + 1
            after = original_pos[sig_idx+add_idx:] + 1
            new_position_ids = torch.cat([before, new_token_pos_id.unsqueeze(0), after], dim=0)
            
            labels_list.append(new_labels)
            position_ids_list.append(new_position_ids)

        combined_embeds_list.append(combined)
        attn_mask_list.append(new_mask)
    
    return_dic = {
        'combined_embeds': torch.cat(combined_embeds_list, dim=0),
        'attn_mask': torch.stack(attn_mask_list, dim=0),
    }
    
    if labels != None:
        return_dic['labels'] = torch.stack(labels_list, dim=0)
        return_dic['position_ids'] = torch.stack(position_ids_list, dim=0)
    
    return return_dic
    