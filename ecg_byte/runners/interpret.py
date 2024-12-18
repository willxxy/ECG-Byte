from tqdm import tqdm
import torch
import time
import numpy as np
import re
from ecg_byte.utils.tokenizer_utils import decode_text, reverse_normalize_all
from ecg_byte.utils.viz_utils import plot_attention_on_signal, plot_text_attention_weights

def get_component_indices(batch, tokenizer):
    tokenized_seq = batch['tokenized_signal'][0]
    labels = batch['quantized_signal_ids_input'][0] if 'quantized_signal_ids_input' in batch else None
    print(f"Tokenized sequence: {tokenized_seq}")
    print(f"Labels: {labels}")
    signal_start = 0
    for i in range(len(tokenized_seq)):
        if tokenized_seq[i] == tokenizer.convert_tokens_to_ids('<sig_start>'):
            signal_start = i + 1
            break
        
    question_start = signal_start
    for i in range(signal_start, len(tokenized_seq)):
        if tokenized_seq[i] == tokenizer.convert_tokens_to_ids('<sig_end>'):
            question_start = i + 1
            break
    
    answer_start = len(tokenized_seq)
    if labels is not None:
        for i in range(question_start, len(labels)):
            if labels[i] != -100 and labels[i] != tokenizer.pad_token_id:
                answer_start = i
                break
    
    print(f"Sequence length: {len(tokenized_seq)}")
    print(f"Indices - signal: {signal_start}, question: {question_start}, answer: {answer_start}")
    
    return signal_start, question_start, answer_start

def interpreter(model, dataloader, tokenizer, vocab, args):
    model.eval()
    signal_attentions, signal_seqs, signal_decodes = [], [], []
    question_attentions, question_seqs = [], []
    answer_attentions, answer_seqs = [], []
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Interpreting {args.model}', position=0, leave=True):
            if batch is None:
                continue
            
            signal_start, question_start, answer_start = get_component_indices(batch, tokenizer)
            
            out = model(batch)
            attention = torch.stack(out.attentions).mean(dim=(0, 2))[0]
            sequence = batch['tokenized_signal'][0]
            
            seq_len = len(sequence)
            signal_seq = sequence[signal_start:question_start]
            signal_att = attention[signal_start:question_start, signal_start:question_start].mean(dim=0)
            
            question_seq = sequence[question_start:answer_start]
            if len(question_seq) > 0:
                question_att = attention[question_start:answer_start, question_start:answer_start].mean(dim=0)
            else:
                continue
            
            answer_seq = sequence[answer_start:seq_len-1]
            if len(answer_seq) > 0:
                answer_att = attention[answer_start:seq_len-1, answer_start:seq_len-1].mean(dim=0)
            else:
                continue
            
            signal_seqs.append(signal_seq.cpu().numpy())
            signal_attentions.append(signal_att.cpu().float().numpy())
            question_seqs.append(question_seq.cpu().numpy())
            question_attentions.append(question_att.cpu().float().numpy())
            answer_seqs.append(answer_seq.cpu().numpy())
            answer_attentions.append(answer_att.cpu().float().numpy())
            
            signal_decoded = tokenizer.decode(signal_seq.cpu().numpy(), skip_special_tokens=True)
            signal_decoded = re.findall(r'signal_(\d+)', signal_decoded)
            signal_decoded = [int(i) for i in signal_decoded]
            expanded_attention = expand_attention(signal_decoded, signal_att.cpu().float().numpy(), vocab)
            attention_array = np.array(expanded_attention)
            attention_array = attention_array.reshape((12, 500))
            signal_decoded = decode_text(signal_decoded, vocab)
            reversed_signal = reverse_normalize_all(np.array(list(signal_decoded)).reshape((12, 500)), dataloader.dataset.percentiles)
            signal_decodes.append(signal_decoded)
            answer_tokens = [tokenizer.decode(token_id) for token_id in answer_seq.cpu().numpy()]
            question_tokens = [tokenizer.decode(token_id) for token_id in question_seq.cpu().numpy()]
            
            if count <=20:
                for i in range(12):
                    plot_attention_on_signal(batch['signal'][0].cpu().numpy(), attention_array, i, count)
                plot_text_attention_weights(question_tokens + answer_tokens, np.concatenate([question_att.cpu().float().numpy(), answer_att.cpu().float().numpy()]), count)
            count +=1
            
            if args.dev and len(signal_seqs) >= 5:
                break
    
    return {
        'signal': {'sequences': signal_seqs, 'attentions': signal_attentions, 'signal' : signal_decodes},
        'question': {'sequences': question_seqs, 'attentions': question_attentions},
        'answer': {'sequences': answer_seqs, 'attentions': answer_attentions}
    }
    
def expand_attention(encoded_ids, attention_sequence, vocab):
    expanded_attention = []
    for id, attention_value in zip(encoded_ids, attention_sequence):
        length = len(vocab[id])
        expanded_attention.extend([attention_value] * length)
    return expanded_attention