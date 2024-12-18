import numpy as np
import argparse

import rust_bpe
from ecg_byte.utils.tokenizer_utils import normalize_all, load_vocab_and_merges, track_encoding
from ecg_byte.utils.viz_utils import generate_distinct_colors, visualize_bpe_encoding

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--tokenizer', type = str, default = 'tokenizer_3500_100_mimic2', help='Please choose the tokenizer')
    parser.add_argument('--sample_signal', type = str, default = './data/seg_ecg_qa_ptb/ecg/train/ecg_299_1.npy', help='Please specify the path to a sample ecg')
    parser.add_argument('--percentiles', type = str, default = './data/mimic_dataset_stats_500_unseg.npy', help='Please specify the path to the percentiles')
    return parser.parse_args()

def main():
    desired_order = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    signal = np.load(args.sample_signal)
    vocab, merges = load_vocab_and_merges(f'./data/{args.tokenizer}.pkl')
    percentiles = np.load(args.percentiles, allow_pickle=True).item()
    _, str_signal = normalize_all(signal, percentiles)
    str_signal = ''.join(str_signal.flatten())
    encoded_ids = rust_bpe.encode_text(str_signal, merges)
    print(len(encoded_ids))
    
    count_len = 0
    
    global_id_to_color = {}
    norm_full_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    for lead in range(signal.shape[0]):
        single_lead = signal[lead]
        norm_signal = norm_full_signal[lead]
        lead_name = desired_order[lead]
        _, single_lead_str = normalize_all(single_lead, percentiles)
        single_lead_str = ''.join(single_lead_str.flatten())
        single_lead_encoded_ids = rust_bpe.encode_text(single_lead_str, merges)
        encoded_ids, segment_map = track_encoding(single_lead_str, merges)
        
        new_ids = set(encoded_ids) - set(global_id_to_color.keys())
        if new_ids:
            new_colors = generate_distinct_colors(len(new_ids))
            global_id_to_color.update(zip(sorted(new_ids), new_colors))
        
        visualize_bpe_encoding(norm_signal, encoded_ids, segment_map, 
                            lead_index=lead, 
                            lead_name=lead_name,
                            id_to_color=global_id_to_color)
        count_len += len(single_lead_encoded_ids)

    print(count_len)
    
if __name__ == '__main__':
    main()