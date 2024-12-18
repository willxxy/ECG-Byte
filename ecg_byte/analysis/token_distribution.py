import rust_bpe
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import glob
import argparse

from ecg_byte.utils.tokenizer_utils import load_vocab_and_merges, analyze_token_distribution
from ecg_byte.utils.viz_utils import plot_distributions

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--tokenizer', type = str, default = 'tokenizer_1756_100_mimic2', help='Please choose the tokenizer')
    parser.add_argument('--list_of_paths', type = str, default = './data/seg_ecg_qa_ptb/ecg/val/*.npy', help='Please specify the path to the list of paths')
    parser.add_argument('--percentiles', type = str, default = './data/mimic_dataset_stats_500_unseg.npy', help='Please specify the path to the percentiles')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    list_of_paths = glob.glob(args.list_of_paths)
    _, loaded_merges = load_vocab_and_merges(f'./data/{args.tokenizer}.pkl')
    percentiles = np.load(args.percentiles, allow_pickle=True).item()
    
    token_counts, token_lengths = analyze_token_distribution(list_of_paths, loaded_merges, percentiles, 6)
    
    plot_distributions(token_counts, token_lengths, int(args.tokenizer.split('_')[1]))