import numpy as np
import rust_bpe
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from collections import Counter

from ecg_byte.utils.file_utils import load_npy

ALPHABET = list('abcdefghijklmnopqrstuvwxyz')

def normalize_all(signal, percentiles):
    normalized = (signal - (percentiles['percentile_1'] -0.5)) / ((percentiles['percentile_99']+0.5) - (percentiles['percentile_1']-0.5) + 1e-6)
    clipped_normalized = np.clip(normalized, 0, 1)
    scaled_signal = np.minimum(np.floor(clipped_normalized * len(ALPHABET)), len(ALPHABET)-1).astype(np.uint8)
    symbol_signal = np.vectorize(lambda x: ALPHABET[x])(scaled_signal)
    return clipped_normalized, symbol_signal


def reverse_normalize_all(symbol_signal, percentiles):
    min_vals = percentiles['percentile_1'] - 0.5
    max_vals = percentiles['percentile_99'] + 0.5
    scaled_signal = np.vectorize(lambda x: ALPHABET.index(x))(symbol_signal)
    clipped_normalized = scaled_signal / (len(ALPHABET) - 1)
    original_signal = clipped_normalized * (max_vals - min_vals) + min_vals
    return original_signal

def analyze_token_distribution(test_data, merges, percentiles, num_workers=None):
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(analyze_single_ecg, ((path, merges, percentiles) for path in test_data)),
            total=len(test_data),
            desc=f'Analyzing token distribution with {num_workers} workers'
        ))
    
    token_counts = Counter()
    token_lengths = []
    for count, length in results:
        token_counts.update(count)
        token_lengths.append(length)
    
    return token_counts, token_lengths

def analyze_single_ecg(args):
    path, merges, percentiles = args
    signal = np.load(path)
    _, norm_signal = normalize_all(signal, percentiles)
    single_lead_str = ''.join(norm_signal.flatten())
    all_encoded_ids = encode_text(single_lead_str, merges)
    all_encoded_ids = list(all_encoded_ids)
    return Counter(all_encoded_ids), len(all_encoded_ids)

def process_ecg(ecg, percentiles):
    ecg = load_npy(ecg)
    _, symbol_signal = normalize_all(ecg, percentiles)
    return ''.join(symbol_signal.flatten())


def save_vocab_and_merges(vocab, merges, filename):
    with open(filename, 'wb') as f:
        pickle.dump((vocab, merges), f)

def load_vocab_and_merges(filename):
    with open(filename, 'rb') as f:
        vocab, merges = pickle.load(f)
    return vocab, merges

def encode_text(text, merges):
    ids = rust_bpe.encode_text(text, merges)
    return ids

def decode_text(encoded_ids, vocab):
    decoded_text = ''.join(vocab[id] for id in encoded_ids)
    return decoded_text

def process_large_file(file_path, percentiles, num_processes, n=None):
    def file_path_generator():
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if n is not None and i >= n:
                    break
                yield line.strip()
    
    file_paths = list(file_path_generator())
    
    with mp.Pool(processes=num_processes) as pool:
        partial_process_ecg = partial(process_ecg, percentiles=percentiles)
        ecg_strings = list(tqdm(pool.imap(partial_process_ecg, file_paths), total=len(file_paths), desc="Processing ECGs"))
    
    return ''.join(ecg_strings)

def track_encoding(text, merges, verbose = True):
    ids = list(text.encode('utf-8'))
    segment_map = [(i, i+1) for i in range(len(ids))]
    
    if verbose != True:
        for batch in merges:
            pair, new_id = batch
            new_ids = []
            new_segment_map = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                    new_ids.append(new_id)
                    new_segment_map.append((segment_map[i][0], segment_map[i+1][1]))
                    i += 2
                else:
                    new_ids.append(ids[i])
                    new_segment_map.append(segment_map[i])
                    i += 1
            ids = new_ids
            segment_map = new_segment_map
    else:
        for batch in tqdm(merges, desc = 'Tracking Encoding'):
            pair, new_id = batch
            new_ids = []
            new_segment_map = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                    new_ids.append(new_id)
                    new_segment_map.append((segment_map[i][0], segment_map[i+1][1]))
                    i += 2
                else:
                    new_ids.append(ids[i])
                    new_segment_map.append(segment_map[i])
                    i += 1
            ids = new_ids
            segment_map = new_segment_map
    
    return ids, segment_map
