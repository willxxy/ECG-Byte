import os
import pickle
import pickle
import json
import numpy as np
import glob
import re
import random

def load_vocab_and_merges(filename):
    with open(filename, 'rb') as f:
        vocab, merges = pickle.load(f)
    return vocab, merges

def ensure_directory_exists(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory ensured: {directory_path}")
    except Exception as e:
        print(f"Error ensuring directory {directory_path}: {str(e)}")

def load_vocab_and_merges(filename):
    with open(filename, 'rb') as f:
        vocab, merges = pickle.load(f)
    return vocab, merges

def open_json(path_to_file):
    with open(path_to_file) as json_file:
        return json.load(json_file)
    
def load_npy(file_path):
    return np.load(file_path)


def align_signal_text_files(signal_dir, text_dir):
    signal_files = glob.glob(os.path.join(signal_dir, '*.npy'))
    text_files = glob.glob(os.path.join(text_dir, '*.json'))

    def extract_indices(filename):
        match = re.search(r'(\d+)_(\d+)', os.path.basename(filename))
        if match:
            return tuple(map(int, match.groups()))
        return None

    signal_dict = {extract_indices(f): f for f in signal_files if extract_indices(f)}
    text_dict = {extract_indices(f): f for f in text_files if extract_indices(f)}

    common_indices = sorted(set(signal_dict.keys()) & set(text_dict.keys()))

    aligned_signals = [signal_dict[idx] for idx in common_indices]
    aligned_texts = [text_dict[idx] for idx in common_indices]

    return aligned_signals, aligned_texts


def sample_N_percent_indices(length, N=0.1):
    sample_size = max(1, int(length * N))  # Ensure at least 1 item is sampled
    return random.sample(range(length), sample_size)

def sample_N_percent_from_lists(list1, list2 = None, N=0.05):
    if list2 != None:
        if len(list1) != len(list2):
            raise ValueError("Both lists must have the same length")
    sampled_indices = sample_N_percent_indices(len(list1), N)
    sampled_list1 = [list1[i] for i in sampled_indices]
    if list2 == None:
        return sampled_list1
    sampled_list2 = [list2[i] for i in sampled_indices]
    return sampled_list1, sampled_list2