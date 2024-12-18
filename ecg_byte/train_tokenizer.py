import numpy as np
import time
import argparse

from ecg_byte.utils.tokenizer_utils import *
from ecg_byte.utils.viz_utils import plot_original_vs_decoded

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_merges', type = int, default = 3500, help='Please choose the vocabulary size')
    parser.add_argument('--sampled_files', type=str, default = None, help='Please specify the path to the .txt file of sampled ecgs')
    parser.add_argument('--num_processes', type=int, default=2, help='Number of processes for multiprocessing')
    parser.add_argument('--percentiles', type=str, default = None, help = 'Please specify the path to the calculated percentiles')
    parser.add_argument('--train', action = 'store_true', default = None, help = 'Please specify whether to train the tokenizer')
    parser.add_argument('--loaded', type = str, default = None, help = 'If you want to just load the tokenizer, please specify the path to the .pkl file.')
    return parser.parse_args()


def main(args):
    percentiles = np.load(args.percentiles, allow_pickle=True).item()

    if args.train:
        num_processes = args.num_processes
        num_merges = args.num_merges
        all_string_signals = process_large_file(args.sampled_files, percentiles, args.num_processes)
        print(f"Total ECGs processed: {len(list(all_string_signals))}")
        print(list(all_string_signals)[:100])
        start_time = time.time()
        ids, vocab, merges = rust_bpe.byte_pair_encoding(all_string_signals, num_merges, num_processes)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Byte pair encoding executed in {execution_time:.2f} seconds")
        print("Shared vocabulary across all ECGs:")
        print(f"Original length: {len(all_string_signals)}")
        print(f"Encoded length: {len(ids)}")
        print(f"Compression ratio: {len(all_string_signals) / len(ids):.2f}X")
        print(f"Vocabulary size: {len(vocab)}")
        tokenizer_file_name = f'./data/tokenizer_{args.num_merges}.pkl'
        save_vocab_and_merges(vocab, merges, tokenizer_file_name)
        print(f"Vocabulary and merges saved to {tokenizer_file_name}")
    ###
    if args.loaded == None:
        args.loaded = tokenizer_file_name
    loaded_vocab, loaded_merges = load_vocab_and_merges(args.loaded)
    print(f"Loaded vocabulary and merges from {args.loaded}")

    new_ecg_signal = np.load('./data/seg_ecg_qa_ptb_500/ecg/train/ecg_10_1.npy')
    new_ecg_text = process_ecg('./data/seg_ecg_qa_ptb_500/ecg/train/ecg_10_1.npy', percentiles=percentiles)
    print(f"Processed ECG signal to text (first 100 characters): {new_ecg_text[:100]}...")
    print(f"Total tokens: {len(new_ecg_text)}")

    # # Encode the new ECG text
    encoded_ecg = encode_text(new_ecg_text, loaded_merges)
    print(f"Encoded ECG (first 20 tokens): {encoded_ecg[:20]}...")
    print(f"Total tokens: {len(encoded_ecg)}")
    print(f"Compression ratio: {len(new_ecg_text) / len(encoded_ecg):.2f}X")

    decoded_text = decode_text(encoded_ecg, loaded_vocab)
    print(f"Decoded text (first 100 characters): {decoded_text[:100]}...")
    print(decoded_text == new_ecg_text)

    decoded_signal = reverse_normalize_all(np.array(list(decoded_text)).reshape(new_ecg_signal.shape), percentiles)
    max_diff = np.max(np.abs(new_ecg_signal - decoded_signal))
    print(f"Maximum difference between original and decoded: {max_diff}")

    plot_original_vs_decoded(decoded_signal, lead_index=5, original_array = new_ecg_signal)
    
if __name__ == "__main__":
    args = get_args()
    main(args)