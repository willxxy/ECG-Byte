from sklearn.model_selection import train_test_split
import argparse
import glob

from ecg_byte.utils.preprocess_utils import *
from ecg_byte.utils.file_utils import open_json

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seg_len', type=int, default=None, help='Please choose the segment length')
    parser.add_argument('--data', type = str, default = None, help = 'Please choose the dataset')
    return parser.parse_args()

def main(args):
    if args.data == 'mimic':
        all_data = open_json('./data/ecg_chat_data/pretrain_mimic.json')
    elif args.data == 'ptb':
        sampling_frequency=500
        datafolder='./data/ptb/'
        task='superdiagnostic'
        outputfolder='./data/ptb/'
        preprocess_ptb(datafolder, outputfolder, task, sampling_frequency, args)
    elif args.data == 'ecg_qa_mimic':
        all_data = glob.glob('./data/ecg-qa/output/mimic-iv-ecg/template/*/*.json') + glob.glob('./data/ecg-qa/output/mimic-iv-ecg/paraphrased/*/*.json')
        all_data = setup_ecg_qa(all_data)
    elif args.data == 'ecg_qa_ptb':
        all_data = glob.glob('./data/ecg-qa/output/ptbxl/template/*/*.json') + glob.glob('./data/ecg-qa/output/ptbxl/paraphrased/*/*.json')
        all_data = setup_ecg_qa(all_data)
        
    # Process and save each split
    if args.data != 'ptb':
        print('Total amount of instances: ', len(all_data))
        
        if args.seg_len == 2500:
            dataset_stats = compute_global_stats(all_data, args)
            np.save(f'./data/{args.data}_dataset_stats.npy', dataset_stats)
        
        indices = np.arange(len(all_data))
        train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.6, random_state=42)
        
        print('Train size', len(train_indices))
        print('Val size', len(val_indices))
        print('Test size', len(test_indices))
        
        train_instances = [all_data[i] for i in train_indices]
        val_instances = [all_data[i] for i in val_indices]
        test_instances = [all_data[i] for i in test_indices]
        
        process_and_save_split(train_instances, 'train', args)
        process_and_save_split(val_instances, 'val', args)
        process_and_save_split(test_instances, 'test', args)

if __name__ == '__main__':
    args = get_args()
    main(args)