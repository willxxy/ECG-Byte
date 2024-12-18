
import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

from ecg_byte.utils.preprocess_utils import analyze_morphologies, stratified_sampling

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--unseg_dir', type = str, default = None, help = 'Please specify the path to the unsegmented ecgs')
    parser.add_argument('--max_clusters', type = int, default = 200, help = 'Please choose the maximum number of clusters to consider')
    parser.add_argument('--n_samples', type = int, default = 200000, help = 'Please choose the number of samples to consider')
    return parser.parse_args()

def main(args):
    os.makedirs(f'./data/pngs', exist_ok=True)
    print("Analyzing ECG morphologies...")
    file_paths, clusters, n_clusters = analyze_morphologies(args.unseg_dir, args.max_clusters, args.n_samples)

    print(f"Optimal number of clusters: {n_clusters}")
    print("Performing stratified sampling...")
    sampled_files = stratified_sampling(file_paths, clusters, args.n_samples)

    print(f"Sampled {len(sampled_files)} files.")
    
    with open(f"./data/sampled_ecg_files_{args.n_samples}.txt", "w") as f:
        for file in sampled_files:
            f.write(f"{file}\n")

    print(f"Sampling complete. Sampled file list saved to './data/sampled_ecg_files_{args.n_samples}.txt'.")
    print("Cluster analysis plots saved as './data/cluster_analysis.png'.")
    
if __name__ == "__main__":
    args = get_args()
    main(args)
    