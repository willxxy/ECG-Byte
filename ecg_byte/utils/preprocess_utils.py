import json
from tqdm import tqdm
import wfdb 
from scipy import signal
import numpy as np
import pywt
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import random
import matplotlib.pyplot as plt
import pandas as pd
import ast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
from multiprocessing import Pool, cpu_count
import torch

from ecg_byte.utils.file_utils import load_npy

### Preprocessing MIMIC IV

def check_nan_inf(data, step_name):
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(f"Warning: NaN or inf values detected after {step_name}")
        # print(f"NaN count: {np.sum(np.isnan(data))}")
        # print(f"Inf count: {np.sum(np.isinf(data))}")
        # Optionally, replace NaN and inf values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

def reorder_indices(signals):
    current_order = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    desired_order = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    order_mapping = {lead: index for index, lead in enumerate(current_order)}
    new_indices = [order_mapping[lead] for lead in desired_order]
    return signals[:, new_indices]

def wavelet_denoise(ecg_data, wavelet='db6', level=4, epsilon=1e-10):
    denoised_ecg = np.zeros_like(ecg_data)
    for i in range(ecg_data.shape[1]):
        coeffs = pywt.wavedec(ecg_data[:, i], wavelet, level=level)
        median_abs = np.median(np.abs(coeffs[-level]))
        if median_abs == 0:
            threshold = 0
        else:
            threshold = median_abs / 0.6745
        
        def safe_threshold(c):
            thresholded = pywt.threshold(c, threshold, mode='soft')
            return np.where(np.isfinite(thresholded) & (np.abs(c) > epsilon), thresholded, 0)
        
        new_coeffs = [coeffs[0]] + [safe_threshold(c) for c in coeffs[1:]]
        denoised_ecg[:, i] = pywt.waverec(new_coeffs, wavelet)
    
    # Replace any remaining NaN or inf values with zeros
    denoised_ecg = np.nan_to_num(denoised_ecg, nan=0.0, posinf=0.0, neginf=0.0)
    return denoised_ecg

def advanced_ecg_filter(ecg_data, fs=500, notch_freqs=[50, 60], highcut=100.0):
    filtered_ecg = ecg_data.copy()
    
    quality_factor = 30.0
    for notch_freq in notch_freqs:
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
        filtered_ecg = signal.filtfilt(b_notch, a_notch, filtered_ecg, axis=0)

    lowcut = 0.5
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    order = 4

    b_band, a_band = signal.butter(order, [low, high], btype='band')
    filtered_ecg = signal.filtfilt(b_band, a_band, filtered_ecg, axis=0)

    baseline_cutoff = 0.05
    baseline_low = baseline_cutoff / nyquist
    b_baseline, a_baseline = signal.butter(order, baseline_low, btype='high')
    filtered_ecg = signal.filtfilt(b_baseline, a_baseline, filtered_ecg, axis=0)

    return filtered_ecg

def nsample_ecg(ecg_data, orig_fs, target_fs):
    num_samples, num_leads = ecg_data.shape
    duration = num_samples / orig_fs
    t_original = np.linspace(0, duration, num_samples, endpoint=True)
    t_target = np.linspace(0, duration, int(num_samples * target_fs / orig_fs), endpoint=True)
    
    downsampled_data = np.zeros((len(t_target), num_leads))
    for lead in range(num_leads):
        f = interpolate.interp1d(t_original, ecg_data[:, lead], kind='cubic', bounds_error=False, fill_value="extrapolate")
        downsampled_data[:, lead] = f(t_target)
    return downsampled_data

def segment_ecg(ecg_data, text_data, seg_len):
    time_length, _ = ecg_data.shape
    num_segments = time_length // seg_len
    
    ecg_data_segmented = []
    text_data_segmented = []
    
    for i in range(num_segments):
        start_idx = i * seg_len
        end_idx = (i + 1) * seg_len
        ecg_data_segmented.append(ecg_data[start_idx:end_idx, :])
        text_data_segmented.append(text_data)
    
    return np.array(ecg_data_segmented), text_data_segmented

def process_instance(instance, args):
    if args.data == 'mimic':
        conversation = instance['conversations']
        path_to_file = f"./data/mimic/{instance['ecg']}"
    elif args.data in ['ecg_qa_mimic','ecg_qa_ptb']:
        conversation = [instance['question_type'], instance['question'], instance['answer']]
        if args.data == 'ecg_qa_ptb':
            path_to_file = os.path.join('./data', instance['ecg_path'][0].lstrip('./').lstrip('../'))
        elif args.data == 'ecg_qa_mimic':
            path_to_file = '.' + instance['ecg_path'][0][instance['ecg_path'][0].find('/data'):]
    try:
        if args.data in ['mimic', 'ecg_qa_mimic', 'ecg_qa_ptb']:
            signals, fields = wfdb.rdsamp(path_to_file)
            sampling_rate = fields['fs']
            assert sampling_rate == 500
            assert signals.shape[1] == 12
            assert signals.shape[0] == 5000
        
            
        if np.any(np.isnan(signals)) or np.any(np.isinf(signals)):
            print(f"Warning: NaN values detected in {path_to_file}. Skipping this instance.")
            return None, None
        
        signals = check_nan_inf(signals, "reading")
        
        if args.data in ['mimic', 'ecg_qa_mimic']:
            signals = reorder_indices(signals)
            signals = check_nan_inf(signals, "reordering")
        
        filtered_signals = advanced_ecg_filter(signals, fs = sampling_rate)
        filtered_signals = check_nan_inf(filtered_signals, "advanced filtering")
        
        filtered_signals = wavelet_denoise(filtered_signals)
        filtered_signals = check_nan_inf(filtered_signals, "wavelet denoising")

        if args.data in ['mimic', 'ecg_qa_mimic', 'ecg_qa_ptb']:
            filtered_signals = nsample_ecg(filtered_signals, 500, 250)
            
        filtered_signals = check_nan_inf(filtered_signals, "resampling")

        ecg_data_seg, text_data_seg = segment_ecg(filtered_signals, conversation, args.seg_len)
        
        ecg_data_seg = check_nan_inf(ecg_data_seg, "segmentation")
        if np.any(np.isnan(ecg_data_seg)) or np.any(np.isinf(ecg_data_seg)):
            print(f"Warning: NaN values detected in {path_to_file}. Skipping this instance.")
            return None, None
        
        return ecg_data_seg, text_data_seg
    except Exception as e:
        print(f"Error processing {path_to_file}: {str(e)}. Skipping this instance.")
        return None, None


def compute_global_stats(instances, args, sample_size=100000):
    global_min = np.inf
    global_max = -np.inf
    samples_for_percentiles = []
    total_samples_collected = 0
    max_samples = sample_size
    skipped_count = 0
    futures = []

    num_cores = 12
    batch_size = 1000
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for i in tqdm(range(0, len(instances), batch_size), desc="Processing instances"):
            batch = instances[i:i + batch_size]
            for idx, instance in enumerate(batch):
                print(f"Processing instance {i + idx + 1}/{len(instances)}")
                future = executor.submit(process_instance, instance, args)
                futures.append(future)
                print(f"Submitted instance {i + idx + 1}/{len(instances)}")
        for future in tqdm(futures, total=len(instances)):
            ecg_data_seg, _ = future.result()
            if ecg_data_seg is None:
                skipped_count += 1
                continue
            for seg in ecg_data_seg:
                batch_min = np.min(seg)
                batch_max = np.max(seg)
                global_min = min(global_min, batch_min)
                global_max = max(global_max, batch_max)
                if total_samples_collected < max_samples:
                    num_new_samples = min(max_samples - total_samples_collected, seg.size)
                    indices = np.random.choice(seg.size, num_new_samples, replace=False)
                    samples = seg.flat[indices]
                    samples_for_percentiles.extend(samples)
                    total_samples_collected += num_new_samples

    samples_for_percentiles = np.array(samples_for_percentiles)
    percentile_1 = np.percentile(samples_for_percentiles, 1)
    percentile_99 = np.percentile(samples_for_percentiles, 99)

    dataset_stats = {'global_min': global_min, 'global_max': global_max, 
                     'percentile_1': percentile_1, 'percentile_99': percentile_99,
                     'skipped_instances': skipped_count}

    print(f"Total instances skipped due to NaN values: {skipped_count}")
    return dataset_stats

def process_and_save_instance(instance, index, split_name, args):
    ecg_data_seg, text_data_seg = process_instance(instance, args)
    if ecg_data_seg is None:
        return None
    for j in range(ecg_data_seg.shape[0]):
        ecg_segment = ecg_data_seg[j, :, :].T  # Shape: (12, seg_len)
        text_segment = text_data_seg[j]
        # Save ecg_segment and text_segment
        np.save(f'./data/{args.data}_{args.seg_len}/ecg/{split_name}/ecg_{index}_{j}.npy', ecg_segment)
        with open(f'./data/{args.data}_{args.seg_len}/text/{split_name}/text_{index}_{j}.json', 'w') as f:
            json.dump(text_segment, f)
    return True

def process_and_save_split(instances, split_name, args):
    os.makedirs(f'./data/{args.data}_{args.seg_len}/ecg/{split_name}', exist_ok=True)
    os.makedirs(f'./data/{args.data}_{args.seg_len}/text/{split_name}', exist_ok=True)
    
    skipped_count = 0
    num_cores = 4
    
    try:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(process_and_save_instance, instance, i, split_name, args) 
                       for i, instance in enumerate(instances)]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    if result is None:
                        skipped_count += 1
                except Exception as e:
                    print(f"Error processing instance: {str(e)}")
                    skipped_count += 1
    
    except Exception as e:
        print(f"Error in process_and_save_split: {str(e)}")
    
    finally:
        print(f"Total instances skipped in {split_name} split: {skipped_count}")



### Sampling ECG

def extract_features(ecg, sampling_rate=250):
    features = []
    
    for lead in range(ecg.shape[0]):
        lead_signal = ecg[lead, :]
        
        # Basic statistical features
        features.extend([
            np.mean(lead_signal),
            np.std(lead_signal),
            np.max(lead_signal),
            np.min(lead_signal),
            np.median(lead_signal),
            np.percentile(lead_signal, 25),
            np.percentile(lead_signal, 75)
        ])
        
        # Frequency domain features
        freqs, psd = signal.welch(lead_signal, fs=sampling_rate, nperseg=1024)
        total_power = np.sum(psd)
        features.extend([
            total_power,  # Total power
            np.max(psd),  # Peak frequency power
            freqs[np.argmax(psd)],  # Dominant frequency
        ])
        
        # Spectral centroid with NaN handling
        if total_power > 0:
            spectral_centroid = np.sum(freqs * psd) / total_power
        else:
            spectral_centroid = 0  # or another appropriate default value
        features.append(spectral_centroid)
        
        peaks, _ = signal.find_peaks(lead_signal, height=0.5*np.max(lead_signal), distance=0.2*sampling_rate)
        if len(peaks) > 1:
            # Heart rate
            rr_intervals = np.diff(peaks) / sampling_rate
            heart_rate = 60 / np.mean(rr_intervals)
            features.append(heart_rate)
            
            # Heart rate variability
            hrv = np.std(rr_intervals)
            features.append(hrv)
            
            # QRS duration (simplified)
            qrs_duration = np.mean([find_qrs_duration(lead_signal, peak, sampling_rate) for peak in peaks])
            features.append(qrs_duration)
        else:
            features.extend([0, 0, 0])  # Placeholder values if no peaks found
        
        # T-wave features (simplified)
        t_wave_amp = find_t_wave_amplitude(lead_signal, peaks)
        features.append(t_wave_amp)
        
        # ST segment features (simplified)
        st_deviation = find_st_deviation(lead_signal, peaks, sampling_rate)
        features.append(st_deviation)
        
        coeffs = pywt.wavedec(lead_signal, 'db4', level=5)
        features.extend([np.mean(np.abs(c)) for c in coeffs])
        
        # Non-linear features
        features.append(np.mean(np.abs(np.diff(lead_signal))))  # Average absolute difference
        features.append(np.sqrt(np.mean(np.square(np.diff(lead_signal)))))  # Root mean square of successive differences
    
    return np.array(features)


def find_qrs_duration(ecg, peak, sampling_rate):
    # Simplified QRS duration estimation
    window = int(0.1 * sampling_rate)  # 100 ms window
    start = max(0, peak - window)
    end = min(len(ecg), peak + window)
    qrs_segment = ecg[start:end]
    return np.sum(np.abs(qrs_segment) > 0.1 * np.max(qrs_segment)) / sampling_rate

def find_t_wave_amplitude(ecg, peaks):
    if len(peaks) < 2:
        return 0
    t_wave_region = ecg[peaks[-2]:peaks[-1]]
    return np.max(t_wave_region) - np.min(t_wave_region)

def find_st_deviation(ecg, peaks, sampling_rate):
    if len(peaks) < 2:
        return 0
    st_point = peaks[-1] + int(0.08 * sampling_rate)  # 80 ms after R peak
    if st_point < len(ecg):
        return ecg[st_point] - ecg[peaks[-1]]
    return 0

def analyze_morphologies(directory, max_clusters=100, subset_size=10000):
    all_features = []
    file_paths = []

    print("Loading and extracting features from ECG files...")
    count = 0
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            file_paths.append(file_path)
            ecg = load_npy(file_path)
            features = extract_features(ecg)
            all_features.append(features)
            if count == subset_size:
                break
            count+=1

    all_features = np.array(all_features)

    # Perform PCA
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    features_pca = pca.fit_transform(all_features)
    del all_features

    # Scale after PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_pca)
    del features_pca
    
    # Determine optimal number of clusters
    n_clusters = find_optimal_clusters(features_scaled, max_clusters)

    print(f"Optimal number of clusters determined: {n_clusters}")
    print("Clustering all data...")

    # Use KMeans for final clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)

    # Try DBSCAN if KMeans results are not satisfactory
    if len(np.unique(clusters)) < 3:
        print("KMeans produced too few clusters. Trying DBSCAN...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(features_scaled)

    return file_paths, clusters, len(np.unique(clusters))

def find_optimal_clusters(data, max_clusters):
    inertias = []
    silhouette_scores = []

    for k in tqdm(range(2, max_clusters + 1), desc="Finding optimal clusters"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_, sample_size=10000))

    # Plot elbow curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')

    plt.tight_layout()
    plt.savefig('./pngs/cluster_analysis.png')
    plt.close()

    # Find the elbow point
    elbow_point = find_elbow_point(inertias)

    # Find the maximum silhouette score
    max_silhouette = max(silhouette_scores)
    max_silhouette_clusters = silhouette_scores.index(max_silhouette) + 2

    print(f"Elbow method suggests {elbow_point} clusters")
    print(f"Highest silhouette score at {max_silhouette_clusters} clusters")

    # Choose the smaller of the two as a conservative estimate
    optimal_clusters = min(elbow_point, max_silhouette_clusters)
    print(f"Chosen number of clusters: {optimal_clusters}")

    return optimal_clusters

def find_elbow_point(inertias):
    # Simple method to find the elbow point
    diffs = np.diff(inertias)
    elbow_point = np.argmin(diffs) + 2  # +2 because we started from 2 clusters
    return elbow_point

def stratified_sampling(file_paths, clusters, n_samples=100000):
    unique_clusters = np.unique(clusters)
    samples_per_cluster = n_samples // len(unique_clusters)
    
    sampled_files = []
    for cluster in tqdm(unique_clusters, desc="Sampling from clusters"):
        cluster_files = [file_paths[i] for i in range(len(file_paths)) if clusters[i] == cluster]
        sampled_files.extend(random.sample(cluster_files, min(samples_per_cluster, len(cluster_files))))
    
    # If we haven't reached n_samples, randomly sample from the remaining files
    remaining_samples = n_samples - len(sampled_files)
    if remaining_samples > 0:
        remaining_files = list(set(file_paths) - set(sampled_files))
        sampled_files.extend(random.sample(remaining_files, min(remaining_samples, len(remaining_files))))
    
    return sampled_files


### Preprocessing PTB-XL
def load_dataset(path, sampling_rate, release=False):
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, path)

    return X, Y

def process_single_signal(signal_meta):
    """Process a single ECG signal with all filtering steps.
    
    Args:
        signal_meta: Tuple of (signal, metadata)
    Returns:
        Processed signal array
    """
    signal, meta = signal_meta
    return nsample_ecg(wavelet_denoise(advanced_ecg_filter(signal)), 500, 250)

def parallel_process_signals(data, n_processes=6):

    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_signal, data),
            total=len(data),
            desc='Filtering ECG (Parallel)'
        ))
    
    return np.array(results)

def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:                
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = parallel_process_signals(data)                    
            check_nan_inf(data, 'ptb')
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data

def compute_label_aggregations(df, folder, ctype):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df

def select_data(XX,YY, ctype, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter 
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter 
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    # save LabelBinarizer
    with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb

def translate_german_to_english_batch(texts):
    try:
        if isinstance(texts, list):
            texts = np.array(texts)
        
        if not isinstance(texts, np.ndarray):
            raise ValueError("Input must be a numpy array or list")
        if texts.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {texts.shape}")
        if len(texts) == 0:
            raise ValueError("Input array cannot be empty")
            
        valid_mask = np.array([bool(text and str(text).strip()) for text in texts])
        valid_texts = texts[valid_mask]
        
        if len(valid_texts) == 0:
            raise ValueError("All input texts are empty")
            
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en", cache_dir='./../.huggingface')
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en", cache_dir='./../.huggingface').to(device)
        
        batch_size = 32
        translations = []
        
        for i in tqdm(range(0, len(valid_texts), batch_size), desc = 'Translating files'):
            batch_texts = valid_texts[i:i + batch_size]
            
            encoded = tokenizer(list(batch_texts), return_tensors="pt", padding=True, truncation=True)
            encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_length=128,
                )
            
            batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(batch_translations)
        
        result = np.empty_like(texts, dtype=object)
        result[valid_mask] = translations
        result[~valid_mask] = ''
        
        return result
        
    except ValueError as e:
        raise e
    except Exception as e:
        raise Exception(f"Translation error: {str(e)}")

    
def segment_ecg_ptb(X, y_reports, segment_length):
    num_instances, time_length, n_channels = X.shape
    num_segments = time_length // segment_length

    X_segmented = []
    y_reports_segmented = []
    
    for i in range(num_instances):
        for j in range(num_segments):
            start_idx = j * segment_length
            end_idx = (j + 1) * segment_length
            segment = X[i, start_idx:end_idx, :]
            X_segmented.append(segment)
            y_reports_segmented.append(y_reports[i])

    X_segmented = np.array(X_segmented)
    y_reports_segmented = np.array(y_reports_segmented)

    return X_segmented, y_reports_segmented

def preprocess_ptb(data_folder, output_folder, task, sf, args):
    data, raw_labels = load_dataset(data_folder, sf)
    labels = compute_label_aggregations(raw_labels, data_folder, task)
    data, labels, Y, _ = select_data(data, labels, task, min_samples=0, outputfolder=output_folder)
    
    X_train = data[labels.strat_fold < 8]
    y_train = Y[labels.strat_fold < 8]
    y_train = np.argmax(y_train, axis=1)
    y_reports_train = translate_german_to_english_batch(np.array(labels.report[labels.strat_fold < 8]))
    segmented_X_train, segmented_y_train = segment_ecg_ptb(X_train, y_reports_train, args.seg_len)
    transposed_X_train = np.transpose(segmented_X_train, (0, 2, 1))
    print(transposed_X_train.shape, segmented_y_train.shape)
    
    X_val = data[labels.strat_fold == 8]
    y_val = Y[labels.strat_fold == 8]
    y_val = np.argmax(y_val, axis=1)
    y_reports_val = translate_german_to_english_batch(np.array(labels.report[labels.strat_fold == 8]))
    segmented_X_val, segmented_y_val = segment_ecg_ptb(X_val, y_reports_val, args.seg_len)
    transposed_X_val = np.transpose(segmented_X_val, (0, 2, 1))
    print(transposed_X_val.shape, segmented_y_val.shape)
    
    X_test = data[labels.strat_fold > 8]
    y_test = Y[labels.strat_fold > 8]
    y_test = np.argmax(y_test, axis=1)
    y_reports_test = translate_german_to_english_batch(np.array(labels.report[labels.strat_fold > 8]))
    segmented_X_test, segmented_y_test = segment_ecg_ptb(X_test, y_reports_test, args.seg_len)
    transposed_X_test = np.transpose(segmented_X_test, (0, 2, 1))
    print(transposed_X_test.shape, segmented_y_test.shape)
    
    os.makedirs(f'./data/{args.data}_{args.seg_len}/ecg/train', exist_ok=True)
    os.makedirs(f'./data/{args.data}_{args.seg_len}/ecg/val', exist_ok=True)
    os.makedirs(f'./data/{args.data}_{args.seg_len}/ecg/test', exist_ok=True)
    
    os.makedirs(f'./data/{args.data}_{args.seg_len}/text/train', exist_ok=True)
    os.makedirs(f'./data/{args.data}_{args.seg_len}/text/val', exist_ok=True)
    os.makedirs(f'./data/{args.data}_{args.seg_len}/text/test', exist_ok=True)
    
    for i in tqdm(range(transposed_X_train.shape[0]), desc = 'Saving Train ECG'):
        ecg_segment = transposed_X_train[i]
        text_segment = segmented_y_train[i]
        np.save(f'./data/{args.data}_{args.seg_len}/ecg/train/ecg_{i}_{i}.npy', ecg_segment)
        with open(f'./data/{args.data}_{args.seg_len}/text/train/text_{i}_{i}.json', 'w') as f:
            json.dump(text_segment, f)
    
    for i in tqdm(range(transposed_X_val.shape[0]), desc = 'Saving Val ECG'):
        ecg_segment = transposed_X_val[i]
        text_segment = segmented_y_val[i]
        np.save(f'./data/{args.data}_{args.seg_len}/ecg/val/ecg_{i}_{i}.npy', ecg_segment)
        with open(f'./data/{args.data}_{args.seg_len}/text/val/text_{i}_{i}.json', 'w') as f:
            json.dump(text_segment, f)
            
    for i in tqdm(range(transposed_X_test.shape[0]), desc = 'Saving Test ECG'):
        ecg_segment = transposed_X_test[i]
        text_segment = segmented_y_test[i]
        np.save(f'./data/{args.data}_{args.seg_len}/ecg/test/ecg_{i}_{i}.npy', ecg_segment)
        with open(f'./data/{args.data}_{args.seg_len}/text/test/text_{i}_{i}.json', 'w') as f:
            json.dump(text_segment, f)
            
            
### Preprocess ECG QA
def setup_ecg_qa(glob_paths):
    data = []
    for fname in sorted(glob_paths):
        with open(fname, "r") as f:
            loaded_file = json.load(f)
            filtered_list = [item for item in loaded_file if item['question_type'] in ['single-verify', 'single-choose', 'single-query']]
            data.extend(filtered_list)
    return data