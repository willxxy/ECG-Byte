#!/bin/bash

## Training
python train_tokenizer.py \
--percentiles=./data/mimic_dataset_stats.npy \
--num_merges=3500 \
--num_processes=6 \
--sampled_files=./data/sampled_ecg_files_200000.txt \
--train