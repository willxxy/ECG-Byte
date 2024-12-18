#!/bin/bash

python interp_analysis.py \
--model=meta-llama/Llama-3.2-1B \
--tokenizer_check=tokenizer_3500_100_mimic2 \
--batch_size=1 \
--pad_to_max=2046 \
--device=cuda:3 \
--toy \
--peft \
--num_merges=3500 \
--bins=100 \
--percentiles=./data/mimic_dataset_stats_500_unseg.npy \
--checkpoint=meta-llama/Llama-3.2-1B_ptb_qa_0.0001_0.9_0.99_1e-08_0.01_500_2_1_0.99_3500_100_2046_True_True \
--dataset=ptb_qa \
--interpret

