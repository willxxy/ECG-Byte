#!/bin/bash

### MAIN TRAINING SCRIPT

### MAIN RESULTS WITH OURS
python main.py \
--model=meta-llama/Llama-3.2-1B \
--tokenizer_check=tokenizer_3500 \
--batch_size=2 \
--pad_to_max=1020 \
--peft \
--num_merges=3500 \
--epochs=1 \
--percentiles=./data/mimic_dataset_stats.npy \
--dataset=ecg_qa_ptb_500 \
--gpus=0,1,2,3 \
--dis \
--ports=12359