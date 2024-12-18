#!/bin/bash
### OURS BEST
python main.py \
--model=meta-llama/Llama-3.2-1B \
--tokenizer_check=tokenizer_3500 \
--batch_size=1 \
--pad_to_max=1020 \
--device=cuda:1 \
--peft \
--num_merges=3500 \
--checkpoint=meta-llama/Llama-3.2-1B_ecg_qa_ptb_500_0.0001_0.9_0.99_1e-08_0.01_500_2_1_0.99_3500_1020_False \
--inference \
--dataset=ecg_qa_ptb_500 \
--percentiles=./data/mimic_dataset_stats.npy

### L_MERL
python finetune.py \
--model=resnet_model \
--batch_size=1 \
--pad_to_max=1022 \
--device=cuda:4 \
--peft \
--dataset=ptb_qa \
--inference \
--first_check=resnet_mimic_0.0001_0.9_0.99_1e-08_0.01_500_64_20_0.99_1000_100_1000_False_True \
--checkpoint=resnet_model_ptb_qa_0.0001_0.9_0.99_1e-08_0.01_500_2_1_0.99_3500_1022_False