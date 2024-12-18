#!/bin/bash

python finetune.py \
--model=resnet_model \
--batch_size=2 \
--pad_to_max=1022 \
--peft \
--epochs=1 \
--dataset=ptb_qa \
--first_check=resnet_mimic_0.0001_0.9_0.99_1e-08_0.01_500_64_20_0.99_1000_100_1000_False_True \
--gpus=0,1 \
--dis \
--ports=12359