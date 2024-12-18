#!/bin/bash

python pretrain.py \
--model=clip \
--batch_size=64 \
--device=cuda:2 \
--peft \
--epochs=20 \
--seg \
--dataset=mimic \
--gpus=0,1,2,3 \
--dis \
--log

python pretrain.py \
--model=vit \
--batch_size=64 \
--device=cuda:2 \
--peft \
--epochs=20 \
--seg \
--dataset=mimic \
--gpus=0,1,2,3 \
--dis \
--log

python pretrain.py \
--model=clip_vit \
--batch_size=64 \
--device=cuda:2 \
--peft \
--epochs=20 \
--seg \
--dataset=mimic \
--gpus=0,1,2,3 \
--dis \
--log

python pretrain.py \
--model=resnet \
--batch_size=64 \
--device=cuda:3 \
--peft \
--epochs=20 \
--seg \
--dataset=mimic \
--gpus=2,3 \
--dis \
--log