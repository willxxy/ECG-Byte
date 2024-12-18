#!/bin/bash
# python preprocess/preprocess_ecg.py --seg_len=2500 --data=mimic
# python preprocess/preprocess_ecg.py --seg_len=500 --data=mimic
# python preprocess/preprocess_ecg.py --seg_len=500 --data=mimic

# python preprocess/preprocess_ecg.py --seg_len=500 --data=ptb
# python preprocess/preprocess_ecg.py --seg_len=500 --data=ptb

# python preprocess/preprocess_ecg.py --seg_len=250 --data=ecg_qa_ptb
python preprocess/preprocess_ecg.py --seg_len=500 --data=ecg_qa_ptb
# python preprocess/preprocess_ecg.py --seg_len=1250 --data=ecg_qa_ptb
# python preprocess/preprocess_ecg.py --data=ecg_qa_ptb
# python preprocess/preprocess_ecg.py --seg_len=500 --data=ecg_qa_mimic

# python preprocess/new_preprocess_ecg.py --seg_len=2500 --base_data=ptb
# python preprocess/new_preprocess_ecg.py --seg_len=500 --base_data=ptb