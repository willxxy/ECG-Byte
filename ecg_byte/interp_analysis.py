import torch
torch.set_num_threads(2)
import random
import numpy as np
from torch.utils.data import DataLoader
import argparse
import gc
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import login

from data_loader import EGMTokenDataset
from ecg_byte.models.llm import LLM
from ecg_byte.utils.file_utils import *
from ecg_byte.utils.viz_utils import *
from ecg_byte.utils.model_utils import count_parameters
from ecg_byte.runners.interpret import interpreter

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--lr', type = float, default = 1e-4, help='Please choose the learning rate')
    parser.add_argument('--batch_size', type = int, default = 128, help='Please choose the batch size')
    parser.add_argument('--epochs', type = int, default = 150, help='Please choose the number of epochs')
    parser.add_argument('--device', type = str, default = None, help='Please choose the device')
    parser.add_argument('--dataset', type = str, default = 'mimic_500', help='Please choose the dataset')
    parser.add_argument('--model', type = str, default = None, help='Please choose the model')
    parser.add_argument('--beta1', type = float, default = 0.9, help='Please choose beta 1 for optimizer')
    parser.add_argument('--beta2', type = float, default = 0.99, help='Please choose beta 2 for optimizer')
    parser.add_argument('--eps', type = float, default = 1e-8, help='Please choose epsilon for optimizer')
    parser.add_argument('--warmup', type = int, default = 500, help = 'Please choose the number of warmup steps for the optimizer' )
    parser.add_argument('--weight_decay', type = float, default = 1e-2, help = 'Please choose the weight decay')
    parser.add_argument('--decay', type = float, default = 0.99, help='Please choose the decay') # 0.99 a bit smoother loss and perplexity (actually does not differ much) but reconstructions are near identical
    parser.add_argument('--seed', type = int, default = 0, help='Please choose the seed')
    parser.add_argument('--patience', type = int, default = 5, help='Please choose the patience')
    parser.add_argument('--dev', action = 'store_true', help = 'Please choose whether to use development mode or not')
    parser.add_argument('--inference', action = 'store_true', help = 'Please choose whether to inference or not')
    parser.add_argument('--interpret', action = 'store_true', help = 'Please choose whether to interpret or not')
    parser.add_argument('--checkpoint', type = str, help = 'Please specify the checkpoint ')
    parser.add_argument('--log', action = 'store_true', help = 'Please choose whether to log or not')
    parser.add_argument('--tokenizer_check', type = str, help = 'Please specify the tokenizer')
    parser.add_argument('--num_merges', type = int, default = 1000, help = 'Please specify the vocab size') 
    parser.add_argument('--pad_to_max', type = int, default = 1000, help = 'Please specify the pad to max size') 
    parser.add_argument('--toy', action = 'store_true', help = 'Please choose whether to use toy dataset or not')
    parser.add_argument('--peft', action = 'store_true', default = None, help = 'Please choose whether to use PEFT or not')
    parser.add_argument('--percentiles', type = str, default = None, help = 'Please choose the percentiles computed during preprocessing')
    return parser.parse_args()

def main(args):
    device = torch.device(args.device)

    if args.dev:
        args.epochs=2
        
    print('Loading API key')
    with open('./../.huggingface/api_keys.txt', 'r') as file:
        file_contents = file.readlines()
    api_key = file_contents[0].strip()

    login(token = api_key)
    
    print('Collecting Garbage')
    gc.collect()
    print('Emptying CUDA Cache')
    torch.cuda.empty_cache()
    print(f'Setting Seed to {args.seed}')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vocab, merges = load_vocab_and_merges(f'./data/{args.tokenizer_check}.pkl')
    print('Number of Merges', args.num_merges)
    
    directory_path = f'./runs/{args.seed}/{args.model}_{args.dataset}_{args.lr}_{args.beta1}_{args.beta2}_{args.eps}_{args.weight_decay}_{args.warmup}_{args.batch_size}_{args.epochs}_{args.decay}_{args.num_merges}_{args.pad_to_max}_{args.toy}'
    
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type = TaskType.CAUSAL_LM,
        )

    print('Initializing Model')
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir = './../.huggingface')
    llm = AutoModelForCausalLM.from_pretrained(args.model, cache_dir = './../.huggingface', torch_dtype=torch.bfloat16)
    new_ids = list(vocab.keys())
    new_ids = [f'signal_{str(ids)}' for ids in new_ids]
    tokenizer.add_tokens(new_ids)
    tokenizer.add_tokens(['<sig_start>'], special_tokens=True)
    tokenizer.add_tokens(['<sig_end>'], special_tokens=True)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    llm.config.pad_token_id = tokenizer.pad_token_id
    llm.resize_token_embeddings(len(tokenizer))
    
    if args.peft:
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()
    model = LLM(llm, args)
    model = model.to(device)
    model_hidden_size = model.llm.config.hidden_size
    find_unused_parameters = False
    
    print(f'Total number of parameters: {count_parameters(model)}')
    
    print(f'Loading {args.dataset}')            
    test_signals, test_texts = align_signal_text_files(f'./data/{args.dataset}/ecg/test', f'./data/{args.dataset}/text/test')
    print(len(test_signals))
    print(len(test_texts))
    for s, t in zip(test_signals[:5], test_texts[:5]):
        print(f"Signal: {os.path.basename(s)} | Text: {os.path.basename(t)}")
    if args.toy:
        test_signals, test_texts = sample_N_percent_from_lists(test_signals, test_texts)
    print(len(test_signals))
    print(len(test_texts))
    test_data = EGMTokenDataset(test_signals, test_texts, vocab, merges, args = args, tokenizer = tokenizer)
    test_loader = DataLoader(test_data,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True)
    print(f'Interpreting {args.checkpoint}')
    checkpoint = torch.load(f'./runs/{args.seed}/{args.checkpoint}/best_model.pth', map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    print('Model Loaded')
    
    out = interpreter(model, test_loader, tokenizer, vocab, args)
    
    print('Interpretation Complete')

if __name__ == '__main__':
    args = get_args()
    main(args)