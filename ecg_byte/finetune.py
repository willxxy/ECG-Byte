import torch
torch.set_num_threads(2)
import random
import numpy as np
from torch.utils.data import DataLoader
import argparse
import gc
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, \
                            AutoProcessor, CLIPModel, AutoImageProcessor, ViTForMaskedImageModeling
from torch.optim import Adam
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import login

from data_loader import ECGCLIPFinetune
from scheduler import ScheduledOptim
from ecg_byte.models.llm import CLIP_LLM, VIT_LLM, CLIP_VIT_LLM, ResNet_LLM
from ecg_byte.models.clip import CLIP
from ecg_byte.models.vit import VIT
from ecg_byte.models.clip_vit import CLIPVIT
from ecg_byte.utils.file_utils import *
from ecg_byte.utils.viz_utils import *
from ecg_byte.utils.model_utils import count_parameters, early_stopping, run_statistical_analysis
from ecg_byte.runners.train import trainer, validater
from ecg_byte.runners.inference import tester
from ecg_byte.models.merl import ResNet101, ResNetPretrain

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
    parser.add_argument('--checkpoint', type = str, help = 'Please specify the checkpoint ')
    parser.add_argument('--log', action = 'store_true', help = 'Please choose whether to log or not')
    parser.add_argument('--dis', action = 'store_true', help = 'Please choose whether to distributed training or not')
    parser.add_argument('--pad_to_max', type = int, default = 1000, help = 'Please specify the pad to max size') 
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU ids to use (e.g., "0,1,2")')
    parser.add_argument('--ports', type=str, default='12355', help='Comma-separated list of ports to use (e.g., "12355,12356,12357")')
    parser.add_argument('--toy', action = 'store_true', help = 'Please choose whether to use toy dataset or not')
    parser.add_argument('--peft', action = 'store_true', default = None, help = 'Please choose whether to use PEFT or not')
    parser.add_argument('--first_check', type = str, default = None, help = 'Please choose the correct first stage checkpoint to load')
    return parser.parse_args()

def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.ports
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    
    args = get_args()
    if args.dis:
        gpu_ids = [int(id) for id in args.gpus.split(',')]
        local_rank = gpu_ids[rank]
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        args.device = device
        setup(rank, world_size, args)
    else:
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

    directory_path = f'./runs/{args.seed}/{args.model}_{args.dataset}_{args.lr}_{args.beta1}_{args.beta2}_{args.eps}_{args.weight_decay}_{args.warmup}_{args.batch_size}_{args.epochs}_{args.decay}_{args.pad_to_max}_{args.toy}'

    if args.log:
        wandb.init(
            project = 'bpe-trans',
            name = f'{args.seed}_{args.model}_{args.dataset}_{args.lr}_{args.beta1}_{args.beta2}_{args.eps}_{args.weight_decay}_{args.warmup}_{args.batch_size}_{args.epochs}_{args.decay}_{args.toy}',
            config = {
                'model' : args.model,
                'dataset' : args.dataset,
                'lr' : args.lr,
                'beta1' : args.beta1,
                'beta2' : args.beta2,
                'eps' : args.eps,
                'weight_decay' : args.weight_decay,
                'warmup' : args.warmup,
                'batch_size' : args.batch_size,
                'epochs' : args.epochs,
                'decay' : args.decay,
                'seed' : args.seed,
                'pad_to_max' : args.pad_to_max,
                'toy' : args.toy,
                }
        )
    
    print('Creating Data Directory')
    ensure_directory_exists('./data')
    
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
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B', cache_dir = './../.huggingface')
    llm = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B', cache_dir = './../.huggingface', torch_dtype=torch.bfloat16)
    
    if args.model == 'clip_model':
        backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './../.huggingface').to(device)
        clip_tokenizer = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32",cache_dir = './../.huggingface')
        backbone_model = CLIP(backbone, args)
        vit_tokenizer = None
        strict = True
    elif args.model == 'vit_model':
        backbone = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = './../.huggingface').to(device)
        vit_tokenizer = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = './../.huggingface')
        args.num_patches = (backbone.config.image_size // backbone.config.patch_size) ** 2
        backbone_model = VIT(backbone, args)
        clip_tokenizer = None
        strict = True
    elif args.model == 'clip_vit_model':
        model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = './../.huggingface').to(device)
        vit_tokenizer = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = './../.huggingface')
        args.num_patches = (model.config.image_size // model.config.patch_size) ** 2
        model_hidden_size = model.config.hidden_size
        vit_model = VIT(model, args, device)
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './../.huggingface').to(device)
        clip_tokenizer = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32",cache_dir = './../.huggingface')
        clip_model = CLIP(model, args, device)
        backbone_model = CLIPVIT(clip_model, vit_model, args)
        strict = True
    elif args.model == 'resnet_model':
        clip_tokenizer = None
        vit_tokenizer = None
        resnet_model = ResNet101().to(device)
        backbone_model = ResNetPretrain(resnet_model, lm = None, device = device, args = args).to(device)
        model_hidden_size = 256
        strict = False
        
        
    checkpoint = torch.load(f'./runs/{args.seed}/{args.first_check}/best_model.pth', map_location=args.device)
    backbone_model.load_state_dict(checkpoint['model'], strict = strict)
    backbone_model = backbone_model.to(torch.bfloat16)
    
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    llm.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.add_tokens(['<sig_start>'], special_tokens=True)
    tokenizer.add_tokens(['<sig_end>'], special_tokens=True)
    tokenizer.add_tokens(['<signal>'], special_tokens=True)
    llm.resize_token_embeddings(len(tokenizer))
    
    if args.peft:
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()
    
    if args.model == 'clip_model':
        model = CLIP_LLM(llm, backbone_model.clip, args)
        find_unused_parameters = False
    elif args.model == 'vit_model':
        model = VIT_LLM(llm, backbone_model.vit, args)
        find_unused_parameters = False
    elif args.model == 'clip_vit_model':
        model = CLIP_VIT_LLM(llm, backbone_model.clip.to(device), backbone_model.vit.to(device), device, args)
        find_unused_parameters = True
    elif args.model == 'resnet_model':
        model = ResNet_LLM(llm, backbone_model, args)
        find_unused_parameters = False
    model = model.to(device)
    model_hidden_size = model.llm.config.hidden_size
        
    print(f'Total number of parameters: {count_parameters(model)}')
    
    if args.dis:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)
    
    print(f'Loading {args.dataset}')            
    if args.inference:
        test_signals, test_texts = align_signal_text_files(f'./data/{args.dataset}/ecg/test', f'./data/{args.dataset}/text/test')
        print(len(test_signals))
        print(len(test_texts))
        for s, t in zip(test_signals[:5], test_texts[:5]):
            print(f"Signal: {os.path.basename(s)} | Text: {os.path.basename(t)}")
        if args.toy:
            test_signals, test_texts = sample_N_percent_from_lists(test_signals, test_texts, 0.25)
        print(len(test_signals))
        print(len(test_texts))
        test_data = ECGCLIPFinetune(test_signals, test_texts, args = args, 
                                        tokenizer = tokenizer, processor_clip = clip_tokenizer, processor_vit = vit_tokenizer,
                                        )
        test_loader = DataLoader(test_data,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True)
        print(f'Inferencing {args.checkpoint}')
        seeds = [0, 42, 123, 456, 789]
        all_seed_results = []
        for seed in seeds:
            print(f'Setting Seed to {seed}')
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            checkpoint = torch.load(f'./runs/{args.seed}/{args.checkpoint}/best_model.pth', map_location=args.device)
            
            model.load_state_dict(checkpoint['model'])
            print('Model Loaded')
            seed_results = tester(model, test_loader, tokenizer, args)
            all_seed_results.append(seed_results)    
            with open(f'./runs/{args.seed}/{args.checkpoint}/seed_{seed}_results_{args.dataset}.json', 'w') as f:
                json.dump({
                    'averages': seed_results['metrics'],
                    'qa_results': seed_results['qa_results']
                }, f)
        
        stats_results = run_statistical_analysis(all_seed_results)
        with open(f'./runs/{args.seed}/{args.checkpoint}/statistical_analysis_{args.dataset}.json', 'w') as f:
            json.dump(stats_results, f)
        
        # Print results
        print("\nStatistical Results Across All Seeds:")
        print("-" * 50)
        for metric, stats in stats_results.items():
            print(f"\n{metric}:")
            print(f"Mean: {stats['mean']:.2f}")
            print(f"Std Dev: {stats['std']:.2f}")
            print(f"95% CI: [{stats['conf_interval'][0]:.2f}, {stats['conf_interval'][1]:.2f}]")
        
        print('Inference Complete')
    else:
        train_signals, train_texts = align_signal_text_files(f'./data/{args.dataset}/ecg/train', f'./data/{args.dataset}/text/train')
        val_signals, val_texts = align_signal_text_files(f'./data/{args.dataset}/ecg/val', f'./data/{args.dataset}/text/val')
        print(len(train_signals))
        print(len(train_texts))
        print(len(val_signals))
        print(len(val_texts))
        for s, t in zip(train_signals[:5], train_texts[:5]):
            print(f"Signal: {os.path.basename(s)} | Text: {os.path.basename(t)}")
        if args.toy:
            train_signals, train_texts = sample_N_percent_from_lists(train_signals, train_texts, 0.25)
        print(len(train_signals))
        print(len(train_texts))
        print(len(val_signals))
        print(len(val_texts))
        training_data = ECGCLIPFinetune(train_signals, train_texts, args = args, 
                                        tokenizer = tokenizer, processor_clip = clip_tokenizer, processor_vit = vit_tokenizer)
        validation_data = ECGCLIPFinetune(val_signals, val_texts, args = args, 
                                          tokenizer = tokenizer, processor_clip = clip_tokenizer, processor_vit = vit_tokenizer)
        
        if args.dis:
            train_sampler = torch.utils.data.distributed.DistributedSampler(training_data, 
                                                                            num_replicas=world_size, 
                                                                            rank=rank,
                                                                            seed = args.seed,
                                                                            shuffle = True)
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        training_loader = DataLoader(training_data, 
                                    batch_size=args.batch_size, 
                                    shuffle=shuffle, 
                                    pin_memory=True,
                                    sampler = train_sampler)
        
        validation_loader = DataLoader(validation_data, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    pin_memory=True, 
                                    )

            
        optimizer = ScheduledOptim(
        Adam(filter(lambda x: x.requires_grad, model.parameters()),
            betas=(args.beta1, args.beta2), eps=args.eps, lr = args.lr, weight_decay=args.weight_decay), model_hidden_size, args.warmup)

        print('Creating Log Directory')
        ensure_directory_exists(directory_path)

        train_loss = []
        val_loss = []

        all_epochs = []

        for epoch in range(args.epochs):
            all_epochs.append(epoch)

            train_dic = trainer(model, training_loader, optimizer, args, epoch, directory_path)
            train_loss.append(train_dic['average_loss'])
            print(f"Training - Epoch: {epoch+1}\nTrain Loss: {train_dic['average_loss']}")

            val_dic = validater(model, validation_loader, args, epoch)
            val_loss.append(val_dic['average_loss'])
            print(f"Validating - Epoch: {epoch+1}\nVal Loss: {val_dic['average_loss']}")

            if args.log:
                wandb.log({
                    'train_epoch_loss': train_dic['average_loss'],
                    'val_epoch_loss': val_dic['average_loss'],
                    'epoch': epoch
                })

            early_stop = early_stopping(val_loss, patience=args.patience, delta=0.01)
            if early_stop:
                print('Validation loss has stopped decreasing. Early stopping...')
                break
            
            if args.dis:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
                
            checkpoint = {
                'model' : model_state_dict,
                'config_file' : 'config',
                'epoch' : epoch
            }

            if val_dic['average_loss'] <= min(val_loss):
                if args.dis:
                    if dist.get_rank() == 0:
                        torch.save(checkpoint, f'{directory_path}/best_model.pth')
                        print(f"Model saved at epoch: {epoch+1}")
                else:
                    torch.save(checkpoint, f'{directory_path}/best_model.pth')
                    print(f"Model saved at epoch: {epoch+1}")
            
            print('-----------------------------------------------------------')

        if args.log:
            wandb.finish()
        
        if args.dis:
            cleanup()
            
        plot_train_val_loss(train_loss, val_loss, directory_path)
        print('Training Finished')

if __name__ == '__main__':
    args = get_args()
    world_size = len(args.gpus.split(','))
    if args.dis:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        rank = 0
        world_size = 1
        main(rank, world_size)