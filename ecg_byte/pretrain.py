import torch
torch.set_num_threads(2)
import random
import numpy as np
from torch.utils.data import DataLoader
import argparse
import gc
import wandb
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, ViTForMaskedImageModeling, \
                            AutoModel, AutoTokenizer
from torch.optim import Adam
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from huggingface_hub import login

from data_loader import ECGCLIPPretrain
from scheduler import ScheduledOptim
from ecg_byte.models.clip import CLIP
from ecg_byte.models.vit import VIT
from ecg_byte.models.clip_vit import CLIPVIT
from ecg_byte.utils.file_utils import *
from ecg_byte.utils.viz_utils import *
from ecg_byte.utils.model_utils import count_parameters
from ecg_byte.runners.train import trainer
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
    parser.add_argument('--seed', type = int, default = 0, help='Please choose the seed')
    parser.add_argument('--patience', type = int, default = 5, help='Please choose the patience')
    parser.add_argument('--dev', action = 'store_true', help = 'Please choose whether to use development mode or not')
    parser.add_argument('--checkpoint', type = str, help = 'Please specify the checkpoint ')
    parser.add_argument('--log', action = 'store_true', help = 'Please choose whether to log or not')
    parser.add_argument('--dis', action = 'store_true', help = 'Please choose whether to distributed training or not')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU ids to use (e.g., "0,1,2")')
    parser.add_argument('--ports', type=str, default='12356', help='Comma-separated list of ports to use (e.g., "12355,12356,12357")')
    parser.add_argument('--percentiles', type = str, default = None, help = 'Please choose the percentiles computed during preprocessing')
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

    directory_path = f'./runs/{args.seed}/{args.model}_{args.dataset}_{args.lr}_{args.beta1}_{args.beta2}_{args.eps}_{args.weight_decay}_{args.warmup}_{args.batch_size}_{args.epochs}'

    if args.log:
        wandb.init(
            project = 'bpe-trans',
            name = f'{args.seed}_{args.model}_{args.dataset}_{args.lr}_{args.beta1}_{args.beta2}_{args.eps}_{args.weight_decay}_{args.warmup}_{args.batch_size}_{args.epochs}',
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
                'seed' : args.seed,
                }
        )
    
    print('Creating Data Directory')
    ensure_directory_exists('./data')

    print('Initializing Model')
    vit_tokenizer = None
    resnet_tokenizer = None
    clip_tokenizer = None
    if args.model == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './../.huggingface').to(device)
        clip_tokenizer = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32",cache_dir = './../.huggingface')
        model_hidden_size = 768
        model = CLIP(model, args)
        find_unused_parameters = False
    elif args.model == 'vit':
        model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = './../.huggingface').to(device)
        vit_tokenizer = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = './../.huggingface')
        args.num_patches = (model.config.image_size // model.config.patch_size) ** 2
        model_hidden_size = model.config.hidden_size
        model = VIT(model, args)
        find_unused_parameters = False
    elif args.model == 'clip_vit':
        model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = './../.huggingface').to(device)
        vit_tokenizer = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = './../.huggingface')
        args.num_patches = (model.config.image_size // model.config.patch_size) ** 2
        model_hidden_size = model.config.hidden_size
        vit_model = VIT(model, args)
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './../.huggingface').to(device)
        clip_tokenizer = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32",cache_dir = './../.huggingface')
        clip_model = CLIP(model, args)
        model = CLIPVIT(clip_model, vit_model, args)
        find_unused_parameters = True
    elif args.model == 'resnet':
        resnet_model = ResNet101().to(device)
        lm = AutoModel.from_pretrained('ncbi/MedCPT-Query-Encoder', cache_dir = './../.huggingface').to(device)
        resnet_tokenizer = AutoTokenizer.from_pretrained('ncbi/MedCPT-Query-Encoder', cache_dir = './../.huggingface')
        model = ResNetPretrain(resnet_model, lm, device, args).to(device)
        for i in range(9):
            for param in list(model.lm.encoder.layer[i].parameters()):
                param.requires_grad = False
        model_hidden_size = 256
        find_unused_parameters = True
        
    model = model.to(device)
    
    print(f'Total number of parameters: {count_parameters(model)}')
    
    if args.dis:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)
    
    print(f'Loading {args.dataset}')            
    train_signals, train_texts = align_signal_text_files(f'./data/{args.dataset}/ecg/train', f'./data/{args.dataset}/text/train')
    print(len(train_signals))
    print(len(train_texts))
    for s, t in zip(train_signals[:5], train_texts[:5]):
        print(f"Signal: {os.path.basename(s)} | Text: {os.path.basename(t)}")
    print(len(train_signals))
    print(len(train_texts))
    training_data = ECGCLIPPretrain(train_signals, train_texts, clip_tokenizer = clip_tokenizer, 
                                    vit_tokenizer = vit_tokenizer, resnet_tokenizer = resnet_tokenizer, args = args)
    
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

        if args.log:
            wandb.log({
                'train_epoch_loss': train_dic['average_loss'],
                'epoch': epoch
            })
        
        if args.dis:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
            
        checkpoint = {
            'model' : model_state_dict,
            'config_file' : 'config',
            'epoch' : epoch
        }

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

if __name__ == '__main__':
    args = get_args()
    world_size = len(args.gpus.split(','))
    if args.dis:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        rank = 0
        world_size = 1
        main(rank, world_size)
