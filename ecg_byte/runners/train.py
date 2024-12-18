from tqdm import tqdm
import torch
import wandb
import torch.distributed as dist
import gc

def trainer(model, dataloader, optimizer, args, epoch, directory_path):
    model.train()
    if args.dis:
        dataloader.sampler.set_epoch(epoch)
    total_loss = 0
    len_of_batch = 0
    dev_count = 0
    

    for step, batch in enumerate(tqdm(dataloader, desc=f'Training {args.model}', position=0, leave=True)):
        if batch is None:
            print(f"Skipping invalid batch at step {step}")
            continue
        
        try:
            optimizer.zero_grad()
            out = model(batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step_and_update_lr()
            
            total_loss += loss.item()
            len_of_batch += 1
            
            if args.log:
                wandb.log({"train_step_loss": loss.item(), "epoch": epoch, "train_step": step})
            
            if ((step + 1) % 50 == 0) and args.toy != True:
                if args.dis:
                    train_model_state_dict = model.module.state_dict()
                else:
                    train_model_state_dict = model.state_dict()
                train_checkpoint = {
                    'model': train_model_state_dict,
                    'epoch': epoch
                }
                if args.dis:
                    dist.barrier()
                    if dist.get_rank() == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                        torch.save(train_checkpoint, f'{directory_path}/best_train_model_{epoch}_{step}.pth')
                        print(f"Best model saved at epoch: {epoch+1} {step}")
                else:
                    torch.save(train_checkpoint, f'{directory_path}/best_train_model_{epoch}_{step}.pth')
                    print(f"Best model saved at epoch: {epoch+1} {step}")
            
            if args.dev:
                dev_count += 1
                if dev_count == 10:
                    break
        except Exception as e:
            print(f"Error during training at step {step}: {e}")
            continue

    if len_of_batch == 0:  # Handle case where all batches are invalid
        print("No valid batches for training.")
        average_loss = float('inf')
    else:
        average_loss = total_loss / len_of_batch
    
    return_dic = {
        'average_loss': average_loss
    }
    return return_dic

def validater(model, dataloader, args, epoch):
    model.eval()
    total_loss = 0
    len_of_batch = 0
    dev_count = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc=f'Validating {args.model}', position=0, leave=True)):
            # Skip invalid batches
            if batch is None:
                print(f"Skipping invalid batch at step {step}")
                continue
            
            try:
                out = model(batch)
                loss = out.loss
                total_loss += loss.item()
                len_of_batch += 1
                
                if args.log:
                    wandb.log({"val_step_loss": loss.item(), "epoch": epoch, "val_step": step})
                
                if args.dev:
                    dev_count += 1
                    if dev_count == 10:
                        break
            except Exception as e:
                print(f"Error during validation at step {step}: {e}")
                continue

    if len_of_batch == 0:  # Handle case where all batches are invalid
        print("No valid batches for validation.")
        average_loss = float('inf')
    else:
        average_loss = total_loss / len_of_batch
    
    return_dic = {
        'average_loss': average_loss
    }
    return return_dic
