from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from torch.utils.data import DataLoader

import time
import math
import os
from dataclasses import dataclass
import torch


from BERT import SBERT
from multi_task_output import Multi_task_output
from dataset import create_loader

#hyperparams for BERT-base (can be adjusted depend on the selected pre-trained model)
B = 4
T = 512
@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    

# Load a mini SNLI dataset
dataset = load_dataset("snli")

#training & validation sets (only select 1000 training samples for demo purpose)
train_loader = iter(create_loader(dataset["train"].select(range(1000)), batch_size= B, shuffle = True))
val_loader = iter(create_loader(dataset["validation"].select(range(500)), batch_size = B, shuffle = True))

#ddp if there are multiple gpu available
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda: {ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0          #this process will do logging, checking etc
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"----using device----: {device}")



#init our multi_task_model here

model = SBERT(Config())
multi_task_model = Multi_task_output(model, 768, num_labels = 3)    #hard code hyperparams here
multi_task_model.to(device)

if ddp:
    multi_task_model = DDP(multi_task_model, device_ids=[ddp_local_rank])
raw_model = multi_task_model.module if ddp else multi_task_model


# Learning rate scheduling
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_step = 2000

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_step:
        return min_lr
    decay_ratio = min(1, max(0, (it - warmup_steps) / (max_step - warmup_steps)))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = torch.optim.AdamW(multi_task_model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

# Training loop, set max_step = 2000 for demo purpose

for step in range(max_step):
    t0 = time.time()
    
    # Validation step
    if step % 100 == 0:
        multi_task_model.eval()
        with torch.no_grad():
            try:
                batch = next(val_loader)
            except StopIteration:
                train_iter = iter(val_loader)
                batch = next(val_loader)
                
            x = batch["sentence_features"]
            for tensor in x:
                tensor.to(device)
            y = batch["label"].to(device)
            
            _, loss = multi_task_model(x, y)
        if ddp_rank == 0:
            print(f"Validation loss: {loss.item():.4f}")

    # Training step
    multi_task_model.train()
    try:
        batch = next(train_loader)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_loader)
    

    x = batch["sentence_features"]            
    for tensor in x:
        tensor.to(device)
    y = batch["label"].to(device)   
    
    optimizer.zero_grad()
    _, loss = multi_task_model(x, y)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(multi_task_model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    
    torch.cuda.synchronize()
    t1 = time.time()

    if ddp_rank == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}, Time: {t1 - t0:.2f}s")

if ddp:
    destroy_process_group()