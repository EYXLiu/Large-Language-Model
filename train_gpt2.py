import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F 

# code from previous implementation (nanoGPT, tokenizerGPT)

# sets to train on: slimpajama, fineweb, hellaswag (multiple choice with most likely completion)
# hellaswag used mostly for evaluation, not for testing

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # 3 is an ugly number (no power of 2)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.scale_down = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size()[-1]))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        # flash attention, fused kernal that is significantly faster (4 blocks fused into 1) (7.6 times faster according to paper)
        # cannot be found by torch.compile, algorithm is rewritten
        # more flops than before
        # mindful of memory hierarchy, fewer reads and writes to HBM, never materializes attention matrix, never read/written to HBM -> millions of numbers per head
        # load from HBM to SRAM, compute attention factor with respect to K, Q, V, and scale by right normalization factor then add up
        # based on flashattention and flashattention2 paper
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module): 

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # smoother relu
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.scale_down = 1

    def forward(self, x):   
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) #ffn

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) 
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    # 50527 is a very ugly number, no powers of 2, odd, etc.. 
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        #wte shared due to similarities in how it works (semantically similar should be embedded in similar places and should have similar probabilities) + save space 
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self.__init__weights)
        
    # initialize weights following gpt2, so normal distribution across 0.02 for Linear modules, if bias then initialize to 0 -> default is uniform distribution, and embeddings same as Linear
    # initialized layerNorm is scale 1, offset 0, which is good
    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'scale_down'):
                std *= (2 * self.config.n_layer) ** -0.5 # scale it down so that the std/var don't increase as you add layers, keeps it around the average std
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # typically std is 1/sqrt(num of features) -> reasonably close to 0.02 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    #batch and token indices, 
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        while idx.size(1) < max_new_tokens:
            with torch.no_grad():
                logits, loss = model(idx)
                logits = logits[:, -1, :]
                logits = logits / 3.0
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                idx = torch.cat((idx, xcol), dim=1)
        return idx
        

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 25 is an ugly number 
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        print(f"loading weights from {model_type}")
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] 
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys)
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    
    def configure_optimizers(self, wd, lr, device):
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # create optim groups, any parameters that are 2D+ are weights decayed, otherwise no
        # matmul and embeddings yes, biases and layernorms no
        optim_groups = [
            {'params': decay_params, 'weight_decay': wd},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        # since fused is a lot faster, check if it's avaliable
        # fused means the kernels are all fused into a single kernal, call one kernal that updates them
        import inspect
        fused_avaliable = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avaliable and 'cuda' in device
        # not avaliable pensive
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        data_root = "edu_fineweb10B" #file directory
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        
        self.reset()
        
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        
    def next_batch(self):
        B, T = self.B, self.T
        batch = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = batch[:-1].view(B, T)
        y = batch[1:].view(B, T)
        
        self.current_position += B * T
        
        if self.current_position + (B * T + 1)> len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y
    
import time

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
ddp = int(os.environ.get('RANK', -1)) != -1
ddp_rank = None
ddp_local_rank = None
ddp_world_size = None
if ddp:
    assert torch.cuda.is_available() # we need cuda for this to work (this is processing on multiple GPUS)
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # make sure only prints for one of them, so the first
else :
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    device = 'cpu'

# gradient accumulation, simulate arbitrary batch size
# run longer, process multiple sequences and add up all gradients to simulate big batch size
# going to add gradients together but not reset, do one single update once they accumulate
# total_batch_size = 524288 # ~0.5M
B = 4 # 64 or 32
T = 32 # 1024 or 2048
# grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

# need to pass ddp_rank (process_rank) and ddp_world_size(num_processes) into train_loader
# DataLoaderLite change
#   self.process_rank = process_rank
#   self.num_processes = num_processes
#   
#   self.current_position = self.B * self.T * self.process_rank
#   
#   self.current_position += B * T * self.num_processes (position advances by entire chunk)
#   if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens)
#       self.current_position = self.B * self.T * self.process_rank

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train") # B=16, T=1024 -> choose numbers with 2, runs more efficiently
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
# since we cant use higher batch sizes (my computer will explode) we use gradient accumulation

# we change from highest in gpu (FP32) to lower because we don't need that accurate precision m23 -> m10
torch.set_float32_matmul_precision('high') # sets to tensorflow32 if avaliable, much faster 


model = GPT(GPTConfig(vocab_size=50304)) # override, add fake tokens, so that the numbers are nicer 
# made wte larger (slight waste of space, never used)
# when used in classifier, network needs to learn that it's never used and has to push it down to 0, however not much different from everything else that isn't in the dataset
# functionally nothing breaks, but just uses slightly more memory, but kernels are much more efficient when computing and usually is faster
model.eval()
model.to(device)
# torch.compile, similar to gcc, reduces python overhead and gpu read/writes
# analyze entire thing, knows exactly whats going to happen, and instead of going layer by layer it, it optimizes the process (eg. compiles entire nn as one object with no python)
# read/write -> doesn't story intermediate steps to HBM and gets it again for next step (stops travel time between HBM and GPU) -> going to memory takes a lot of time, we want to minimize that
# stored on GPU SRAM, which cannot store a lot but accessing values is a lot faster (19TB/s vs 1.5TB/s)
# should always use unless debugging
# hellaswag doesn't work with torch.compile for some reason so turn it off when testing
model = torch.compile(model)
#logits, loss = model(x, y)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # wrap model into DDP
    # in a forward pass, behaves the same
    # in a backward pass, averages the gradients and sets to each model (synchronizes gradients)
raw_model = model.module if ddp else model
# based on gpt3 model, cosine decay with warmup up to a point and then continues at 10% of the value
# gpt3 doesn't go to max_steps, it goes to when theres 260 billion tokens 
max_lr = 6e-4
min_lr = max_lr * 0.1
# warmup and max matches gpt3
warmup_steps = 50 # 715 
max_steps = 1500 # 19073
def get_lr(it):
    # warmup region runs at x learning rate
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # after optimization, go at min learning rate
    if it > max_steps:
        return min_lr
    # use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    
optimizer = raw_model.configure_optimizers(wd=0.1, lr=6e-4, device=device)

for i in range(max_steps):
    t0 = time.time()
    last_step = (i == max_steps - 1)
    if i % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    # every x number of steps and last step check hellaswag and log it
    # every x number of steps and last step log model validation
    model.train()
    optimizer.zero_grad()
    # loss_accum = 0.0
    #for step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
        # based on pytorch documentation, Automatic Model Precision package (run with mixed precision because bfloat is e5, meaning it doesn't take as long to pull memory)
        # many models are restrained not by the calculation speed, but by the speed at which it can pull values from memory
        # torch.float16 requires gradient scalers, use bfloat16
        # some also do not cast: eg. softmax or 
        # with torch.autocast(device_type=device, dtype=torch.bfloat16): # currently on cpu, so cannot cast to bfloat16
    logits, loss = model(x, y)
        # don't autocast to backward and optimizer because pytorch does not recommend
        # loss = loss / grad_accum_steps # add normalizer because it is summ of loss, not proportion of loss
        # loss_accum += loss.detach()
    # if ddp: 
        # model.require_backward_grad_sync = (step == grad_accum - 1) # makes it so it doesn't backwards loss when outside the big batch, but only averages after the entire batch(DDP)
    loss.backward()
    # if ddp:
        # dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # gets the averaged loss, not per 
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip global norm -> might get unlucky during optimization and get really high loss -> high gradient, and shoke model
    # determine learning rate 
    lr = get_lr(i)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    #torch.cuda.synchronize() # wait for GPU to finish running and sync cores if you have multiple if necessary
    t1 = time.time()
    dt = (t1 - t0) *1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    if master_process: 
        if (i % 10 == 0): 
            print(f"step: {i} | loss: {loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | time: {dt:.2f}ms | tokens: {tokens_per_sec:.2f}")

max_length = 30
returned = 5

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(returned, 1)
x = tokens.to(device)

torch.manual_seed(1)
idx = model.generate(x, max_length)
        
for i in range(returned):
    tokens = idx[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

if ddp:
    destroy_process_group()