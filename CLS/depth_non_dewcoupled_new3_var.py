import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# -----------------------------------------------------------------------------
# Hyperparameters & Setup
# -----------------------------------------------------------------------------
# FIXED parameters
width = 256
import os

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import random
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from torch.autograd import Function
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, reduce, repeat

# -----------------------------------------------------------------------------
# Hyperparameters & Setup
# -----------------------------------------------------------------------------
# FIXED parameters
width = 256
epochs = 20           
seed = 1
seeds = [42, 111, 1356]
epochs = 20
patch_size = 8
T = 1
num_head = 1
set_variance = 1.0

# VARIABLE parameters (Iterate over these)
depths = [3, 6, 9]

# Generate 10 logarithmically spaced learning rates from 0.1 to 3.0
base_lrs = (2 ** np.linspace(np.log2(0.1), np.log2(3.0), 10)).tolist()

# Print the generated learning rates rounded to 3 decimal places for easy checking
formatted_lrs = [f"{lr:.3f}" for lr in base_lrs]
print(f"Generated base_lrs: {formatted_lrs}")

# 1. Seed Setting
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(seed)

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 512
test_batch_size = 512

trainset_full = torchvision.datasets.CIFAR10(
    root='../dataset', train=True, download=True, transform=transform_train)

validset_full = torchvision.datasets.CIFAR10(
    root='../dataset', train=True, download=False, transform=transform_test)

valid_size = 5000
num_train = len(trainset_full)

g = torch.Generator()
g.manual_seed(seed)
perm = torch.randperm(num_train, generator=g).tolist()
valid_idx = perm[:valid_size]
train_idx = perm[valid_size:]

trainset = Subset(trainset_full, train_idx)
validset = Subset(validset_full, valid_idx)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=test_batch_size, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

def img_to_patch_pt(x, patch_size, flatten_channels=True):
    x = x.permute(0, 2, 3, 1)
    B, H, W, C = x.shape
    x = x.reshape(B, H//patch_size, patch_size, W//patch_size, patch_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, -1, *x.shape[3:]) 
    if flatten_channels:
        x = x.reshape(B, x.shape[1], -1)
    return x

# -----------------------------------------------------------------------------
# Model Components: FNN (Pre-Norm)
# -----------------------------------------------------------------------------

def relu(x):
    return torch.relu(x)

def relu_prime(x):
    return (x > 0).to(x.dtype)

class DecoupledFNNBlockFn(Function):
    @staticmethod
    def forward(ctx, h, h_norm, W_fwd, W_bwd, W_fwd_copy, alpha, inv_sqrt_n, phi, phi_prime):
        # We save h_norm because activations/projections are based on it
        ctx.save_for_backward(h_norm, W_fwd, W_bwd)
        ctx.alpha = alpha
        ctx.phi = phi
        ctx.phi_prime = phi_prime

        # 1. Apply activation directly to input h_norm (Pre-Norm)
        a = phi(h_norm)
        
        # 2. Project Down using W
        y_down = a @ W_fwd.T
        
        # 3. Residual Connection (h is the residual stream, y_down is the branch)
        y = h + alpha * y_down

        # Reference for decoupled logic (using h_norm)
        a_ref = phi(h_norm)
        y_fwd_ref = a @ W_fwd_copy.T

        return y, torch.zeros_like(h), y_down - y_fwd_ref

    @staticmethod
    def backward(ctx, grad_y, grad_x_fwd_diff=None, grad_y_fwd_diff=None):
        h_norm, W_fwd, W_bwd = ctx.saved_tensors
        alpha = ctx.alpha
        phi = ctx.phi
        phi_prime = ctx.phi_prime

        # --- UPDATED BACKWARD FORMULA FOR PRE-NORM ---
        
        # 1. Recompute activation 'a' and its derivative w.r.t 'h_norm'
        a = phi(h_norm)
        da_dh_norm = phi_prime(h_norm) 

        # 2. Gradient for W_fwd
        grad_W_fwd = alpha * torch.einsum("...i,...j->ij", grad_y, a)

        # Grad for branch 'h_norm': Backprop through W, then Phi
        grad_a = alpha * (grad_y @ W_bwd) 
        grad_h_norm = grad_a * da_dh_norm

        # Residual gradient
        grad_h = grad_y.clone()

        # ===== tie bwd grads (Kolen-Pollack style) =====
        grad_W_bwd = grad_W_fwd.clone()
        
        return (
            grad_h,      # Residual gradient
            grad_h_norm, # Normed input gradient
            grad_W_fwd,
            grad_W_bwd,
            None,        # W_fwd_copy
            None,        # alpha
            None,        # inv_sqrt_n
            None,        # phi
            None,        # phi_prime
        )

class DecoupledFNN(nn.Module):
    def __init__(self, n, T, L, phi=relu, phi_prime=relu_prime):
        super().__init__()
        self.n = n
        self.T = T
        self.L = L
        self.phi = phi
        self.phi_prime = phi_prime

        # PRE-NORM Layer
        self.norm = nn.RMSNorm(n, elementwise_affine=False)

        self.W_fwd = nn.Parameter(torch.randn(n, n))
        self.W_bwd = nn.Parameter(torch.randn(n, n))

        nn.init.normal_(self.W_fwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.W_bwd, mean=0.0, std=set_variance)
        
        self.W_fwd_copy = self.W_fwd.clone().detach().cuda()

        self.alpha = math.sqrt(T / (L * n))
        self.inv_sqrt_n = 1.0 / math.sqrt(n)

    def forward(self, h):
        # ---> ADD THIS LINE <---
        self.W_fwd_copy.copy_(self.W_fwd.detach())

        # Calculate Normed Input
        h_norm = self.norm(h)
        
        return DecoupledFNNBlockFn.apply(
            h,          # Residual backbone
            h_norm,     # Normed input for branch
            self.W_fwd,
            self.W_bwd,
            self.W_fwd_copy,
            self.alpha,
            self.inv_sqrt_n,
            self.phi,
            self.phi_prime,
        )

# -----------------------------------------------------------------------------
# Model Components: Attention (Pre-Norm)
# -----------------------------------------------------------------------------

def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)

class DecoupledAttBlockFn(Function):
    @staticmethod
    def forward(
        ctx,
        h,                      # [b, n, d_model] - RESIDUAL
        h_norm,                 # [b, n, d_model] - NORMALIZED
        q_fwd, k_fwd, o_fwd,
        q_bwd, k_bwd, o_bwd,
        q_fwd_copy, k_fwd_copy, o_fwd_copy,
        alpha, inv_sqrt_n, num_heads, softmax,
        softmax_prime=None  
    ):
        # Save h_norm for backward (v_* are unused)
        ctx.save_for_backward(h_norm, q_fwd, k_fwd, o_fwd, q_bwd, k_bwd, o_bwd, q_fwd_copy, k_fwd_copy, o_fwd_copy)
        ctx.alpha = alpha
        ctx.inv_sqrt_n = inv_sqrt_n
        ctx.num_heads = num_heads
        ctx.softmax = softmax
        ctx.softmax_prime = softmax_prime

        b, n, d_model = h.shape
        head_dim = d_model // num_heads

        # 1. Projections (using h_norm for Pre-Norm)
        query = h_norm @ q_fwd.T   
        key   = h_norm @ k_fwd.T   
        value = h_norm              

        queries = inv_sqrt_n * rearrange(query, "b s (h d) -> b h s d", h=num_heads)
        keys    = inv_sqrt_n * rearrange(key,   "b s (h d) -> b h s d", h=num_heads)
        values  = rearrange(value, "b s (h d) -> b h s d", h=num_heads)

        # 2. Attention Mechanism
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys) 

        # 3. Reference Calculations
        query_ref = h_norm @ q_fwd_copy.T
        key_ref   = h_norm @ k_fwd_copy.T       

        queries_ref = inv_sqrt_n * rearrange(query_ref, "b s (h d) -> b h s d", h=num_heads)
        keys_ref    = inv_sqrt_n * rearrange(key_ref,   "b s (h d) -> b h s d", h=num_heads)
        
        energy_ref = torch.einsum("bhqd, bhkd -> bhqk", queries_ref, keys_ref)

        scaling = 1.0 / head_dim
        att_minus = F.softmax((energy - energy_ref)*scaling, dim=-1)

        attn = softmax(scaling * energy, dim=-1)

        out = torch.einsum("bhqk, bhkd -> bhqd", attn, values)
        out_merge = rearrange(out, "b h n d -> b n (h d)") 
        out_proj = out_merge @ o_fwd.T                      

        y = h + alpha * out_proj
        
        return y, queries-queries_ref, keys-keys_ref, torch.zeros_like(values), scaling*(energy-energy_ref), att_minus

    @staticmethod
    def backward(ctx, grad_y, q_diff=None, k_diff=None, v_diff=None, logits_diff=None, attn_diff=None):
        # Retrieve saved tensors
        h_norm, q_fwd, k_fwd, o_fwd, q_bwd, k_bwd, o_bwd, q_fwd_copy, k_fwd_copy, o_fwd_copy = ctx.saved_tensors
        alpha = ctx.alpha
        inv_sqrt_n = ctx.inv_sqrt_n
        num_heads = ctx.num_heads
        softmax = ctx.softmax
        b, n, d_model = h_norm.shape
        head_dim = d_model // num_heads

        # Recompute Forward using h_norm
        query = h_norm @ q_fwd.T
        key   = h_norm @ k_fwd.T
        value = h_norm 

        Q = inv_sqrt_n * rearrange(query, "b s (h d) -> b h s d", h=num_heads)
        K = inv_sqrt_n * rearrange(key, "b s (h d) -> b h s d", h=num_heads)
        V = rearrange(value, "b s (h d) -> b h s d", h=num_heads) 

        energy = torch.einsum("bhqd, bhkd -> bhqk", Q, K)
        scaling = 1.0 / head_dim
        logits = scaling * energy
        A = softmax(logits, dim=-1)

        out = torch.einsum("bhqk, bhkd -> bhqd", A, V)
        out_merge = rearrange(out, "b h n d -> b n (h d)")

        # --- Backward Logic ---
        # 1. Residual Gradient
        grad_h = grad_y.clone()

        # 2. Compute branch gradients
        grad_out_proj = alpha * grad_y
        grad_o_fwd = grad_out_proj.reshape(-1, d_model).T @ out_merge.reshape(-1, d_model)

        grad_out_merge = grad_out_proj @ o_bwd
        grad_out = rearrange(grad_out_merge, "b n (h d) -> b h n d", h=num_heads)

        grad_A = torch.einsum("bhqd, bhkd -> bhqk", grad_out, V)
        grad_V = torch.einsum("bhqk, bhqd -> bhkd", A, grad_out) 

        tmp = (grad_A * A).sum(dim=-1, keepdim=True)
        grad_logits = A * (grad_A - tmp)
        grad_energy = scaling * grad_logits

        grad_Q = torch.einsum("bhqk, bhkd -> bhqd", grad_energy, K)
        grad_K = torch.einsum("bhqk, bhqd -> bhkd", grad_energy, Q)

        grad_query = rearrange(inv_sqrt_n * grad_Q, "b h s d -> b s (h d)")
        grad_key   = rearrange(inv_sqrt_n * grad_K, "b h s d -> b s (h d)")
        
        grad_value = rearrange(grad_V, "b h s d -> b s (h d)") 

        h_flat = h_norm.reshape(-1, d_model)
        grad_query_flat = grad_query.reshape(-1, d_model)
        grad_key_flat   = grad_key.reshape(-1, d_model)

        grad_q_fwd = grad_query_flat.T @ h_flat
        grad_k_fwd = grad_key_flat.T   @ h_flat
        
        # 3. Branch Input Gradient (h_norm)
        grad_h_norm_from_q = grad_query @ q_bwd
        grad_h_norm_from_k = grad_key   @ k_bwd
        grad_h_norm_from_v = grad_value 
        
        grad_h_norm = grad_h_norm_from_q + grad_h_norm_from_k + grad_h_norm_from_v

        grad_q_bwd = grad_q_fwd.clone()
        grad_k_bwd = grad_k_fwd.clone()
        grad_o_bwd = grad_o_fwd.clone()

        grad_alpha = None
        grad_inv_sqrt_n = None
        grad_num_heads = None
        grad_softmax = None
        grad_softmax_prime = None

        return (
            grad_h,      # Residual
            grad_h_norm, # Normed Branch
            grad_q_fwd, grad_k_fwd, grad_o_fwd, 
            grad_q_bwd, grad_k_bwd, grad_o_bwd,                     
            None, None, None,                     
            grad_alpha,
            grad_inv_sqrt_n,
            grad_num_heads,
            grad_softmax,
            grad_softmax_prime
        )

class DecoupledAttention(nn.Module):
    def __init__(self, n, T, L, num_heads, softmax=softmax, softmax_prime=None):
        super().__init__()
        self.n = n
        self.T = T
        self.L = L
        self.num_heads = num_heads
        self.softmax = softmax
        self.softmax_prime = None

        # PRE-NORM Layer
        self.norm = nn.RMSNorm(n, elementwise_affine=False)

        # forward weights
        self.q_fwd = nn.Parameter(torch.randn(n, n))
        self.k_fwd = nn.Parameter(torch.randn(n, n))
        self.o_fwd = nn.Parameter(torch.randn(n, n))

        # full decoupled
        self.q_bwd = nn.Parameter(torch.randn(n, n))
        self.k_bwd = nn.Parameter(torch.randn(n, n))
        self.o_bwd = nn.Parameter(torch.randn(n, n))

        # weight initial
        nn.init.normal_(self.q_fwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.k_fwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.o_fwd, mean=0.0, std=set_variance)

        nn.init.normal_(self.q_bwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.k_bwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.o_bwd, mean=0.0, std=set_variance)

        # weight copy
        self.q_fwd_copy = self.q_fwd.clone().detach().cuda()
        self.k_fwd_copy = self.k_fwd.clone().detach().cuda()
        self.o_fwd_copy = self.o_fwd.clone().detach().cuda()
        
        self.alpha = math.sqrt(T / (L * n)) 
        self.inv_sqrt_n = 1.0 / math.sqrt(n) 

    def forward(self, h):
        # ---> ADD THESE THREE LINES <---
        self.q_fwd_copy.copy_(self.q_fwd.detach())
        self.k_fwd_copy.copy_(self.k_fwd.detach())
        self.o_fwd_copy.copy_(self.o_fwd.detach())

        # Calculate Normed Input
        h_norm = self.norm(h)

        return DecoupledAttBlockFn.apply(
            h,          # Residual
            h_norm,     # Normed
            self.q_fwd,
            self.k_fwd,
            self.o_fwd,
            self.q_bwd,
            self.k_bwd,
            self.o_bwd,
            self.q_fwd_copy,
            self.k_fwd_copy,
            self.o_fwd_copy,
            self.alpha,
            self.inv_sqrt_n,
            self.num_heads,
            self.softmax,
            self.softmax_prime,
        )

# -----------------------------------------------------------------------------
# Main ViT Model
# -----------------------------------------------------------------------------

class DecoupledVitModel(nn.Module):
    def __init__(self, n, L, seqlen, input_dim,num_heads, T):
        super().__init__()
        
        self.embed_dim = n
        self.input_dim = input_dim
        self.L = L

        self.embed = nn.Linear(self.input_dim, self.embed_dim, bias=False)  
        self.cls_token_embed = nn.Parameter(torch.randn(1,1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1+seqlen, self.embed_dim)) 
        self.num_head = num_heads

        self.attention_h = []
        for _ in range(L):
            self.attention_h.append(DecoupledAttention(n, T, L, num_heads, softmax, None))
        self.mlp_h = []
        for _ in range(L):
            self.mlp_h.append(DecoupledFNN(n, T, L, relu, relu_prime))

        self.attention = nn.ModuleList(self.attention_h)
        self.fnn = nn.ModuleList(self.mlp_h)

        self.class_head = nn.Linear(self.embed_dim, 10, bias=False)

        #unit-variance initalizaion
        nn.init.normal_(self.embed.weight, mean=0.0, std=set_variance)
        nn.init.normal_(self.cls_token_embed, mean=0.0, std=set_variance)
        nn.init.normal_(self.pos_embed, mean=0.0, std=set_variance)
        nn.init.normal_(self.class_head.weight, mean=0.0, std=set_variance)
        
    def forward(self, x,flag_return_mid=False):
        B, T, _ = x.shape
        x = self.embed(x)/math.sqrt(self.input_dim)

        cls_tokens = repeat(self.cls_token_embed, '() n e -> b n e', b=B)
        x = torch.cat([cls_tokens, x], dim=1) 
        h = math.sqrt(1/2)*(x + self.pos_embed)
        
        mid_layers = {}

        # middle transformer blocks
        for l in range(self.L):
            # attention block
            h,q_diff, k_diff, v_diff, logits_diff, attn_diff = self.attention[l](h)
            if flag_return_mid:
                mid_layers["layer{}_logits".format(l)] = logits_diff.clone().detach().cpu().numpy()
                mid_layers["layer{}_attn".format(l)] = attn_diff.clone().detach().cpu().numpy()
                mid_layers["layer{}_q".format(l)] = q_diff.clone().detach().cpu().numpy()
                mid_layers["layer{}_k".format(l)] = k_diff.clone().detach().cpu().numpy()
                mid_layers["layer{}_v".format(l)] = v_diff.clone().detach().cpu().numpy()
            
            # fnn block
            h,_,_ = self.fnn[l](h)
        
        # classification head
        classfiy_h = (1/self.embed_dim)*self.class_head(h[:,0,:])
        return classfiy_h,mid_layers

# -----------------------------------------------------------------------------
# Utils & Training Loop
# -----------------------------------------------------------------------------

def calculate_vit_sequence_length(total_dim=3072, channels=3, patch_size=4):
    if total_dim % channels != 0:
        raise ValueError(f"Total dim {total_dim} not divisible by channels {channels}.")
    total_pixels = total_dim // channels
    image_side = int(math.sqrt(total_pixels)) # Changed to int
    
    num_patches_h = image_side // patch_size
    num_patches_w = image_side // patch_size
    sequence_length = num_patches_h * num_patches_w
    return sequence_length

def set_optimizer_with_different_lr_decoupled(model, width, depth, base_lr=0.0):
    params = []
    scale_factor = width * math.sqrt(depth)
    
    for name, param in model.named_parameters():
        if "q_fwd" in name or "k_fwd" in name:
            params.append({'params': param, 'lr': base_lr * scale_factor})
        elif "o_bwd" in name:
            params.append({'params': param, 'lr': base_lr * scale_factor})
        elif any(k in name for k in ["o_fwd", "q_bwd", "k_bwd", "W_fwd", "W_bwd"]):
            params.append({'params': param, 'lr': base_lr * width})
        else:
            params.append({'params': param, 'lr': base_lr * width})
            
    optimizer = optim.SGD(params)
    return optimizer



# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_path = "/home/ztang13/workspace/SparseAttention/CLS/data" 
if not os.path.exists(my_path):
    os.makedirs(my_path)



print("=================================================================================")
print(f"STARTING EXPERIMENTS | Depths: {depths} | LRs: {base_lrs} | Seeds: {seeds}")
print("=================================================================================")

for depth in depths:
    task_name = "depth-decoupled-prenorm-var01ep{}-depth:{}-width:{}-seeds_averaged".format(epochs, depth, width)
    
    results = {
        "task_info": {
            "width": width, "depth": depth, "T": T, "seeds_used": seeds, "epochs": epochs, "base_lrs": base_lrs
        },
        "summary": {},    
        "training_results": {} 
    }
    
    print(f"\nProcessing Depth {depth}...")

    for base_lr in base_lrs:
        print(f"   -> LR: {base_lr} Started across {len(seeds)} independent seeds")
        
        lr_key = f"lr_{base_lr}"
        results["training_results"][lr_key] = {
            "epochs": list(range(1, epochs + 1)),
            "train_loss": [],
            "test_loss": [],
            "valid_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "valid_accuracy": [],
            "train_epoch_loss": []
        }
        
        num_seeds = len(seeds)
        hist_train_loss = np.zeros((num_seeds, epochs))
        hist_test_loss = np.zeros((num_seeds, epochs))
        hist_valid_loss = np.zeros((num_seeds, epochs))
        hist_train_acc = np.zeros((num_seeds, epochs))
        hist_test_acc = np.zeros((num_seeds, epochs))
        hist_valid_acc = np.zeros((num_seeds, epochs))
        hist_train_epoch_loss = np.zeros((num_seeds, epochs))

        # --- SEED LOOP ---
        for s_idx, current_seed in enumerate(seeds):
            # Reset random seeds so model weights initialize uniquely for this seed
            set_seed(current_seed)

            seqlen = int(calculate_vit_sequence_length(total_dim=3072, channels=3, patch_size=patch_size))
            input_dim = int(3072/seqlen)
            
            model = DecoupledVitModel(L=depth, n=width, seqlen=seqlen, input_dim=input_dim, num_heads=num_head, T=T).to(device)
            optimizer = set_optimizer_with_different_lr_decoupled(model, width, depth, base_lr=base_lr)
            criterion = nn.CrossEntropyLoss()

            all_train_loss = [] 

            # --- EPOCH LOOP ---
            for epoch in range(epochs):
                model.train()
                for batch_idx, (data, target) in enumerate(tqdm(trainloader, desc=f"D{depth}|LR{base_lr}|S{current_seed}|Ep{epoch+1} Train", leave=False)):
                    data, target = data.to(device), target.to(device)
                    data = img_to_patch_pt(data, patch_size=patch_size, flatten_channels=True)
                    optimizer.zero_grad()
                    output, mid_layers = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    all_train_loss.append(loss.item())
                    optimizer.step()
                
                model.eval()
                all_train_loss_after_epoch = []
                all_test_loss = []
                all_acc = []
                all_train_accuracy = []
                
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(tqdm(testloader, desc="Test", leave=False)):
                        data, target = data.to(device), target.to(device)
                        data = img_to_patch_pt(data, patch_size=patch_size, flatten_channels=True)
                        output, mid_layers = model(data)
                        loss = criterion(output, target)
                        acc = (output.argmax(dim=1) == target).float().mean()
                        all_test_loss.append(loss.item())
                        all_acc.append(acc.item())
                
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(tqdm(trainloader, desc="Train Eval", leave=False)):
                        data, target = data.to(device), target.to(device)
                        data = img_to_patch_pt(data, patch_size=patch_size, flatten_channels=True)
                        output, mid_layers = model(data)
                        loss = criterion(output, target)
                        all_train_loss_after_epoch.append(loss.item())
                        _, predicted = output.max(1)
                        correct = predicted.eq(target).sum().item()
                        accuracy = correct / target.size(0)
                        all_train_accuracy.append(accuracy)
                
                all_valid_loss = []
                all_valid_accuracy = []
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(tqdm(validloader, desc="Valid", leave=False)):
                        data, target = data.to(device), target.to(device)
                        data = img_to_patch_pt(data, patch_size=patch_size, flatten_channels=True)
                        output, mid_layers = model(data)
                        loss = criterion(output, target)
                        all_valid_loss.append(loss.item())
                        acc = (output.argmax(dim=1) == target).float().mean()
                        all_valid_accuracy.append(acc.item())

                # Calculate current values
                curr_train_loss_val = float(np.mean(all_train_loss))
                curr_test_loss_val = float(np.mean(all_test_loss))
                curr_valid_loss_val = float(np.mean(all_valid_loss))    
                curr_train_acc_val = float(np.mean(all_train_accuracy))
                curr_test_acc_val = float(np.mean(all_acc))
                curr_valid_acc_val = float(np.mean(all_valid_accuracy)) 
                curr_train_epoch_loss_val = float(np.mean(all_train_loss_after_epoch))

                # Populate history arrays for this specific seed and epoch
                hist_train_loss[s_idx, epoch] = curr_train_loss_val 
                hist_test_loss[s_idx, epoch] = curr_test_loss_val
                hist_valid_loss[s_idx, epoch] = curr_valid_loss_val    
                hist_train_acc[s_idx, epoch] = curr_train_acc_val
                hist_test_acc[s_idx, epoch] = curr_test_acc_val
                hist_valid_acc[s_idx, epoch] = curr_valid_acc_val 
                hist_train_epoch_loss[s_idx, epoch] = curr_train_epoch_loss_val
                
                # --- PRINT: Per Seed, Per Epoch ---
                print(f"   [Seed {current_seed} | Ep {epoch+1}/{epochs}] "
                      f"Train Loss: {curr_train_epoch_loss_val:.4f}, Train Acc: {curr_train_acc_val:.4f} | "
                      f"Valid Loss: {curr_valid_loss_val:.4f}, Valid Acc: {curr_valid_acc_val:.4f} | "
                      f"Test Loss: {curr_test_loss_val:.4f}, Test Acc: {curr_test_acc_val:.4f}")

        # ==========================================================
        # After all seeds finish for this LR: Average the results
        # ==========================================================
        
        avg_train_loss = np.mean(hist_train_loss, axis=0)
        avg_test_loss = np.mean(hist_test_loss, axis=0)
        avg_valid_loss = np.mean(hist_valid_loss, axis=0)
        avg_train_acc = np.mean(hist_train_acc, axis=0)
        avg_test_acc = np.mean(hist_test_acc, axis=0)
        avg_valid_acc = np.mean(hist_valid_acc, axis=0)
        avg_train_epoch_loss = np.mean(hist_train_epoch_loss, axis=0)

        # --- PRINT: Averaged Per Epoch ---
        print(f"\n   --- AVERAGED RESULTS ACROSS {num_seeds} SEEDS (LR: {base_lr}) ---")
        for ep in range(epochs):
            print(f"   [Averaged | Ep {ep+1}/{epochs}] "
                  f"Train Loss: {avg_train_epoch_loss[ep]:.4f}, Train Acc: {avg_train_acc[ep]:.4f} | "
                  f"Valid Loss: {avg_valid_loss[ep]:.4f}, Valid Acc: {avg_valid_acc[ep]:.4f} | "
                  f"Test Loss: {avg_test_loss[ep]:.4f}, Test Acc: {avg_test_acc[ep]:.4f}")
        print("   -----------------------------------------------------------------")

        best_epoch_idx = np.argmin(avg_valid_loss) 
        best_epoch = best_epoch_idx + 1
        
        best_valid_loss_val = avg_valid_loss[best_epoch_idx]
        best_valid_acc_val = avg_valid_acc[best_epoch_idx]
        best_test_loss_val = avg_test_loss[best_epoch_idx]
        best_test_acc_val = avg_test_acc[best_epoch_idx]
        best_train_loss_val = avg_train_epoch_loss[best_epoch_idx]
        best_train_acc_val = avg_train_acc[best_epoch_idx]

        current_results = results["training_results"][lr_key]
        current_results["train_loss"] = avg_train_loss.tolist()
        current_results["test_loss"] = avg_test_loss.tolist()
        current_results["valid_loss"] = avg_valid_loss.tolist()
        current_results["train_accuracy"] = avg_train_acc.tolist()
        current_results["test_accuracy"] = avg_test_acc.tolist()
        current_results["valid_accuracy"] = avg_valid_acc.tolist()
        current_results["train_epoch_loss"] = avg_train_epoch_loss.tolist()

        results["summary"][lr_key] = {
            "best_epoch": int(best_epoch),
            "best_test_acc": float(best_test_acc_val),
            "min_test_loss": float(best_test_loss_val),
            "best_valid_acc": float(best_valid_acc_val),
            "best_valid_loss": float(best_valid_loss_val),
            "best_epoch_train_acc": float(best_train_acc_val),
            "best_epoch_train_loss": float(best_train_loss_val)
        }

        print(f"   -> LR {base_lr} Best Result (Ep {best_epoch}): Test Acc {best_test_acc_val:.4f}, Test Loss {best_test_loss_val:.4f}, Valid Acc {best_valid_acc_val:.4f}, Valid Loss {best_valid_loss_val:.4f}, Train Acc {best_train_acc_val:.4f}, Train Loss {best_train_loss_val:.4f}")

        with open(f"{my_path}/{task_name}.json", 'w') as f:
            json.dump(results, f, indent=2)

print("\nAll experiments completed.")