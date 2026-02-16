import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
seed = 1
epochs = 30
patch_size = 8
T = 1
num_head = 1
set_variance = 1.0

# VARIABLE parameters (Iterate over these)
depths = [3, 6, 9]
base_lrs = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.0]

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
    def forward(ctx, h, h_norm, U_fwd, U_bwd, W_fwd, W_bwd, U_fwd_copy, W_fwd_copy,
                alpha, inv_sqrt_n, phi, phi_prime):
        # We save h_norm because activations/projections are based on it
        ctx.save_for_backward(h_norm, W_fwd)
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
        h_norm, W_fwd = ctx.saved_tensors
        alpha = ctx.alpha
        phi = ctx.phi
        phi_prime = ctx.phi_prime

        # --- UPDATED BACKWARD FORMULA FOR PRE-NORM ---
        
        # 1. Recompute activation 'a' and its derivative w.r.t 'h_norm'
        a = phi(h_norm)
        da_dh_norm = phi_prime(h_norm) 

        # 2. Gradient for W_fwd
        grad_W_fwd = alpha * torch.einsum("...i,...j->ij", grad_y, a)

        # 3. Gradient w.r.t Inputs
        # Grad for residual 'h': Direct pass-through
        grad_h = grad_y 

        # Grad for branch 'h_norm': Backprop through W, then Phi
        grad_a = alpha * (grad_y @ W_fwd) 
        grad_h_norm = grad_a * da_dh_norm 
        
        return (
            grad_h,      # Residual gradient
            grad_h_norm, # Normed input gradient
            None,        # grad_U_fwd (Unused)
            None,        # grad_U_bwd (Unused)
            grad_W_fwd,
            None,        # grad_W_bwd (Unused)
            None,        # U_fwd_copy
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

        # forward weights
        self.U_fwd = nn.Parameter(torch.randn(n, n))
        self.W_fwd = nn.Parameter(torch.randn(n, n))

        # fully decoupled
        self.U_bwd = nn.Parameter(torch.randn(n, n))
        self.W_bwd = nn.Parameter(torch.randn(n, n))

        nn.init.normal_(self.U_fwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.W_fwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.U_bwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.W_bwd, mean=0.0, std=set_variance)
        
        self.U_fwd_copy = self.U_fwd.clone().detach().cuda()
        self.W_fwd_copy = self.W_fwd.clone().detach().cuda()

        self.alpha = math.sqrt(T / (L * n))
        self.inv_sqrt_n = 1.0 / math.sqrt(n)

    def forward(self, h):
        # Calculate Normed Input
        h_norm = self.norm(h)
        
        return DecoupledFNNBlockFn.apply(
            h,          # Residual backbone
            h_norm,     # Normed input for branch
            self.U_fwd,
            self.U_bwd,
            self.W_fwd,
            self.W_bwd,
            self.U_fwd_copy,
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
        q_fwd, k_fwd, v_fwd, o_fwd,    
        q_bwd, k_bwd, v_bwd, o_bwd,    
        q_fwd_copy, k_fwd_copy, v_fwd_copy, o_fwd_copy,
        alpha,                  
        inv_sqrt_n,             
        num_heads,              
        softmax,                
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
        value_ref = h_norm          

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

        grad_out_merge = grad_out_proj @ o_fwd
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
        grad_h_norm_from_q = grad_query @ q_fwd
        grad_h_norm_from_k = grad_key   @ k_fwd
        grad_h_norm_from_v = grad_value 
        
        grad_h_norm = grad_h_norm_from_q + grad_h_norm_from_k + grad_h_norm_from_v

        grad_alpha = None
        grad_inv_sqrt_n = None
        grad_num_heads = None
        grad_softmax = None
        grad_softmax_prime = None

        return (
            grad_h,      # Residual
            grad_h_norm, # Normed Branch
            grad_q_fwd, grad_k_fwd, None, grad_o_fwd, 
            None, None, None, None,                    
            None, None, None, None,                    
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
        # Calculate Normed Input
        h_norm = self.norm(h)

        return DecoupledAttBlockFn.apply(
            h,          # Residual
            h_norm,     # Normed
            self.q_fwd,
            self.k_fwd,
            None,       # v_fwd
            self.o_fwd,
            self.q_bwd,
            self.k_bwd,
            None,       # v_bwd
            self.o_bwd,
            self.q_fwd_copy,
            self.k_fwd_copy,
            None,       # v_fwd_copy
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
    # Note: Using the passed 'depth' variable for scaling
    scale_factor = width * math.sqrt(depth)
    
    for name, param in model.named_parameters():
        if "q_fwd" in name or "k_fwd" in name or "v_fwd" in name or "U_fwd" in name:
            params.append({'params': param, 'lr': base_lr * scale_factor})
        elif "o_bwd" in name or "W_bwd" in name:
            params.append({'params': param, 'lr': base_lr * scale_factor})
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
print(f"STARTING EXPERIMENTS | Depths: {depths} | LRs: {base_lrs}")
print("=================================================================================")

for depth in depths:
    # We save ONE json per Depth, covering all LRs
    task_name = "depth-decoupled-prenorm-var01ep30-depth:{}-width:{}-seed:{}".format(depth, width, seed)
    
    # Structure to hold results
    results = {
        "task_info": {
            "width": width, "depth": depth, "T": T, "seed": seed, "epochs": epochs, "base_lrs": base_lrs
        },
        "summary": {},    # To store the Best Acc/Loss for each LR
        "training_results": {} # To store per-epoch logs for each LR (matches your old structure key)
    }
    
    print(f"\nProcessing Depth {depth}...")

    for base_lr in base_lrs:
        set_seed(seed)

        seqlen = int(calculate_vit_sequence_length(total_dim=3072, channels=3, patch_size=patch_size))
        input_dim = int(3072/seqlen)
        
        # Re-init model for each depth/LR combination
        model = DecoupledVitModel(L=depth, n=width, seqlen=seqlen, input_dim=input_dim, num_heads=num_head, T=T).to(device)
        optimizer = set_optimizer_with_different_lr_decoupled(model, width, depth, base_lr=base_lr)
        criterion = nn.CrossEntropyLoss()

        lr_key = f"lr_{base_lr}"
        results["training_results"][lr_key] = {
            "epochs": [],
            "train_loss": [],
            "test_loss": [],
            "valid_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "valid_accuracy": [],
            "train_epoch_loss": []
        }
        
        # Track best performance for this LR
        best_test_acc = 0.0
        min_test_loss = float('inf')
        best_epoch = -1
        
        # --- NEW TRACKING VARIABLES ---
        best_epoch_train_acc = 0.0
        best_epoch_train_loss = 0.0
        
        all_train_loss = []
        
        print(f"   -> LR: {base_lr} Started")
        
        for epoch in range(epochs):
            # 1. Train Step
            model.train()
            for batch_idx, (data, target) in enumerate(tqdm(trainloader, desc=f"Ep {epoch+1} Train")):
                data, target = data.to(device), target.to(device)
                data = img_to_patch_pt(data, patch_size=patch_size, flatten_channels=True)
                optimizer.zero_grad()
                output, mid_layers = model(data)
                loss = criterion(output, target)
                loss.backward()
                all_train_loss.append(loss.item())
                optimizer.step()
            
            # 2. Evaluation Step
            model.eval()
            all_train_loss_after_epoch = []
            all_test_loss = []
            all_acc = []
            all_train_accuracy = []
            
            # Test Loop
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(testloader, desc="Test")):
                    data, target = data.to(device), target.to(device)
                    data = img_to_patch_pt(data, patch_size=patch_size, flatten_channels=True)
                    output, mid_layers = model(data)
                    loss = criterion(output, target)
                    acc = (output.argmax(dim=1) == target).float().mean()
                    all_test_loss.append(loss.item())
                    all_acc.append(acc.item())
            
            # Train Eval Loop
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(trainloader, desc="Train Eval")):
                    data, target = data.to(device), target.to(device)
                    data = img_to_patch_pt(data, patch_size=patch_size, flatten_channels=True)
                    output, mid_layers = model(data)
                    loss = criterion(output, target)
                    all_train_loss_after_epoch.append(loss.item())
                    _, predicted = output.max(1)
                    correct = predicted.eq(target).sum().item()
                    accuracy = correct / target.size(0)
                    all_train_accuracy.append(accuracy)
            
            # Valid Loop
            all_valid_loss = []
            all_valid_accuracy = []
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(validloader, desc="Valid")):
                    data, target = data.to(device), target.to(device)
                    data = img_to_patch_pt(data, patch_size=patch_size, flatten_channels=True)
                    output, mid_layers = model(data)
                    loss = criterion(output, target)
                    all_valid_loss.append(loss.item())
                    acc = (output.argmax(dim=1) == target).float().mean()
                    all_valid_accuracy.append(acc.item())

            # Store metrics
            current_results = results["training_results"][lr_key]
            curr_train_loss = float(np.mean(all_train_loss)) 
            curr_test_loss = float(np.mean(all_test_loss))
            curr_valid_loss = float(np.mean(all_valid_loss))
            curr_train_acc = float(np.mean(all_train_accuracy))
            curr_test_acc = float(np.mean(all_acc))
            curr_valid_acc = float(np.mean(all_valid_accuracy))
            curr_train_epoch_loss = float(np.mean(all_train_loss_after_epoch))

            current_results["epochs"].append(epoch + 1)
            current_results["train_loss"].append(curr_train_loss)
            current_results["test_loss"].append(curr_test_loss)
            current_results["valid_loss"].append(curr_valid_loss)
            current_results["train_accuracy"].append(curr_train_acc)
            current_results["test_accuracy"].append(curr_test_acc)
            current_results["valid_accuracy"].append(curr_valid_acc)
            current_results["train_epoch_loss"].append(curr_train_epoch_loss)
            
            # --- UPDATED BEST LOGIC ---
            if curr_test_acc > best_test_acc:
                best_test_acc = curr_test_acc
                min_test_loss = curr_test_loss
                best_epoch = epoch + 1
                best_epoch_train_acc = curr_train_acc
                best_epoch_train_loss = curr_train_epoch_loss

            print(f"LR: {base_lr}, Epoch: {epoch+1}, Train Loss: {curr_train_epoch_loss:.4f}, Test Loss: {curr_test_loss:.4f}, Train Acc: {curr_train_acc:.4f}, Test Acc: {curr_test_acc:.4f}")
        
        # Save Summary for this LR
        results["summary"][lr_key] = {
            "best_test_acc": best_test_acc,
            "min_test_loss": min_test_loss,
            "best_epoch": best_epoch,
            "best_epoch_train_acc": best_epoch_train_acc,
            "best_epoch_train_loss": best_epoch_train_loss
        }
        
        # --- UPDATED FINAL PRINT ---
        print(f"   -> LR {base_lr} Done. Best Result: Test Acc {best_test_acc:.4f}, Test Loss {min_test_loss:.4f}, Train Acc {best_epoch_train_acc:.4f}, Train Loss {best_epoch_train_loss:.4f} (Ep {best_epoch})")

        # Save to JSON
        with open(f"{my_path}/{task_name}.json", 'w') as f:
            json.dump(results, f, indent=2)

print("All experiments completed.")