import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import random
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
from torch import nn

import torch
from torch import Tensor, nn
import argparse

import json
from tqdm import tqdm

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, reduce, repeat
from torch.autograd import Function
from torch.utils.data import Subset


# --- Hyperparameters Updated here ---
width = 256 # 128, 256
# depth = 9  <-- Moved to loop below
depths = [3, 6, 9]

set_variance = 1.0
# base_lrs = [0.2, 0.4, 0.6, 0.8] <-- Updated list
base_lrs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

epochs = 30
seed = 1
T = 1
num_head = 1
patch_size = 8

# 1. 定义设置种子的函数
def set_seed(seed=42):
    """
    设置所有相关的随机种子以确保结果可复现。
    """
    random.seed(seed)                # Python built-in random
    np.random.seed(seed)             # Numpy random
    torch.manual_seed(seed)          # PyTorch CPU random
    torch.cuda.manual_seed(seed)     # PyTorch GPU random
    torch.cuda.manual_seed_all(seed) # PyTorch Multi-GPU random
    
    # 确保卷积操作也是确定性的（可能会降低一点速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置 DataLoader 的生成器种子 (PyTorch >= 1.9 需要)
    os.environ['PYTHONHASHSEED'] = str(seed)

# 2. 调用函数设置种子 (你可以修改 seed 的数值)
set_seed(seed)

#prepare data

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
# 1) 两份 train=True 数据集：train 用增强，valid 用 test 的 normalize（不做随机增强）
trainset_full = torchvision.datasets.CIFAR10(
    root='../dataset', train=True, download=True, transform=transform_train)

validset_full = torchvision.datasets.CIFAR10(
    root='../dataset', train=True, download=False, transform=transform_test)

# 2) 固定切分：valid_size 可自行调整（常用 5000）
valid_size = 5000
num_train = len(trainset_full)

g = torch.Generator()
g.manual_seed(seed)
perm = torch.randperm(num_train, generator=g).tolist()
valid_idx = perm[:valid_size]
train_idx = perm[valid_size:]

trainset = Subset(trainset_full, train_idx)
validset = Subset(validset_full, valid_idx)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)

validloader = torch.utils.data.DataLoader(
    validset, batch_size=test_batch_size, shuffle=False)

testset = torchvision.datasets.CIFAR10(
    root='../dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')
    
def img_to_patch_pt(x, patch_size, flatten_channels=True):
    """
    adapt torchvision datasets
    Inputs:
        x - torch.Tensor representing the image of shape [B, H, W, C]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                                as a feature vector instead of a image grid.
    """
    x = x.permute(0, 2, 3, 1)
    B, H, W, C = x.shape
    x = x.reshape(B, H//patch_size, patch_size, W//patch_size, patch_size, C)
    #x = x.transpose(0, 1, 3, 2, 4, 5)    # [B, H', W', p_H, p_W, C]
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, -1, *x.shape[3:])   # [B, H'*W', p_H, p_W, C]
    if flatten_channels:
        x = x.reshape(B, x.shape[1], -1) # [B, H'*W', p_H*p_W*C]
    return x


# model-FNN
import torch.nn as nn

def relu(x):
    return torch.relu(x)

def relu_prime(x):
    return (x > 0).to(x.dtype)

class DecoupledFNNBlockFn(Function):
    @staticmethod
    def forward(ctx, h, U_fwd, U_bwd, W_fwd, W_bwd, U_fwd_copy, W_fwd_copy,
                alpha, inv_sqrt_n, phi, phi_prime):
        ctx.save_for_backward(h, U_fwd, U_bwd, W_fwd, W_bwd, U_fwd_copy, W_fwd_copy)
        ctx.alpha = alpha
        ctx.inv_sqrt_n = inv_sqrt_n
        ctx.phi = phi
        ctx.phi_prime = phi_prime

        x = inv_sqrt_n * (h @ U_fwd.T)      # [..., n]
        a = phi(x)                          # [..., n]
        y_down = a @ W_fwd.T                # [..., n]
        y = h + alpha * y_down              # [..., n]

        x_fwd_ref = inv_sqrt_n * (h @ U_fwd_copy.T)
        y_fwd_ref = a @ W_fwd_copy.T

        return y, x - x_fwd_ref, y_down - y_fwd_ref

    @staticmethod
    def backward(ctx, grad_y, grad_x_fwd_diff=None, grad_y_fwd_diff=None):
        h, U_fwd, U_bwd, W_fwd, W_bwd, U_fwd_copy, W_fwd_copy = ctx.saved_tensors
        alpha = ctx.alpha
        inv_sqrt_n = ctx.inv_sqrt_n
        phi = ctx.phi
        phi_prime = ctx.phi_prime

        # recompute
        x = inv_sqrt_n * (h @ U_fwd.T)      # [..., n]
        a = phi(x)                          # [..., n]
        da_dx = phi_prime(x)                # [..., n]

        # ===== grad W_fwd: sum over all leading dims (B,S,...) =====
        # grad_W[i,j] = alpha * sum_{...} grad_y[..., i] * a[..., j]
        grad_W_fwd = alpha * torch.einsum("...i,...j->ij", grad_y, a)

        # ===== decoupled path through W: use W_bwd to backprop to a =====
        grad_a = alpha * (grad_y @ W_fwd)       # [..., n]

        # ===== activation =====
        grad_x = grad_a * da_dx                 # [..., n]

        # ===== grad U_fwd =====
        # grad_U[i,j] = (1/sqrt(n)) * sum_{...} grad_x[..., i] * h[..., j]
        grad_U_fwd = inv_sqrt_n * torch.einsum("...i,...j->ij", grad_x, h)

        # ===== decoupled path to h through U: use U_bwd =====
        grad_h_from_x = inv_sqrt_n * (grad_x @ U_fwd)  # [..., n]
        grad_h = grad_y + grad_h_from_x                # [..., n]

        ## tie bwd grads
        #grad_U_bwd = grad_U_fwd.clone()
        #grad_W_bwd = grad_W_fwd.clone()

        return (
            grad_h,
            grad_U_fwd,
            None,
            grad_W_fwd,
            None,
            None,  # U_fwd_copy
            None,  # W_fwd_copy
            None,  # alpha
            None,  # inv_sqrt_n
            None,  # phi
            None,  # phi_prime
        )

class DecoupledFNN(nn.Module):
    def __init__(self, n, T, L, phi=relu, phi_prime=relu_prime):
        super().__init__()
        self.n = n
        self.T = T
        self.L = L
        self.phi = phi
        self.phi_prime = phi_prime

        # forward weights
        self.U_fwd = nn.Parameter(torch.randn(n, n))
        self.W_fwd = nn.Parameter(torch.randn(n, n))

        # fully decoupled
        self.U_bwd = nn.Parameter(torch.randn(n, n))
        self.W_bwd = nn.Parameter(torch.randn(n, n))

        # # depth aware lr
        # self.U_bwd = nn.Parameter(self.U_fwd.detach().clone())
        # self.W_bwd = nn.Parameter(self.W_fwd.detach().clone())

        nn.init.normal_(self.U_fwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.W_fwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.U_bwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.W_bwd, mean=0.0, std=set_variance)
        
        self.U_fwd_copy = self.U_fwd.clone().detach().cuda()
        self.W_fwd_copy = self.W_fwd.clone().detach().cuda()

        self.alpha = math.sqrt(T / (L * n))
        self.inv_sqrt_n = 1.0 / math.sqrt(n)

    def forward(self, h):
        return DecoupledFNNBlockFn.apply(
            h,
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

## mdoel-Attention
import torch

def softmax(x, dim=-1):
    """
    计算 Softmax 激活函数。
    
    公式: S(x_i) = exp(x_i) / sum(exp(x_j))
    为了数值稳定性，通常实现为: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    # PyTorch 官方实现已经包含了数值稳定性处理
    return torch.softmax(x, dim=dim)

import torch
from torch.autograd import Function
from einops import rearrange

class DecoupledAttBlockFn(Function):
    @staticmethod
    def forward(
        ctx,
        h,                  # [b, n, d_model]
        q_fwd, k_fwd, v_fwd, o_fwd,   # [d_model, d_model]
        q_bwd, k_bwd, v_bwd, o_bwd,   # [d_model, d_model]
        q_fwd_copy, k_fwd_copy, v_fwd_copy, o_fwd_copy, # [d_model, d_model]
        alpha,              # scalar
        inv_sqrt_n,         # scalar
        num_heads,          # int
        softmax,            # callable: softmax(x, dim=...)
        softmax_prime=None  # (可不使用；softmax 的反传用标准公式更稳)
    ):
        # 保存必要张量
        ctx.save_for_backward(h, q_fwd, k_fwd, v_fwd, o_fwd, q_bwd, k_bwd, v_bwd, o_bwd, q_fwd_copy, k_fwd_copy, v_fwd_copy, o_fwd_copy)
        ctx.alpha = alpha
        ctx.inv_sqrt_n = inv_sqrt_n
        ctx.num_heads = num_heads
        ctx.softmax = softmax
        ctx.softmax_prime = softmax_prime

        b, n, d_model = h.shape
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads

        # 线性投影
        query = h @ q_fwd.T   # [b, n, d_model]
        key   = h @ k_fwd.T   # [b, n, d_model]
        value = h @ v_fwd.T   # [b, n, d_model]

        queries = inv_sqrt_n * rearrange(query, "b s (h d) -> b h s d", h=num_heads)
        keys = inv_sqrt_n * rearrange(key, "b s (h d) -> b h s d", h=num_heads)
        values = inv_sqrt_n * rearrange(value, "b s (h d) -> b h s d", h=num_heads)

    
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)  # [b,h,n,n]

        #vansih-q,k,v,attention logits,attention scores
        query_ref = h@ q_fwd_copy.T  # [b, n, d_model]
        key_ref = h@ k_fwd_copy.T    # [b, n, d_model]
        value_ref = h@ v_fwd_copy.T  # [b, n, d_model]
        
        queries_ref = inv_sqrt_n * rearrange(query_ref, "b s (h d) -> b h s d", h=num_heads)
        keys_ref = inv_sqrt_n * rearrange(key_ref, "b s (h d) -> b h s d", h=num_heads)
        values_ref = inv_sqrt_n * rearrange(value_ref, "b s (h d) -> b h s d", h=num_heads)
        energy_ref = torch.einsum("bhqd, bhkd -> bhqk", queries_ref, keys_ref)  # [b,h,n,n]

        scaling = 1.0 / head_dim
        att_minus = F.softmax((energy - energy_ref)*scaling, dim=-1)

        attn = softmax(scaling * energy, dim=-1)  # [b,h,n,n]

        out = torch.einsum("bhqk, bhkd -> bhqd", attn, values)  # [b,h,n,d]
        out_merge = rearrange(out, "b h n d -> b n (h d)")        # [b,n,d_model]
        out_proj = out_merge @ o_fwd.T                            # [b,n,d_model]

        y = h + alpha * out_proj
        return y, queries-queries_ref, keys-keys_ref, values-values_ref, scaling*(energy-energy_ref), att_minus

    @staticmethod
    def backward(ctx, grad_y,q_diff=None, k_diff=None, v_diff=None, logits_diff=None, attn_diff=None):
        """
        grad_y: [b, n, d_model]
        """
        h, q_fwd, k_fwd, v_fwd, o_fwd, q_bwd, k_bwd, v_bwd, o_bwd, q_fwd_copy, k_fwd_copy, v_fwd_copy, o_fwd_copy = ctx.saved_tensors
        alpha = ctx.alpha
        inv_sqrt_n = ctx.inv_sqrt_n
        num_heads = ctx.num_heads
        softmax = ctx.softmax
        b, n, d_model = h.shape
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads

        query = h @ q_fwd.T
        key   = h @ k_fwd.T
        value = h @ v_fwd.T

        Q = inv_sqrt_n * rearrange(query, "b s (h d) -> b h s d", h=num_heads)
        K = inv_sqrt_n * rearrange(key, "b s (h d) -> b h s d", h=num_heads)
        V = inv_sqrt_n * rearrange(value, "b s (h d) -> b h s d", h=num_heads)

        energy = torch.einsum("bhqd, bhkd -> bhqk", Q, K)  # [b,h,n,n]
        scaling = 1.0 / head_dim
        logits = scaling * energy
        A = softmax(logits, dim=-1)  # attention, [b,h,n,n]

        out = torch.einsum("bhqk, bhkd -> bhqd", A, V)       # [b,h,n,d]
        out_merge = rearrange(out, "b h n d -> b n (h d)")   # [b,n,d_model]
        # out_proj = out_merge @ o_fwd.T  (不必显式再算)

        # ===== 2) residual：y = h + alpha*out_proj =====
        grad_h = grad_y.clone()                 # skip connection
        grad_out_proj = alpha * grad_y          # [b,n,d_model]

        # ===== 3) out_proj = out_merge @ o_fwd.T =====
        # grad_o_fwd: [d_model, d_model]
        grad_o_fwd = grad_out_proj.reshape(-1, d_model).T @ out_merge.reshape(-1, d_model)

        # Use o_fwd instead of o_bwd for backprop to previous layer
        grad_out_merge = grad_out_proj @ o_fwd                # [b,n,d_model]
        grad_out = rearrange(grad_out_merge, "b n (h d) -> b h n d", h=num_heads)  # [b,h,n,d]

        # ===== 4) out = A @ V =====
        # out[b,h,q,d] = sum_k A[b,h,q,k] * V[b,h,k,d]
        grad_A = torch.einsum("bhqd, bhkd -> bhqk", grad_out, V)       # [b,h,n,n]
        grad_V = torch.einsum("bhqk, bhqd -> bhkd", A, grad_out)       # [b,h,n,d]

        tmp = (grad_A * A).sum(dim=-1, keepdim=True)                 # [b,h,n,1]
        grad_logits = A * (grad_A - tmp)                              # [b,h,n,n]
        grad_energy = scaling * grad_logits                           # [b,h,n,n]

        # ===== 6) energy = Q @ K^T =====
        # energy[b,h,q,k] = sum_d Q[b,h,q,d] * K[b,h,k,d]
        grad_Q = torch.einsum("bhqk, bhkd -> bhqd", grad_energy, K)    # [b,h,n,d]
        grad_K = torch.einsum("bhqk, bhqd -> bhkd", grad_energy, Q)    # [b,h,n,d]

        grad_query = rearrange(inv_sqrt_n * grad_Q, "b h s d -> b s (h d)")
        grad_key   = rearrange(inv_sqrt_n * grad_K, "b h s d -> b s (h d)")
        grad_value = rearrange(inv_sqrt_n * grad_V, "b h s d -> b s (h d)")

        # ===== 9) query = h @ q_fwd.T 等 =====
        h_flat = h.reshape(-1, d_model)
        grad_query_flat = grad_query.reshape(-1, d_model)
        grad_key_flat   = grad_key.reshape(-1, d_model)
        grad_value_flat = grad_value.reshape(-1, d_model)

        # 参数梯度（标准链式法则，使用 forward weights）
        grad_q_fwd = grad_query_flat.T @ h_flat   # [d_model, d_model]
        grad_k_fwd = grad_key_flat.T   @ h_flat
        grad_v_fwd = grad_value_flat.T @ h_flat

        # [CHANGE]: Use q_fwd, k_fwd, v_fwd for backprop to h (Instead of _bwd)
        grad_h_from_q = grad_query @ q_fwd        # [b,n,d_model]
        grad_h_from_k = grad_key   @ k_fwd
        grad_h_from_v = grad_value @ v_fwd
        grad_h = grad_h + grad_h_from_q + grad_h_from_k + grad_h_from_v

        ## ===== 10) 绑定 bwd 权重梯度（保持 Δ 固定） =====
        #grad_q_bwd = grad_q_fwd.clone()
        #grad_k_bwd = grad_k_fwd.clone()
        #grad_v_bwd = grad_v_fwd.clone()
        #grad_o_bwd = grad_o_fwd.clone()

        # 其它非张量/超参不求导
        grad_alpha = None
        grad_inv_sqrt_n = None
        grad_num_heads = None
        grad_softmax = None
        grad_softmax_prime = None
        #-----
        grad_q_fwd_copy = None
        grad_k_fwd_copy = None
        grad_v_fwd_copy = None
        grad_o_fwd_copy = None

        return (
            grad_h,
            grad_q_fwd, grad_k_fwd, grad_v_fwd, grad_o_fwd,
            None, None, None, None,
            None, None, None, None,
            grad_alpha,
            grad_inv_sqrt_n,
            grad_num_heads,
            grad_softmax,
            grad_softmax_prime,
        )

class DecoupledAttention(nn.Module):
    def __init__(self, n, T, L, num_heads,softmax=softmax, softmax_prime=None):
        super().__init__()
        self.n = n #width 
        self.T = T # T
        self.L = L #depth
        self.num_heads = num_heads
        self.softmax = softmax
        self.softmax_prime = None

        # forward weigths
        self.q_fwd = nn.Parameter(torch.randn(n, n))
        self.k_fwd = nn.Parameter(torch.randn(n, n))
        self.v_fwd = nn.Parameter(torch.randn(n, n))
        self.o_fwd = nn.Parameter(torch.randn(n, n))

        # full decoupled
        self.q_bwd = nn.Parameter(torch.randn(n, n))
        self.k_bwd = nn.Parameter(torch.randn(n, n))
        self.v_bwd = nn.Parameter(torch.randn(n, n))
        self.o_bwd = nn.Parameter(torch.randn(n, n))

        # # depth aware lr
        # self.q_bwd = nn.Parameter(self.q_fwd.detach().clone())
        # self.k_bwd = nn.Parameter(self.k_fwd.detach().clone())
        # self.v_bwd = nn.Parameter(self.v_fwd.detach().clone())
        # self.o_bwd = nn.Parameter(self.o_fwd.detach().clone())

        # weight inital
        nn.init.normal_(self.q_fwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.k_fwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.v_fwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.o_fwd, mean=0.0, std=set_variance)

        nn.init.normal_(self.q_bwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.k_bwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.v_bwd, mean=0.0, std=set_variance)
        nn.init.normal_(self.o_bwd, mean=0.0, std=set_variance)

        # weight copy
        self.q_fwd_copy = self.q_fwd.clone().detach().cuda()
        self.k_fwd_copy = self.k_fwd.clone().detach().cuda()
        self.v_fwd_copy = self.v_fwd.clone().detach().cuda()
        self.o_fwd_copy = self.o_fwd.clone().detach().cuda()
        
        self.alpha = math.sqrt(T / (L * n)) #attention output scaling
        self.inv_sqrt_n = 1.0 / math.sqrt(n) # width scaling

    def forward(self, h):
        return DecoupledAttBlockFn.apply(
            h,
            self.q_fwd,
            self.k_fwd,
            self.v_fwd,
            self.o_fwd,
            self.q_bwd,
            self.k_bwd,
            self.v_bwd,
            self.o_bwd,
            self.q_fwd_copy,
            self.k_fwd_copy,
            self.v_fwd_copy,
            self.o_fwd_copy,
            self.alpha,
            self.inv_sqrt_n,
            self.num_heads,
            self.softmax,
            self.softmax_prime,
        )

class DecoupledVitModel(nn.Module):
    def __init__(self, n, L, seqlen, input_dim,num_heads, T):
        super().__init__()
        
        self.embed_dim = n
        self.input_dim = input_dim
        self.L = L

        self.embed = nn.Linear(self.input_dim, self.embed_dim, bias=False)  # 48 input features per patch
        self.cls_token_embed = nn.Parameter(torch.randn(1,1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1+seqlen, self.embed_dim))  #
        self.num_head = num_heads

        self.attention_h = []
        for _ in range(L):
            #n, T, L, num_heads,softmax=softmax, softmax_prime=None
            self.attention_h.append(DecoupledAttention(n, T, L, num_heads, softmax, None))
        self.mlp_h = []#nn.ModuleList
        for _ in range(L):
            # n, T, L, phi=relu, phi_prime=relu_prime
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
        # 假设每个patch之间是independent的
        B, T, _ = x.shape
        x = self.embed(x)/math.sqrt(self.input_dim)

        cls_tokens = repeat(self.cls_token_embed, '() n e -> b n e', b=B)#/math.sqrt(48)
        x = torch.cat([cls_tokens, x], dim=1) #prepending the cls token
        h = math.sqrt(1/2)*(x + self.pos_embed)#/math.sqrt(65)
        # mid embedding
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
        #h = self.class_head(h[:,0,:]) # only use cls token
        classfiy_h = (1/self.embed_dim)*self.class_head(h[:,0,:])
        return classfiy_h,mid_layers

import math

def calculate_vit_sequence_length(total_dim=3072, channels=3, patch_size=4):
    """
    计算 Vision Transformer (ViT) 的序列长度 (Number of Patches)。
    
    参数:
        total_dim (int): 图像的总维度 (H * W * C)。默认为 3072 (32x32x3)。
        channels (int): 图像的通道数。默认为 3。
        patch_size (int): Patch 的边长 x。
        
    返回:
        int: 序列长度 (Patch 的数量)。
    """
    
    # 1. 计算图像的总像素数 (H * W)
    if total_dim % channels != 0:
        raise ValueError(f"总维度 {total_dim} 无法被通道数 {channels} 整除。")
    
    total_pixels = total_dim // channels
    
    # 2. 计算图像的边长 (假设图像是正方形 H=W)
    # math.isqrt 计算整数平方根
    image_side = math.isqrt(total_pixels)
    
    # 验证是否真的是正方形
    if image_side * image_side != total_pixels:
        raise ValueError(f"计算出的像素总数 {total_pixels} 不是一个完全平方数，无法构成正方形图像。")
    
    #print(f"推断图像尺寸: {image_side} x {image_side} x {channels}")
    
    # 3. 检查 Patch 大小是否合法
    if image_side % patch_size != 0:
        raise ValueError(f"图像边长 {image_side} 无法被 Patch 大小 {patch_size} 整除。")
    
    # 4. 计算序列长度 (Number of Patches)
    # 序列长度 = (H / P) * (W / P)
    num_patches_h = image_side // patch_size
    num_patches_w = image_side // patch_size
    sequence_length = num_patches_h * num_patches_w
    
    return sequence_length

def set_optimizer_with_different_lr_decoupled(model,width,depth, base_lr=0.0):
    """
    forward: NsqrtL-Q,K,V,U -- N-O,W,
    backward: N-Q,K,V,U -- NsqrtL-O,W
    """
    params = []
    for name, param in model.named_parameters():
        if "q_fwd" in name or "k_fwd" in name or "v_fwd" in name or "U_fwd" in name:
            params.append({'params': param, 'lr': base_lr*width*math.sqrt(depth)})
        elif "o_bwd" in name or "W_bwd" in name:
            params.append({'params': param, 'lr': base_lr*width*math.sqrt(depth)})
        else:
            params.append({'params': param, 'lr': base_lr*width})
    optimizer = optim.SGD(params)
    return optimizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# Main Training Logic (Modified for Loop)
# -----------------------------------------------------------------------------

# Output path
my_path = "/home/ztang13/workspace/SparseAttention/CLS/data" 
if not os.path.exists(my_path):
    os.makedirs(my_path)

print("=================================================================================")
print(f"STARTING GRID SEARCH | Depths: {depths} | LRs: {base_lrs}")
print("=================================================================================")

# --- Outer Loop: DEPTH ---
for depth in depths:
    # Update task name for current depth
    task_name = "depth-decoupled-var01ep30-depth:{}-width:{}-seed:{}".format(depth, width, seed)
    
    # Re-initialize results for current depth
    results = {
        "task_info": {
            "width": width,
            "depth": depth,
            "T": T,
            "seed": seed,
            "epochs": epochs,
            "base_lrs": base_lrs
        },
        "training_results": {}
    }
    
    print(f"\n[Processing Depth {depth}] Saving to: {task_name}.json")

    # --- Inner Loop: LEARNING RATE ---
    for base_lr in base_lrs:
        set_seed(seed)

        seqlen = int(calculate_vit_sequence_length(total_dim=3072, channels=3, patch_size=patch_size))
        input_dim = int(3072/seqlen)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model with current DEPTH
        model = DecoupledVitModel(L=depth,n=width,seqlen=seqlen,input_dim=input_dim,num_heads=num_head,T=T).to(device)
        
        # Optimizer with scaling based on current DEPTH
        optimizer = set_optimizer_with_different_lr_decoupled(model,width,depth, base_lr=base_lr)

        criterion = nn.CrossEntropyLoss()
        
        # Initialize storage for current LR
        results["training_results"][f"lr_{base_lr}"] = {
            "epochs": [],
            "train_loss": [],
            "test_loss": [],
            "valid_loss": [],           # <-- add
            "train_accuracy": [],
            "test_accuracy": [],
            "valid_accuracy": [],       # <-- add
            "train_epoch_loss": []
        }
        
        all_train_loss = []
        
        # --- NEW: Tracking variables for Best metrics ---
        best_test_acc = 0.0
        best_test_loss = 0.0
        best_train_acc = 0.0
        best_train_loss = 0.0
        best_epoch = -1
        
        print(f"   -> LR: {base_lr} Started")
        
        for epoch in range(epochs):
            # 1. Train
            model.train()
            for batch_idx, (data, target) in enumerate(tqdm(trainloader, desc=f"D{depth}|LR{base_lr}|Ep{epoch+1} Train", leave=False)):
                data, target = data.to(device), target.to(device)
                data = img_to_patch_pt(data, patch_size=patch_size, flatten_channels=True)
                optimizer.zero_grad()
                output, mid_layers = model(data)
                loss = criterion(output, target)
                loss.backward()
                all_train_loss.append(loss.item())
                optimizer.step()
            
            # 2. Evaluation
            model.eval()
            all_train_loss_after_epoch = []
            all_test_loss = []
            all_acc = []
            all_train_accuracy = []
            
            # Test Loop
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(testloader, desc="Testing", leave=False)):
                    data, target = data.to(device), target.to(device)
                    data = img_to_patch_pt(data, patch_size=patch_size, flatten_channels=True)
                    output, mid_layers = model(data)
                    loss = criterion(output, target)
                    acc = (output.argmax(dim=1) == target).float().mean()
                    all_test_loss.append(loss.item())
                    all_acc.append(acc.item())
            
            # Train Eval Loop
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
            
            # Valid Loop
            all_valid_loss = []
            all_valid_accuracy = []
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(validloader, desc="Validation", leave=False)):
                    data, target = data.to(device), target.to(device)
                    data = img_to_patch_pt(data, patch_size=patch_size, flatten_channels=True)
                    output, mid_layers = model(data)
                    loss = criterion(output, target)
                    all_valid_loss.append(loss.item())
                    acc = (output.argmax(dim=1) == target).float().mean()
                    all_valid_accuracy.append(acc.item())

            # Metrics for current epoch
            curr_train_loss = float(np.mean(all_train_loss_after_epoch))
            curr_train_acc = float(np.mean(all_train_accuracy))
            curr_test_loss = float(np.mean(all_test_loss))
            curr_test_acc = float(np.mean(all_acc))
            
            # --- NEW: Update BEST metrics based on test accuracy ---
            if curr_test_acc > best_test_acc:
                best_test_acc = curr_test_acc
                best_test_loss = curr_test_loss
                best_train_acc = curr_train_acc
                best_train_loss = curr_train_loss
                best_epoch = epoch + 1

            # Save current epoch results
            current_results = results["training_results"][f"lr_{base_lr}"]
            current_results["epochs"].append(epoch + 1)
            current_results["train_loss"].append(float(np.mean(all_train_loss)))
            current_results["test_loss"].append(curr_test_loss)
            current_results["valid_loss"].append(float(np.mean(all_valid_loss)))
            current_results["train_accuracy"].append(curr_train_acc)
            current_results["test_accuracy"].append(curr_test_acc)
            current_results["valid_accuracy"].append(float(np.mean(all_valid_accuracy)))
            current_results["train_epoch_loss"].append(curr_train_loss)

            # Print per-epoch stats
            print(f"LR: {base_lr}, Epoch: {epoch+1}, Train Loss: {curr_train_loss:.4f}, Test Loss: {curr_test_loss:.4f}, Train Acc: {curr_train_acc:.4f}, Test Acc: {curr_test_acc:.4f}")
            
            # Save JSON after every epoch
            with open(f"{my_path}/{task_name}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            model.train()

        # --- NEW: Final Summary Print after all epochs for this LR ---
        print(f"   -> LR {base_lr} Done. Best Result: Test Acc {best_test_acc:.4f}, Test Loss {best_test_loss:.4f}, Train Acc {best_train_acc:.4f}, Train Loss {best_train_loss:.4f} (Ep {best_epoch})")
        
        # Optional: Save Best metrics into the summary part of results
        results["summary"] = results.get("summary", {})
        results["summary"][f"lr_{base_lr}"] = {
            "best_epoch": best_epoch,
            "test_acc": best_test_acc,
            "test_loss": best_test_loss,
            "train_acc": best_train_acc,
            "train_loss": best_train_loss
        }

print("\nAll experiments completed.")