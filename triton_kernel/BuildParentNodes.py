import os
# ==========================================
# CRITICAL FIX: GPU SELECTION MUST BE FIRST
# ==========================================
# Set this before importing torch or calling torch.cuda to avoid OOM
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import torch
import triton
import triton.language as tl
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


# ------------------------------------------------------------------
#  Forward Kernel (Context Saving + Adaptive Block Size)
# ------------------------------------------------------------------
@triton.jit
def build_parent_nodes_forward_kernel(
    # Pointers
    Q_ptr, Kp_ptr, Vp_ptr, Kc_ptr, Vc_ptr,
    Out_ptr, W_ptr,
    
    # Strides
    sq_b, sq_n, sq_h, sq_d,
    skp_b, skp_n, skp_h, skp_d,
    skc_b, skc_n, skc_h, skc_d,
    so_b, so_n, so_h, so_d,
    sw_b, sw_n, sw_h, sw_3,
    
    # Constants
    sm_scale,
    H: tl.constexpr, 
    BLOCK_H: tl.constexpr, 
    D: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr  # <--- Now represents the Tile Size, not full D
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # Pointers
    q_base_ptr = Q_ptr + (b_idx * sq_b) + (node_idx * sq_n)
    kp_base_ptr = Kp_ptr + (b_idx * skp_b) + (node_idx * skp_n)
    
    child0_idx = 2 * node_idx
    child1_idx = 2 * node_idx + 1
    kc0_base_ptr = Kc_ptr + (b_idx * skc_b) + (child0_idx * skc_n)
    kc1_base_ptr = Kc_ptr + (b_idx * skc_b) + (child1_idx * skc_n)
    
    # Accumulators
    score_self = tl.zeros([BLOCK_H], dtype=tl.float32)
    score_c0   = tl.zeros([BLOCK_H], dtype=tl.float32)
    score_c1   = tl.zeros([BLOCK_H], dtype=tl.float32)

    # --- Loop over D (Tiling) ---
    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        mask_load = mask_h[:, None] & mask_d[None, :]
        
        q = tl.load(q_base_ptr + (offs_h[:, None] * sq_h) + (offs_d[None, :] * sq_d), mask=mask_load, other=0.0)
        kp = tl.load(kp_base_ptr + (offs_h[:, None] * skp_h) + (offs_d[None, :] * skp_d), mask=mask_load, other=0.0)
        kc0 = tl.load(kc0_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d), mask=mask_load, other=0.0)
        kc1 = tl.load(kc1_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d), mask=mask_load, other=0.0)
        
        score_self += tl.sum(q * kp, axis=1)
        score_c0   += tl.sum(q * kc0, axis=1)
        score_c1   += tl.sum(q * kc1, axis=1)

    # Softmax
    score_self = score_self * sm_scale
    score_c0   = score_c0 * sm_scale
    score_c1   = score_c1 * sm_scale

    max_score = tl.maximum(score_self, tl.maximum(score_c0, score_c1))
    exp_self = tl.exp(score_self - max_score)
    exp_c0   = tl.exp(score_c0 - max_score)
    exp_c1   = tl.exp(score_c1 - max_score)
    denom = exp_self + exp_c0 + exp_c1 + 1e-9
    
    w_self = exp_self / denom
    w_c0   = exp_c0 / denom
    w_c1   = exp_c1 / denom

    # Save Weights
    w_base = W_ptr + (b_idx * sw_b) + (node_idx * sw_n) + (offs_h * sw_h)
    tl.store(w_base + 0 * sw_3, w_self, mask=mask_h)
    tl.store(w_base + 1 * sw_3, w_c0,   mask=mask_h)
    tl.store(w_base + 2 * sw_3, w_c1,   mask=mask_h)

    # Weighted Sum
    out_base_ptr = Out_ptr + (b_idx * so_b) + (node_idx * so_n)
    vp_base_ptr = Vp_ptr + (b_idx * skp_b) + (node_idx * skp_n)
    vc0_base_ptr = Vc_ptr + (b_idx * skc_b) + (child0_idx * skc_n)
    vc1_base_ptr = Vc_ptr + (b_idx * skc_b) + (child1_idx * skc_n)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]
        
        vp  = tl.load(vp_base_ptr  + (offs_h[:, None] * skp_h) + (offs_d[None, :] * skp_d), mask=mask_op, other=0.0)
        vc0 = tl.load(vc0_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d), mask=mask_op, other=0.0)
        vc1 = tl.load(vc1_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d), mask=mask_op, other=0.0)
        
        out_val = (w_self[:, None] * vp) + (w_c0[:, None] * vc0) + (w_c1[:, None] * vc1)
        tl.store(out_base_ptr + (offs_h[:, None] * so_h) + (offs_d[None, :] * so_d), out_val, mask=mask_op)


# ------------------------------------------------------------------
#  Backward Kernel (Optimized + Safe Tiling)
# ------------------------------------------------------------------
@triton.jit
def build_parent_nodes_backward_kernel(
    DO_ptr, W_ptr,
    Q_ptr, Kp_ptr, Vp_ptr, Kc_ptr, Vc_ptr,
    DQ_ptr, DKp_ptr, DVp_ptr, DKc_ptr, DVc_ptr,
    sq_b, sq_n, sq_h, sq_d,
    skp_b, skp_n, skp_h, skp_d,
    skc_b, skc_n, skc_h, skc_d,
    sw_b, sw_n, sw_h, sw_3,
    sdo_b, sdo_n, sdo_h, sdo_d,
    sdq_b, sdq_n, sdq_h, sdq_d,
    sdkp_b, sdkp_n, sdkp_h, sdkp_d,
    sdkc_b, sdkc_n, sdkc_h, sdkc_d,
    sm_scale,
    H: tl.constexpr, BLOCK_H: tl.constexpr, 
    D: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    
    # 1. Load Weights
    w_base = W_ptr + (b_idx * sw_b) + (node_idx * sw_n) + (offs_h * sw_h)
    w_self = tl.load(w_base + 0 * sw_3, mask=mask_h, other=0.0)
    w_c0   = tl.load(w_base + 1 * sw_3, mask=mask_h, other=0.0)
    w_c1   = tl.load(w_base + 2 * sw_3, mask=mask_h, other=0.0)

    # 2. Compute dV and dP
    do_ptr_base = DO_ptr + (b_idx * sdo_b) + (node_idx * sdo_n)
    vp_ptr_base = Vp_ptr + (b_idx * skp_b) + (node_idx * skp_n)
    
    c0_idx = 2 * node_idx; c1_idx = 2 * node_idx + 1
    vc0_ptr_base = Vc_ptr + (b_idx * skc_b) + (c0_idx * skc_n)
    vc1_ptr_base = Vc_ptr + (b_idx * skc_b) + (c1_idx * skc_n)
    
    dvp_ptr_base = DVp_ptr + (b_idx * sdkp_b) + (node_idx * sdkp_n)
    dvc0_ptr_base = DVc_ptr + (b_idx * sdkc_b) + (c0_idx * sdkc_n)
    dvc1_ptr_base = DVc_ptr + (b_idx * sdkc_b) + (c1_idx * sdkc_n)

    dp_self = tl.zeros([BLOCK_H], dtype=tl.float32)
    dp_c0   = tl.zeros([BLOCK_H], dtype=tl.float32)
    dp_c1   = tl.zeros([BLOCK_H], dtype=tl.float32)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        mask = mask_h[:, None] & mask_d[None, :]
        
        do = tl.load(do_ptr_base + (offs_h[:, None]*sdo_h) + (offs_d[None, :]*sdo_d), mask=mask, other=0.0)
        vp = tl.load(vp_ptr_base + (offs_h[:, None]*skp_h) + (offs_d[None, :]*skp_d), mask=mask, other=0.0)
        vc0 = tl.load(vc0_ptr_base + (offs_h[:, None]*skc_h) + (offs_d[None, :]*skc_d), mask=mask, other=0.0)
        vc1 = tl.load(vc1_ptr_base + (offs_h[:, None]*skc_h) + (offs_d[None, :]*skc_d), mask=mask, other=0.0)
        
        tl.store(dvp_ptr_base + (offs_h[:, None]*sdkp_h) + (offs_d[None, :]*sdkp_d), do * w_self[:, None], mask=mask)
        tl.store(dvc0_ptr_base + (offs_h[:, None]*sdkc_h) + (offs_d[None, :]*sdkc_d), do * w_c0[:, None], mask=mask)
        tl.store(dvc1_ptr_base + (offs_h[:, None]*sdkc_h) + (offs_d[None, :]*sdkc_d), do * w_c1[:, None], mask=mask)

        dp_self += tl.sum(vp * do, axis=1)
        dp_c0   += tl.sum(vc0 * do, axis=1)
        dp_c1   += tl.sum(vc1 * do, axis=1)

    # 3. Compute dS
    sum_w_dp = (w_self * dp_self) + (w_c0 * dp_c0) + (w_c1 * dp_c1)
    ds_self = w_self * (dp_self - sum_w_dp) * sm_scale
    ds_c0   = w_c0   * (dp_c0   - sum_w_dp) * sm_scale
    ds_c1   = w_c1   * (dp_c1   - sum_w_dp) * sm_scale

    # 4. Compute dQ and dK
    off_p_base = (b_idx * sq_b) + (node_idx * sq_n)
    q_ptr_base  = Q_ptr + off_p_base
    kp_ptr_base = Kp_ptr + (b_idx * skp_b) + (node_idx * skp_n)
    
    off_c0 = (b_idx * skc_b) + (c0_idx * skc_n)
    off_c1 = (b_idx * skc_b) + (c1_idx * skc_n)
    kc0_ptr_base = Kc_ptr + off_c0
    kc1_ptr_base = Kc_ptr + off_c1
    
    dq_ptr_base = DQ_ptr + (b_idx * sdq_b) + (node_idx * sdq_n)
    dkp_ptr_base = DKp_ptr + (b_idx * sdkp_b) + (node_idx * sdkp_n)
    dkc0_ptr_base = DKc_ptr + (b_idx * sdkc_b) + (c0_idx * sdkc_n)
    dkc1_ptr_base = DKc_ptr + (b_idx * sdkc_b) + (c1_idx * sdkc_n)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        mask = mask_h[:, None] & mask_d[None, :]

        q = tl.load(q_ptr_base + (offs_h[:, None]*sq_h) + (offs_d[None, :]*sq_d), mask=mask, other=0.0)
        kp = tl.load(kp_ptr_base + (offs_h[:, None]*skp_h) + (offs_d[None, :]*skp_d), mask=mask, other=0.0)
        kc0 = tl.load(kc0_ptr_base + (offs_h[:, None]*skc_h) + (offs_d[None, :]*skc_d), mask=mask, other=0.0)
        kc1 = tl.load(kc1_ptr_base + (offs_h[:, None]*skc_h) + (offs_d[None, :]*skc_d), mask=mask, other=0.0)

        dq_val = (ds_self[:, None] * kp) + (ds_c0[:, None] * kc0) + (ds_c1[:, None] * kc1)
        tl.store(dq_ptr_base + (offs_h[:, None]*sdq_h) + (offs_d[None, :]*sdq_d), dq_val, mask=mask)
        
        tl.store(dkp_ptr_base + (offs_h[:, None]*sdkp_h) + (offs_d[None, :]*sdkp_d), ds_self[:, None] * q, mask=mask)
        tl.store(dkc0_ptr_base + (offs_h[:, None]*sdkc_h) + (offs_d[None, :]*sdkc_d), ds_c0[:, None] * q, mask=mask)
        tl.store(dkc1_ptr_base + (offs_h[:, None]*sdkc_h) + (offs_d[None, :]*sdkc_d), ds_c1[:, None] * q, mask=mask)


# ------------------------------------------------------------------
#  Wrapper Class (Updates BLOCK_SIZE logic)
# ------------------------------------------------------------------
class BuildParentNodesFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q_p, K_p, V_p, K_c, V_c):
        Q_p = Q_p.contiguous(); K_p = K_p.contiguous(); V_p = V_p.contiguous()
        K_c = K_c.contiguous(); V_c = V_c.contiguous()
        
        B, P, H, D = Q_p.shape
        assert K_c.shape[1] == 2 * P, "Child count mismatch"
        
        Out = torch.empty_like(Q_p)
        Weights = torch.empty((B, P, H, 3), device=Q_p.device, dtype=torch.float32)
        
        grid = (P, B)
        BLOCK_H = triton.next_power_of_2(H)
        
        # [OPTIMIZATION] Cap BLOCK_SIZE to avoid register spilling for large D
        # 128 is a safe sweet spot for shared memory usage.
        BLOCK_SIZE = min(128, triton.next_power_of_2(D))
        sm_scale = 1.0 / math.sqrt(D)

        build_parent_nodes_forward_kernel[grid](
            Q_p, K_p, V_p, K_c, V_c, Out, Weights,
            *Q_p.stride(), *K_p.stride(), *K_c.stride(),
            *Out.stride(), *Weights.stride(),
            sm_scale, H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        ctx.save_for_backward(Q_p, K_p, V_p, K_c, V_c, Weights)
        ctx.constants = (sm_scale, H, BLOCK_H, D, BLOCK_SIZE)
        return Out

    @staticmethod
    def backward(ctx, grad_output):
        Q_p, K_p, V_p, K_c, V_c, Weights = ctx.saved_tensors
        sm_scale, H, BLOCK_H, D, BLOCK_SIZE = ctx.constants
        
        grad_output = grad_output.contiguous()
        dQ = torch.empty_like(Q_p); dKp = torch.empty_like(K_p); dVp = torch.empty_like(V_p)
        dKc = torch.empty_like(K_c); dVc = torch.empty_like(V_c)
        
        grid = (P, B) = (Q_p.shape[1], Q_p.shape[0])
        
        build_parent_nodes_backward_kernel[grid](
            grad_output, Weights,
            Q_p, K_p, V_p, K_c, V_c,
            dQ, dKp, dVp, dKc, dVc,
            *Q_p.stride(), *K_p.stride(), *K_c.stride(),
            *Weights.stride(),
            *grad_output.stride(), *dQ.stride(), *dKp.stride(), *dKc.stride(),
            sm_scale, H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        return dQ, dKp, dVp, dKc, dVc

def build_parent_nodes(Q_p, K_p, V_p, K_c, V_c):
    return BuildParentNodesFunc.apply(Q_p, K_p, V_p, K_c, V_c)


def build_hierarchical_index_lookup_table(seq_len, device="cuda", dtype=torch.int32):
    """
    Vectorized version: Builds index table without Python loops over seq_len.
    Drastically faster for large sequences.

    Build a hierarchical index lookup table storing only neighbor nodes
    across different levels (excluding the leaf itself).

    The relationship follows:
        level 0: n ^ 1
        level 1: (n // 2 + seq_len) ^ 1
        level 2: (n1 // 2 + seq_len) ^ 1
        ...
    Stops when the next index exceeds total_nodes - 2.

    Args:
        seq_len (int): number of leaf tokens (must be a power of 2)
        device (str): 'cuda' or 'cpu'
        dtype (torch.dtype): index dtype (default int32)

    Returns:
        idx_map: [seq_len, level_num] int tensor
    """
    assert (seq_len & (seq_len - 1)) == 0, "seq_len must be a power of 2"

    total_nodes = 2 * seq_len - 1
    max_valid = total_nodes - 2
    level_num = int(math.log2(seq_len))

    # Initialize Tensors
    causal_mask = torch.zeros((seq_len, level_num), dtype=torch.bool, device=device)
    idx_map = torch.full((seq_len, level_num), -1, dtype=dtype, device=device)

    # Start with the leaf nodes [0, 1, 2, ..., seq_len-1]
    n_cur = torch.arange(seq_len, device=device, dtype=torch.int64)

    for lvl in range(level_num):
        # Vectorized Logic
        if lvl == 0:
            n_next = n_cur ^ 1
            pair = n_cur
        else:
            # Vectorized formula
            n_next = (n_cur // 2 + seq_len) ^ 1
            pair = (n_cur // 2 + seq_len)

        # Boundary Check (Vectorized)
        # We use a mask to prevent writing invalid indices, 
        # effectively mimicking the 'break' in the loop
        valid_mask = n_next <= max_valid
        
        # Causal Masking Logic (Vectorized)
        # pair < n_next means neighbor is in the "future"
        mask_step = (pair < n_next) & valid_mask

        # Update Tables (Batch assignment)
        # We only update where valid_mask is True
        idx_map[:, lvl] = torch.where(valid_mask, n_next.to(dtype), idx_map[:, lvl])
        causal_mask[:, lvl] = torch.where(valid_mask, mask_step, causal_mask[:, lvl])

        # Climb up via the neighbor for the next iteration
        n_cur = n_next

    return idx_map, causal_mask

class HierarchicalSparseAttentionTriton(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # --- Y Updates (Bottom-Up) ---
        self.Wq_y = nn.Linear(dim, dim, bias=False)
        self.Wk_y = nn.Linear(dim, dim, bias=False)
        self.Wv_y = nn.Linear(dim, dim, bias=False)
        self.out_proj_y = nn.Linear(dim, dim)

        # --- X Updates (Top-Down) ---
        self.Wq_x = nn.Linear(dim, dim, bias=False)
        self.Wk_x = nn.Linear(dim, dim, bias=False)
        self.Wv_x = nn.Linear(dim, dim, bias=False)
        self.out_proj_x = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        
        self.cached_idx_table = None
        self.cached_causal_mask = None
        self.cached_seq_len = -1

        self.sizes = None
        self.offsets = None

    def _get_lookup_table(self, L, device):
        """
        Smart retrieval: Returns cached table if L matches, otherwise recomputes.
        """
        # Check if we can reuse the cache
        # 1. Cache exists
        # 2. Sequence length matches
        # 3. Device matches (crucial for moving model CPU <-> GPU)
        if (self.cached_idx_table is not None and 
            self.cached_seq_len == L and 
            self.cached_idx_table.device == device):
            return self.cached_idx_table, self.cached_causal_mask

        # If not, recompute and update cache
        idx_table, mask = build_hierarchical_index_lookup_table(L, device=device, dtype=torch.int64)
        
        self.cached_idx_table = idx_table
        self.cached_causal_mask = mask
        self.cached_seq_len = L
        
        return idx_table, mask

    @staticmethod
    def generate_span_input_Y(x):
        """Initializes Y by averaging adjacent pairs of X."""
        B, N, D = x.shape
        Y_levels = []
        curr = x
        while curr.size(1) > 1:
            L = curr.size(1)
            even = L - (L % 2)
            curr_pairs = curr[:, :even, :].reshape(B, even // 2, 2, D)
            parents = 0.5 * curr_pairs[:, :, 0, :] + 0.5 * curr_pairs[:, :, 1, :]
            Y_levels.append(parents)
            curr = parents
        
        if not Y_levels:
            return None
            
        return torch.cat(Y_levels, dim=1)

    @staticmethod
    def build_level_info(N):
        sizes = []
        curr = N
        while curr > 1:
            sizes.append(curr // 2)
            curr = curr // 2
        offsets = [0]
        for s in sizes[:-1]:
            offsets.append(offsets[-1] + s)
        return sizes, offsets


    def cross_update_Y_Ref(self, x, y_in):
        """
        Architecture: Recursive Parent-Self Attention.
        
        Refactored Layout: (Batch, Heads, Sequence, Head_Dim) aka (B, H, N, Dh).
        This matches standard PyTorch MultiheadAttention implementation details.
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        assert y_in is not None
        assert y_in.size(1) == N - 1, f"y_in size {y_in.size(1)} mismatch! Expected N-1 ({N-1}) for N={N} leaves."

        if self.sizes is None: self.sizes, self.offsets = self.build_level_info(N)

        # -------------------------------------------------------
        # OPTIMIZATION: Global Parent Projection (Standard Layout)
        # -------------------------------------------------------
        # Input: (B, Total_Parents, D)
        # 1. Project -> (B, Total_Parents, H, Dh)
        # 2. Transpose -> (B, H, Total_Parents, Dh)
        Q_p_all = self.Wq_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        K_p_all = self.Wk_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        V_p_all = self.Wv_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        
        new_Y_levels = []
        prev_sources = x 

        for level, parent_count in enumerate(self.sizes):
            # ---------------------------------------------------
            # 1. Prepare Children
            # ---------------------------------------------------
            useful_len = parent_count * 2
            children = prev_sources[:, :useful_len, :] 
            
            # Project Children -> (B, H, 2*P, Dh)
            K_c = self.Wk_y(children).view(B, -1, H, Dh).transpose(1, 2)
            V_c = self.Wv_y(children).view(B, -1, H, Dh).transpose(1, 2)
            V_c = self.dropout(V_c)

            # ---------------------------------------------------
            # 2. Reshape Children to Pairs
            # ---------------------------------------------------
            # Current: (B, H, 2*P, Dh)
            # Target:  (B, H, P, 2, Dh)  <-- Group pairs together
            K_c_pairs = K_c.view(B, H, parent_count, 2, Dh)
            V_c_pairs = V_c.view(B, H, parent_count, 2, Dh)

            # ---------------------------------------------------
            # 3. Slice Parents
            # ---------------------------------------------------
            # Current: (B, H, Total_Parents, Dh)
            # Target:  (B, H, P, Dh)
            offset = self.offsets[level]
            Q_p = Q_p_all[:, :, offset : offset + parent_count, :]
            K_p = K_p_all[:, :, offset : offset + parent_count, :]
            V_p = V_p_all[:, :, offset : offset + parent_count, :]

            # ---------------------------------------------------
            # 4. Form Attention Pool
            # ---------------------------------------------------
            # Parent (Self) needs to be unsqueezed to match Child pairs structure
            # K_p: (B, H, P, Dh) -> (B, H, P, 1, Dh)
            # Pool: Concat along the 'pair' dim -> (B, H, P, 3, Dh)
            # Pool Order: [Parent(Self), Child_Left, Child_Right]
            K_pool = torch.cat([K_p.unsqueeze(3), K_c_pairs], dim=3)
            V_pool = torch.cat([V_p.unsqueeze(3), V_c_pairs], dim=3)

            # ---------------------------------------------------
            # 5. Attention Scores
            # ---------------------------------------------------
            # Q: (B, H, P, Dh) -> (B, H, P, 1, Dh)
            # K_pool: (B, H, P, 3, Dh) -> Transpose to (B, H, P, Dh, 3)
            
            # Matmul: (B,H,P,1,Dh) @ (B,H,P,Dh,3) -> (B, H, P, 1, 3)
            # This computes 3 scores per parent: Self, Left, Right
            logits = torch.matmul(Q_p.unsqueeze(3), K_pool.transpose(-1, -2))
            logits = logits / math.sqrt(Dh)
            
            weights = F.softmax(logits, dim=-1) # (B, H, P, 1, 3)

            # ---------------------------------------------------
            # 6. Weighted Sum
            # ---------------------------------------------------
            # V_pool: (B, H, P, 3, Dh)
            # Matmul: (B,H,P,1,3) @ (B,H,P,3,Dh) -> (B, H, P, 1, Dh)
            attn_out = torch.matmul(weights, V_pool)
            
            # ---------------------------------------------------
            # 7. Merge Heads
            # ---------------------------------------------------
            # 1. Squeeze the '1' dim: (B, H, P, Dh)
            # 2. Transpose back to Sequence First: (B, P, H, Dh)
            # 3. Flatten/Reshape: (B, P, D)
            attn_out = attn_out.squeeze(3).transpose(1, 2).contiguous().reshape(B, parent_count, D)
            
            new_Y_levels.append(attn_out)
            prev_sources = attn_out

        return self.out_proj_y(torch.cat(new_Y_levels, dim=1))

    def cross_update_Y(self, x, y_in):
        """
        Optimized bottom-up update using Triton Kernel with Parent-Self Attention.
        
        Logic:
          - Pool = [Parent (Self), Child_Left, Child_Right]
          - Q = Parent
          - K, V = Pool
          
        Key Optimization:
          - Global projection of all Parents (Q, K, V) at once.
          - Single fused kernel handles the 3-way attention and reduction.
          - Maintains [Batch, Node, Head, Dim] layout to avoid transposes.
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        
        assert y_in is not None, "y_in cannot be None"
        assert y_in.size(1) == N - 1, f"y_in size {y_in.size(1)} mismatch! Expected N-1 ({N-1}) for N={N} leaves."

        if self.sizes is None: self.sizes, self.offsets = self.build_level_info(N)

        # -------------------------------------------------------
        # OPTIMIZATION 1: Global Parent Projection
        # -------------------------------------------------------
        # We project ALL parents at once for Q, K, and V.
        # This allows the parent to attend to itself (Parent-Self Attention).
        # Layout: [B, Total_Parents, H, Dh] (Native Linear output, no transpose)
        Q_p_all = self.Wq_y(y_in).view(B, -1, H, Dh)
        K_p_all = self.Wk_y(y_in).view(B, -1, H, Dh)
        V_p_all = self.Wv_y(y_in).view(B, -1, H, Dh)
        
        new_Y_levels = []
        prev_sources = x # Starts as the leaves (first layer children)
        
        for level, parent_count in enumerate(self.sizes):
            offset = self.offsets[level]

            # ---------------------------------------------------
            # 1. Prepare Parents (Slicing)
            # ---------------------------------------------------
            # Slice Q, K, V for this specific level's parents
            # Shape: [B, Parent_Count, H, Dh]
            Q_p = Q_p_all[:, offset : offset + parent_count, :, :]
            K_p = K_p_all[:, offset : offset + parent_count, :, :]
            V_p = V_p_all[:, offset : offset + parent_count, :, :]
            
            # ---------------------------------------------------
            # 2. Prepare Children (Projection)
            # ---------------------------------------------------
            useful_len = parent_count * 2
            # Slice strictly to useful length to ensure even pairs
            children_in = prev_sources[:, :useful_len, :]
            
            # Project Children Keys/Values (Must be done in loop as prev_sources changes)
            # Shape: [B, 2*P, H, Dh]
            K_c = self.Wk_y(children_in).view(B, -1, H, Dh)
            V_c = self.Wv_y(children_in).view(B, -1, H, Dh)
            V_c = self.dropout(V_c)

            # ---------------------------------------------------
            # 3. Triton Kernel (3-Way Attention)
            # ---------------------------------------------------
            # Replaces: Reshape -> Cat -> Dot -> Softmax -> Weighted Sum
            # Input:  Parents (Q,K,V) and Children (K,V)
            # Output: Updated Parents [B, P, H, Dh]
            updated_heads = build_parent_nodes(Q_p, K_p, V_p, K_c, V_c)

            # ---------------------------------------------------
            # 4. Merge Heads
            # ---------------------------------------------------
            # [B, P, H, Dh] -> [B, P, D]
            updated_merged = updated_heads.reshape(B, parent_count, D)
            
            new_Y_levels.append(updated_merged)
            prev_sources = updated_merged

        # Concatenate and Project
        Y_new = torch.cat(new_Y_levels, dim=1)
        Y_new = self.out_proj_y(Y_new)
        
        return Y_new

    def update_X_from_Y(self, x, y, mask=None):
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # [Check] Enforce strict existence
        assert y is not None, "y (parents) cannot be None during top-down update"

        # 1. Combine Input (Leaves + Parents)
        XY = torch.cat([x, y], dim=1)
        
        # 2. Separate Projections (No Fused Wkv)
        # Q: [B, N, H, Dh]
        Q = self.Wq_x(x).view(B, N, H, Dh)
        
        # K, V: [B, Total_Nodes, H, Dh]
        # We project K and V separately using your existing layers
        K_full = self.Wk_x(XY).view(B, -1, H, Dh)
        V_full = self.Wv_x(XY).view(B, -1, H, Dh)
        
        # Apply dropout to V
        V_full = self.dropout(V_full)

        # 3. Ensure Contiguity
        # The Triton kernel does pointer arithmetic. While Linear() outputs are usually 
        # contiguous, operations like Dropout or View can sometimes create strides 
        # that break optimized kernels. We enforce it here to be safe.
        if not K_full.is_contiguous(): K_full = K_full.contiguous()
        if not V_full.is_contiguous(): V_full = V_full.contiguous()

        # Get Topology
        idx_table, neighbor_mask = self._get_lookup_table(N, device=x.device)

        # DECISION LOGIC:
        # If the user passed a mask (indicating causality is on), use neighbor_mask.
        # If mask is None (indicating full/bidirectional), pass None.
        active_mask = neighbor_mask if mask is not None else None

        output_leaf_heads = hierarchical_fused_attention(
            Q, K_full, V_full, 
            idx_table, active_mask
        )
    
        return output_leaf_heads.view(B, N, D)

    def _standard_attention(self, Q, K, V, mask):
        # Q, K, V are already [B, H, N, Dh]
        D_head = Q.size(-1)
        
        # Optimization: removed explicit .contiguous() calls.
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(D_head)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(attn, V), attn

    def forward(self, query, key, value, y=None, mask=None, return_attention=False):
        x = query 
        B, L_Q, D = x.size()
        H, Dh = self.num_heads, self.head_dim
        L_K = key.size(1)
        L_V = value.size(1)

        if L_Q == L_K == L_V and y is not None:
            # Hierarchical Update
            output_leaf = self.update_X_from_Y(x, y, mask)
            output = self.out_proj_x(output_leaf)
            return (output, None) if return_attention else output
        else:
            # Standard Attention Fallback
            # --- Inline Split ---
            Q = self.Wq_x(query).view(B, L_Q, H, Dh).transpose(1, 2)
            K = self.Wk_x(key).view(B, L_K, H, Dh).transpose(1, 2)
            V = self.Wv_x(value).view(B, L_V, H, Dh).transpose(1, 2)
            
            output_leaf, attn_weights = self._standard_attention(Q, K, V, mask)
            
            # --- Inline Merge ---
            output = output_leaf.transpose(1, 2).reshape(B, L_Q, D)
            output = self.out_proj_x(output)
            
            return (output, attn_weights) if return_attention else output


from torch.profiler import profile, record_function, ProfilerActivity


import os
# ==========================================
# [CRITICAL] SET GPU BEFORE IMPORTING TORCH
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import triton
import math
from torch.profiler import profile, record_function, ProfilerActivity

# [Insert your HierarchicalSparseAttentionTriton class and kernels here...]
# ... (Assuming they are defined in the file above) ...

def run_full_suite():
    print(f"{'='*60}")
    print("1. CORRECTNESS CHECK (Float16)")
    print(f"{'='*60}")

    # 1. Setup Dimensions
    # We use N=2048 for correctness to be fast, but FP16 allows larger if needed.
    B, N, D, H = 2, 32768, 64, 16 
    dim = H * D
    
    # 2. Initialize Model 
    # Ensure model is in Float16
    model = HierarchicalSparseAttentionTriton(dim, H, dropout=0.0).cuda().to(torch.float16)
    
    # 3. Create Inputs (Float16)
    dtype = torch.float16
    x = torch.randn(B, N, dim, device='cuda', dtype=dtype)
    y = torch.randn(B, N - 1, dim, device='cuda', dtype=dtype)
    
    print(f"Input Shapes -> X: {x.shape}, Y: {y.shape}, Dtype: {x.dtype}")
    
    assert y.shape[1] == N - 1, f"Sanity Check Failed"

    # -------------------------------------------------
    # 4. Run PyTorch Reference Path
    # -------------------------------------------------
    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = y.clone().detach().requires_grad_(True)
    
    model.sizes = None; model.offsets = None 
    out_ref = model.cross_update_Y_Ref(x_ref, y_ref)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    # -------------------------------------------------
    # 5. Run Triton Kernel Path
    # -------------------------------------------------
    x_tri = x.clone().detach().requires_grad_(True)
    y_tri = y.clone().detach().requires_grad_(True)
    
    model.sizes = None; model.offsets = None
    out_tri = model.cross_update_Y(x_tri, y_tri)
    loss_tri = out_tri.sum()
    loss_tri.backward()
    
    # -------------------------------------------------
    # 6. Compare Results
    # -------------------------------------------------
    # Move to float32 for accurate diff calculation
    diff_out = (out_ref - out_tri).abs().max().item()
    diff_grad_x = (x_ref.grad - x_tri.grad).abs().max().item()
    diff_grad_y = (y_ref.grad - y_tri.grad).abs().max().item()
    
    print(f"Max Diff Output:   {diff_out:.6f}")
    print(f"Max Diff Grad X:   {diff_grad_x:.6f}")
    print(f"Max Diff Grad Y:   {diff_grad_y:.6f}")
    
    # Relaxed tolerance for FP16 (1e-2 is typical for accumulation noise)
    tol = 1e-2 
    try:
        assert torch.allclose(out_ref, out_tri, atol=tol), "Forward pass mismatch!"
        assert torch.allclose(x_ref.grad, x_tri.grad, atol=tol), "Gradient X mismatch!"
        assert torch.allclose(y_ref.grad, y_tri.grad, atol=tol), "Gradient Y mismatch!"
        print(f"SUCCESS: Triton kernel matches PyTorch reference (within FP16 tolerance).")
    except AssertionError as e:
        print(f"\n{e}")
        # Don't stop, proceed to benchmark to see speed even if precision is slightly off
        
    # ==========================================================================
    # 2. PERFORMANCE BENCHMARK (Float16 - Large Scale)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("2. SPEED BENCHMARK (Float16 - N=32768)")
    print(f"{'='*60}")

    # Config: Massive scale
    B, N, D, H = 16, 2024, 768, 12 
    # B, N, D, H = 16, 4096, 1024, 16 # Alternative config

    print(f"Config: B={B}, N={N}, D={D}, H={H}, dtype={dtype}")

    # Re-init model
    model = HierarchicalSparseAttentionTriton(dim=D, num_heads=H, dropout=0.0).to('cuda').to(dtype)
    model.eval() 

    # Create large inputs 
    x = torch.randn(B, N, D, device='cuda', dtype=dtype, requires_grad=True)
    y_in = torch.randn(B, N-1, D, device='cuda', dtype=dtype, requires_grad=True)
    
    model.sizes = None
    # Dry run 
    out_warm = model.cross_update_Y(x, y_in)
    out_warm.sum().backward()
    model.zero_grad(); x.grad = None; y_in.grad = None

    # --- Timing Setup ---
    num_warmup = 5
    num_trials = 20 
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # A. Measure PyTorch Reference 
    print("  Running PyTorch Reference (FWD + BWD)...")
    for _ in range(num_warmup): 
        out = model.cross_update_Y_Ref(x, y_in)
        out.sum().backward()
        model.zero_grad(); x.grad = None; y_in.grad = None

    torch.cuda.synchronize()
    start.record()
    for _ in range(num_trials):
        out = model.cross_update_Y_Ref(x, y_in)
        out.sum().backward()
        model.zero_grad(); x.grad = None; y_in.grad = None
    end.record()
    torch.cuda.synchronize()
    ms_ref = start.elapsed_time(end)

    # B. Measure Triton Kernel
    print("  Running Triton Kernel (FWD + BWD)...")
    for _ in range(num_warmup): 
        out = model.cross_update_Y(x, y_in)
        out.sum().backward()
        model.zero_grad(); x.grad = None; y_in.grad = None

    torch.cuda.synchronize()
    start.record()
    for _ in range(num_trials):
        out = model.cross_update_Y(x, y_in)
        out.sum().backward()
        model.zero_grad(); x.grad = None; y_in.grad = None
    end.record()
    torch.cuda.synchronize()
    ms_opt = start.elapsed_time(end)

    # Results
    print("-" * 50)
    print(f"  PyTorch Avg Time (Fwd+Bwd): {ms_ref/num_trials:.3f} ms")
    print(f"  Triton  Avg Time (Fwd+Bwd): {ms_opt/num_trials:.3f} ms")
    print(f"  >>> Speedup: {ms_ref/ms_opt:.2f}x")
    print("-" * 50)

    # ==========================================================================
    # 3. PROFILER (Float16)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("3. DETAILED PROFILING")
    print(f"{'='*60}")

    print("Profiling Triton Kernel Trace...")
    model.zero_grad(); x.grad = None; y_in.grad = None

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("Triton_Step"):
            for _ in range(3): 
                out = model.cross_update_Y(x, y_in)
                out.sum().backward()
                model.zero_grad(); x.grad = None; y_in.grad = None

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

if __name__ == "__main__":
    run_full_suite()