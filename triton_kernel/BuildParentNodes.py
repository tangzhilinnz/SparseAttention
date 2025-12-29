import torch
import triton
import triton.language as tl
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


# ------------------------------------------------------------------
#  1. Forward Kernel (Standardized Argument Order)
# ------------------------------------------------------------------
@triton.jit
def build_parent_nodes_forward_kernel(
    # Input Pointers
    Q_ptr,              # [B, P, H, D]
    Kp_ptr, Vp_ptr,     # [B, P, H, D]
    Kc_ptr, Vc_ptr,     # [B, 2*P, H, D]
    
    # Output Pointer (Moved to LAST pointer arg for safety)
    Out_ptr,            # [B, P, H, D]

    # Strides (Inputs)
    sq_b, sq_n, sq_h, sq_d,
    skp_b, skp_n, skp_h, skp_d,
    skc_b, skc_n, skc_h, skc_d,
    
    # Strides (Output)
    so_b, so_n, so_h, so_d,
    
    # Constants
    sm_scale,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr, 
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    # -----------------------------------------------------------
    # 1. Grid & Indices
    # -----------------------------------------------------------
    offs_h = tl.arange(0, BLOCK_H)
    # [CRITICAL] Guard against invalid heads if BLOCK_H > H
    mask_h = offs_h < H

    # -----------------------------------------------------------
    # 2. Base Pointers Setup
    # -----------------------------------------------------------
    # Parent (Self) pointers
    q_base_ptr = Q_ptr + (b_idx * sq_b) + (node_idx * sq_n)
    kp_base_ptr = Kp_ptr + (b_idx * skp_b) + (node_idx * skp_n)
    
    # Child pointers (2 children per parent)
    child0_idx = 2 * node_idx
    child1_idx = 2 * node_idx + 1
    
    kc0_base_ptr = Kc_ptr + (b_idx * skc_b) + (child0_idx * skc_n)
    kc1_base_ptr = Kc_ptr + (b_idx * skc_b) + (child1_idx * skc_n)
    
    # -----------------------------------------------------------
    # 3. Compute Scores (3-Way: Self, Left, Right)
    # -----------------------------------------------------------
    score_self = tl.zeros([BLOCK_H], dtype=tl.float32)
    score_c0   = tl.zeros([BLOCK_H], dtype=tl.float32)
    score_c1   = tl.zeros([BLOCK_H], dtype=tl.float32)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        mask_load = mask_h[:, None] & mask_d[None, :]
        
        # Load Q
        ptr_q = q_base_ptr + (offs_h[:, None] * sq_h) + (offs_d[None, :] * sq_d)
        q = tl.load(ptr_q, mask=mask_load, other=0.0)
        
        # Load K (Self)
        ptr_kp = kp_base_ptr + (offs_h[:, None] * skp_h) + (offs_d[None, :] * skp_d)
        kp = tl.load(ptr_kp, mask=mask_load, other=0.0)

        # Load K (Children)
        ptr_kc0 = kc0_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d)
        ptr_kc1 = kc1_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d)
        kc0 = tl.load(ptr_kc0, mask=mask_load, other=0.0)
        kc1 = tl.load(ptr_kc1, mask=mask_load, other=0.0)
        
        # Accumulate Dot Products
        score_self += tl.sum(q * kp, axis=1)
        score_c0   += tl.sum(q * kc0, axis=1)
        score_c1   += tl.sum(q * kc1, axis=1)

    # -----------------------------------------------------------
    # 4. Softmax (over 3 elements)
    # -----------------------------------------------------------
    score_self = score_self * sm_scale
    score_c0   = score_c0 * sm_scale
    score_c1   = score_c1 * sm_scale

    # Numerical stability
    max_score = tl.maximum(score_self, tl.maximum(score_c0, score_c1))
    
    exp_self = tl.exp(score_self - max_score)
    exp_c0   = tl.exp(score_c0 - max_score)
    exp_c1   = tl.exp(score_c1 - max_score)
    
    # 1e-9 is optional but safe
    denom = exp_self + exp_c0 + exp_c1 + 1e-9
    
    w_self = exp_self / denom
    w_c0   = exp_c0 / denom
    w_c1   = exp_c1 / denom

    # -----------------------------------------------------------
    # 5. Weighted Sum & Store
    # -----------------------------------------------------------
    out_base_ptr = Out_ptr + (b_idx * so_b) + (node_idx * so_n)
    vp_base_ptr = Vp_ptr + (b_idx * skp_b) + (node_idx * skp_n)
    
    # Values for children (Assuming Vc follows same stride logic as Kc)
    vc0_base_ptr = Vc_ptr + (b_idx * skc_b) + (child0_idx * skc_n)
    vc1_base_ptr = Vc_ptr + (b_idx * skc_b) + (child1_idx * skc_n)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]
        
        # Calculate Pointers for V
        ptr_vp  = vp_base_ptr  + (offs_h[:, None] * skp_h) + (offs_d[None, :] * skp_d)
        ptr_vc0 = vc0_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d)
        ptr_vc1 = vc1_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d)
        ptr_out = out_base_ptr + (offs_h[:, None] * so_h) + (offs_d[None, :] * so_d)
        
        vp  = tl.load(ptr_vp, mask=mask_op, other=0.0)
        vc0 = tl.load(ptr_vc0, mask=mask_op, other=0.0)
        vc1 = tl.load(ptr_vc1, mask=mask_op, other=0.0)
        
        # Weighted Sum
        out_val = (w_self[:, None] * vp) + (w_c0[:, None] * vc0) + (w_c1[:, None] * vc1)
        
        tl.store(ptr_out, out_val, mask=mask_op)


# ------------------------------------------------------------------
#  2. Backward Kernel (Fixed: Explicit Strides & Uniqueness Check)
# ------------------------------------------------------------------
@triton.jit
def build_parent_nodes_backward_kernel(
    # Gradients (Input)
    DO_ptr,              # Gradient of Output [B, P, H, D]
    
    # Originals (Input - Needed to recompute scores)
    Q_ptr, Kp_ptr, Vp_ptr, Kc_ptr, Vc_ptr,
    
    # Gradients (Output)
    DQ_ptr,              # Grad of Q [B, P, H, D]
    DKp_ptr, DVp_ptr,    # Grad of Parent K/V [B, P, H, D]
    DKc_ptr, DVc_ptr,    # Grad of Child K/V  [B, 2*P, H, D]

    # Strides (Input)
    sq_b, sq_n, sq_h, sq_d,
    skp_b, skp_n, skp_h, skp_d,
    skc_b, skc_n, skc_h, skc_d,
    
    # Strides (Gradient Outputs) - [CRITICAL FIX: Explicit strides for safety]
    sdo_b, sdo_n, sdo_h, sdo_d,     # For DO
    sdq_b, sdq_n, sdq_h, sdq_d,     # For DQ
    sdkp_b, sdkp_n, sdkp_h, sdkp_d, # For DKp/DVp
    sdkc_b, sdkc_n, sdkc_h, sdkc_d, # For DKc/DVc
    
    # Constants
    sm_scale,
    H: tl.constexpr, BLOCK_H: tl.constexpr, 
    D: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    
    # --- 1. Pointer Arithmetic (Forward) ---
    # Parent Offsets
    off_p_base = (b_idx * sq_b) + (node_idx * sq_n)
    q_ptr_base  = Q_ptr + off_p_base
    kp_ptr_base = Kp_ptr + (b_idx * skp_b) + (node_idx * skp_n)
    
    # Child Offsets
    c0_idx = 2 * node_idx
    c1_idx = 2 * node_idx + 1
    
    off_c0 = (b_idx * skc_b) + (c0_idx * skc_n)
    off_c1 = (b_idx * skc_b) + (c1_idx * skc_n)
    
    kc0_ptr_base = Kc_ptr + off_c0
    kc1_ptr_base = Kc_ptr + off_c1

    # --- 2. Pointer Arithmetic (Gradients) ---
    # [FIX] Use explicit strides for all gradients
    do_ptr_base = DO_ptr + (b_idx * sdo_b) + (node_idx * sdo_n)
    dq_ptr_base = DQ_ptr + (b_idx * sdq_b) + (node_idx * sdq_n)
    
    dkp_ptr_base = DKp_ptr + (b_idx * sdkp_b) + (node_idx * sdkp_n)
    dvp_ptr_base = DVp_ptr + (b_idx * sdkp_b) + (node_idx * sdkp_n)
    
    dkc0_ptr_base = DKc_ptr + (b_idx * sdkc_b) + (c0_idx * sdkc_n)
    dkc1_ptr_base = DKc_ptr + (b_idx * sdkc_b) + (c1_idx * sdkc_n)
    
    dvc0_ptr_base = DVc_ptr + (b_idx * sdkc_b) + (c0_idx * sdkc_n)
    dvc1_ptr_base = DVc_ptr + (b_idx * sdkc_b) + (c1_idx * sdkc_n)

    # --- 3. Pass 1: Recompute Attention Scores ---
    score_self = tl.zeros([BLOCK_H], dtype=tl.float32)
    score_c0   = tl.zeros([BLOCK_H], dtype=tl.float32)
    score_c1   = tl.zeros([BLOCK_H], dtype=tl.float32)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        mask = mask_h[:, None] & mask_d[None, :]
        
        # Pointers
        _q = tl.load(q_ptr_base + offs_h[:, None]*sq_h + offs_d[None, :]*sq_d, mask=mask, other=0.0)
        _kp = tl.load(kp_ptr_base + offs_h[:, None]*skp_h + offs_d[None, :]*skp_d, mask=mask, other=0.0)
        _kc0 = tl.load(kc0_ptr_base + offs_h[:, None]*skc_h + offs_d[None, :]*skc_d, mask=mask, other=0.0)
        _kc1 = tl.load(kc1_ptr_base + offs_h[:, None]*skc_h + offs_d[None, :]*skc_d, mask=mask, other=0.0)
        
        score_self += tl.sum(_q * _kp, axis=1)
        score_c0   += tl.sum(_q * _kc0, axis=1)
        score_c1   += tl.sum(_q * _kc1, axis=1)

    # Softmax Recomputation
    score_self = score_self * sm_scale
    score_c0   = score_c0   * sm_scale
    score_c1   = score_c1   * sm_scale
    
    max_score = tl.maximum(score_self, tl.maximum(score_c0, score_c1))
    exp_self = tl.exp(score_self - max_score)
    exp_c0   = tl.exp(score_c0 - max_score)
    exp_c1   = tl.exp(score_c1 - max_score)
    denom    = exp_self + exp_c0 + exp_c1 + 1e-9
    
    w_self = exp_self / denom
    w_c0   = exp_c0 / denom
    w_c1   = exp_c1 / denom
    
    # --- 4. Pass 2: Accumulate Gradients for Softmax (dP) ---
    dp_self = tl.zeros([BLOCK_H], dtype=tl.float32)
    dp_c0   = tl.zeros([BLOCK_H], dtype=tl.float32)
    dp_c1   = tl.zeros([BLOCK_H], dtype=tl.float32)
    
    # Need Vp, Vc pointers again (Forward input)
    vp_ptr_base = Vp_ptr + (b_idx * skp_b) + (node_idx * skp_n)
    vc0_ptr_base = Vc_ptr + (b_idx * skc_b) + (c0_idx * skc_n)
    vc1_ptr_base = Vc_ptr + (b_idx * skc_b) + (c1_idx * skc_n)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        mask = mask_h[:, None] & mask_d[None, :]
        
        # Load Gradients (DO) and Values (V)
        _do = tl.load(do_ptr_base + offs_h[:, None]*sdo_h + offs_d[None, :]*sdo_d, mask=mask, other=0.0)
        
        _vp = tl.load(vp_ptr_base + offs_h[:, None]*skp_h + offs_d[None, :]*skp_d, mask=mask, other=0.0)
        _vc0 = tl.load(vc0_ptr_base + offs_h[:, None]*skc_h + offs_d[None, :]*skc_d, mask=mask, other=0.0)
        _vc1 = tl.load(vc1_ptr_base + offs_h[:, None]*skc_h + offs_d[None, :]*skc_d, mask=mask, other=0.0)
        
        # [CRITICAL NOTE]: We use tl.store because this is a STRICT TREE.
        # Each child node (2*n, 2*n+1) has exactly ONE parent.
        # If your topology is a DAG (children shared), you MUST use tl.atomic_add here.
        
        # 4a. Compute dV (Store immediately)
        # dV = w * DO
        tl.store(dvp_ptr_base + offs_h[:, None]*sdkp_h + offs_d[None, :]*sdkp_d, _do * w_self[:, None], mask=mask)
        tl.store(dvc0_ptr_base + offs_h[:, None]*sdkc_h + offs_d[None, :]*sdkc_d, _do * w_c0[:, None], mask=mask)
        tl.store(dvc1_ptr_base + offs_h[:, None]*sdkc_h + offs_d[None, :]*sdkc_d, _do * w_c1[:, None], mask=mask)

        # 4b. Accumulate dP precursors (DO . V)
        dp_self += tl.sum(_vp * _do, axis=1)
        dp_c0   += tl.sum(_vc0 * _do, axis=1)
        dp_c1   += tl.sum(_vc1 * _do, axis=1)

    # --- 5. Compute Softmax Gradients (dS) ---
    sum_w_dp = (w_self * dp_self) + (w_c0 * dp_c0) + (w_c1 * dp_c1)
    
    ds_self = w_self * (dp_self - sum_w_dp) * sm_scale
    ds_c0   = w_c0   * (dp_c0   - sum_w_dp) * sm_scale
    ds_c1   = w_c1   * (dp_c1   - sum_w_dp) * sm_scale

    # --- 6. Pass 3: Compute dQ and dK ---
    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        mask = mask_h[:, None] & mask_d[None, :]

        # Load Q and K again
        _q = tl.load(q_ptr_base + offs_h[:, None]*sq_h + offs_d[None, :]*sq_d, mask=mask, other=0.0)
        _kp = tl.load(kp_ptr_base + offs_h[:, None]*skp_h + offs_d[None, :]*skp_d, mask=mask, other=0.0)
        _kc0 = tl.load(kc0_ptr_base + offs_h[:, None]*skc_h + offs_d[None, :]*skc_d, mask=mask, other=0.0)
        _kc1 = tl.load(kc1_ptr_base + offs_h[:, None]*skc_h + offs_d[None, :]*skc_d, mask=mask, other=0.0)

        # dQ = sum(dS * K)
        # Note: dQ writes to same address as Q, so we use sdq strides
        _dq = (ds_self[:, None] * _kp) + (ds_c0[:, None] * _kc0) + (ds_c1[:, None] * _kc1)
        tl.store(dq_ptr_base + offs_h[:, None]*sdq_h + offs_d[None, :]*sdq_d, _dq, mask=mask)
        
        # dK = dS * Q
        tl.store(dkp_ptr_base + offs_h[:, None]*sdkp_h + offs_d[None, :]*sdkp_d, ds_self[:, None] * _q, mask=mask)
        tl.store(dkc0_ptr_base + offs_h[:, None]*sdkc_h + offs_d[None, :]*sdkc_d, ds_c0[:, None] * _q, mask=mask)
        tl.store(dkc1_ptr_base + offs_h[:, None]*sdkc_h + offs_d[None, :]*sdkc_d, ds_c1[:, None] * _q, mask=mask)


# ------------------------------------------------------------------
#  3. Wrapper Class (Fixed: Arguments, Asserts, and Indentation)
# ------------------------------------------------------------------
class BuildParentNodesFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q_p, K_p, V_p, K_c, V_c):       
        """
        Inputs:
          Q_p, K_p, V_p: [B, Parent_Count, H, D]
          K_c, V_c:      [B, Child_Count, H, D]
        """
        # 1. Enforce Contiguity
        Q_p = Q_p.contiguous()
        K_p = K_p.contiguous()
        V_p = V_p.contiguous()
        K_c = K_c.contiguous()
        V_c = V_c.contiguous()
        
        B, P, H, D = Q_p.shape

        # --- SAFETY CHECK (Prevent Segfaults) ---
        C = K_c.shape[1]
        assert C == 2 * P, f"Shape Mismatch: Child count {C} must be exactly 2x Parent count {P}"
        # ----------------------------------------
        
        # 2. Allocate Output
        Out = torch.empty_like(Q_p)
        
        # 3. Kernel Config
        grid = (P, B)
        # Safe block size calculation
        BLOCK_H = triton.next_power_of_2(H)
        BLOCK_SIZE = triton.next_power_of_2(D)
        sm_scale = 1.0 / math.sqrt(D)

        # 4. Launch Forward
        build_parent_nodes_forward_kernel[grid](
            # Inputs
            Q_p, K_p, V_p, K_c, V_c, 
            # Output (Last pointer arg)
            Out, 
            # Strides
            Q_p.stride(0), Q_p.stride(1), Q_p.stride(2), Q_p.stride(3),
            K_p.stride(0), K_p.stride(1), K_p.stride(2), K_p.stride(3),
            K_c.stride(0), K_c.stride(1), K_c.stride(2), K_c.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            # Constants
            sm_scale=sm_scale,
            H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        # 5. Save for Backward
        ctx.save_for_backward(Q_p, K_p, V_p, K_c, V_c)
        ctx.constants = (sm_scale, H, BLOCK_H, D, BLOCK_SIZE)
        
        return Out

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Retrieve Tensors
        Q_p, K_p, V_p, K_c, V_c = ctx.saved_tensors
        sm_scale, H, BLOCK_H, D, BLOCK_SIZE = ctx.constants
        
        # Ensure gradient is contiguous for reading
        grad_output = grad_output.contiguous()
        
        # 2. Allocate Gradients
        # We use empty_like because our kernel performs a DIRECT STORE (Overwrite).
        dQ = torch.empty_like(Q_p)
        dKp = torch.empty_like(K_p)
        dVp = torch.empty_like(V_p)
        dKc = torch.empty_like(K_c)
        dVc = torch.empty_like(V_c)
        
        # 3. Launch Backward
        B, P = Q_p.shape[0], Q_p.shape[1]
        grid = (P, B)
        
        build_parent_nodes_backward_kernel[grid](
            # Gradients Input
            grad_output,
            # Originals
            Q_p, K_p, V_p, K_c, V_c,
            # Gradients Output
            dQ, dKp, dVp, dKc, dVc,
            
            # Strides (Inputs)
            Q_p.stride(0), Q_p.stride(1), Q_p.stride(2), Q_p.stride(3),
            K_p.stride(0), K_p.stride(1), K_p.stride(2), K_p.stride(3),
            K_c.stride(0), K_c.stride(1), K_c.stride(2), K_c.stride(3),
            
            # Strides (Gradients - Explicitly Passed)
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            dKp.stride(0), dKp.stride(1), dKp.stride(2), dKp.stride(3),
            dKc.stride(0), dKc.stride(1), dKc.stride(2), dKc.stride(3),

            # Constants
            sm_scale=sm_scale,
            H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        return dQ, dKp, dVp, dKc, dVc


# Helper Function
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
        
        Optimization:
          - Pre-allocates Y_new to avoid torch.cat() overhead.
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        
        assert y_in is not None, "y_in cannot be None"
        assert y_in.size(1) == N - 1, f"y_in size {y_in.size(1)} mismatch! Expected N-1 ({N-1}) for N={N} leaves."

        if self.sizes is None: self.sizes, self.offsets = self.build_level_info(N)

        # -------------------------------------------------------
        # 1. Global Parent Projection
        # -------------------------------------------------------
        Q_p_all = self.Wq_y(y_in).view(B, -1, H, Dh)
        K_p_all = self.Wk_y(y_in).view(B, -1, H, Dh)
        V_p_all = self.Wv_y(y_in).view(B, -1, H, Dh)
        
        # -------------------------------------------------------
        # 2. Pre-allocate Output Buffer (The Optimization)
        # -------------------------------------------------------
        # Instead of accumulating a list, we allocate the final shape immediately.
        # Shape: [B, N-1, D] matching y_in
        Y_new = torch.empty((B, N - 1, D), device=x.device, dtype=x.dtype)
        
        prev_sources = x # Starts as the leaves
        
        for level, parent_count in enumerate(self.sizes):
            offset = self.offsets[level]

            # ---------------------------------------------------
            # A. Prepare Parents (Slicing)
            # ---------------------------------------------------
            Q_p = Q_p_all[:, offset : offset + parent_count, :, :]
            K_p = K_p_all[:, offset : offset + parent_count, :, :]
            V_p = V_p_all[:, offset : offset + parent_count, :, :]
            
            # ---------------------------------------------------
            # B. Prepare Children (Projection)
            # ---------------------------------------------------
            useful_len = parent_count * 2
            children_in = prev_sources[:, :useful_len, :]
            
            K_c = self.Wk_y(children_in).view(B, -1, H, Dh)
            V_c = self.Wv_y(children_in).view(B, -1, H, Dh)
            V_c = self.dropout(V_c)

            # ---------------------------------------------------
            # C. Triton Kernel
            # ---------------------------------------------------
            updated_heads = build_parent_nodes(Q_p, K_p, V_p, K_c, V_c)

            # ---------------------------------------------------
            # D. Merge Heads & Store Directly
            # ---------------------------------------------------
            updated_merged = updated_heads.reshape(B, parent_count, D)
            
            # [OPTIMIZED WRITE] 
            # Write directly into the pre-allocated buffer slice.
            # This avoids creating a transient tensor that needs to be copied later.
            Y_new[:, offset : offset + parent_count, :] = updated_merged
            
            # Update pointer for next level
            # Note: We can iterate on the slice view to save memory if strict, 
            # but using updated_merged is safe and fast enough.
            prev_sources = updated_merged

        # -------------------------------------------------------
        # 3. Final Projection
        # -------------------------------------------------------
        # Y_new is already contiguous and full. Pass directly to Linear.
        return self.out_proj_y(Y_new)

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


import torch
import triton
import math
from torch.profiler import profile, record_function, ProfilerActivity

# [Insert your HierarchicalSparseAttentionTriton class, Triton Kernels, and Wrappers here]
# ... (Assuming they are defined in the same file above this script) ...

def run_full_suite():
    print(f"{'='*60}")
    print("1. CORRECTNESS CHECK (Small Scale)")
    print(f"{'='*60}")

    # 1. Setup Dimensions for Correctness
    #B, N, D, H = 16, 2048, 128, 8
    B, N, D, H = 16, 4096, 128, 16 # Uncomment for even heavier load
    dim = H * D
    
    # 2. Initialize Model (Dropout=0 for determinism)
    model = HierarchicalSparseAttentionTriton(dim, H, dropout=0.0).cuda()
    
    # 3. Create Inputs
    x = torch.randn(B, N, dim, device='cuda', dtype=torch.float32)
    
    # SANITY CHECK: Y must be N-1 for a binary tree
    # Level 1 (N/2) + Level 2 (N/4) ... + Root (1) = N-1
    y = torch.randn(B, N - 1, dim, device='cuda', dtype=torch.float32)
    
    print(f"Input Shapes -> X: {x.shape}, Y: {y.shape}")
    
    # Explicit Sanity Check
    assert y.shape[1] == N - 1, f"Sanity Check Failed: Y dim {y.shape[1]} != N-1 ({N-1})"

    # -------------------------------------------------
    # 4. Run PyTorch Reference Path (Gradients)
    # -------------------------------------------------
    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = y.clone().detach().requires_grad_(True)
    
    model.sizes = None; model.offsets = None # Reset cache
    out_ref = model.cross_update_Y_Ref(x_ref, y_ref)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    # -------------------------------------------------
    # 5. Run Triton Kernel Path (Gradients)
    # -------------------------------------------------
    x_tri = x.clone().detach().requires_grad_(True)
    y_tri = y.clone().detach().requires_grad_(True)
    
    model.sizes = None; model.offsets = None # Reset cache
    out_tri = model.cross_update_Y(x_tri, y_tri)
    loss_tri = out_tri.sum()
    loss_tri.backward()
    
    # -------------------------------------------------
    # 6. Compare Results
    # -------------------------------------------------
    diff_out = (out_ref - out_tri).abs().max().item()
    diff_grad_x = (x_ref.grad - x_tri.grad).abs().max().item()
    diff_grad_y = (y_ref.grad - y_tri.grad).abs().max().item()
    
    print(f"Max Diff Output:   {diff_out:.8f}")
    print(f"Max Diff Grad X:   {diff_grad_x:.8f}")
    print(f"Max Diff Grad Y:   {diff_grad_y:.8f}")
    
    try:
        assert torch.allclose(out_ref, out_tri, atol=1e-4), "Forward pass mismatch!"
        assert torch.allclose(x_ref.grad, x_tri.grad, atol=1e-4), "Gradient X mismatch!"
        assert torch.allclose(y_ref.grad, y_tri.grad, atol=1e-4), "Gradient Y mismatch!"
        print(f"SUCCESS: Triton kernel matches PyTorch reference.")
    except AssertionError as e:
        print(f"\n{e}")
        return # Stop if correctness fails

    # ==========================================================================
    # 2. PERFORMANCE BENCHMARK (Forward + Backward)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("2. SPEED BENCHMARK (Forward + Backward)")
    print(f"{'='*60}")

    # Config: Larger size to stress GPU
    #B, N, D, H = 16, 2048, 1024, 8
    B, N, D, H = 16, 4096, 2048, 16 # Uncomment for even heavier load

    dtype = torch.float16 
    print(f"Config: B={B}, N={N}, D={D}, H={H}, dtype={dtype}")

    # Re-init model in fp16
    model = HierarchicalSparseAttentionTriton(dim=D, num_heads=H, dropout=0.0).to('cuda').to(dtype)
    model.eval() # Disable dropout overhead to benchmark pure kernel speed

    # Create large inputs with REQUIRES_GRAD=True for backward pass
    x = torch.randn(B, N, D, device='cuda', dtype=dtype, requires_grad=True)
    y_in = torch.randn(B, N-1, D, device='cuda', dtype=dtype, requires_grad=True)
    
    # Sanity check again for large scale
    assert y_in.shape[1] == N - 1, "Large Scale Sanity Check Failed"

    # Reset cache once
    model.sizes = None
    # Dry run to compile kernels/allocate buffers
    out_warm = model.cross_update_Y(x, y_in)
    out_warm.sum().backward()
    model.zero_grad()
    x.grad = None
    y_in.grad = None

    # --- Timing Setup ---
    num_warmup = 5
    num_trials = 50 
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # A. Measure PyTorch Reference (Forward + Backward)
    print("  Running PyTorch Reference (FWD + BWD)...")
    # Warmup
    for _ in range(num_warmup): 
        out = model.cross_update_Y_Ref(x, y_in)
        out.sum().backward()
        model.zero_grad(); x.grad = None; y_in.grad = None

    torch.cuda.synchronize()
    start.record()
    for _ in range(num_trials):
        out = model.cross_update_Y_Ref(x, y_in)
        out.sum().backward()
        # Reset grads to simulate distinct steps
        model.zero_grad(); x.grad = None; y_in.grad = None
    end.record()
    torch.cuda.synchronize()
    ms_ref = start.elapsed_time(end)

    # B. Measure Triton Kernel (Forward + Backward)
    print("  Running Triton Kernel (FWD + BWD)...")
    # Warmup
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
    # 3. PROFILER (Forward + Backward)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("3. DETAILED PROFILING (Triton Fwd + Bwd)")
    print(f"{'='*60}")

    print("Profiling Triton Kernel Trace...")

    # Clean up before profiling
    model.zero_grad(); x.grad = None; y_in.grad = None

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("Triton_Step"):
            for _ in range(5): # Run a few times
                out = model.cross_update_Y(x, y_in)
                out.sum().backward()
                model.zero_grad(); x.grad = None; y_in.grad = None

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

if __name__ == "__main__":
    run_full_suite()