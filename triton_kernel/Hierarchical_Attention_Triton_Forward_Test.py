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
        Weights = torch.empty((B, P, H, 3), device=Q_p.device, dtype=Q_p.dtype)
        
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
            num_warps=2
        )
        
        ctx.save_for_backward(Q_p, K_p, V_p, K_c, V_c, Weights)
        ctx.constants = (sm_scale, H, BLOCK_H, D, BLOCK_SIZE)
        return Out

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Retrieve Tensors
        Q_p, K_p, V_p, K_c, V_c, Weights = ctx.saved_tensors
        sm_scale, H, BLOCK_H, D, BLOCK_SIZE = ctx.constants
        
        # Ensure gradient is contiguous
        grad_output = grad_output.contiguous()
        
        # 2. Allocate Gradients
        # [FIX #2] Use empty_like instead of zeros_like because the kernel 
        # fully overwrites these buffers unconditionally using tl.store.
        # This saves 5x memset operations.
        dQ = torch.empty_like(Q_p)
        dKp = torch.empty_like(K_p)
        dVp = torch.empty_like(V_p)
        dKc = torch.empty_like(K_c)
        dVc = torch.empty_like(V_c)
        
        # 3. Launch Backward
        B, P = Q_p.shape[0], Q_p.shape[1]
        grid = (P, B)
        
        build_parent_nodes_backward_kernel[grid](
            # Inputs
            grad_output,
            Weights, 
            Q_p, K_p, V_p, K_c, V_c,
            
            # Outputs
            dQ, dKp, dVp, dKc, dVc,
            
            # Strides (Inputs)
            *Q_p.stride(), *K_p.stride(), *K_c.stride(),
            *Weights.stride(),
            
            # Strides (Gradients)
            *grad_output.stride(), *dQ.stride(), *dKp.stride(), *dKc.stride(),

            # Constants
            sm_scale=sm_scale,
            H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2
        )
        
        return dQ, dKp, dVp, dKc, dVc

def build_parent_nodes(Q_p, K_p, V_p, K_c, V_c):
    return BuildParentNodesFunc.apply(Q_p, K_p, V_p, K_c, V_c)

## ============================================================================================= ##
## ============================================================================================= ##
## ============================================================================================= ##
## ============================================================================================= ##


@triton.jit
def hierarchical_attention_forward_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, 
    Lookup_ptr, Mask_ptr, 
    Out_ptr, W_ptr, 
    
    # Strides (Q)
    sq_b, sq_n, sq_h, sq_d,
    # Strides (K)
    sk_b, sk_n, sk_h, sk_d,
    # Strides (V) <--- [FIX]: Explicit V strides added
    sv_b, sv_n, sv_h, sv_d,
    # Strides (Topology)
    sl_n, sl_lvl,
    # Strides (Out)
    so_b, so_n, so_h, so_d,
    # Strides (Weights)
    sw_b, sw_n, sw_h, sw_lvl,
    
    # Constants
    sm_scale,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_LEVELS: tl.constexpr, 
    HAS_MASK: tl.constexpr 
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    # 1. Setup
    h_idx = tl.arange(0, BLOCK_H)
    mask_h = h_idx < H
    offs_d = tl.arange(0, BLOCK_D)
    offs_lvl = tl.arange(0, BLOCK_LEVELS)
    mask_lvl = offs_lvl < LEVELS

    # 2. Load Topology
    off_lookup = node_idx * sl_n + offs_lvl * sl_lvl
    neighbor_indices = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl, other=0)
    
    neighbor_mask_val = tl.zeros([BLOCK_LEVELS], dtype=tl.int1)
    if HAS_MASK:
        # User confirmed: 1 = Mask Out (Ignore), 0 = Valid (Keep)
        val_int8 = tl.load(Mask_ptr + off_lookup, mask=mask_lvl, other=1).to(tl.int8)
        neighbor_mask_val = (val_int8 != 0)

    # 3. Base Pointers
    k_batch_base = K_ptr + b_idx * sk_b
    
    # [FIX]: Use V specific stride for batch
    v_batch_base = V_ptr + b_idx * sv_b
    
    off_node_self = node_idx * sk_n
    off_node_cross = neighbor_indices * sk_n 

    q_ptr = Q_ptr + (b_idx * sq_b) + (node_idx * sq_n) + \
            (h_idx[:, None] * sq_h) + (offs_d[None, :] * sq_d)

    acc_self = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_cross = tl.zeros([BLOCK_H, BLOCK_LEVELS], dtype=tl.float32)

    # 4. Score Loop
    for off_d_start in range(0, D, BLOCK_D):
        cur_offs_d = off_d_start + offs_d
        d_mask = cur_offs_d < D
        mask_q = mask_h[:, None] & d_mask[None, :]
        
        q = tl.load(q_ptr, mask=mask_q, other=0.0)
        
        # --- K SELF ---
        ptr_k_self = k_batch_base + off_node_self + \
                     (h_idx[:, None] * sk_h) + (cur_offs_d[None, :] * sk_d)
        k_self = tl.load(ptr_k_self, mask=mask_q, other=0.0)
        acc_self += tl.sum(q * k_self, axis=1)

        # --- K CROSS ---
        ptr_k_cross = k_batch_base + \
                      off_node_cross[None, :, None] + \
                      (h_idx[:, None, None] * sk_h) + \
                      (cur_offs_d[None, None, :] * sk_d)
        
        mask_k = mask_h[:, None, None] & mask_lvl[None, :, None] & d_mask[None, None, :]
        k_cross = tl.load(ptr_k_cross, mask=mask_k, other=0.0)
        
        acc_cross += tl.sum(q[:, None, :] * k_cross, axis=2)
        q_ptr += BLOCK_D * sq_d

    # 5. Softmax
    acc_self = acc_self * sm_scale
    acc_cross = acc_cross * sm_scale
    
    # Apply Masking (1 = Mask Out / -inf)
    mask_broadcast = (offs_lvl >= LEVELS)
    if HAS_MASK:
        mask_broadcast = mask_broadcast | neighbor_mask_val
        
    acc_cross = tl.where(mask_broadcast[None, :], -float('inf'), acc_cross)
    
    max_cross = tl.max(acc_cross, axis=1)
    max_all = tl.maximum(acc_self, max_cross)
    
    exp_self = tl.exp(acc_self - max_all)
    exp_cross = tl.exp(acc_cross - max_all[:, None])
    
    denom = exp_self + tl.sum(exp_cross, axis=1)
    w_self = exp_self / denom 
    w_cross = exp_cross / denom[:, None]

    # Save Weights
    w_base_ptr = W_ptr + (b_idx * sw_b) + (node_idx * sw_n) + (h_idx * sw_h)
    tl.store(w_base_ptr + (0 * sw_lvl), w_self, mask=mask_h)
    
    w_cross_ptr = w_base_ptr[:, None] + ((1 + offs_lvl[None, :]) * sw_lvl)
    tl.store(w_cross_ptr, w_cross, mask=mask_h[:, None] & mask_lvl[None, :])

    # 6. Weighted Sum Loop
    out_base_ptr = Out_ptr + (b_idx * so_b) + (node_idx * so_n) + \
                   (h_idx[:, None] * so_h) + (offs_d[None, :] * so_d)

    for off_d_start in range(0, D, BLOCK_D):
        cur_offs_d = off_d_start + offs_d
        d_mask = cur_offs_d < D
        mask_op = mask_h[:, None] & d_mask[None, :]

        # --- V SELF ---
        # [FIX]: Use sv_h and sv_d
        ptr_v_self = v_batch_base + off_node_self + \
                     (h_idx[:, None] * sv_h) + (cur_offs_d[None, :] * sv_d)
        v_self = tl.load(ptr_v_self, mask=mask_op, other=0.0)
        out_acc = w_self[:, None] * v_self
        
        # --- V CROSS ---
        # [FIX]: Use sv_h and sv_d
        ptr_v_cross = v_batch_base + \
                      off_node_cross[None, :, None] + \
                      (h_idx[:, None, None] * sv_h) + \
                      (cur_offs_d[None, None, :] * sv_d)
                      
        mask_v = mask_h[:, None, None] & mask_lvl[None, :, None] & d_mask[None, None, :]
        v_cross = tl.load(ptr_v_cross, mask=mask_v, other=0.0)
        
        out_acc += tl.sum(w_cross[:, :, None] * v_cross, axis=1)
        
        tl.store(out_base_ptr, out_acc.to(Out_ptr.dtype.element_ty), mask=mask_op)
        out_base_ptr += BLOCK_D * so_d


# ------------------------------------------------------------------
#  Backward Kernel 1: Score Gradient (Computes dS)
# ------------------------------------------------------------------
@triton.jit
def hierarchical_attention_backward_dS_kernel(
    DO_ptr, W_ptr, V_ptr, Lookup_ptr, DS_ptr, Mask_ptr,
    # Strides for dO
    sdo_b, sdo_n, sdo_h, sdo_d,
    # Strides for W
    sw_b, sw_n, sw_h, sw_lvl,
    # Strides for V (Now separate from K)
    sv_b, sv_n, sv_h, sv_d,
    # Strides for Lookup/Topology
    sl_n, sl_lvl,
    # Strides for DS
    sds_b, sds_n, sds_h, sds_lvl,
    
    sm_scale,
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    LEVELS: tl.constexpr, BLOCK_LEVELS: tl.constexpr,
    HAS_MASK: tl.constexpr # <--- Added Flag
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    h_idx = tl.arange(0, BLOCK_H)
    mask_h = h_idx < H
    offs_lvl = tl.arange(0, BLOCK_LEVELS)
    
    # -----------------------------------------------------------
    # 1. Mask Logic & Topology Load
    # -----------------------------------------------------------
    # Boundary Check: Are we within the max levels?
    mask_lvl_bounds = offs_lvl < LEVELS
    off_lookup = node_idx * sl_n + offs_lvl * sl_lvl
    
    neighbor_indices = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl_bounds, other=0)

    # Combined Mask: (In Bounds) AND (Not Masked by User)
    # Start with bounds
    mask_valid_cross = mask_lvl_bounds
    
    if HAS_MASK:
        # Load User Mask (1 = Ignore/Masked, 0 = Keep)
        val_int8 = tl.load(Mask_ptr + off_lookup, mask=mask_lvl_bounds, other=1).to(tl.int8)
        mask_valid_cross = mask_valid_cross & (val_int8 == 0)

    # -----------------------------------------------------------
    # 2. Load Weights (W)
    # -----------------------------------------------------------
    w_base = W_ptr + (b_idx * sw_b) + (node_idx * sw_n) + (h_idx * sw_h)
    
    # Self Weight
    w_self = tl.load(w_base + (0 * sw_lvl), mask=mask_h, other=0.0)
    
    # Cross Weight
    # DEFENSIVE: Apply mask_valid_cross. If masked, w_cross becomes 0.0.
    w_cross = tl.load(w_base[:, None] + ((1 + offs_lvl[None, :]) * sw_lvl), 
                      mask=mask_h[:, None] & mask_valid_cross[None, :], other=0.0)

    # -----------------------------------------------------------
    # 3. Compute dP = dot(dO, V)
    # -----------------------------------------------------------
    dp_self = tl.zeros([BLOCK_H], dtype=tl.float32)
    dp_cross = tl.zeros([BLOCK_H, BLOCK_LEVELS], dtype=tl.float32)

    # Use explicit SV strides for V pointer calculation
    v_batch_base = V_ptr + b_idx * sv_b
    do_batch_base = DO_ptr + (b_idx * sdo_b) + (node_idx * sdo_n)
    
    off_node_self = node_idx * sv_n
    off_node_cross = neighbor_indices * sv_n

    for off_d in range(0, D, BLOCK_D):
        offs_d = off_d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]

        # Load dO
        do = tl.load(do_batch_base + (h_idx[:, None]*sdo_h) + (offs_d[None, :]*sdo_d), 
                     mask=mask_op, other=0.0)
        
        # Load V Self (using sv strides)
        ptr_v_self = v_batch_base + off_node_self + (h_idx[:, None]*sv_h) + (offs_d[None, :]*sv_d)
        v_self = tl.load(ptr_v_self, mask=mask_op, other=0.0)
        
        # Load V Cross (using sv strides)
        ptr_v_cross = v_batch_base + off_node_cross[None, :, None] + \
                      (h_idx[:, None, None]*sv_h) + (offs_d[None, None, :]*sv_d)
        
        # DEFENSIVE: Apply mask_valid_cross to V load.
        # This prevents loading 'NaNs' or garbage from padding tokens.
        mask_v_cross = mask_h[:, None, None] & mask_valid_cross[None, :, None] & mask_d[None, None, :]
        v_cross = tl.load(ptr_v_cross, mask=mask_v_cross, other=0.0)

        dp_self += tl.sum(do * v_self, axis=1)
        dp_cross += tl.sum(do[:, None, :] * v_cross, axis=2)

    # -----------------------------------------------------------
    # 4. Compute dS (Softmax Gradient)
    # -----------------------------------------------------------
    sum_wdp = (w_self * dp_self) + tl.sum(w_cross * dp_cross, axis=1)
    
    ds_self = w_self * (dp_self - sum_wdp) * sm_scale
    ds_cross = w_cross * (dp_cross - sum_wdp[:, None]) * sm_scale

    # -----------------------------------------------------------
    # 5. Store dS
    # -----------------------------------------------------------
    ds_base = DS_ptr + (b_idx * sds_b) + (node_idx * sds_n) + (h_idx * sds_h)
    
    tl.store(ds_base + (0 * sds_lvl), ds_self, mask=mask_h)
    
    ds_cross_ptr = ds_base[:, None] + ((1 + offs_lvl[None, :]) * sds_lvl)
    
    # SAFE STORE: We use mask_lvl_bounds (bounds only), NOT mask_valid_cross.
    # Why? If a node is masked, ds_cross is 0.0 (because w_cross was 0.0).
    # We WANT to write this 0.0 to memory to overwrite any garbage in uninitialized DS.
    # If we masked the store, DS would retain random values from empty_like().
    tl.store(ds_cross_ptr, ds_cross, mask=mask_h[:, None] & mask_lvl_bounds[None, :])


# ==================================================================
#  BACKWARD KERNEL 2a: SPECIALIZED LEAF KERNEL (Level 0)
# ==================================================================
# Hardcoded for Level 0: 
# - Self Interaction
# - Single Neighbor Gather (Width=1)
# - No Loops, No Branching overhead
@triton.jit
def hierarchical_attention_backward_dK_dV_leaf_kernel(
    DS_ptr, Q_ptr, W_ptr, DO_ptr, Gather_Table_ptr,
    DK_ptr, DV_ptr,
    sds_b, sds_n, sds_h, sds_lvl,
    sq_b, sq_n, sq_h, sq_d,
    sw_b, sw_n, sw_h, sw_lvl,
    sdo_b, sdo_n, sdo_h, sdo_d,
    sdk_b, sdk_node, sdk_h, sdk_d,
    sg_node, sg_dim,
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr
):
    node_id = tl.program_id(0)
    b_idx = tl.program_id(1)

    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # -----------------------------------------------------------
    # 1. Pre-Load DS and W (Invariant across D)
    # -----------------------------------------------------------
    # These are scalars per head, so we can load them once and reuse them
    # for every chunk of D. This saves significant memory bandwidth.

    # --- Self Pointers ---
    off_ds = (b_idx * sds_b) + (node_id * sds_n)
    off_w = (b_idx * sw_b) + (node_id * sw_n)
    
    ds_self = tl.load(DS_ptr + off_ds + (offs_h[:, None] * sds_h), mask=mask_h[:, None], other=0.0)
    w_self = tl.load(W_ptr + off_w + (offs_h[:, None] * sw_h), mask=mask_h[:, None], other=0.0)

    # --- Sibling Pointers & Check ---
    tab_ptr = Gather_Table_ptr + (node_id * sg_node)
    sibling_leaf = tl.load(tab_ptr + 0)
    
    has_sibling = (sibling_leaf != -1)
    ds_sib = tl.zeros([BLOCK_H, 1], dtype=tl.float32)
    w_sib = tl.zeros([BLOCK_H, 1], dtype=tl.float32)

    if has_sibling:
        w_idx = 1
        off_ds_sib = (b_idx * sds_b) + (sibling_leaf * sds_n) + (w_idx * sds_lvl)
        off_w_sib = (b_idx * sw_b) + (sibling_leaf * sw_n) + (w_idx * sw_lvl)
        
        ds_sib = tl.load(DS_ptr + off_ds_sib + (offs_h[:, None] * sds_h), mask=mask_h[:, None], other=0.0)
        w_sib = tl.load(W_ptr + off_w_sib + (offs_h[:, None] * sw_h), mask=mask_h[:, None], other=0.0)

    # -----------------------------------------------------------
    # 2. Loop over Dimension D (The Fix)
    # -----------------------------------------------------------
    # We process D in chunks of BLOCK_D.
    off_q = (b_idx * sq_b) + (node_id * sq_n)
    off_do = (b_idx * sdo_b) + (node_id * sdo_n)
    
    # Sibling Q/dO Base Pointers
    off_q_sib = (b_idx * sq_b) + (sibling_leaf * sq_n)
    off_do_sib = (b_idx * sdo_b) + (sibling_leaf * sdo_n)

    for off_d_start in range(0, D, BLOCK_D):
        offs_d = off_d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]

        # --- Self Computation ---
        q_self = tl.load(Q_ptr + off_q + (offs_h[:, None] * sq_h) + (offs_d[None, :] * sq_d), mask=mask_op, other=0.0)
        do_self = tl.load(DO_ptr + off_do + (offs_h[:, None] * sdo_h) + (offs_d[None, :] * sdo_d), mask=mask_op, other=0.0)
        
        # dK = dS * Q | dV = W * dO
        dk_acc = ds_self * q_self
        dv_acc = w_self * do_self

        # --- Sibling Computation ---
        if has_sibling:
            q_sib = tl.load(Q_ptr + off_q_sib + (offs_h[:, None] * sq_h) + (offs_d[None, :] * sq_d), mask=mask_op, other=0.0)
            do_sib = tl.load(DO_ptr + off_do_sib + (offs_h[:, None] * sdo_h) + (offs_d[None, :] * sdo_d), mask=mask_op, other=0.0)
            
            #dk_acc += ds_sib * q_sib
            #dv_acc += w_sib * do_sib
            dk_acc += ds_sib * q_sib
            dv_acc += w_sib * do_sib


        # --- Store Chunk ---
        off_out = (b_idx * sdk_b) + (node_id * sdk_node)
        tl.store(DK_ptr + off_out + (offs_h[:, None] * sdk_h) + (offs_d[None, :] * sdk_d), dk_acc, mask=mask_op)
        tl.store(DV_ptr + off_out + (offs_h[:, None] * sdk_h) + (offs_d[None, :] * sdk_d), dv_acc, mask=mask_op)

@triton.jit
def hierarchical_attention_backward_low_level_kernel(
    # Inputs
    DS_ptr, Q_ptr, W_ptr, DO_ptr, Gather_Table_ptr,
    DK_ptr, DV_ptr,
    # Strides
    sds_b, sds_n, sds_h, sds_lvl,
    sq_b, sq_n, sq_h, sq_d,
    sw_b, sw_n, sw_h, sw_lvl,
    sdo_b, sdo_n, sdo_h, sdo_d,
    sdk_b, sdk_node, sdk_h, sdk_d,
    sg_node, sg_dim,
    # Constants
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    N: tl.constexpr,
    MAX_LEVEL: tl.constexpr
):
    pid = tl.program_id(0)
    b_idx = tl.program_id(1)

    # ------------------------------------------------------------------
    # 1. OPTIMIZED GEOMETRIC LOOP (Fully Unrolled & Constant Folded)
    # ------------------------------------------------------------------
    # Pre-calculate the constant shift for Node ID.
    # Mathematical Fact: (current_node_offset - current_block_offset) is ALWAYS == N.
    # Therefore, node_id is ALWAYS (N + pid) regardless of the level.
    # This removes 2 additions from inside the loop.
    node_id = N + pid
    
    # Use static_range to ensure the compiler unrolls this loop 100%.
    # Each iteration becomes a standalone block of PTX code.
    target_level = 0
    
    # We maintain the 'if' structure because it allows for "Early Exit".
    # (Threads that match Level 1 don't need to do work for Level 2).
    # We use a 'found' flag to prevent overwriting if we continue checking.
    found = 0
    
    for lvl in tl.static_range(1, MAX_LEVEL + 1):
        # 1. Calculate Bounds CONSTANTLY (No accumulation dependency)
        # Level starts at: N - N/(2^(L-1))
        # Level ends at:   N - N/(2^L)
        # Since N and lvl are constants, Triton calculates these at compile time.
        
        # e.g. Level 1: Start=0, End=N/2
        # e.g. Level 2: Start=N/2, End=3N/4
        lvl_start = N - (N >> (lvl - 1))
        lvl_end   = N - (N >> lvl)
        
        # 2. Check (Pure Comparison, No Arithmetic)
        # We check 'if not found' first to simulate an 'else-if' chain efficiently
        if found == 0:
            # We only need to check the upper bound 'lvl_end' 
            # because the lower bound is implicitly handled by the previous iteration's failure
            # (or it's 0 for the first level).
            if pid < lvl_end:
                target_level = lvl
                # We already calculated node_id = N + pid outside!
                found = 1

    # ------------------------------------------------------------------
    # 2. GATHER LOGIC (With Early Stop)
    # ------------------------------------------------------------------  
    # Gather Table Lookup
    tab_ptr = Gather_Table_ptr + (node_id * sg_node)
    child_start_base = tl.load(tab_ptr + 0)

    # [OPTIMIZATION] EARLY STOP
    # If this node has no children, we are done. 
    # Since dK and dV are initialized to 0.0 by torch.zeros_like(), 
    # we don't need to write anything.
    if child_start_base == -1:
        return

    # --- Everything below is SKIPPED for leaf/invalid nodes ---

    num_children = 1 << target_level
    w_idx = target_level + 1

    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    
    off_out_base = (b_idx * sdk_b) + (node_id * sdk_node) + (offs_h[:, None] * sdk_h)
    
    ptr_ds_base = DS_ptr + (b_idx * sds_b) + (w_idx * sds_lvl)
    ptr_w_base  = W_ptr  + (b_idx * sw_b)  + (w_idx * sw_lvl)
    ptr_q_base  = Q_ptr  + (b_idx * sq_b)
    ptr_do_base = DO_ptr + (b_idx * sdo_b)

    for off_d_start in range(0, D, BLOCK_D):
        offs_d = off_d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]

        dk_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

        # We already checked has_children above, so we know it's True here
        off_hq_d  = (offs_h[:, None] * sq_h)  + (offs_d[None, :] * sq_d)
        off_hdo_d = (offs_h[:, None] * sdo_h) + (offs_d[None, :] * sdo_d)

        for k in range(num_children):
            child_idx = child_start_base + k
            
            ptr_ds = ptr_ds_base + (child_idx * sds_n) + (offs_h * sds_h)
            ptr_w  = ptr_w_base  + (child_idx * sw_n)  + (offs_h * sw_h)
            ds = tl.load(ptr_ds, mask=mask_h, other=0.0)[:, None]
            w  = tl.load(ptr_w,  mask=mask_h, other=0.0)[:, None]
            
            ptr_q  = ptr_q_base  + (child_idx * sq_n)  + off_hq_d
            ptr_do = ptr_do_base + (child_idx * sdo_n) + off_hdo_d
            q  = tl.load(ptr_q,  mask=mask_op, other=0.0)
            do = tl.load(ptr_do, mask=mask_op, other=0.0)
            
            #dk_acc += ds * q
            #dv_acc += w * do
            dk_acc += ds * q.to(tl.float32)
            dv_acc += w * do.to(tl.float32)

        # Store Result
        ptr_dk = DK_ptr + off_out_base + (offs_d[None, :] * sdk_d)
        ptr_dv = DV_ptr + off_out_base + (offs_d[None, :] * sdk_d)
        tl.store(ptr_dk, dk_acc, mask=mask_op)
        tl.store(ptr_dv, dv_acc, mask=mask_op)

@triton.jit
def hierarchical_attention_backward_high_level_kernel(
    # Inputs
    DS_ptr, Q_ptr, W_ptr, DO_ptr, Gather_Table_ptr,
    DK_ptr, DV_ptr,
    # Strides
    sds_b, sds_n, sds_h, sds_lvl,
    sq_b, sq_n, sq_h, sq_d,
    sw_b, sw_n, sw_h, sw_lvl,
    sdo_b, sdo_n, sdo_h, sdo_d,
    sdk_b, sdk_node, sdk_h, sdk_d,
    sg_node, sg_dim,
    # Constants
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    N: tl.constexpr,
    START_LEVEL: tl.constexpr
):
    pid = tl.program_id(0)
    b_idx = tl.program_id(1)

    # ------------------------------------------------------------------
    # 1. BITWISE PID DECODING (Extreme Optimization)
    # ------------------------------------------------------------------
    # Constants derived from N and START_LEVEL
    # SHIFT_GRID = log2(BLOCKS_PER_LVL) = log2(N) - (START_LEVEL - 1)
    # Note: We rely on the compiler to fold these constants.
    
    # Blocks Per Level = N / 2^(START_LEVEL - 1)
    BLOCKS_PER_LVL: tl.constexpr = N >> (START_LEVEL - 1)
    BLOCK_MASK: tl.constexpr = BLOCKS_PER_LVL - 1
    
    # 1. Which Level? (Integer Div -> Right Shift?)
    # Since BLOCKS_PER_LVL is power of 2, 'pid // BLOCKS' is a Shift.
    # However, 'pid' is dynamic, so we can just use integer div, 
    # Triton/LLVM optimizes 'div by power-of-2' into a shift automatically.
    # But for 'rem', using AND is explicitly cleaner.
    
    lvl_offset = pid // BLOCKS_PER_LVL
    target_level = START_LEVEL + lvl_offset
    
    # 2. Relative Index (Modulo -> Bitwise AND)
    rem = pid & BLOCK_MASK
    
    # 3. Calculate Split-K Parameters
    # shift_val = L - (START_LEVEL - 1)
    shift_val = target_level - (START_LEVEL - 1)
    
    # split_k = 1 << shift_val. 
    # split_k_mask = split_k - 1 = (1 << shift_val) - 1.
    split_k_mask = (1 << shift_val) - 1
    
    # 4. Decode Node vs Split (Bitwise)
    node_local = rem >> shift_val           # Division by split_k
    split_id   = rem & split_k_mask         # Modulo split_k
    
    # 5. Global Node ID
    start_node_global = (2 * N) - (N >> (target_level - 1))
    node_id = start_node_global + node_local

    # ------------------------------------------------------------------
    # 2. EARLY EXIT (The Optimization)
    # ------------------------------------------------------------------
    # Load Table Entry FIRST.
    tab_ptr = Gather_Table_ptr + (node_id * sg_node)
    child_start_base = tl.load(tab_ptr + 0)
    
    # If -1, this node is empty. Exit immediately.
    # This prevents all subsequent math, pointer arithmetic, and atomic locking.
    if child_start_base == -1:
        return

    # ------------------------------------------------------------------
    # 3. CONSTANT LOOP SETUP
    # ------------------------------------------------------------------
    # Only reached if node has children.
    
    # Ratio = 2^L / 2^(L - (START-1)) = 2^(START-1)
    CHILDREN_PER_SPLIT: tl.constexpr = 1 << (START_LEVEL - 1)
    
    w_idx = target_level + 1
    
    # Start K
    start_k = split_id * CHILDREN_PER_SPLIT

    # Base Pointers (Hoisted)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    off_out_base = (b_idx * sdk_b) + (node_id * sdk_node) + (offs_h[:, None] * sdk_h)
    
    ptr_ds_base = DS_ptr + (b_idx * sds_b) + (w_idx * sds_lvl)
    ptr_w_base  = W_ptr  + (b_idx * sw_b)  + (w_idx * sw_lvl)
    ptr_q_base  = Q_ptr  + (b_idx * sq_b)
    ptr_do_base = DO_ptr + (b_idx * sdo_b)

    # ------------------------------------------------------------------
    # 4. MAIN LOOP
    # ------------------------------------------------------------------
    for off_d_start in range(0, D, BLOCK_D):
        offs_d = off_d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]

        dk_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

        # Note: We don't need 'if has_children' check here because 
        # we would have returned above if it was False.
        off_hq_d  = (offs_h[:, None] * sq_h)  + (offs_d[None, :] * sq_d)
        off_hdo_d = (offs_h[:, None] * sdo_h) + (offs_d[None, :] * sdo_d)

        # Constant Unrolled Loop
        for k_offset in range(CHILDREN_PER_SPLIT):
            # Calculate offsets (Pure ALU, no branches)
            k = start_k + k_offset
            child_idx = child_start_base + k
            
            # Pointers
            ptr_ds = ptr_ds_base + (child_idx * sds_n) + (offs_h * sds_h)
            ptr_w  = ptr_w_base  + (child_idx * sw_n)  + (offs_h * sw_h)
            
            # Load
            ds = tl.load(ptr_ds, mask=mask_h, other=0.0)[:, None]
            w  = tl.load(ptr_w,  mask=mask_h, other=0.0)[:, None]
            
            ptr_q  = ptr_q_base  + (child_idx * sq_n)  + off_hq_d
            ptr_do = ptr_do_base + (child_idx * sdo_n) + off_hdo_d
            
            q  = tl.load(ptr_q,  mask=mask_op, other=0.0)
            do = tl.load(ptr_do, mask=mask_op, other=0.0)
            
            # FMA
            #dk_acc += ds * q
            #dv_acc += w * do
            dk_acc += ds * q.to(tl.float32)
            dv_acc += w * do.to(tl.float32)

        # [ATOMIC STORE] 
        # We only reach here if children exist, so we never atomic_add to empty nodes.
        ptr_dk = DK_ptr + off_out_base + (offs_d[None, :] * sdk_d)
        ptr_dv = DV_ptr + off_out_base + (offs_d[None, :] * sdk_d)
        tl.atomic_add(ptr_dk, dk_acc, mask=mask_op)
        tl.atomic_add(ptr_dv, dv_acc, mask=mask_op)

# ------------------------------------------------------------------
#  Backward Kernel 3: Compute dQ (Small Kernel)
# ------------------------------------------------------------------
@triton.jit
def hierarchical_attention_backward_dQ_kernel(
    DS_ptr, K_ptr, Lookup_ptr, DQ_ptr, Mask_ptr,
    sds_b, sds_n, sds_h, sds_lvl,
    sk_b, sk_n, sk_h, sk_d,
    sl_n, sl_lvl,
    sdq_b, sdq_n, sdq_h, sdq_d,
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, 
    BLOCK_D: tl.constexpr,
    LEVELS: tl.constexpr, 
    HAS_MASK: tl.constexpr
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    h_idx = tl.arange(0, BLOCK_H)
    mask_h = h_idx < H

    # -----------------------------------------------------------
    # 1. Pre-calculate Base Pointers
    # -----------------------------------------------------------
    # DS Base: [Node, Head]
    ds_base = DS_ptr + (b_idx * sds_b) + (node_idx * sds_n) + (h_idx * sds_h)
    
    # DQ Base: [Node, Head]
    dq_base = DQ_ptr + (b_idx * sdq_b) + (node_idx * sdq_n) + (h_idx[:, None] * sdq_h)

    # K Batch Base: [Batch]
    k_batch_base = K_ptr + b_idx * sk_b

    # Pre-load Self DS (reused across D loop)
    # Shape: [BLOCK_H]
    ds_self = tl.load(ds_base + (0 * sds_lvl), mask=mask_h, other=0.0)

    # -----------------------------------------------------------
    # 2. Outer Loop over D (Chunked for Registers)
    # -----------------------------------------------------------
    for off_d_start in range(0, D, BLOCK_D):
        offs_d = off_d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]

        # -------------------------------------------------------
        # A. Process Self (Level 0)
        # -------------------------------------------------------
        # K Self Pointer: [Node, Head, D_Chunk]
        off_k_self = (node_idx * sk_n) + \
                     (h_idx[:, None] * sk_h) + \
                     (offs_d[None, :] * sk_d)
                     
        k_self = tl.load(k_batch_base + off_k_self, mask=mask_op, other=0.0)

        # Init Accumulator (FP32 for Precision)
        # dQ = dS_self * K_self
        dq_acc = ds_self[:, None].to(tl.float32) * k_self.to(tl.float32)

        # -------------------------------------------------------
        # B. Inner Loop over Levels (The Optimization)
        # -------------------------------------------------------
        for lvl_idx in range(LEVELS):
            # 1. Load Topology for this level
            off_lookup = node_idx * sl_n + lvl_idx * sl_lvl
            
            # Since 'node_idx' and 'lvl_idx' are valid, no mask needed for lookup
            p_idx = tl.load(Lookup_ptr + off_lookup)

            # 2. Check Validity (Mask + Topology)
            is_valid = (p_idx != -1)
            if HAS_MASK:
                mask_val = tl.load(Mask_ptr + off_lookup).to(tl.int8)
                is_valid = is_valid & (mask_val == 0)

            # 3. Load dS for this level [BLOCK_H]
            # Offset: (lvl + 1) because level 0 is Self
            ds_ptr_lvl = ds_base + ((lvl_idx + 1) * sds_lvl)
            
            # Mask logic: Head must be valid AND Edge must be valid
            mask_load = mask_h & is_valid
            
            ds_cross = tl.load(ds_ptr_lvl, mask=mask_load, other=0.0)

            # 4. Load K for this Parent [BLOCK_H, BLOCK_D]
            # Safe Parent Index (redirect invalid to 0)
            safe_p_idx = tl.where(is_valid, p_idx, 0)
            
            # Pointer: [Parent_Node, Head, D_Chunk]
            off_k_cross = (safe_p_idx * sk_n) + \
                          (h_idx[:, None] * sk_h) + \
                          (offs_d[None, :] * sk_d)
            
            # Mask logic for K: (Head & Valid_Edge & Dim)
            mask_k = mask_load[:, None] & mask_d[None, :]
            
            k_cross = tl.load(k_batch_base + off_k_cross, mask=mask_k, other=0.0)

            # 5. Accumulate (FP32)
            # dq += dS_cross * K_cross
            dq_acc += ds_cross[:, None].to(tl.float32) * k_cross.to(tl.float32)

        # -------------------------------------------------------
        # C. Write Result
        # -------------------------------------------------------
        # Cast back to original dtype
        off_dq_out = offs_d[None, :] * sdq_d
        tl.store(dq_base + off_dq_out, dq_acc.to(DQ_ptr.dtype.element_ty), mask=mask_op)

class HierarchicalAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, idx_table, gather_table, mask_table=None):
        # Alignment checks
        Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous()
        idx_table = idx_table.contiguous()
        if mask_table is not None: mask_table = mask_table.contiguous()

        B, N, H, D = Q.shape
        LEVELS = idx_table.shape[1]
        
        Out = torch.empty_like(Q)
        
        # Save weights for backward: [B, N, H, 1 + LEVELS]
        Weights = torch.empty((B, N, H, 1 + LEVELS), device=Q.device, dtype=torch.float32)
        
        HAS_MASK = (mask_table is not None)
        mask_ptr_safe = mask_table if HAS_MASK else Q # Dummy ptr
        
        grid = (N, B)
        BLOCK_H = triton.next_power_of_2(H)
        BLOCK_LEVELS = triton.next_power_of_2(LEVELS)
        BLOCK_D = min(64, triton.next_power_of_2(D))
        sm_scale = 1.0 / math.sqrt(D)
        
        hierarchical_attention_forward_kernel[grid](
            Q, K, V,
            idx_table, mask_ptr_safe,
            Out, Weights,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *idx_table.stride(),
            *Out.stride(), *Weights.stride(),
            sm_scale=sm_scale,
            H=H, BLOCK_H=BLOCK_H,
            D=D, LEVELS=LEVELS,
            BLOCK_D=BLOCK_D, BLOCK_LEVELS=BLOCK_LEVELS,
            HAS_MASK=HAS_MASK,
            num_warps=2
        )
        
        # [UPDATE] Save mask_table for backward. PyTorch handles 'None' correctly.
        ctx.save_for_backward(Q, K, V, idx_table, gather_table, Weights, mask_table)
        ctx.constants = (sm_scale, H, BLOCK_H, D, BLOCK_D, LEVELS, BLOCK_LEVELS)
        return Out

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Retrieve Tensors
        Q, K, V, idx_table, gather_table, Weights, mask_table = ctx.saved_tensors
        sm_scale, H, BLOCK_H, D, BLOCK_D, LEVELS, BLOCK_LEVELS = ctx.constants

        # View as 4D
        grad_output = grad_output.contiguous()
        B, N = Q.shape[0], Q.shape[1]
        grad_output_4d = grad_output.view(B, N, H, D)
        
        # 2. Compute dS (Main Stream)
        DS = torch.empty_like(Weights)
        grid_ds = (N, B)
        HAS_MASK = (mask_table is not None)
        mask_ptr_safe = mask_table if HAS_MASK else Weights
        
        hierarchical_attention_backward_dS_kernel[grid_ds](
            grad_output_4d, Weights, V, idx_table, DS, mask_ptr_safe,
            *grad_output_4d.stride(), *Weights.stride(), *V.stride(), 
            *idx_table.stride(), *DS.stride(),            
            sm_scale, H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_D=32, 
            LEVELS=LEVELS, BLOCK_LEVELS=BLOCK_LEVELS, HAS_MASK=HAS_MASK, num_warps=2
        )

        # --- SETUP PARALLELISM ---
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        dQ = torch.empty_like(Q)

        # --- BRANCH 2: dK/dV (Dependent on dS) ---
        
        # Step A: Leaf Kernel (Level 0)
        grid_leaf = (N, B)
        hierarchical_attention_backward_dK_dV_leaf_kernel[grid_leaf](
            DS, Q, Weights, grad_output_4d, gather_table,
            dK, dV,
            *DS.stride(), *Q.stride(), *Weights.stride(), 
            *grad_output_4d.stride(), *dK.stride(), *gather_table.stride(),
            H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_D=BLOCK_D, num_warps=2
        )

        # --- Dynamic CUTOFF_LEVEL Logic ---
        # Default is 6, adjusted based on sequence length N
        if N <= 1025:
            CUTOFF_LEVEL = 5
        elif N <= 2048:
            CUTOFF_LEVEL = 6
        elif N <= 4096:
            CUTOFF_LEVEL = 8
        else:
            # Covers large scale N (e.g., 2048*64, 2048*256)
            CUTOFF_LEVEL = 10
        
        # --- KERNEL A: Low Levels (Split=1) ---
        if LEVELS >= 1:
            limit = min(LEVELS, CUTOFF_LEVEL)
            # Total blocks = N - (N >> limit)
            total_blocks_low = N - (N >> limit)
            
            grid_low = (total_blocks_low, B)
            
            hierarchical_attention_backward_low_level_kernel[grid_low](
                DS, Q, Weights, grad_output_4d, gather_table,
                dK, dV,
                *DS.stride(), *Q.stride(), *Weights.stride(),
                *grad_output_4d.stride(), *dK.stride(), *gather_table.stride(),
                H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_D=BLOCK_D,
                N=N, 
                MAX_LEVEL=limit, 
                num_warps=4
            )
        
        # --- KERNEL B: High Levels (Split>1) ---
        if LEVELS > CUTOFF_LEVEL:
            num_high_levels = LEVELS - CUTOFF_LEVEL
            
            # Constant blocks per level = N >> (CUTOFF)
            # Actually, logic dictates: N >> (START_LEVEL - 1)
            # If START_LEVEL=9, we need N >> 8.
            blocks_per_lvl = N >> CUTOFF_LEVEL
            
            total_blocks_high = blocks_per_lvl * num_high_levels
            
            grid_high = (total_blocks_high, B)
            
            hierarchical_attention_backward_high_level_kernel[grid_high](
                DS, Q, Weights, grad_output_4d, gather_table,
                dK, dV,
                *DS.stride(), *Q.stride(), *Weights.stride(),
                *grad_output_4d.stride(), *dK.stride(), *gather_table.stride(),
                H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_D=BLOCK_D,
                N=N,
                START_LEVEL=CUTOFF_LEVEL + 1,
                num_warps=2
            )

        # --- BRANCH 1: dQ (Independent) ---
        grid_dq = (N, B)
        hierarchical_attention_backward_dQ_kernel[grid_dq](
            DS, K, idx_table, dQ, mask_ptr_safe,
            *DS.stride(), *K.stride(), *idx_table.stride(), *dQ.stride(),
            H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_D=32, LEVELS=LEVELS,
            HAS_MASK=HAS_MASK, num_warps=2
        )
            
        return dQ, dK, dV, None, None, None

def hierarchical_fused_attention(Q, K, V, idx_table, gather_table, mask_table=None):
    """
    Wrapper for the custom autograd function.
    """
    return HierarchicalAttentionFunc.apply(Q, K, V, idx_table, gather_table, mask_table)


## ============================================================================================= ##
## ============================================================================================= ##
## ============================================================================================= ##
## ============================================================================================= ##


def build_tree_topology(seq_len, is_causal=True, device="cuda", dtype=torch.int32):
    """
    Builds ALL topology tables needed for Hierarchical Attention (Forward & Backward).
    
    Args:
        seq_len (int): Number of leaf tokens (must be power of 2).
        is_causal (bool): Forward/Backward causal masking mode.
        
    Returns:
        dict: {
            "forward_idx": [seq_len, levels] int32 (Who leaves attend to),
            "forward_mask": [seq_len, levels] bool (Causal mask),
            "backward_gather": [total_nodes, 3] int32 (Start, End, Level for Gather)
        }
    """
    assert (seq_len & (seq_len - 1)) == 0, "seq_len must be power of 2"
    
    total_nodes = 2 * seq_len - 1
    level_num = int(math.log2(seq_len))
    max_valid = total_nodes - 2
    
    # ===============================================================
    # 1. Forward Table Construction (Leaf -> Neighbors)
    # ===============================================================
    forward_idx = torch.full((seq_len, level_num), -1, dtype=dtype, device=device)
    forward_mask = torch.zeros((seq_len, level_num), dtype=torch.bool, device=device)
    
    # Vectorized "Climb Up" Logic
    n_cur = torch.arange(seq_len, device=device, dtype=torch.int64)
    
    for lvl in range(level_num):
        if lvl == 0:
            n_next = n_cur ^ 1
            pair = n_cur
        else:
            # Parent of n is (n // 2 + seq_len)
            parent = (n_cur // 2 + seq_len)
            n_next = parent ^ 1 # Sibling of parent
            pair = parent

        valid_mask = n_next <= max_valid
        
        # Causal Logic: pair < n_next means neighbor is in the Future
        is_future = pair < n_next
        
        # Store Index
        forward_idx[:, lvl] = torch.where(valid_mask, n_next.to(dtype), forward_idx[:, lvl])
        
        # Store Mask (If causal=True, block future connections)
        if is_causal:
            forward_mask[:, lvl] = torch.where(valid_mask, is_future, forward_mask[:, lvl])
        
        n_cur = n_next # Climb

    # ===============================================================
    # 2. Backward Table Construction (Node -> Leaf Range)
    # ===============================================================
    subtree_ranges = torch.zeros((total_nodes, 2), dtype=torch.int64, device=device)
    node_levels = torch.zeros(total_nodes, dtype=dtype, device=device)
    
    # Initialize Leaves (Level 0)
    leaves = torch.arange(seq_len, device=device)
    subtree_ranges[:seq_len, 0] = leaves
    subtree_ranges[:seq_len, 1] = leaves + 1
    # Leaves are effectively Level 0 (Self/Sibling interaction)
    node_levels[:seq_len] = 0 
    
    curr_start = 0
    curr_count = seq_len
    
    for lvl in range(level_num):
        next_start = curr_start + curr_count
        next_count = curr_count // 2
        
        parents = torch.arange(next_start, next_start + next_count, device=device)
        indices = torch.arange(next_count, device=device)
        left_children = curr_start + 2 * indices
        right_children = curr_start + 2 * indices + 1
        
        subtree_ranges[parents, 0] = subtree_ranges[left_children, 0]
        subtree_ranges[parents, 1] = subtree_ranges[right_children, 1]
        
        # --- CRITICAL FIX ---
        # Old: node_levels[parents] = lvl 
        # New: node_levels[parents] = lvl + 1
        # Reason: 
        # Leaves (lvl 0 in loop) handle forward_idx[:, 0] -> stored as Level 0
        # Parents created in lvl=0 loop handle forward_idx[:, 1] -> stored as Level 1
        node_levels[parents] = lvl + 1
        # --------------------
        
        curr_start = next_start
        curr_count = next_count

    # Gather Logic
    all_nodes = torch.arange(total_nodes, device=device)
    siblings = all_nodes ^ 1
    siblings = torch.clamp(siblings, max=total_nodes-1)

    gather_info = torch.full((total_nodes, 3), -1, dtype=dtype, device=device)

    if is_causal:
        should_gather = siblings > all_nodes
    else:
        should_gather = siblings != all_nodes

    # Mask out Root (it has no valid sibling to gather from)
    should_gather = should_gather & (siblings != all_nodes)

    valid_nodes = all_nodes[should_gather]
    valid_siblings = siblings[should_gather]

    gather_info[valid_nodes, 0] = subtree_ranges[valid_siblings, 0].to(dtype)
    gather_info[valid_nodes, 1] = subtree_ranges[valid_siblings, 1].to(dtype)
    gather_info[valid_nodes, 2] = node_levels[valid_nodes] # Now uses the corrected levels


    ## ===============================================================
    ## 3. DEBUG: Print Line by Line
    ## ===============================================================
    #print("\n=== Gather Info Table ===")
    #print(f"{'Node':<6} | {'Start':<6} | {'End':<6} | {'Level':<6}")
    #print("-" * 30)
    #
    ## Convert to CPU list for clean iteration
    #gi_cpu = gather_info.detach().cpu().numpy()
    #
    #for i in range(total_nodes):
    #    s, e, l = gi_cpu[i]
    #    # Only print nodes that actually gather something (optional, remove 'if' to see all)
    #    if s != -1: 
    #        print(f"{i:<6} | {s:<6} | {e:<6} | {l:<6}")
    #    else:
    #        print(f"{i:<6} | {'-1':<6} | {'-1':<6} | {'-1':<6}")
    #        
    #print("=========================\n")


    return {
        "forward_idx": forward_idx,
        "forward_mask": forward_mask,
        "backward_gather": gather_info
    }

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

        # --- Topology Cache ---
        # We cache the entire dictionary returned by build_tree_topology
        self.cached_tables = None       
        self.cached_seq_len = -1
        self.cached_is_causal = None  # To invalidate cache if switching causal/non-causal

        self.sizes = None
        self.offsets = None

    def _get_lookup_table(self, L, is_causal, device):
        """
        Smart retrieval: Returns cached topology tables if (L, mode, device) match.
        """
        # 1. Check if cache is valid
        if (self.cached_tables is not None and 
            self.cached_seq_len == L and 
            self.cached_is_causal == is_causal and
            self.cached_tables['forward_idx'].device == device):
            return self.cached_tables

        # 2. Recompute using the unified builder
        # This builds Forward Indices, Forward Masks, and Backward Gather Ranges
        tables = build_tree_topology(L, is_causal=is_causal, device=device, dtype=torch.int32)
        
        # 3. Update Cache
        self.cached_tables = tables
        self.cached_seq_len = L
        self.cached_is_causal = is_causal
        
        return tables

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
          - [FIX #1]: Use torch.split to avoid SliceBackward explosion.
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        
        assert y_in is not None, "y_in cannot be None"
        assert y_in.size(1) == N - 1, f"y_in size {y_in.size(1)} mismatch! Expected N-1 ({N-1}) for N={N} leaves."

        if self.sizes is None: self.sizes, self.offsets = self.build_level_info(N)

        # -------------------------------------------------------
        # 1. Global Parent Projection
        # -------------------------------------------------------
        # We project ALL parents at once for Q, K, and V.
        # This allows the parent to attend to itself (Parent-Self Attention).
        # Layout: [B, Total_Parents, H, Dh] (Native Linear output, no transpose)
        Q_p_all = self.Wq_y(y_in).view(B, -1, H, Dh)
        K_p_all = self.Wk_y(y_in).view(B, -1, H, Dh)
        V_p_all = self.Wv_y(y_in).view(B, -1, H, Dh)
        
        # [FIX #1] Pre-split tensors to avoid creating hundreds of slice views in the loop
        # This creates 1 SplitBackward node instead of N SliceBackward nodes.
        Q_p_levels = torch.split(Q_p_all, self.sizes, dim=1)
        K_p_levels = torch.split(K_p_all, self.sizes, dim=1)
        V_p_levels = torch.split(V_p_all, self.sizes, dim=1)
        
        new_Y_levels = []
        
        prev_sources = x # Starts as the leaves (first layer children)
        
        # [OPTIMIZATION] Project Level 0 Leaves ONCE here.
        # This removes the projection overhead for 50% of the total nodes.
        K_leaves = self.Wk_y(prev_sources).view(B, -1, H, Dh)
        V_leaves = self.Wv_y(prev_sources).view(B, -1, H, Dh)
        V_leaves = self.dropout(V_leaves)

        for level, parent_count in enumerate(self.sizes):
            offset = self.offsets[level]

            # ---------------------------------------------------
            # 1. Prepare Parents (Using Pre-Split Tensors)
            # ---------------------------------------------------
            # [FIX #1] Access list instead of slicing tensor
            Q_p = Q_p_levels[level]
            K_p = K_p_levels[level]
            V_p = V_p_levels[level]
            
            # ---------------------------------------------------
            # 2. Prepare Children (Projection)
            # ---------------------------------------------------
            if level == 0:
                # FAST PATH: Use pre-projected leaves
                K_c = K_leaves
                V_c = V_leaves
                V_c = self.dropout(V_c)
            else:
                # STANDARD PATH: Project output of previous level
                # Must slice prev_sources because it is a changing reference
                children_in = prev_sources 
                
                # Project Children Keys/Values
                K_c = self.Wk_y(children_in).view(B, -1, H, Dh)
                V_c = self.Wv_y(children_in).view(B, -1, H, Dh)      
                V_c = self.dropout(V_c)

            # ---------------------------------------------------
            # 3. Triton Kernel (3-Way Attention)
            # ---------------------------------------------------
            # Replaces: Reshape -> Cat -> Dot -> Softmax -> Weighted Sum
            updated_heads = build_parent_nodes(Q_p, K_p, V_p, K_c, V_c)

            # ---------------------------------------------------
            # 4. Merge Heads
            # ---------------------------------------------------
            updated_merged = updated_heads.reshape(B, parent_count, D)
            new_Y_levels.append(updated_merged)
            
            # Update pointer for next level
            prev_sources = updated_merged

        # Concatenate and Project
        Y_new = torch.cat(new_Y_levels, dim=1)
        # Concatenate and Project
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
        if not Q.is_contiguous(): Q = Q.contiguous()
        if not K_full.is_contiguous(): K_full = K_full.contiguous()
        if not V_full.is_contiguous(): V_full = V_full.contiguous()

        # 4. Get Topology Tables
        # Determine causality based on user input (mask presence implies causal)
        is_causal = (mask is not None)
        
        tables = self._get_lookup_table(N, is_causal, device=x.device)
        
        idx_table = tables["forward_idx"]
        neighbor_mask = tables["forward_mask"]
        gather_table = tables["backward_gather"]
        
        # If active_mask is None, kernel assumes full visibility.
        # If is_causal, we pass the pre-computed mask from the table.
        active_mask = neighbor_mask if is_causal else None

        output_leaf_heads = hierarchical_fused_attention(
            Q, K_full, V_full, 
            idx_table,      # Forward Topology
            gather_table,
            active_mask     # Mask Table
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

class HierarchicalSparseAttention(nn.Module):
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

        # --- Topology Cache ---
        # We cache the entire dictionary returned by build_tree_topology
        self.cached_tables = None       
        self.cached_seq_len = -1
        self.cached_is_causal = None  # To invalidate cache if switching causal/non-causal

        self.sizes = None
        self.offsets = None

    def _get_lookup_table(self, L, is_causal, device):
        """
        Smart retrieval: Returns cached topology tables if (L, mode, device) match.
        """
        # 1. Check if cache is valid
        if (self.cached_tables is not None and 
            self.cached_seq_len == L and 
            self.cached_is_causal == is_causal and
            self.cached_tables['forward_idx'].device == device):
            return self.cached_tables

        # 2. Recompute using the unified builder
        # This builds Forward Indices, Forward Masks, and Backward Gather Ranges
        tables = build_tree_topology(L, is_causal=is_causal, device=device, dtype=torch.int32)
        
        # 3. Update Cache
        self.cached_tables = tables
        self.cached_seq_len = L
        self.cached_is_causal = is_causal
        
        return tables

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

    def cross_update_Y(self, x, y_in):
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

    def update_X_from_Y(self, x, y, mask=None):
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        if y is None: return x

        # Concatenate inputs once to allow unified K/V calculation
        XY = torch.cat([x, y], dim=1)

        # --- Inline Split Heads ---
        Q = self.Wq_x(x).view(B, N, H, Dh).transpose(1, 2)
        
        # Calculate K, V on combined input
        kv_input = self.Wk_x(XY).view(B, -1, H, Dh).transpose(1, 2)
        v_input = self.Wv_x(XY).view(B, -1, H, Dh).transpose(1, 2)

        K_full = kv_input
        V_full = self.dropout(v_input)

        # [FIX HERE] Pass is_causal based on mask existence
        is_causal = (mask is not None)
        tables = self._get_lookup_table(N, is_causal=is_causal, device=x.device)

        idx_table = tables["forward_idx"]
        neighbor_causal_mask = tables["forward_mask"]
            
        # Self: The leaves attending to themselves
        K_self = K_full[:, :, :N, :]                  
        V_self = V_full[:, :, :N, :]                  

        # Gather neighbors using index table
        # Note: -1 indices wrap to the last element, but masked_fill below handles validity
        gather_indices = idx_table.long() 
        
        neighbors_k = K_full[:, :, gather_indices, :] 
        neighbors_v = V_full[:, :, gather_indices, :]

        # Compute Self Logits
        self_logits = torch.einsum('b h n d, b h n d -> b h n', Q, K_self) / math.sqrt(Dh)

        # Compute Neighbor Logits
        neighbors_logits = torch.einsum('b h l x d, b h l n d -> b h l n', Q.unsqueeze(3), neighbors_k)
        neighbors_logits = neighbors_logits / math.sqrt(Dh)

        # Apply Causal Mask
        if mask is not None:
            neighbors_logits = neighbors_logits.masked_fill(neighbor_causal_mask, float('-inf'))

        # Concatenate (Self + Neighbors)
        all_v = torch.cat([V_self.unsqueeze(3), neighbors_v], dim=3)              
        all_logits = torch.cat([self_logits.unsqueeze(3), neighbors_logits], dim=3) 

        # Attention Softmax & Weighted Sum
        max_logits = all_logits.max(dim=-1, keepdim=True)[0]
        weights = F.softmax(all_logits - max_logits, dim=-1)              
            
        output_leaf = torch.einsum('b h l n, b h l n d -> b h l d', weights, all_v)

        # --- Inline Merge Heads ---
        return output_leaf.transpose(1, 2).reshape(B, N, D)

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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=80000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (part of state_dict but not a trained parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        # self.pe: [1, Max_Len, Dim] -> Sliced to [1, Seq_Len, Dim]
        # Broadcasting adds pe to every item in the batch automatically
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- IMPROVED DECODER LAYER (Pre-LN Architecture) ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, triton=False, dropout=0.1):
        super().__init__()
        # Note: dim=d_model to match your previous code hyperparameters
        if triton:
            self.self_attn = HierarchicalSparseAttentionTriton(d_model, num_heads, dropout)
        else:
            self.self_attn = HierarchicalSparseAttention(d_model, num_heads, dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # --- NEW: Norm for Y ---
        self.norm_y = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y, mask=None, return_attention=False):
 
        # 1. Update Y (Hierarchy) with Residual + Norm
        # We use 'norm_y(y)' as input to be safe, similar to Pre-LN for x
        y_norm = self.norm_y(y)
        
        # Calculate the update (delta)
        y_delta = self.self_attn.cross_update_Y(x, y_in=y_norm)
        
        # Apply Residual Connection to Y
        y_next = y + self.dropout(y_delta)

    
        # PRE-LAYER NORMALIZATION (Apply Norm BEFORE Attention)
        # This significantly improves stability and convergence speed
        
        # Norm -> Attention -> Add
        norm_x = self.norm1(x)

        if return_attention:
            attn_output, self_attn_weights = self.self_attn(norm_x, norm_x, norm_x, y=y_next, mask=mask, return_attention=True)
        else:
            attn_output = self.self_attn(norm_x, norm_x, norm_x, y=y_next, mask=mask)
            
        x = x + self.dropout(attn_output) # Residual
        
        # FF
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        if return_attention:
            return x, y_next, self_attn_weights
        return x, y_next

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, triton=False, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, triton, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Weight Tying (Optional but good for PPL)
        self.fc_out.weight = self.embedding.weight
        self.d_model = d_model
        
        # --- Apply Initialization ---
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        # Initialize weights with small std (0.02) to prevent high starting loss
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def make_causal_mask(self, x):
        seq_len = x.shape[1]
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, trg, return_attention=False):
        x = self.embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Initialize Y from the embeddings using the static method
        y = HierarchicalSparseAttention.generate_span_input_Y(x)
        
        trg_mask = self.make_causal_mask(trg)

        attentions = [] if return_attention else None
        
        for layer in self.layers:
            if return_attention:
                x, y, attention = layer(x, y, mask=None, return_attention=True)
                attentions.append(attention)
            else:
                x, y = layer(x, y, mask=trg_mask)
        
        output = self.fc_out(x)
        return output

from torch.profiler import profile, record_function, ProfilerActivity

import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def print_header(msg):
    print(f"\n{'='*60}\n {msg}\n{'='*60}")

def get_peak_memory_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3)

def run_transformer_benchmark():
    # --------------------------------------------------------------------------
    # 0. CONFIGURATION
    # --------------------------------------------------------------------------
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Use the device defined in your environment/kernels
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)

    # Hyperparameters
    VOCAB_SIZE = 50000
    D_MODEL = 768
    NUM_HEADS = 12
    D_FF = 3072
    NUM_LAYERS = 12
    DROPOUT = 0.0 
    
    # Batch config
    B = 8    
    SEQ_LEN = 2048
    
    # Variables for the randn snippet
    B_check = B
    N_check = SEQ_LEN
    dim = D_MODEL
    check_dtype = torch.float16  # Matches model.half()
    
    print_header(f"TRANSFORMER BENCHMARK (Random Float Input)\n B={B}, L={SEQ_LEN}, Layers={NUM_LAYERS}\n Device: {torch.cuda.get_device_name(device)}")

    # --------------------------------------------------------------------------
    # 1. SETUP & WEIGHT SYNCHRONIZATION
    # --------------------------------------------------------------------------
    print("Initializing Standard PyTorch Model...")
    model_ref = TransformerLM(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS, 
        d_ff=D_FF, num_layers=NUM_LAYERS, triton=False, dropout=DROPOUT
    ).to(device).half() 

    print("Initializing Triton Optimized Model...")
    model_opt = TransformerLM(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS, 
        d_ff=D_FF, num_layers=NUM_LAYERS, triton=True, dropout=DROPOUT
    ).to(device).half() 

    print("Synchronizing weights for Sanity Check...")
    model_opt.load_state_dict(model_ref.state_dict())

    # --- INPUT GENERATION (RANDN) ---
    # Using the requested snippet with mapped variables
    x = torch.randn(B_check, N_check, dim, device='cuda', dtype=check_dtype).clamp(-1, 1)

    # --------------------------------------------------------------------------
    # HELPER: Forward Wrapper (Handles Float Inputs by skipping Embedding)
    # --------------------------------------------------------------------------
    def run_forward(model, x_input):
        if x_input.is_floating_point():
            # Bypass Embedding Layer for raw float inputs
            # Scale x just like the embedding layer would have
            x_local = x_input * math.sqrt(model.d_model)
            x_local = model.pos_encoding(x_local)
            
            # Generate Y hierarchy
            y = HierarchicalSparseAttention.generate_span_input_Y(x_local)
            
            # Create mask (reuse the model's method, but pass dummy x for shape)
            trg_mask = model.make_causal_mask(x_input)
            
            # Run Layers
            for layer in model.layers:
                x_local, y = layer(x_local, y, mask=trg_mask)
            
            return model.fc_out(x_local)
        else:
            # Standard Integer Path
            return model(x_input)

    # --------------------------------------------------------------------------
    # 2. NUMERICAL SANITY CHECK
    # --------------------------------------------------------------------------
    print_header("2. NUMERICAL SANITY CHECK")
    
    # Forward
    print("Running Forward Pass...")
    out_ref = run_forward(model_ref, x)
    out_opt = run_forward(model_opt, x)
    
    diff = (out_ref - out_opt).abs().max().item()
    print(f"Forward Pass Max Difference: {diff:.6f}")
    if diff > 1e-1: 
        print("WARNING: Large forward difference detected!")
    else:
        print(">> Forward Pass Check: PASSED")

    # Backward
    print("Running Backward Pass...")
    grad_output = torch.randn_like(out_ref)
    
    # Check Ref
    model_ref.zero_grad()
    out_ref.backward(grad_output)
    # We check the first Linear layer grad since we skipped embedding
    grad_ref = model_ref.layers[0].feed_forward.fc1.weight.grad.clone()

    # Check Opt
    model_opt.zero_grad()
    out_opt.backward(grad_output)
    grad_opt = model_opt.layers[0].feed_forward.fc1.weight.grad.clone()

    grad_diff = (grad_ref - grad_opt).abs().max().item()
    print(f"Backward Pass (Grad) Max Difference: {grad_diff:.6f}")
    if grad_diff > 1.0: 
        print("WARNING: Large gradient difference detected!")
    else:
        print(">> Backward Pass Check: PASSED")

    del out_ref, out_opt, grad_output, grad_ref, grad_opt
    torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # 3. PROFILING (Chrome Traces)
    # --------------------------------------------------------------------------
    print_header("3. KERNEL PROFILING")
    
    print(f"Profiling Triton Model (1 Step)...")
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            out = run_forward(model_opt, x)
            loss = out.mean()
            loss.backward()

    trace_name = "hierarchical_triton_trace.json"
    prof.export_chrome_trace(trace_name)
    print(f">> Trace exported to '{trace_name}'. Open in chrome://tracing")
    
    # Increase row limit to see small kernels
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
    torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # 4. PERFORMANCE BENCHMARKING
    # --------------------------------------------------------------------------
    print_header("4. PERFORMANCE BENCHMARK (Latency & Memory)")

    def benchmark_model(model, name, steps=10, warmup=5):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Warmup
        for _ in range(warmup):
            out = run_forward(model, x)
            loss = out.mean()
            loss.backward()
            model.zero_grad()
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(steps):
            out = run_forward(model, x)
            loss = out.mean()
            loss.backward()
            model.zero_grad()
        end_event.record()
        torch.cuda.synchronize()
        
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / steps
        max_mem_gb = get_peak_memory_gb()
        
        print(f"[{name}]")
        print(f"  Avg Step Time: {avg_time_ms:.2f} ms")
        print(f"  Peak Memory:   {max_mem_gb:.2f} GB")
        return avg_time_ms, max_mem_gb

    # Run Benchmark
    time_ref, mem_ref = benchmark_model(model_ref, "Standard PyTorch (FP16)")
    time_opt, mem_opt = benchmark_model(model_opt, "Triton Optimized (FP16)")

    # --------------------------------------------------------------------------
    # 5. FINAL SUMMARY
    # --------------------------------------------------------------------------
    print_header("FINAL RESULTS")
    speedup = time_ref / time_opt
    mem_saving = (1 - (mem_opt / mem_ref)) * 100
    
    print(f"{'Metric':<20} | {'PyTorch':<15} | {'Triton':<15} | {'Delta':<10}")
    print("-" * 70)
    print(f"{'Time (ms)':<20} | {time_ref:<15.2f} | {time_opt:<15.2f} | {speedup:.2f}x FASTER")
    print(f"{'Memory (GB)':<20} | {mem_ref:<15.2f} | {mem_opt:<15.2f} | {mem_saving:.2f}% LESS")
    print("-" * 70)

if __name__ == "__main__":
    run_transformer_benchmark()