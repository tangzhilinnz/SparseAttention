import triton
import triton.language as tl

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