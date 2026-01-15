import triton
import triton.language as tl

# ------------------------------------------------------------------
#  Forward Kernel
# ------------------------------------------------------------------
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
    # Strides (V)
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

    # 2. Load Topology (Safe Loading)
    off_lookup = node_idx * sl_n + offs_lvl * sl_lvl
    
    # [FIX] Load with -1 default to detect invalid neighbors
    neighbor_indices = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl, other=-1)
    
    # [FIX] Create validity mask and safe indices for pointer math
    valid_neighbor_mask = (neighbor_indices != -1)
    safe_indices = tl.where(valid_neighbor_mask, neighbor_indices, 0)
    
    neighbor_mask_val = tl.zeros([BLOCK_LEVELS], dtype=tl.int1)
    if HAS_MASK:
        # User confirmed: 1 = Mask Out (Ignore), 0 = Valid (Keep)
        val_int8 = tl.load(Mask_ptr + off_lookup, mask=mask_lvl, other=1).to(tl.int8)
        neighbor_mask_val = (val_int8 != 0)

    # 3. Base Pointers
    k_batch_base = K_ptr + b_idx * sk_b
    v_batch_base = V_ptr + b_idx * sv_b
    
    off_node_self = node_idx * sk_n
    
    # [FIX] Calculate cross offsets using safe indices (redirects -1 to 0)
    off_node_cross_k = safe_indices * sk_n 
    off_node_cross_v = safe_indices * sv_n

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
        # [FIX] Use safe_indices offsets
        ptr_k_cross = k_batch_base + \
                      off_node_cross_k[None, :, None] + \
                      (h_idx[:, None, None] * sk_h) + \
                      (cur_offs_d[None, None, :] * sk_d)
        
        # [FIX] Apply valid_neighbor_mask to the load
        mask_k = mask_h[:, None, None] & mask_lvl[None, :, None] & d_mask[None, None, :] & valid_neighbor_mask[None, :, None]
        k_cross = tl.load(ptr_k_cross, mask=mask_k, other=0.0)
        
        acc_cross += tl.sum(q[:, None, :] * k_cross, axis=2)
        q_ptr += BLOCK_D * sq_d

    # 5. Softmax
    acc_self = acc_self * sm_scale
    acc_cross = acc_cross * sm_scale
    
    # Apply Masking (1 = Mask Out / -inf)
    # [FIX] Mask out if Level is out of bounds OR Neighbor is -1
    mask_broadcast = (offs_lvl >= LEVELS) | (~valid_neighbor_mask)
    
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
        ptr_v_self = v_batch_base + (node_idx * sv_n) + \
                     (h_idx[:, None] * sv_h) + (cur_offs_d[None, :] * sv_d)
        v_self = tl.load(ptr_v_self, mask=mask_op, other=0.0)
        out_acc = w_self[:, None] * v_self
        
        # --- V CROSS ---
        # [FIX] Use safe offsets and valid_mask for V
        ptr_v_cross = v_batch_base + \
                      off_node_cross_v[None, :, None] + \
                      (h_idx[:, None, None] * sv_h) + \
                      (cur_offs_d[None, None, :] * sv_d)
                      
        mask_v = mask_h[:, None, None] & mask_lvl[None, :, None] & d_mask[None, None, :] & valid_neighbor_mask[None, :, None]
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
    HAS_MASK: tl.constexpr
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    h_idx = tl.arange(0, BLOCK_H)
    mask_h = h_idx < H
    offs_lvl = tl.arange(0, BLOCK_LEVELS)
    
    # -----------------------------------------------------------
    # 1. Mask Logic & Topology Load
    # -----------------------------------------------------------
    mask_lvl_bounds = offs_lvl < LEVELS
    off_lookup = node_idx * sl_n + offs_lvl * sl_lvl
    
    # [FIX] Load with -1 default
    neighbor_indices = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl_bounds, other=-1)

    # [FIX] Create Safe Indices
    valid_mask = (neighbor_indices != -1)
    safe_indices = tl.where(valid_mask, neighbor_indices, 0)

    # Combined Mask: (In Bounds) AND (Not -1)
    mask_valid_cross = mask_lvl_bounds & valid_mask
    
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
    w_cross = tl.load(w_base[:, None] + ((1 + offs_lvl[None, :]) * sw_lvl), 
                      mask=mask_h[:, None] & mask_valid_cross[None, :], other=0.0)

    # -----------------------------------------------------------
    # 3. Compute dP = dot(dO, V)
    # -----------------------------------------------------------
    dp_self = tl.zeros([BLOCK_H], dtype=tl.float32)
    dp_cross = tl.zeros([BLOCK_H, BLOCK_LEVELS], dtype=tl.float32)

    v_batch_base = V_ptr + b_idx * sv_b
    do_batch_base = DO_ptr + (b_idx * sdo_b) + (node_idx * sdo_n)
    
    off_node_self = node_idx * sv_n
    # [FIX] Safe Cross Offset
    off_node_cross = safe_indices * sv_n

    for off_d in range(0, D, BLOCK_D):
        offs_d = off_d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]

        # Load dO
        do = tl.load(do_batch_base + (h_idx[:, None]*sdo_h) + (offs_d[None, :]*sdo_d), 
                     mask=mask_op, other=0.0)
        
        # Load V Self
        ptr_v_self = v_batch_base + off_node_self + (h_idx[:, None]*sv_h) + (offs_d[None, :]*sv_d)
        v_self = tl.load(ptr_v_self, mask=mask_op, other=0.0)
        
        # Load V Cross
        # [FIX] Use Safe Indices and Mask
        ptr_v_cross = v_batch_base + off_node_cross[None, :, None] + \
                      (h_idx[:, None, None]*sv_h) + (offs_d[None, None, :]*sv_d)
        
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
    
    # SAFE STORE: We store 0s to invalid slots to clean up memory
    tl.store(ds_cross_ptr, ds_cross, mask=mask_h[:, None] & mask_lvl_bounds[None, :])


# ==================================================================
#  BACKWARD KERNEL 2a: SPECIALIZED LEAF KERNEL (Level 0)
#  (Unchanged logic, just context preserved)
# ==================================================================
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
    # 2. Loop over Dimension D
    # -----------------------------------------------------------
    off_q = (b_idx * sq_b) + (node_id * sq_n)
    off_do = (b_idx * sdo_b) + (node_id * sdo_n)
    
    off_q_sib = (b_idx * sq_b) + (sibling_leaf * sq_n)
    off_do_sib = (b_idx * sdo_b) + (sibling_leaf * sdo_n)

    for off_d_start in range(0, D, BLOCK_D):
        offs_d = off_d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]

        # --- Self Computation ---
        q_self = tl.load(Q_ptr + off_q + (offs_h[:, None] * sq_h) + (offs_d[None, :] * sq_d), mask=mask_op, other=0.0)
        do_self = tl.load(DO_ptr + off_do + (offs_h[:, None] * sdo_h) + (offs_d[None, :] * sdo_d), mask=mask_op, other=0.0)
        
        dk_acc = ds_self * q_self
        dv_acc = w_self * do_self

        # --- Sibling Computation ---
        if has_sibling:
            q_sib = tl.load(Q_ptr + off_q_sib + (offs_h[:, None] * sq_h) + (offs_d[None, :] * sq_d), mask=mask_op, other=0.0)
            do_sib = tl.load(DO_ptr + off_do_sib + (offs_h[:, None] * sdo_h) + (offs_d[None, :] * sdo_d), mask=mask_op, other=0.0)
            
            dk_acc += ds_sib * q_sib
            dv_acc += w_sib * do_sib

        # --- Store Chunk ---
        off_out = (b_idx * sdk_b) + (node_id * sdk_node)
        tl.store(DK_ptr + off_out + (offs_h[:, None] * sdk_h) + (offs_d[None, :] * sdk_d), dk_acc, mask=mask_op)
        tl.store(DV_ptr + off_out + (offs_h[:, None] * sdk_h) + (offs_d[None, :] * sdk_d), dv_acc, mask=mask_op)

@triton.jit
def hierarchical_attention_backward_low_level_kernel(
    DS_ptr, Q_ptr, W_ptr, DO_ptr, Gather_Table_ptr,
    DK_ptr, DV_ptr,
    sds_b, sds_n, sds_h, sds_lvl,
    sq_b, sq_n, sq_h, sq_d,
    sw_b, sw_n, sw_h, sw_lvl,
    sdo_b, sdo_n, sdo_h, sdo_d,
    sdk_b, sdk_node, sdk_h, sdk_d,
    sg_node, sg_dim,
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    N: tl.constexpr,
    MAX_LEVEL: tl.constexpr
):
    pid = tl.program_id(0)
    b_idx = tl.program_id(1)

    node_id = N + pid
    target_level = 0
    found = 0
    
    for lvl in tl.static_range(1, MAX_LEVEL + 1):
        lvl_start = N - (N >> (lvl - 1))
        lvl_end   = N - (N >> lvl)
        
        if found == 0:
            if pid < lvl_end:
                target_level = lvl
                found = 1

    # Gather Logic
    tab_ptr = Gather_Table_ptr + (node_id * sg_node)
    child_start_base = tl.load(tab_ptr + 0)

    if child_start_base == -1:
        return

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
            
            dk_acc += ds * q.to(tl.float32)
            dv_acc += w * do.to(tl.float32)

        ptr_dk = DK_ptr + off_out_base + (offs_d[None, :] * sdk_d)
        ptr_dv = DV_ptr + off_out_base + (offs_d[None, :] * sdk_d)
        tl.store(ptr_dk, dk_acc, mask=mask_op)
        tl.store(ptr_dv, dv_acc, mask=mask_op)

@triton.jit
def hierarchical_attention_backward_high_level_kernel(
    DS_ptr, Q_ptr, W_ptr, DO_ptr, Gather_Table_ptr,
    DK_ptr, DV_ptr,
    sds_b, sds_n, sds_h, sds_lvl,
    sq_b, sq_n, sq_h, sq_d,
    sw_b, sw_n, sw_h, sw_lvl,
    sdo_b, sdo_n, sdo_h, sdo_d,
    sdk_b, sdk_node, sdk_h, sdk_d,
    sg_node, sg_dim,
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    N: tl.constexpr,
    START_LEVEL: tl.constexpr
):
    pid = tl.program_id(0)
    b_idx = tl.program_id(1)

    BLOCKS_PER_LVL: tl.constexpr = N >> (START_LEVEL - 1)
    BLOCK_MASK: tl.constexpr = BLOCKS_PER_LVL - 1
    
    lvl_offset = pid // BLOCKS_PER_LVL
    target_level = START_LEVEL + lvl_offset
    
    rem = pid & BLOCK_MASK
    shift_val = target_level - (START_LEVEL - 1)
    split_k_mask = (1 << shift_val) - 1
    
    node_local = rem >> shift_val
    split_id   = rem & split_k_mask
    
    start_node_global = (2 * N) - (N >> (target_level - 1))
    node_id = start_node_global + node_local

    tab_ptr = Gather_Table_ptr + (node_id * sg_node)
    child_start_base = tl.load(tab_ptr + 0)
    
    if child_start_base == -1:
        return

    CHILDREN_PER_SPLIT: tl.constexpr = 1 << (START_LEVEL - 1)
    w_idx = target_level + 1
    start_k = split_id * CHILDREN_PER_SPLIT

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

        off_hq_d  = (offs_h[:, None] * sq_h)  + (offs_d[None, :] * sq_d)
        off_hdo_d = (offs_h[:, None] * sdo_h) + (offs_d[None, :] * sdo_d)

        for k_offset in range(CHILDREN_PER_SPLIT):
            k = start_k + k_offset
            child_idx = child_start_base + k
            
            ptr_ds = ptr_ds_base + (child_idx * sds_n) + (offs_h * sds_h)
            ptr_w  = ptr_w_base  + (child_idx * sw_n)  + (offs_h * sw_h)
            
            ds = tl.load(ptr_ds, mask=mask_h, other=0.0)[:, None]
            w  = tl.load(ptr_w,  mask=mask_h, other=0.0)[:, None]
            
            ptr_q  = ptr_q_base  + (child_idx * sq_n)  + off_hq_d
            ptr_do = ptr_do_base + (child_idx * sdo_n) + off_hdo_d
            
            q  = tl.load(ptr_q,  mask=mask_op, other=0.0)
            do = tl.load(ptr_do, mask=mask_op, other=0.0)
            
            dk_acc += ds * q.to(tl.float32)
            dv_acc += w * do.to(tl.float32)

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
    ds_base = DS_ptr + (b_idx * sds_b) + (node_idx * sds_n) + (h_idx * sds_h)
    dq_base = DQ_ptr + (b_idx * sdq_b) + (node_idx * sdq_n) + (h_idx[:, None] * sdq_h)
    k_batch_base = K_ptr + b_idx * sk_b

    # Pre-load Self DS
    ds_self = tl.load(ds_base + (0 * sds_lvl), mask=mask_h, other=0.0)

    # -----------------------------------------------------------
    # 2. Outer Loop over D
    # -----------------------------------------------------------
    for off_d_start in range(0, D, BLOCK_D):
        offs_d = off_d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]

        # -------------------------------------------------------
        # A. Process Self (Level 0)
        # -------------------------------------------------------
        off_k_self = (node_idx * sk_n) + \
                     (h_idx[:, None] * sk_h) + \
                     (offs_d[None, :] * sk_d)
                     
        k_self = tl.load(k_batch_base + off_k_self, mask=mask_op, other=0.0)

        dq_acc = ds_self[:, None].to(tl.float32) * k_self.to(tl.float32)

        # -------------------------------------------------------
        # B. Inner Loop over Levels
        # -------------------------------------------------------
        for lvl_idx in range(LEVELS):
            # 1. Load Topology for this level
            off_lookup = node_idx * sl_n + lvl_idx * sl_lvl
            
            p_idx = tl.load(Lookup_ptr + off_lookup)

            # 2. Check Validity [FIX] Check for -1
            is_valid = (p_idx != -1)
            if HAS_MASK:
                mask_val = tl.load(Mask_ptr + off_lookup).to(tl.int8)
                is_valid = is_valid & (mask_val == 0)

            # 3. Load dS for this level [BLOCK_H]
            ds_ptr_lvl = ds_base + ((lvl_idx + 1) * sds_lvl)
            mask_load = mask_h & is_valid
            ds_cross = tl.load(ds_ptr_lvl, mask=mask_load, other=0.0)

            # 4. Load K for this Parent [BLOCK_H, BLOCK_D]
            # [FIX] Safe Parent Index (redirect invalid to 0)
            safe_p_idx = tl.where(is_valid, p_idx, 0)
            
            off_k_cross = (safe_p_idx * sk_n) + \
                          (h_idx[:, None] * sk_h) + \
                          (offs_d[None, :] * sk_d)
            
            mask_k = mask_load[:, None] & mask_d[None, :]
            k_cross = tl.load(k_batch_base + off_k_cross, mask=mask_k, other=0.0)

            # 5. Accumulate
            dq_acc += ds_cross[:, None].to(tl.float32) * k_cross.to(tl.float32)

        # -------------------------------------------------------
        # C. Write Result
        # -------------------------------------------------------
        off_dq_out = offs_d[None, :] * sdq_d
        tl.store(dq_base + off_dq_out, dq_acc.to(DQ_ptr.dtype.element_ty), mask=mask_op)