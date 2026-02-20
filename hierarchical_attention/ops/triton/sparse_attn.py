import triton
import triton.language as tl

# ==================================================================
# 1. FORWARD KERNEL
# ==================================================================
@triton.jit
def hierarchical_attention_forward_kernel(
    Q_ptr, K_ptr, V_ptr, 
    Lookup_ptr, Mask_ptr, 
    Out_ptr, W_ptr, 
    
    sq_b, sq_n, sq_h, sq_d,
    sk_b, sk_n, sk_h, sk_d,
    sv_b, sv_n, sv_h, sv_d,
    sl_n, sl_lvl,
    so_b, so_n, so_h, so_d,
    sw_b, sw_n, sw_h, sw_lvl,
    
    sm_scale,
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, LEVELS: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_LEVELS: tl.constexpr, 
    BLOCK_WINDOW: tl.constexpr, 
    HAS_MASK: tl.constexpr, RADIUS: tl.constexpr, N: tl.constexpr 
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    WINDOW_SIZE: tl.constexpr = 2 * RADIUS + 1

    h_idx = tl.arange(0, BLOCK_H)
    mask_h = h_idx < H
    offs_d = tl.arange(0, BLOCK_D)
    offs_lvl = tl.arange(0, BLOCK_LEVELS)
    mask_lvl = offs_lvl < LEVELS

    off_lookup = node_idx * sl_n + offs_lvl * sl_lvl
    neighbor_indices = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl, other=-1)
    
    # Safely handle -1 padding to avoid reading out-of-bounds memory
    is_valid_cross = (neighbor_indices != -1)
    safe_neighbor_cross = tl.where(is_valid_cross, neighbor_indices, 0)
    
    neighbor_mask_val = tl.zeros([BLOCK_LEVELS], dtype=tl.int1)
    if HAS_MASK:
        val_int8 = tl.load(Mask_ptr + off_lookup, mask=mask_lvl, other=1).to(tl.int8)
        neighbor_mask_val = (val_int8 != 0)

    k_batch_base = K_ptr + b_idx * sk_b
    v_batch_base = V_ptr + b_idx * sv_b
    off_node_cross = safe_neighbor_cross * sk_n 

    q_ptr = Q_ptr + (b_idx * sq_b) + (node_idx * sq_n) + \
            (h_idx[:, None] * sq_h) + (offs_d[None, :] * sq_d)

    acc_window = tl.zeros([BLOCK_H, BLOCK_WINDOW], dtype=tl.float32)
    acc_cross = tl.zeros([BLOCK_H, BLOCK_LEVELS], dtype=tl.float32)

    for off_d_start in range(0, D, BLOCK_D):
        cur_offs_d = off_d_start + offs_d
        d_mask = cur_offs_d < D
        mask_q = mask_h[:, None] & d_mask[None, :]
        
        q = tl.load(q_ptr, mask=mask_q, other=0.0)
        
        for w in tl.static_range(0, WINDOW_SIZE):
            offset = w - RADIUS
            neighbor_idx = node_idx + offset

            is_valid_geom = (neighbor_idx >= 0) & (neighbor_idx < N)
            safe_neighbor_idx = tl.where(is_valid_geom, neighbor_idx, 0)
            mask_win_load = mask_q & is_valid_geom

            ptr_k_win = k_batch_base + (safe_neighbor_idx * sk_n) + \
                        (h_idx[:, None] * sk_h) + (cur_offs_d[None, :] * sk_d)
            
            k_val = tl.load(ptr_k_win, mask=mask_win_load, other=0.0)
            
            score = tl.sum(q * k_val, axis=1)
            mask_w = (tl.arange(0, BLOCK_WINDOW) == w)[None, :]
            acc_window = tl.where(mask_w, acc_window + score[:, None], acc_window)

        ptr_k_cross = k_batch_base + off_node_cross[None, :, None] + \
                      (h_idx[:, None, None] * sk_h) + (cur_offs_d[None, None, :] * sk_d)
        
        mask_k = mask_h[:, None, None] & mask_lvl[None, :, None] & d_mask[None, None, :] & is_valid_cross[None, :, None]
        k_cross = tl.load(ptr_k_cross, mask=mask_k, other=0.0)
        
        acc_cross += tl.sum(q[:, None, :] * k_cross, axis=2)
        q_ptr += BLOCK_D * sq_d

    acc_window = acc_window * sm_scale
    acc_cross = acc_cross * sm_scale
    
    offs_w_pad = tl.arange(0, BLOCK_WINDOW)
    mask_w_pad = offs_w_pad < WINDOW_SIZE
    acc_window = tl.where(mask_w_pad[None, :], acc_window, -float('inf'))
    
    for w in tl.static_range(0, WINDOW_SIZE):
        offset = w - RADIUS
        neighbor_idx = node_idx + offset
        
        is_valid_geom = (neighbor_idx >= 0) & (neighbor_idx < N)
        is_causal = (neighbor_idx <= node_idx)

        should_mask = (~is_valid_geom)
        if HAS_MASK:
            should_mask = should_mask | (~is_causal)
        
        mask_w = (tl.arange(0, BLOCK_WINDOW) == w)[None, :]
        acc_window = tl.where(mask_w & should_mask, -float('inf'), acc_window)

    mask_broadcast = (offs_lvl >= LEVELS) | (~is_valid_cross)
    if HAS_MASK:
        mask_broadcast = mask_broadcast | neighbor_mask_val
    acc_cross = tl.where(mask_broadcast[None, :], -float('inf'), acc_cross)
    
    max_cross = tl.max(acc_cross, axis=1)
    max_window = tl.max(acc_window, axis=1)
    max_all = tl.maximum(max_window, max_cross)
    
    exp_window = tl.exp(acc_window - max_all[:, None])
    exp_cross = tl.exp(acc_cross - max_all[:, None])
    
    denom = tl.sum(exp_window, axis=1) + tl.sum(exp_cross, axis=1)
    denom = tl.maximum(denom, 1.0e-5)

    w_window = exp_window / denom[:, None]
    w_cross = exp_cross / denom[:, None]

    w_base_ptr = W_ptr + (b_idx * sw_b) + (node_idx * sw_n) + (h_idx * sw_h)
    offs_win = tl.arange(0, BLOCK_WINDOW)
    mask_win_store = mask_h[:, None] & (offs_win < WINDOW_SIZE)[None, :]
    
    # Memory Clamps
    safe_offs_win = tl.where(offs_win < WINDOW_SIZE, offs_win, 0)
    tl.store(w_base_ptr[:, None] + (safe_offs_win[None, :] * sw_lvl), w_window, mask=mask_win_store)
    
    safe_offs_lvl = tl.where(mask_lvl, offs_lvl, 0)
    w_cross_ptr = w_base_ptr[:, None] + ((WINDOW_SIZE + safe_offs_lvl[None, :]) * sw_lvl)
    tl.store(w_cross_ptr, w_cross, mask=mask_h[:, None] & mask_lvl[None, :])

    out_base_ptr = Out_ptr + (b_idx * so_b) + (node_idx * so_n) + \
                   (h_idx[:, None] * so_h) + (offs_d[None, :] * so_d)

    for off_d_start in range(0, D, BLOCK_D):
        cur_offs_d = off_d_start + offs_d
        d_mask = cur_offs_d < D
        mask_op = mask_h[:, None] & d_mask[None, :]
        out_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

        for w in tl.static_range(0, WINDOW_SIZE):
            offset = w - RADIUS
            neighbor_idx = node_idx + offset
            
            is_valid_geom = (neighbor_idx >= 0) & (neighbor_idx < N)
            safe_neighbor_idx = tl.where(is_valid_geom, neighbor_idx, 0)
            mask_win_v = mask_op & is_valid_geom
            
            ptr_v_win = v_batch_base + (safe_neighbor_idx * sv_n) + \
                        (h_idx[:, None] * sv_h) + (cur_offs_d[None, :] * sv_d)
            
            v_val = tl.load(ptr_v_win, mask=mask_win_v, other=0.0)
            
            mask_w = (tl.arange(0, BLOCK_WINDOW) == w)[None, :]
            w_val = tl.sum(tl.where(mask_w, w_window, 0.0), axis=1)
            out_acc += w_val[:, None] * v_val

        ptr_v_cross = v_batch_base + off_node_cross[None, :, None] + \
                      (h_idx[:, None, None] * sv_h) + (cur_offs_d[None, None, :] * sv_d)
        
        mask_v = mask_h[:, None, None] & mask_lvl[None, :, None] & d_mask[None, None, :] & is_valid_cross[None, :, None]
        v_cross = tl.load(ptr_v_cross, mask=mask_v, other=0.0)
        
        out_acc += tl.sum(w_cross[:, :, None] * v_cross, axis=1)
        
        tl.store(out_base_ptr, out_acc.to(Out_ptr.dtype.element_ty), mask=mask_op)
        out_base_ptr += BLOCK_D * so_d

# ==================================================================
# 2. BACKWARD KERNEL 1: SCORE GRADIENT (dS)
# ==================================================================
@triton.jit
def hierarchical_attention_backward_dS_kernel(
    DO_ptr, W_ptr, V_ptr, Lookup_ptr, DS_ptr, Mask_ptr,
    sdo_b, sdo_n, sdo_h, sdo_d,
    sw_b, sw_n, sw_h, sw_lvl,
    sv_b, sv_n, sv_h, sv_d,
    sl_n, sl_lvl,
    sds_b, sds_n, sds_h, sds_lvl,
    sm_scale,
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    LEVELS: tl.constexpr, BLOCK_LEVELS: tl.constexpr,
    BLOCK_WINDOW: tl.constexpr, 
    HAS_MASK: tl.constexpr,
    RADIUS: tl.constexpr, WINDOW_SIZE: tl.constexpr, N: tl.constexpr
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    h_idx = tl.arange(0, BLOCK_H)
    mask_h = h_idx < H
    offs_lvl = tl.arange(0, BLOCK_LEVELS)
    
    mask_lvl_bounds = offs_lvl < LEVELS
    off_lookup = node_idx * sl_n + offs_lvl * sl_lvl
    neighbor_indices = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl_bounds, other=-1)

    is_valid_cross = (neighbor_indices != -1)
    safe_neighbor_cross = tl.where(is_valid_cross, neighbor_indices, 0)

    mask_valid_cross = mask_lvl_bounds & is_valid_cross
    if HAS_MASK:
        val_int8 = tl.load(Mask_ptr + off_lookup, mask=mask_lvl_bounds, other=1).to(tl.int8)
        mask_valid_cross = mask_valid_cross & (val_int8 == 0)

    w_base = W_ptr + (b_idx * sw_b) + (node_idx * sw_n) + (h_idx * sw_h)
    ds_base = DS_ptr + (b_idx * sds_b) + (node_idx * sds_n) + (h_idx * sds_h)
    
    v_batch_base = V_ptr + b_idx * sv_b
    do_batch_base = DO_ptr + (b_idx * sdo_b) + (node_idx * sdo_n)
    
    offs_win = tl.arange(0, BLOCK_WINDOW)
    mask_win_bounds = offs_win < WINDOW_SIZE
    
    # Memory Clamps
    safe_offs_win = tl.where(mask_win_bounds, offs_win, 0)
    w_window_ptr = w_base[:, None] + (safe_offs_win[None, :] * sw_lvl)
    w_window_cache = tl.load(w_window_ptr, mask=mask_h[:, None] & mask_win_bounds[None, :], other=0.0)

    safe_offs_lvl = tl.where(mask_lvl_bounds, offs_lvl, 0)
    w_cross_ptr = w_base[:, None] + ((WINDOW_SIZE + safe_offs_lvl[None, :]) * sw_lvl)
    w_cross = tl.load(w_cross_ptr, mask=mask_h[:, None] & mask_valid_cross[None, :], other=0.0)

    dp_window = tl.zeros([BLOCK_H, BLOCK_WINDOW], dtype=tl.float32)
    dp_cross = tl.zeros([BLOCK_H, BLOCK_LEVELS], dtype=tl.float32)
    
    off_node_cross = safe_neighbor_cross * sv_n 

    for off_d_start in range(0, D, BLOCK_D):
        offs_d = off_d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]

        do = tl.load(do_batch_base + (h_idx[:, None]*sdo_h) + (offs_d[None, :]*sdo_d), 
                     mask=mask_op, other=0.0)

        for w in tl.static_range(0, WINDOW_SIZE):
            offset = w - RADIUS
            neighbor_idx = node_idx + offset
            
            is_valid_geom = (neighbor_idx >= 0) & (neighbor_idx < N)
            safe_neighbor = tl.where(is_valid_geom, neighbor_idx, 0)
            mask_load = mask_op & is_valid_geom
            
            ptr_v = v_batch_base + (safe_neighbor * sv_n) + \
                    (h_idx[:, None] * sv_h) + (offs_d[None, :] * sv_d)
            
            v = tl.load(ptr_v, mask=mask_load, other=0.0)
            score_dp = tl.sum(do * v, axis=1)
            
            mask_w = (tl.arange(0, BLOCK_WINDOW) == w)[None, :]
            dp_window = tl.where(mask_w, dp_window + score_dp[:, None], dp_window)

        ptr_v_cross = v_batch_base + off_node_cross[None, :, None] + \
                      (h_idx[:, None, None]*sv_h) + (offs_d[None, None, :]*sv_d)
        
        mask_v = mask_h[:, None, None] & mask_valid_cross[None, :, None] & mask_d[None, None, :]
        v_cross = tl.load(ptr_v_cross, mask=mask_v, other=0.0)
        
        dp_cross += tl.sum(do[:, None, :] * v_cross, axis=2)

    sum_wdp = tl.sum(w_window_cache * dp_window, axis=1) + tl.sum(w_cross * dp_cross, axis=1)

    ds_window = w_window_cache * (dp_window - sum_wdp[:, None]) * sm_scale
    ds_win_ptr = ds_base[:, None] + (safe_offs_win[None, :] * sds_lvl)
    tl.store(ds_win_ptr, ds_window, mask=mask_h[:, None] & mask_win_bounds[None, :])

    ds_cross = w_cross * (dp_cross - sum_wdp[:, None]) * sm_scale
    ds_cross_ptr = ds_base[:, None] + ((WINDOW_SIZE + safe_offs_lvl[None, :]) * sds_lvl)
    tl.store(ds_cross_ptr, ds_cross, mask=mask_h[:, None] & mask_lvl_bounds[None, :])

# ==================================================================
# 3. BACKWARD KERNEL 2A: SPECIALIZED WINDOW KERNEL (dK/dV)
# ==================================================================
@triton.jit
def hierarchical_attention_backward_dK_dV_window_kernel(
    DS_ptr, Q_ptr, W_ptr, DO_ptr,
    DK_ptr, DV_ptr,
    sds_b, sds_n, sds_h, sds_lvl,
    sq_b, sq_n, sq_h, sq_d,
    sw_b, sw_n, sw_h, sw_lvl,
    sdo_b, sdo_n, sdo_h, sdo_d,
    sdk_b, sdk_n, sdk_h, sdk_d,
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    RADIUS: tl.constexpr, WINDOW_SIZE: tl.constexpr, N: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    node_id = tl.program_id(0)
    b_idx = tl.program_id(1)

    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    
    off_out = (b_idx * sdk_b) + (node_id * sdk_n)
    
    ds_base = DS_ptr + (b_idx * sds_b)
    w_base  = W_ptr  + (b_idx * sw_b)
    q_base  = Q_ptr  + (b_idx * sq_b)
    do_base = DO_ptr + (b_idx * sdo_b)

    for off_d_start in range(0, D, BLOCK_D):
        offs_d = off_d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]

        dk_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

        for w in tl.static_range(0, WINDOW_SIZE):
            offset = w - RADIUS
            source_idx = node_id - offset 

            is_valid_geom = (source_idx >= 0) & (source_idx < N)
            safe_src = tl.where(is_valid_geom, source_idx, 0)
            
            mask_scalar = mask_h[:, None] & is_valid_geom
            mask_vec = mask_op & is_valid_geom

            off_ds = ds_base + (safe_src * sds_n) + (w * sds_lvl) + (offs_h[:, None] * sds_h)
            off_w  = w_base  + (safe_src * sw_n)  + (w * sw_lvl)  + (offs_h[:, None] * sw_h)
            
            ds_val = tl.load(off_ds, mask=mask_scalar, other=0.0)
            w_val  = tl.load(off_w,  mask=mask_scalar, other=0.0)

            off_q  = q_base  + (safe_src * sq_n)  + (offs_h[:, None] * sq_h)  + (offs_d[None, :] * sq_d)
            off_do = do_base + (safe_src * sdo_n) + (offs_h[:, None] * sdo_h) + (offs_d[None, :] * sdo_d)

            q_src  = tl.load(off_q,  mask=mask_vec, other=0.0)
            do_src = tl.load(off_do, mask=mask_vec, other=0.0)

            dk_acc += ds_val * q_src
            dv_acc += w_val * do_src

        # Level 0 Sibling Cross-Attention Gather
        sibling = node_id ^ 1
        is_valid_geom = sibling < N
        
        if IS_CAUSAL:
            is_valid_cross = is_valid_geom & (sibling > node_id)
        else:
            is_valid_cross = is_valid_geom
            
        safe_sibling = tl.where(is_valid_cross, sibling, 0)
        
        w_idx_lvl0 = WINDOW_SIZE + 0 
        
        mask_cross_scalar = mask_h[:, None] & is_valid_cross
        mask_cross_vec = mask_op & is_valid_cross
        
        off_ds_cross = ds_base + (safe_sibling * sds_n) + (w_idx_lvl0 * sds_lvl) + (offs_h[:, None] * sds_h)
        off_w_cross  = w_base  + (safe_sibling * sw_n)  + (w_idx_lvl0 * sw_lvl)  + (offs_h[:, None] * sw_h)
        
        ds_cross = tl.load(off_ds_cross, mask=mask_cross_scalar, other=0.0)
        w_cross  = tl.load(off_w_cross,  mask=mask_cross_scalar, other=0.0)
        
        off_q_cross  = q_base  + (safe_sibling * sq_n)  + (offs_h[:, None] * sq_h)  + (offs_d[None, :] * sq_d)
        off_do_cross = do_base + (safe_sibling * sdo_n) + (offs_h[:, None] * sdo_h) + (offs_d[None, :] * sdo_d)
        
        q_cross  = tl.load(off_q_cross,  mask=mask_cross_vec, other=0.0)
        do_cross = tl.load(off_do_cross, mask=mask_cross_vec, other=0.0)
        
        dk_acc += ds_cross * q_cross
        dv_acc += w_cross * do_cross

        ptr_dk = DK_ptr + off_out + (offs_h[:, None] * sdk_h) + (offs_d[None, :] * sdk_d)
        ptr_dv = DV_ptr + off_out + (offs_h[:, None] * sdk_h) + (offs_d[None, :] * sdk_d)
        
        tl.store(ptr_dk, dk_acc, mask=mask_op)
        tl.store(ptr_dv, dv_acc, mask=mask_op)

# ==================================================================
# 4. BACKWARD KERNEL 2B: LOW LEVEL PARENTS
# ==================================================================
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
    N: tl.constexpr, MAX_LEVEL: tl.constexpr,
    WINDOW_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    b_idx = tl.program_id(1)

    node_id = N + pid
    target_level = tl.zeros([], dtype=tl.int32)
    
    for lvl in tl.static_range(1, MAX_LEVEL + 1):
        lvl_end = N - (N >> lvl)
        is_match = (pid < lvl_end) & (target_level == 0)
        target_level = tl.where(is_match, lvl, target_level)

    tab_ptr = Gather_Table_ptr + (node_id * sg_node)
    child_start_base = tl.load(tab_ptr + 0)

    if child_start_base == -1:
        return

    num_children = 1 << target_level
    w_idx = WINDOW_SIZE + target_level

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

# ==================================================================
# 5. BACKWARD KERNEL 2C: HIGH LEVEL PARENTS
# ==================================================================
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
    N: tl.constexpr, START_LEVEL: tl.constexpr,
    WINDOW_SIZE: tl.constexpr
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
    w_idx = WINDOW_SIZE + target_level
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

# ==================================================================
# 6. BACKWARD KERNEL 3: COMPUTE dQ
# ==================================================================
@triton.jit
def hierarchical_attention_backward_dQ_kernel(
    DS_ptr, K_ptr, Lookup_ptr, DQ_ptr, Mask_ptr,
    sds_b, sds_n, sds_h, sds_lvl,
    sk_b, sk_n, sk_h, sk_d,
    sl_n, sl_lvl,
    sdq_b, sdq_n, sdq_h, sdq_d,
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    LEVELS: tl.constexpr, 
    HAS_MASK: tl.constexpr,
    RADIUS: tl.constexpr, WINDOW_SIZE: tl.constexpr, N: tl.constexpr
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    ds_base = DS_ptr + (b_idx * sds_b) + (node_idx * sds_n) + (offs_h[:, None] * sds_h)
    k_batch_base = K_ptr + b_idx * sk_b
    dq_base = DQ_ptr + (b_idx * sdq_b) + (node_idx * sdq_n) + (offs_h[:, None] * sdq_h)

    for off_d_start in range(0, D, BLOCK_D):
        offs_d = off_d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]

        dq_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

        for w in tl.static_range(0, WINDOW_SIZE):
            offset = w - RADIUS
            neighbor_idx = node_idx + offset
            
            is_valid_geom = (neighbor_idx >= 0) & (neighbor_idx < N)
            safe_neighbor = tl.where(is_valid_geom, neighbor_idx, 0)
            
            mask_scalar = mask_h[:, None] & is_valid_geom
            mask_vec = mask_op & is_valid_geom

            ds_val = tl.load(ds_base + (w * sds_lvl), mask=mask_scalar, other=0.0)
            off_k = (safe_neighbor * sk_n) + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
            k_val = tl.load(k_batch_base + off_k, mask=mask_vec, other=0.0)

            dq_acc += ds_val * k_val

        for lvl_idx in range(LEVELS):
            off_lookup = node_idx * sl_n + lvl_idx * sl_lvl
            mask_lvl = lvl_idx < LEVELS
            p_idx = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl, other=-1)
            
            is_valid = (p_idx != -1)
            if HAS_MASK:
                mask_val = tl.load(Mask_ptr + off_lookup, mask=mask_lvl, other=1).to(tl.int8)
                is_valid = is_valid & (mask_val == 0)

            ds_cross_ptr = ds_base + ((WINDOW_SIZE + lvl_idx) * sds_lvl)
            ds_cross = tl.load(ds_cross_ptr, mask=mask_h[:, None] & is_valid, other=0.0)

            safe_p_idx = tl.where(is_valid, p_idx, 0)
            mask_k = mask_op & is_valid

            off_k_cross = (safe_p_idx * sk_n) + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
            k_cross = tl.load(k_batch_base + off_k_cross, mask=mask_k, other=0.0)

            dq_acc += ds_cross * k_cross

        off_dq_out = offs_d[None, :] * sdq_d
        tl.store(dq_base + off_dq_out, dq_acc.to(DQ_ptr.dtype.element_ty), mask=mask_op)