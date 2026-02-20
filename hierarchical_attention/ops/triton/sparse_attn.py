import triton
import triton.language as tl

@triton.jit
def hierarchical_attention_forward_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, 
    Lookup_ptr, Mask_ptr, 
    Out_ptr, W_ptr, 
    
    # Strides
    sq_b, sq_n, sq_h, sq_d,
    sk_b, sk_n, sk_h, sk_d,
    sv_b, sv_n, sv_h, sv_d,
    sl_n, sl_lvl,
    so_b, so_n, so_h, so_d,
    sw_b, sw_n, sw_h, sw_lvl,
    
    # Constants
    sm_scale,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_LEVELS: tl.constexpr, 
    BLOCK_WINDOW: tl.constexpr, # [NEW] Power of 2 padded size
    HAS_MASK: tl.constexpr,
    RADIUS: tl.constexpr,
    N: tl.constexpr 
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    WINDOW_SIZE: tl.constexpr = 2 * RADIUS + 1

    # 1. Setup Coordinates
    h_idx = tl.arange(0, BLOCK_H)
    mask_h = h_idx < H
    offs_d = tl.arange(0, BLOCK_D)
    offs_lvl = tl.arange(0, BLOCK_LEVELS)
    mask_lvl = offs_lvl < LEVELS

    # 2. Load Topology (Hierarchy)
    off_lookup = node_idx * sl_n + offs_lvl * sl_lvl
    neighbor_indices = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl, other=0)
    
    neighbor_mask_val = tl.zeros([BLOCK_LEVELS], dtype=tl.int1)
    if HAS_MASK:
        val_int8 = tl.load(Mask_ptr + off_lookup, mask=mask_lvl, other=1).to(tl.int8)
        neighbor_mask_val = (val_int8 != 0)

    # 3. Base Pointers
    k_batch_base = K_ptr + b_idx * sk_b
    v_batch_base = V_ptr + b_idx * sv_b
    off_node_cross = neighbor_indices * sk_n 

    q_ptr = Q_ptr + (b_idx * sq_b) + (node_idx * sq_n) + \
            (h_idx[:, None] * sq_h) + (offs_d[None, :] * sq_d)

    # Accumulators - [FIX] Allocated with BLOCK_WINDOW
    acc_window = tl.zeros([BLOCK_H, BLOCK_WINDOW], dtype=tl.float32)
    acc_cross = tl.zeros([BLOCK_H, BLOCK_LEVELS], dtype=tl.float32)

    # 4. Main Loop over D (Outer Loop)
    for off_d_start in range(0, D, BLOCK_D):
        cur_offs_d = off_d_start + offs_d
        d_mask = cur_offs_d < D
        mask_q = mask_h[:, None] & d_mask[None, :]
        
        q = tl.load(q_ptr, mask=mask_q, other=0.0)
        
        # 4.1. SAFE SLIDING WINDOW LOOP
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
            # Create a [1, BLOCK_WINDOW] mask that is True only at column w
            mask_w = (tl.arange(0, BLOCK_WINDOW) == w)[None, :]
            # Update the whole tensor using tl.where
            acc_window = tl.where(mask_w, acc_window + score[:, None], acc_window)

        # 4.2. HIERARCHICAL CROSS LOOP
        ptr_k_cross = k_batch_base + off_node_cross[None, :, None] + \
                      (h_idx[:, None, None] * sk_h) + (cur_offs_d[None, None, :] * sk_d)
        
        mask_k = mask_h[:, None, None] & mask_lvl[None, :, None] & d_mask[None, None, :]
        k_cross = tl.load(ptr_k_cross, mask=mask_k, other=0.0)
        
        acc_cross += tl.sum(q[:, None, :] * k_cross, axis=2)
        q_ptr += BLOCK_D * sq_d

    # 5. Post-Loop Masking
    acc_window = acc_window * sm_scale
    acc_cross = acc_cross * sm_scale
    
    # [NEW FIX] Mask out the padded region in the block so tl.max ignores it
    offs_w_pad = tl.arange(0, BLOCK_WINDOW)
    mask_w_pad = offs_w_pad < WINDOW_SIZE
    acc_window = tl.where(mask_w_pad[None, :], acc_window, -float('inf'))
    
    # Masking valid bounds
    for w in tl.static_range(0, WINDOW_SIZE):
        offset = w - RADIUS
        neighbor_idx = node_idx + offset
        
        is_valid_geom = (neighbor_idx >= 0) & (neighbor_idx < N)
        is_causal = (neighbor_idx <= node_idx)

        should_mask = (~is_valid_geom)
        if HAS_MASK:
            should_mask = should_mask | (~is_causal)
        
        mask_w = (tl.arange(0, BLOCK_WINDOW) == w)[None, :]
        # Apply -inf only where the column matches `w` AND it should be masked
        acc_window = tl.where(mask_w & should_mask, -float('inf'), acc_window)

    mask_broadcast = (offs_lvl >= LEVELS)
    if HAS_MASK:
        mask_broadcast = mask_broadcast | neighbor_mask_val
    acc_cross = tl.where(mask_broadcast[None, :], -float('inf'), acc_cross)
    
    # 6. Softmax & Reduction
    max_cross = tl.max(acc_cross, axis=1)
    max_window = tl.max(acc_window, axis=1)
    max_all = tl.maximum(max_window, max_cross)
    
    exp_window = tl.exp(acc_window - max_all[:, None])
    exp_cross = tl.exp(acc_cross - max_all[:, None])
    
    denom = tl.sum(exp_window, axis=1) + tl.sum(exp_cross, axis=1)
    denom = tl.maximum(denom, 1.0e-5)

    w_window = exp_window / denom[:, None]
    w_cross = exp_cross / denom[:, None]

    # Save Weights - [FIX] Use masked store for the padded window block
    w_base_ptr = W_ptr + (b_idx * sw_b) + (node_idx * sw_n) + (h_idx * sw_h)
    offs_win = tl.arange(0, BLOCK_WINDOW)
    mask_win_store = mask_h[:, None] & (offs_win < WINDOW_SIZE)[None, :]
    tl.store(w_base_ptr[:, None] + (offs_win[None, :] * sw_lvl), w_window, mask=mask_win_store)
    
    w_cross_ptr = w_base_ptr[:, None] + ((WINDOW_SIZE + offs_lvl[None, :]) * sw_lvl)
    tl.store(w_cross_ptr, w_cross, mask=mask_h[:, None] & mask_lvl[None, :])

    # 7. Weighted Sum (Value Aggregation)
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
            
            # Extract the column first
            v_val = tl.load(ptr_v_win, mask=mask_win_v, other=0.0)
            
            # 1. Create a mask for the exact column w
            mask_w = (tl.arange(0, BLOCK_WINDOW) == w)[None, :]
            
            # 2. Extract column w by zeroing everything else and summing along the window axis
            # This cleanly extracts a [BLOCK_H] shape tensor
            w_val = tl.sum(tl.where(mask_w, w_window, 0.0), axis=1)
            
            # 3. Broadcast to [BLOCK_H, BLOCK_D] and multiply
            out_acc += w_val[:, None] * v_val

        ptr_v_cross = v_batch_base + off_node_cross[None, :, None] + \
                      (h_idx[:, None, None] * sv_h) + (cur_offs_d[None, None, :] * sv_d)
        
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
    sdo_b, sdo_n, sdo_h, sdo_d,
    sw_b, sw_n, sw_h, sw_lvl,
    sv_b, sv_n, sv_h, sv_d,
    sl_n, sl_lvl,
    sds_b, sds_n, sds_h, sds_lvl,
    sm_scale,
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    LEVELS: tl.constexpr, BLOCK_LEVELS: tl.constexpr,
    BLOCK_WINDOW: tl.constexpr, # [NEW] Power of 2 padding
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
    neighbor_indices = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl_bounds, other=0)

    mask_valid_cross = mask_lvl_bounds
    if HAS_MASK:
        val_int8 = tl.load(Mask_ptr + off_lookup, mask=mask_lvl_bounds, other=1).to(tl.int8)
        mask_valid_cross = mask_valid_cross & (val_int8 == 0)

    w_base = W_ptr + (b_idx * sw_b) + (node_idx * sw_n) + (h_idx * sw_h)
    ds_base = DS_ptr + (b_idx * sds_b) + (node_idx * sds_n) + (h_idx * sds_h)
    
    v_batch_base = V_ptr + b_idx * sv_b
    do_batch_base = DO_ptr + (b_idx * sdo_b) + (node_idx * sdo_n)
    
    # [FIX] Load Window Weights using BLOCK_WINDOW
    offs_win = tl.arange(0, BLOCK_WINDOW)
    mask_win_bounds = offs_win < WINDOW_SIZE
    w_window_ptr = w_base[:, None] + (offs_win[None, :] * sw_lvl)
    w_window_cache = tl.load(w_window_ptr, mask=mask_h[:, None] & mask_win_bounds[None, :], other=0.0)

    w_cross_ptr = w_base[:, None] + ((WINDOW_SIZE + offs_lvl[None, :]) * sw_lvl)
    w_cross = tl.load(w_cross_ptr, mask=mask_h[:, None] & mask_valid_cross[None, :], other=0.0)

    # [FIX] Accumulate with BLOCK_WINDOW
    dp_window = tl.zeros([BLOCK_H, BLOCK_WINDOW], dtype=tl.float32)
    dp_cross = tl.zeros([BLOCK_H, BLOCK_LEVELS], dtype=tl.float32)
    
    off_node_cross = neighbor_indices * sv_n 

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

    # [FIX] Store Window DS using bounded mask
    ds_window = w_window_cache * (dp_window - sum_wdp[:, None]) * sm_scale
    ds_win_ptr = ds_base[:, None] + (offs_win[None, :] * sds_lvl)
    tl.store(ds_win_ptr, ds_window, mask=mask_h[:, None] & mask_win_bounds[None, :])

    ds_cross = w_cross * (dp_cross - sum_wdp[:, None]) * sm_scale
    ds_cross_ptr = ds_base[:, None] + ((WINDOW_SIZE + offs_lvl[None, :]) * sds_lvl)
    
    tl.store(ds_cross_ptr, ds_cross, mask=mask_h[:, None] & mask_lvl_bounds[None, :])


# ------------------------------------------------------------------
#  Backward Kernel 2a: SPECIALIZED window KERNEL (Level 0)
# ------------------------------------------------------------------
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
    RADIUS: tl.constexpr, WINDOW_SIZE: tl.constexpr, N: tl.constexpr
):
    node_id = tl.program_id(0)
    b_idx = tl.program_id(1)

    # 1. Coordinate Setup
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    
    # Base Pointers
    off_out = (b_idx * sdk_b) + (node_id * sdk_n)
    
    # Base pointer parts for DS/W that don't depend on window/D
    ds_base = DS_ptr + (b_idx * sds_b)
    w_base  = W_ptr  + (b_idx * sw_b)
    q_base  = Q_ptr  + (b_idx * sq_b)
    do_base = DO_ptr + (b_idx * sdo_b)

    # 2. Main Loop over D (Tiling D to keep registers manageable)
    for off_d_start in range(0, D, BLOCK_D):
        offs_d = off_d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        
        # Combined Mask for Vector Loads [BLOCK_H, BLOCK_D]
        mask_op = mask_h[:, None] & mask_d[None, :]

        dk_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

        # 3. Window Accumulation Loop
        for w in tl.static_range(0, WINDOW_SIZE):
            offset = w - RADIUS
            source_idx = node_id - offset 

            # A. Validity
            is_valid_geom = (source_idx >= 0) & (source_idx < N)
            
            # [Optimization] Early continue logic isn't possible in Triton static loops easily
            # without complex masking. We use the 'safe index + mask' pattern.
            
            safe_src = tl.where(is_valid_geom, source_idx, 0)
            
            # B. Masks
            # Scalar mask for DS/W (broadcasting over Head)
            # Use 'is_valid_geom' (scalar) broadcasted
            mask_scalar = mask_h[:, None] & is_valid_geom
            
            # Vector mask for Q/dO
            mask_vec = mask_op & is_valid_geom

            # C. Load Scalars (DS, W)
            # off: [Batch] + [Source] + [Window_Index] + [Head]
            off_ds = ds_base + (safe_src * sds_n) + (w * sds_lvl) + (offs_h[:, None] * sds_h)
            off_w  = w_base  + (safe_src * sw_n)  + (w * sw_lvl)  + (offs_h[:, None] * sw_h)
            
            ds_val = tl.load(off_ds, mask=mask_scalar, other=0.0)
            w_val  = tl.load(off_w,  mask=mask_scalar, other=0.0)

            # D. Load Vectors (Q, dO)
            # off: [Batch] + [Source] + [Head] + [D]
            off_q  = q_base  + (safe_src * sq_n)  + (offs_h[:, None] * sq_h)  + (offs_d[None, :] * sq_d)
            off_do = do_base + (safe_src * sdo_n) + (offs_h[:, None] * sdo_h) + (offs_d[None, :] * sdo_d)

            q_src  = tl.load(off_q,  mask=mask_vec, other=0.0)
            do_src = tl.load(off_do, mask=mask_vec, other=0.0)

            # E. Fused Multiply Add (FMA)
            # dk += ds (scalar_broadcast) * q (vector)
            dk_acc += ds_val * q_src
            dv_acc += w_val * do_src

        # 4. Store Tile
        ptr_dk = DK_ptr + off_out + (offs_h[:, None] * sdk_h) + (offs_d[None, :] * sdk_d)
        ptr_dv = DV_ptr + off_out + (offs_h[:, None] * sdk_h) + (offs_d[None, :] * sdk_d)
        
        tl.store(ptr_dk, dk_acc, mask=mask_op)
        tl.store(ptr_dv, dv_acc, mask=mask_op)


# ------------------------------------------------------------------
#  Backward Kernel 2b: Low Level Parents
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
#  Backward Kernel 2b: High Level Parents
# ------------------------------------------------------------------
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
    # Strides
    sds_b, sds_n, sds_h, sds_lvl,
    sk_b, sk_n, sk_h, sk_d,
    sl_n, sl_lvl,
    sdq_b, sdq_n, sdq_h, sdq_d,
    # Constants
    H: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    LEVELS: tl.constexpr, 
    HAS_MASK: tl.constexpr,
    RADIUS: tl.constexpr, WINDOW_SIZE: tl.constexpr, N: tl.constexpr
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    # 1. Coordinate Setup [FIX: Define offs_h explicitly]
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # 2. Base Pointers (Using offs_h, not h_idx)
    # DS Base: [Node, Head]
    ds_base = DS_ptr + (b_idx * sds_b) + (node_idx * sds_n) + (offs_h[:, None] * sds_h)
    
    # K Batch Base: [Batch]
    k_batch_base = K_ptr + b_idx * sk_b
    
    # DQ Base: [Node, Head]
    dq_base = DQ_ptr + (b_idx * sdq_b) + (node_idx * sdq_n) + (offs_h[:, None] * sdq_h)

    # -----------------------------------------------------------
    # Outer Loop over D (Chunked for Registers)
    # -----------------------------------------------------------
    for off_d_start in range(0, D, BLOCK_D):
        offs_d = off_d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        
        # Combined Mask for Vector Loads [BLOCK_H, BLOCK_D]
        mask_op = mask_h[:, None] & mask_d[None, :]

        dq_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

        # -------------------------------------------------------
        # A. Window Loop (Replaces Self/Level 0)
        # -------------------------------------------------------
        for w in tl.static_range(0, WINDOW_SIZE):
            offset = w - RADIUS
            neighbor_idx = node_idx + offset
            
            # Validity Check
            is_valid_geom = (neighbor_idx >= 0) & (neighbor_idx < N)
            
            # Safe Indexing (Avoid Negative Pointers)
            safe_neighbor = tl.where(is_valid_geom, neighbor_idx, 0)
            
            # Masks
            # Scalar mask for dS (Head only)
            # [CRITICAL FIX] We MUST mask dS load with geometry to avoid reading 
            # garbage/zeros from invalid window slots that weren't written to.
            mask_scalar = mask_h[:, None] & is_valid_geom
            
            # Vector mask for K (Head & D)
            mask_vec = mask_op & is_valid_geom

            # 1. Load dS for this window position [BLOCK_H]
            # DS Index is exactly 'w'
            ds_val = tl.load(ds_base + (w * sds_lvl), mask=mask_scalar, other=0.0)

            # 2. Load K from neighbor [BLOCK_H, BLOCK_D]
            # Use offs_h (not h_idx)
            off_k = (safe_neighbor * sk_n) + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
            k_val = tl.load(k_batch_base + off_k, mask=mask_vec, other=0.0)

            # 3. Accumulate
            dq_acc += ds_val * k_val

        # -------------------------------------------------------
        # B. Hierarchy Loop (Updated Offsets)
        # -------------------------------------------------------
        for lvl_idx in range(LEVELS):
            # 1. Load Topology
            off_lookup = node_idx * sl_n + lvl_idx * sl_lvl
            
            # [Minor Safety] Mask the topology load for consistency
            # Since lvl_idx < LEVELS (constexpr), this is safe, but clean.
            mask_lvl = lvl_idx < LEVELS
            p_idx = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl, other=-1)
            
            # 2. Check Validity
            is_valid = (p_idx != -1)
            if HAS_MASK:
                mask_val = tl.load(Mask_ptr + off_lookup, mask=mask_lvl, other=1).to(tl.int8)
                is_valid = is_valid & (mask_val == 0)

            # 3. Load dS for this hierarchy level
            # [CRITICAL UPDATE] Offset is now WINDOW_SIZE + lvl_idx
            ds_cross_ptr = ds_base + ((WINDOW_SIZE + lvl_idx) * sds_lvl)
            
            # Strictly mask dS load with is_valid
            ds_cross = tl.load(ds_cross_ptr, mask=mask_h[:, None] & is_valid, other=0.0)

            # 4. Load K for this Parent
            safe_p_idx = tl.where(is_valid, p_idx, 0)
            mask_k = mask_op & is_valid

            # Use offs_h (not h_idx)
            off_k_cross = (safe_p_idx * sk_n) + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
            k_cross = tl.load(k_batch_base + off_k_cross, mask=mask_k, other=0.0)

            # 5. Accumulate
            dq_acc += ds_cross * k_cross

        # -------------------------------------------------------
        # C. Write Result
        # -------------------------------------------------------
        off_dq_out = offs_d[None, :] * sdq_d
        tl.store(dq_base + off_dq_out, dq_acc.to(DQ_ptr.dtype.element_ty), mask=mask_op)