import torch
import math

import triton
import triton.language as tl

import torch.nn.functional as F
import time

# --- (Your existing Triton kernels and helper functions go here) ---
# build_hierarchical_span_nodes, build_hierarchical_logits, 
# build_hierarchical_index_lookup_table, and hierarchical_attention_forward
# should be defined above this point.

# --- New Function for Standard PyTorch Attention ---
def standard_attention_pytorch(q, k, v, H, D):
    """
    Computes standard full attention using PyTorch's optimized implementation.
    This function leverages torch.nn.functional.scaled_dot_product_attention,
    which automatically uses memory-efficient attention (like FlashAttention)
    if the hardware and inputs are suitable.

    Args:
        q (torch.Tensor): Query tensor of shape (B, N, D_h).
        k (torch.Tensor): Key tensor of shape (B, N, D_h).
        v (torch.Tensor): Value tensor of shape (B, N, D_h).
        H (int): Number of attention heads.
        D (int): Dimension per head.

    Returns:
        torch.Tensor: Output tensor of shape (B, N, D_h).
    """
    B, N, D_h = q.shape

    # Reshape from (B, N, D_h) to (B, H, N, D) for multi-head processing
    q = q.view(B, N, H, D).transpose(1, 2)
    k = k.view(B, N, H, D).transpose(1, 2)
    v = v.view(B, N, H, D).transpose(1, 2)

    # Use PyTorch's built-in, highly optimized scaled dot-product attention
    out_pytorch = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    # Reshape back to the original (B, N, D_h) format
    out_pytorch = out_pytorch.transpose(1, 2).contiguous().view(B, N, D_h)
    
    return out_pytorch






























def build_hierarchical_index_lookup_table(seq_len, device="cuda", dtype=torch.int32):
    """
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

    # 1. Initialize Tensors
    # Mask defaults to True (attend to everything), we set False to mask out future tokens.
    causal_mask = torch.full((seq_len, level_num), False, dtype=torch.bool, device=device)
    
    # Map defaults to -1 (padding/invalid)
    idx_map = torch.full((seq_len, level_num), -1, dtype=dtype, device=device)

    for n in range(seq_len):
        n_cur = n # Starts as the leaf index
        
        for lvl in range(level_num):
            # --- 1. Calculate the Neighbor (n_next) and Self/Ancestor (pair) ---
            if lvl == 0:
                n_next = n_cur ^ 1  # Sibling leaf
                pair = n_cur        # The leaf itself
            else:
                # Formula: (Child_Index // 2) + Offset
                # Note: We use n_cur (which is the *neighbor* from prev loop).
                # This works because floor(neighbor / 2) == floor(self / 2).
                n_next = (n_cur // 2 + seq_len) ^ 1 # Uncle
                pair = (n_cur // 2 + seq_len)       # Parent

            # --- 2. Boundary Check ---
            # If the neighbor is the Root or out of bounds
            if n_next > max_valid:
                break

            # --- 3. Causal Masking Logic ---
            # If our Ancestor (pair) appears BEFORE the Neighbor (n_next),
            # it means the Neighbor is in the "Future" (Right branch).
            # We must mask it out.
            if pair < n_next:
                causal_mask[n, lvl] = True

            # --- 4. Update for next iteration ---
            idx_map[n, lvl] = n_next
            n_cur = n_next # Climb up via the neighbor

    return idx_map, causal_mask



import math
import torch
import triton
import triton.language as tl
import time
import torch.nn.functional as F # Import the functional module

# ------------------------------------------------------------------
# -------------------- Hierarchical Span Kernels -------------------
# ------------------------------------------------------------------

# --- Optimization 1: Define Autotuner Configurations ---
# This allows Triton to benchmark different kernel configurations and pick the fastest one
# for the specific hardware and input shapes.
# num_stages tells the compiler how much to pipeline data loads, hiding memory latency.
# num_warps allocates more threads to the work.
#AUTOTUNE_CONFIGS = [
#    triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
#    triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
#    triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
#    triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
#    triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
#]
#
#@triton.autotune(
#    configs=AUTOTUNE_CONFIGS,
#    key=['D_h'], # Autotune based on the dimension we are iterating over
#)
@triton.jit
def build_hierarchical_span_nodes_kernel(
    ALL_Q, ALL_K, ALL_V,        # [B, total_nodes, D_h]
    CHILD_BASE_NODE, OUT_BASE_NODE,
    D_h,
    stride_b, stride_node, stride_d,    # strides for [B, node, D_h]
    BLOCK_SIZE: tl.constexpr
):
    parent_idx = tl.program_id(0)   # 0..n_parents-1
    b_idx = tl.program_id(1)        # 0..B-1

    # --- Optimization 2: Pre-calculate base pointers ---
    b_offset = b_idx * stride_b
    c0_node_offset = CHILD_BASE_NODE + 2 * parent_idx
    c1_node_offset = c0_node_offset + 1
    out_node_offset = OUT_BASE_NODE + parent_idx

    ptr_c0_base = b_offset + c0_node_offset * stride_node
    ptr_c1_base = b_offset + c1_node_offset * stride_node
    ptr_out_base = b_offset + out_node_offset * stride_node

    # --- Pass 1: Dot products. Accumulate in high precision. ---
    dot0 = tl.zeros([1], dtype=tl.float32)
    dot1 = tl.zeros([1], dtype=tl.float32)

    for off in range(0, D_h, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        d_offset = cols * stride_d
        mask_cols = cols < D_h

        # Construct pointers to the current block of data
        ptr_k0 = ALL_K + ptr_c0_base + d_offset
        ptr_k1 = ALL_K + ptr_c1_base + d_offset
        ptr_q_parent = ALL_Q + ptr_out_base + d_offset

        # Load K vectors and the parent Q vector
        k0 = tl.load(ptr_k0, mask=mask_cols, other=0.).to(tl.float32)
        k1 = tl.load(ptr_k1, mask=mask_cols, other=0.).to(tl.float32)
        q_parent = tl.load(ptr_q_parent, mask=mask_cols, other=0.).to(tl.float32)

        # Update dot products
        dot0 += tl.sum(q_parent * k0)
        dot1 += tl.sum(q_parent * k1)

    # --- Softmax weights (Scalar computation, already fast) ---
    scale = 1.0 / tl.sqrt(D_h.to(tl.float32))
    l0, l1 = dot0 * scale, dot1 * scale
    m = tl.maximum(l0, l1)
    e0, e1 = tl.exp(l0 - m), tl.exp(l1 - m)
    denom = e0 + e1 + 1e-6
    w0 = e0 / denom
    w1 = e1 / denom

    # --- Pass 2: Weighted sums ---
    # This second loop is necessary because w0 and w1 depend on the full dot product.
    for off in range(0, D_h, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        d_offset = cols * stride_d
        mask_cols = cols < D_h

        # Construct pointers for all required tensors
        ptr_q0 = ALL_Q + ptr_c0_base + d_offset
        ptr_q1 = ALL_Q + ptr_c1_base + d_offset
        ptr_k0 = ALL_K + ptr_c0_base + d_offset
        ptr_k1 = ALL_K + ptr_c1_base + d_offset
        ptr_v0 = ALL_V + ptr_c0_base + d_offset
        ptr_v1 = ALL_V + ptr_c1_base + d_offset
        ptr_out = ptr_out_base + d_offset # Combined pointer for Q, K, V outputs

        # Load all child data required for the weighted sum
        q0 = tl.load(ptr_q0, mask=mask_cols, other=0.)#.to(tl.float32)
        q1 = tl.load(ptr_q1, mask=mask_cols, other=0.)#.to(tl.float32)
        k0 = tl.load(ptr_k0, mask=mask_cols, other=0.)#.to(tl.float32)
        k1 = tl.load(ptr_k1, mask=mask_cols, other=0.)#.to(tl.float32)
        v0 = tl.load(ptr_v0, mask=mask_cols, other=0.)#.to(tl.float32)
        v1 = tl.load(ptr_v1, mask=mask_cols, other=0.)#.to(tl.float32)

        # Calculate the weighted sums
        q_span = w0 * q0 + w1 * q1
        k_span = w0 * k0 + w1 * k1
        v_span = w0 * v0 + w1 * v1
        v_span = v0 + v1 # Unweighted sum for V

        # Store results back to global memory
        tl.store(ALL_Q + ptr_out, q_span.to(ALL_Q.dtype.element_ty), mask=mask_cols)
        tl.store(ALL_K + ptr_out, k_span.to(ALL_K.dtype.element_ty), mask=mask_cols)
        tl.store(ALL_V + ptr_out, v_span.to(ALL_V.dtype.element_ty), mask=mask_cols)


def build_hierarchical_span_nodes(
    tok_q, tok_k, tok_v,
    ALL_Q, ALL_K, ALL_V,
    dropout_p=0.1,        # <-- Add dropout probability
    is_training=False,    # <-- Add training status flag
    BLOCK_SIZE=256, num_warps=4
):
    """
    Triton wrapper for BD layout.

    tok_q/k/v: [B, N, D_h]
    ALL_Q/K/V: [B, total_nodes, D_h] preallocated
    """
    assert tok_q.is_cuda
    B, N, D_h = tok_q.shape

    level_sizes = []
    cur = N
    while cur // 2 > 0:
        cur = cur // 2
        level_sizes.append(cur)

    level_starts = [N]
    offset = N
    for size in level_sizes[:-1]:
        offset += size
        level_starts.append(offset)

    stride_b, stride_node, stride_d = ALL_Q.stride()

    # Apply dropout to the initial leaf V-vectors if in training mode.
    if dropout_p > 0.0 and is_training:
        tok_v = F.dropout(tok_v, p=dropout_p, training=is_training)

    # copy leaf tokens into ALL buffers (tok_v is now the potentially dropped-out version)
    ALL_Q[:, :N, :].copy_(tok_q)
    ALL_K[:, :N, :].copy_(tok_k)
    ALL_V[:, :N, :].copy_(tok_v)

    for lvl_idx, n_parents in enumerate(level_sizes):
        CHILD_BASE_NODE = 0 if lvl_idx == 0 else level_starts[lvl_idx - 1]
        OUT_BASE_NODE = level_starts[lvl_idx]

        grid = (n_parents, B)  # parent_idx first, then batch_idx
        build_hierarchical_span_nodes_kernel[grid](
            ALL_Q, ALL_K, ALL_V,
            CHILD_BASE_NODE, OUT_BASE_NODE,
            D_h,
            stride_b, stride_node, stride_d,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )

    return level_starts, level_sizes



# ---------------- Optimized Self-Logits (head axis last: RES_SELF shape = (B, Seq, H)) ----------------
@triton.jit
def hierarchical_self_logits_kernel_opt(
    # Pointers
    ALL_Q, ALL_K, RES,
    # Strides for ALL_Q / ALL_K
    stride_b_q, stride_seq_q, stride_h_q,
    stride_b_k, stride_seq_k, stride_h_k,
    # Strides for RES_SELF (B, Seq, H) -> (stride_b_res, stride_seq_res, stride_h_res)
    stride_b_res, stride_seq_res, stride_h_res,
    # Constants
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 1. Coordinate Setup
    token_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    # 2. Base Pointer Arithmetic
    q_base = ALL_Q + b_idx * stride_b_q + token_idx * stride_seq_q
    k_base = ALL_K + b_idx * stride_b_k + token_idx * stride_seq_k

    # 3. Head Offsets (Pre-calculated)
    offs_h = tl.arange(0, H)
    q_head_ptrs = q_base + offs_h[:, None] * stride_h_q
    k_head_ptrs = k_base + offs_h[:, None] * stride_h_k

    #ptr_qk_row = ALL_Q + b_idx * stride_b_q + token_idx * stride_node_q
    #h_indices = tl.arange(0, H)

    # 4. Accumulator
    acc = tl.zeros((H,), dtype=tl.float32)

    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        
        # --- EXTREME OPTIMIZATION: HINTS ---
        # 1. Masking: Only needed if D is not a multiple of BLOCK_D
        mask = offs_d < D
        
        # 2. Contiguity Hint: Tell Triton that 'offs_d' is contiguous.
        # This enables vectorization (merging adjacent loads).
        offs_d = tl.max_contiguous(tl.multiple_of(offs_d, BLOCK_D), BLOCK_D)
        
        # Calculate final pointers for this block
        # Broadcasting: (H, 1) + (1, BLOCK_D) -> (H, BLOCK_D)
        # Note: We implicitly assume stride_d is 1 here by adding offs_d directly.
        q_ptrs = q_head_ptrs + offs_d[None, :]
        k_ptrs = k_head_ptrs + offs_d[None, :]

        # Load
        q_chunk = tl.load(q_ptrs, mask=mask[None, :], other=0.0)
        k_chunk = tl.load(k_ptrs, mask=mask[None, :], other=0.0)

        # Math: FMA (Fused Multiply Add)
        # Accumulate the dot product along the D dimension
        acc += tl.sum(q_chunk * k_chunk, axis=1)

    # 6. Final Scale and Store
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    acc = acc * scale

    # Output pointers
    res_base = RES_ptr + b_idx * stride_b_res + token_idx * stride_seq_res
    res_ptrs = res_base + offs_h * stride_h_res
    
    tl.store(res_ptrs, acc)

    # for d_offset in range(0, D, BLOCK_SIZE_D):
    #     d_indices = d_offset + tl.arange(0, BLOCK_SIZE_D)
    #     d_mask = d_indices < D
    #     offsets = h_indices[:, None] * D + d_indices[None, :]
    #     q_vals = tl.load(ptr_qk_row + offsets, mask=d_mask[None, :], other=0.0)
    #     k_vals = tl.load(ptr_qk_row + offsets, mask=d_mask[None, :], other=0.0)
    #     acc += tl.sum(q_vals.to(tl.float32) * k_vals.to(tl.float32), axis=1)
    # 
    # scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    # scaled_acc = acc * scale
    # 
    # # RES layout: (B, Seq, H). Store at RES[b_idx, token_idx, h]
    # res_ptr_base = RES + b_idx * stride_b_res + token_idx * stride_seq_res
    # res_ptrs = res_ptr_base + h_indices * stride_h_res
    # 
    # tl.store(res_ptrs, scaled_acc)


# ---------------- Optimized Cross-Logits (RES_CROSS shape = (B, M, H)) ----------------
@triton.jit
def hierarchical_cross_logits_kernel_opt(
    # Pointers
    ALL_Q, ALL_K, RES,
    # Strides for ALL_Q / ALL_K
    stride_b_q, stride_node_q, stride_d_q,
    # Strides for RES_CROSS (B, M, H) -> (stride_b_res, stride_m_res, stride_h_res)
    stride_b_res, stride_m_res, stride_h_res,
    # Constants
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    q_node = node_idx
    k_node = node_idx ^ 1

    ptr_q_row = ALL_Q + b_idx * stride_b_q + q_node * stride_node_q
    ptr_k_row = ALL_K + b_idx * stride_b_q + k_node * stride_node_q

    h_indices = tl.arange(0, H)
    acc = tl.zeros((H,), dtype=tl.float32)
    for d_offset in range(0, D, BLOCK_SIZE_D):
        d_indices = d_offset + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_indices < D
        offsets = h_indices[:, None] * D + d_indices[None, :]
        q_vals = tl.load(ptr_q_row + offsets, mask=d_mask[None, :], other=0.0)
        k_vals = tl.load(ptr_k_row + offsets, mask=d_mask[None, :], other=0.0)
        acc += tl.sum(q_vals.to(tl.float32) * k_vals.to(tl.float32), axis=1)

    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    scaled_acc = acc * scale

    # RES layout: (B, M, H). We store at RES[b_idx, k_node, h]
    res_ptr_base = RES + b_idx * stride_b_res + k_node * stride_m_res
    res_ptrs = res_ptr_base + h_indices * stride_h_res

    tl.store(res_ptrs, scaled_acc)


# ---------------- Python Launcher (fixing RES shapes to (B, M, H) and (B, Seq, H)) ----------------
def build_hierarchical_logits(
    ALL_Q, ALL_K,
    H, D,
    seq_len,
    BLOCK_SIZE_D=32,
    num_warps=4,
):
    """
    Launches optimized self- and cross-logits kernels without reshaping inputs.
    ALL_Q shape: (B, total_nodes, D_h) where D_h = H*D
    """
    B, total_nodes, D_h = ALL_Q.shape
    assert D_h == H * D

    if not ALL_Q.is_contiguous() or not ALL_K.is_contiguous():
        ALL_Q = ALL_Q.contiguous()
        ALL_K = ALL_K.contiguous()

    stride_b_q, stride_node_q, stride_d_q = ALL_Q.stride()
    assert stride_d_q == 1, "Input tensors must be contiguous on the last dimension"

    # Cross logits: M = total_nodes - 1 (or whatever your definition)
    M = total_nodes - 1
    # Allocate RES_CROSS with shape (B, M, H)
    RES_CROSS = torch.empty((B, M, H), device=ALL_Q.device, dtype=torch.float32)
    stride_b_r_cross, stride_m_r_cross, stride_h_r_cross = RES_CROSS.stride()

    grid_cross = (M, B)
    hierarchical_cross_logits_kernel_opt[grid_cross](
        ALL_Q, ALL_K, RES_CROSS,
        stride_b_q, stride_node_q, stride_d_q,
        stride_b_r_cross, stride_m_r_cross, stride_h_r_cross,
        H=H, D=D,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps
    )

    # Self logits: allocate RES_SELF with shape (B, seq_len, H)
    RES_SELF = torch.empty((B, seq_len, H), device=ALL_Q.device, dtype=torch.float32)
    stride_b_r_self, stride_seq_r_self, stride_h_r_self = RES_SELF.stride()

    grid_self = (seq_len, B)
    hierarchical_self_logits_kernel_opt[grid_self](
        ALL_Q, ALL_K, RES_SELF,
        stride_b_q, stride_node_q, stride_d_q,
        stride_b_r_self, stride_seq_r_self, stride_h_r_self,
        H=H, D=D,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps
    )

    return RES_SELF, RES_CROSS














@triton.jit
def hierarchical_self_attention_result_kernel(
    # Pointers
    RES_CROSS,      # (B, M, H)
    RES_SELF,       # (B, Seq, H)
    LOOKUP,         # (Seq, LEVELS)
    V_all,          # (B, M, D_h)
    OUT,            # (B, Seq, D_h)

    # Strides
    stride_b_cross, stride_s_cross, stride_h_cross,
    stride_b_self, stride_s_self, stride_h_self,
    stride_b_v, stride_s_v, stride_dh_v,
    stride_b_out, stride_seq_out, stride_dh_out,

    # Constants
    H: tl.constexpr,
    D: tl.constexpr,
    D_h: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    # --- KERNEL SETUP ---
    # This entire kernel instance computes the result for a single token (`token_idx`)
    # from a single batch (`b_idx`).
    token_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    # `lane` represents the parallel threads available in this instance.
    # These threads will be primarily used to fetch the `LEVELS` different cross logits.
    lane = tl.arange(0, BLOCK_SIZE_D)
    lvl_mask = lane < LEVELS

    # `h_idx` creates a vector [0, 1, 2, ..., H-1]. This is the key to head parallelism.
    # Instead of assigning threads to heads, we make every operation aware of all heads
    # by broadcasting across this vector.
    h_idx = tl.arange(0, H)

    # --- LOADING ALL LOGITS FOR ALL HEADS ---
    # Each of the `lane` threads fetches one `node_index` from the LOOKUP table.
    lookup_row_ptr = LOOKUP + token_idx * LEVELS
    node_indices = tl.load(lookup_row_ptr + lane, mask=lvl_mask, other=0).to(tl.int32) # Shape: (BLOCK_SIZE_D,)

    # Now we fetch the cross logits. The pointer math is broadcastable.
    res_cross_b_base = RES_CROSS + b_idx * stride_b_cross
    # `h_idx[:, None]` has shape (H, 1) and `node_indices[None, :]` has shape (1, BLOCK_SIZE_D).
    # Triton broadcasts these to create a (H, BLOCK_SIZE_D) set of pointers.
    # This simultaneously fetches all `LEVELS` of cross logits for all `H` heads.
    ptr_cross = res_cross_b_base + node_indices[None, :] * stride_s_cross + h_idx[:, None] * stride_h_cross
    cross_logits = tl.load(ptr_cross, mask=lvl_mask[None, :], other=-1e9).to(tl.float32) # Shape: (H, BLOCK_SIZE_D)

    # Similarly, we fetch the single self-logit for the current token for all H heads at once.
    res_self_b_base = RES_SELF + b_idx * stride_b_self
    ptr_self = res_self_b_base + token_idx * stride_s_self + h_idx * stride_h_self
    self_logits = tl.load(ptr_self, mask=h_idx < H, other=0.).to(tl.float32) # Shape: (H,)

    # --- PER-HEAD SOFTMAX ---
    # At this point, `cross_logits` holds all cross logits for all heads,
    # and `self_logits` holds the self logit for each head.

    # `tl.max(..., axis=1)` computes the maximum value independently FOR EACH HEAD
    # across its `LEVELS` of cross logits.
    max_cross = tl.max(cross_logits, axis=1) # Shape: (H,)
    # Now find the overall max for each head (comparing its self logit to its max cross logit).
    m = tl.maximum(self_logits, max_cross) # Shape: (H,)

    # The rest of the math is now broadcast. `m` is subtracted from the logits of each head.
    exp_self = tl.exp(self_logits - m)
    exp_cross = tl.exp(cross_logits - m[:, None])
    
    # The denominator is calculated independently for each head.
    pow2 = tl.exp2(lane.to(tl.float32)) * lvl_mask.to(tl.float32)
    pow2 = pow2[None, :]
    # `tl.sum(..., axis=1)` sums the weighted cross logits FOR EACH HEAD.
    denom = exp_self + tl.sum(exp_cross * pow2, axis=1) # Shape: (H,)

    # Final weights are computed FOR EACH HEAD.
    weight_self = exp_self / (denom + 1e-12) # Shape: (H,)
    weight_cross = (exp_cross / (denom[:, None] + 1e-12)) * lvl_mask[None, :] # Shape: (H, BLOCK_SIZE_D)

    # --- PER-HEAD WEIGHTED SUM & STORE ---
    self_node_idx = token_idx
    v_base_ptr = V_all + b_idx * stride_b_v
    out_base_ptr = OUT + b_idx * stride_b_out + token_idx * stride_seq_out

    # This loop processes the D dimension in chunks. The logic inside is also vectorized across heads.
    for d_off in range(0, D, BLOCK_SIZE_D):
        d_indices = d_off + lane
        d_mask_block = d_indices < D
        
        # `d_offsets` calculates the memory locations for a block of the D dimension
        # for all H heads simultaneously.
        d_offsets = h_idx[:, None] * D + d_indices[None, :] # Shape: (H, BLOCK_SIZE_D)

        # Load value vectors for the self-token for all heads.
        ptr_v_self = v_base_ptr + self_node_idx * stride_s_v + d_offsets * stride_dh_v
        v_self = tl.load(ptr_v_self, mask=d_mask_block[None, :], other=0.).to(tl.float32)
        
        # Start calculating the output block FOR EACH HEAD using its specific `weight_self`.
        out_block = weight_self[:, None] * v_self
        
        # Load value vectors for the cross-nodes for all heads.
        ptr_v_cross = v_base_ptr + node_indices[None, :, None] * stride_s_v + d_offsets[:, None, :] * stride_dh_v
        v_cross = tl.load(ptr_v_cross, mask=(lvl_mask[None, :, None] & d_mask_block[None, None, :]), other=0.).to(tl.float32)
        
        # `tl.sum(..., axis=1)` computes the weighted sum of cross-values independently FOR EACH HEAD.
        weighted_v_cross = weight_cross[:, :, None] * v_cross
        out_block += tl.sum(weighted_v_cross, axis=1)

        # Store the final result block. This writes the data for all H heads
        # into their correct interleaved positions in the output tensor in one go.
        out_ptr = out_base_ptr + d_offsets * stride_dh_out
        tl.store(out_ptr, out_block.to(OUT.dtype.element_ty), mask=d_mask_block[None, :])


## --------------------------------------------------------------------------------------

def hierarchical_attention_forward(
    res_cross: torch.Tensor,
    res_self: torch.Tensor,
    lookup: torch.Tensor,
    v_all: torch.Tensor,
    min_block_size_d: int = 32,
    num_warps: int = 4
) -> torch.Tensor:

    B, M, H = res_cross.shape
    _, Seq, _ = res_self.shape
    _, _, D_h = v_all.shape
    _, LEVELS = lookup.shape

    assert D_h % H == 0, "D_h must be divisible by H"
    D = D_h // H

    # MODIFIED: Create the output tensor in the fused (B, Seq, D_h) format
    out = torch.empty((B, Seq, D_h), device=res_cross.device, dtype=v_all.dtype)

    grid = lambda meta: (Seq, B)

    # MODIFIED: Update BLOCK_SIZE_D calculation
    # 1. Assert that the given value is a power of 2
    assert (min_block_size_d & (min_block_size_d - 1) == 0) and min_block_size_d > 0, \
        "min_block_size_d must be a power of 2"
    # 2. Find the minimum block size required by the number of levels
    required_size = triton.next_power_of_2(LEVELS)
    # 3. Use the maximum of the required size and the user-provided minimum
    BLOCK_SIZE_D = max(required_size, min_block_size_d)

    hierarchical_self_attention_result_kernel[grid](
        res_cross, res_self, lookup, v_all, out,
        res_cross.stride(0), res_cross.stride(1), res_cross.stride(2),
        res_self.stride(0), res_self.stride(1), res_self.stride(2),
        v_all.stride(0), v_all.stride(1), v_all.stride(2),
        # MODIFIED: Pass the 3D strides for the new output tensor
        out.stride(0), out.stride(1), out.stride(2),
        H=H,
        D=D,
        D_h=D_h,
        LEVELS=LEVELS,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps
    )

    return out












if __name__ == "__main__":
    import time
    torch.manual_seed(42)

    # --- Parameters ---
    B, H = 64, 8
    N = 256   # sequence length (must be power of 2)
    D = 64         # dimension per head
    num_trials = 50

    assert (N & (N - 1)) == 0, f"N must be a power of 2, got N={N}"
    D_h = D * H
    print(f"Benchmark Parameters: B={B}, N={N}, H={H}, D={D}, D_h={D_h}")

    device = "cuda"

    # --- Random input tensors ---
    tok_q = torch.randn(B, N, D_h, device=device, dtype=torch.float16)
    tok_k = torch.randn_like(tok_q)
    tok_v = torch.randn_like(tok_q)

    # --- Compute hierarchical level sizes ---
    level_sizes = []
    cur = N
    while cur // 2 > 0:
        cur //= 2
        level_sizes.append(cur)
    total_nodes = N + sum(level_sizes)
    print(f"Hierarchy Levels: {level_sizes} | Total nodes = {total_nodes}")

    # --- Allocate hierarchical buffers ---
    ALL_Q = torch.zeros((B, total_nodes, D_h), device=device, dtype=torch.float16)
    ALL_K = torch.zeros_like(ALL_Q)
    ALL_V = torch.zeros_like(ALL_Q)

    ## --- Build hierarchical span nodes ---
    #print("\n--- Building hierarchical span nodes ---")
    #torch.cuda.synchronize()
    #build_hierarchical_span_nodes(tok_q, tok_k, tok_v, ALL_Q, ALL_K, ALL_V, BLOCK_SIZE=256, num_warps=4)
    #torch.cuda.synchronize()
    #
    ## --- Compute hierarchical logits (RES_SELF, RES_CROSS) ---
    #print("\n--- Building hierarchical logits ---")
    #RES_SELF, RES_CROSS = build_hierarchical_logits(
    #    ALL_Q, ALL_K, H, D, seq_len=N, BLOCK_SIZE_D=D, num_warps=H
    #)
    #torch.cuda.synchronize()

    #print(f"RES_SELF.shape: {RES_SELF.shape}  (expected [B, Seq, H])")
    #print(f"RES_CROSS.shape: {RES_CROSS.shape}  (expected [B, M, H])")

    # --- Build lookup table ---
    lookup = build_hierarchical_index_lookup_table(N, device=device)
    print(f"LOOKUP.shape: {lookup.shape}  (Seq, LEVELS)")

    # --- Benchmark Hierarchical Attention Forward ---
    print("\n--- Benchmarking Hierarchical Attention Forward (Triton) ---")
    torch.cuda.synchronize()
    t0_triton = time.time()
    for _ in range(num_trials):
        build_hierarchical_span_nodes(
            tok_q, tok_k, tok_v,
            ALL_Q, ALL_K, ALL_V,
            BLOCK_SIZE=256, num_warps=4
        )
        RES_SELF, RES_CROSS = build_hierarchical_logits(
            ALL_Q, ALL_K,
            H, D,
            seq_len=N,
            BLOCK_SIZE_D=D,   # try 64, 128, 256 for tuning
            num_warps=H       # 2, 4, 8 depending on your GPU
        )
        OUT = hierarchical_attention_forward(
            RES_CROSS, RES_SELF, lookup, ALL_V,
            min_block_size_d=D,
            num_warps=H
        )
    torch.cuda.synchronize()
    total_time_triton = time.time() - t0_triton
    avg_time_triton = total_time_triton / num_trials
    print(f"Hierarchical Attention total runtime: {total_time_triton:.6f} sec")
    print(f"Hierarchical Attention avg runtime:   {avg_time_triton:.6f} sec")
    #print(f"  - OUT shape: {OUT.shape} (expected [B, Seq, D_h])")
    #
    ## Numerical check
    #if torch.isnan(OUT).any() or torch.isinf(OUT).any():
    #    print("Warning: NaN/Inf detected in output!")
    #else:
    #    print("Output numerical check passed (no NaN/Inf)")
    #
    ##print("Sample OUT[0, 0, :16] =", OUT[0, 0, :16].float().cpu().numpy())



    # --- Benchmark Standard Full Attention (PyTorch) ---
    print("\n--- Benchmarking Standard Full Attention (PyTorch) ---")
    torch.cuda.synchronize()
    t0_pytorch = time.time()
    for _ in range(num_trials):
        # Call the standard attention function
        out_pytorch = standard_attention_pytorch(tok_q, tok_k, tok_v, H, D)
    torch.cuda.synchronize()
    total_time_pytorch = time.time() - t0_pytorch
    avg_time_pytorch = total_time_pytorch / num_trials
    
    print(f"Standard Attention total runtime: {total_time_pytorch:.6f} sec")
    print(f"Standard Attention avg runtime:   {avg_time_pytorch:.6f} sec")
    print(f"  - out_pytorch shape: {out_pytorch.shape} (expected [B, Seq, D_h])")

    # Numerical check
    if torch.isnan(out_pytorch).any() or torch.isinf(out_pytorch).any():
        print("Warning: NaN/Inf detected in PyTorch output!")
    else:
        print("PyTorch output numerical check passed (no NaN/Inf)")

    #print("Sample out_pytorch[0, 0, :16] =", out_pytorch[0, 0, :16].float().cpu().numpy())


    # --- Performance Comparison ---
    print("\n--- Performance Comparison ---")
    if avg_time_triton > 0:
        ratio = avg_time_pytorch / avg_time_triton
        print(f"Average time ratio (Standard / Hierarchical): {ratio:.2f}x")
        if ratio > 1:
            print(f"The hierarchical Triton implementation is {ratio:.2f} times FASTER than standard PyTorch attention.")
        else:
            print(f"The hierarchical Triton implementation is {1/ratio:.2f} times SLOWER than standard PyTorch attention.")
    else:
        print("Could not compute ratio due to zero Triton execution time.")