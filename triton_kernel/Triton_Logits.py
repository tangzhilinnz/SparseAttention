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
    # Reduce redundant arithmetic inside the loops by calculating constant offsets once.
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



import torch
import triton
import triton.language as tl
import math

# ---------------- Self-Logits (MODIFIED with internal batch loop) ----------------
@triton.jit
def hierarchical_self_logits_kernel_opt(
    # Pointers & Strides are the same
    ALL_Q, ALL_K, RES,
    stride_b_q, stride_node_q, stride_d_q,
    stride_b_res, stride_seq_res, stride_h_res,
    
    # MODIFIED: Added B as a constant
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Grid is now 1D, for the token dimension only
    token_idx = tl.program_id(0)

    # --- MODIFIED: Loop over each batch item sequentially WITHIN the kernel ---
    for b_idx in range(B):
        # The internal logic is the original vectorized version, now inside the loop
        ptr_qk_row = ALL_Q + b_idx * stride_b_q + token_idx * stride_node_q
        h_indices = tl.arange(0, H)
        acc = tl.zeros((H,), dtype=tl.float32)
        for d_offset in range(0, D, BLOCK_SIZE_D):
            d_indices = d_offset + tl.arange(0, BLOCK_SIZE_D)
            d_mask = d_indices < D
            offsets = h_indices[:, None] * D + d_indices[None, :]
            q_vals = tl.load(ptr_qk_row + offsets, mask=d_mask[None, :], other=0.0)
            k_vals = tl.load(ptr_qk_row + offsets, mask=d_mask[None, :], other=0.0)
            acc += tl.sum(q_vals.to(tl.float32) * k_vals.to(tl.float32), axis=1)

        scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
        scaled_acc = acc * scale

        res_ptr_base = RES + b_idx * stride_b_res + token_idx * stride_seq_res
        res_ptrs = res_ptr_base + h_indices * stride_h_res
        tl.store(res_ptrs, scaled_acc)


# ---------------- Cross-Logits (MODIFIED with internal batch loop) ----------------
@triton.jit
def hierarchical_cross_logits_kernel_opt(
    # Pointers & Strides are the same
    ALL_Q, ALL_K, RES,
    stride_b_q, stride_node_q, stride_d_q,
    stride_b_res, stride_m_res, stride_h_res,

    # MODIFIED: Added B as a constant
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Grid is now 1D, for the node dimension only
    node_idx = tl.program_id(0)

    # --- MODIFIED: Loop over each batch item sequentially WITHIN the kernel ---
    for b_idx in range(B):
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

        res_ptr_base = RES + b_idx * stride_b_res + k_node * stride_m_res
        res_ptrs = res_ptr_base + h_indices * stride_h_res
        tl.store(res_ptrs, scaled_acc)


# ---------------- Python Launcher (MODIFIED with 1D grid) ----------------
def build_hierarchical_logits(
    ALL_Q, ALL_K,
    H, D,
    seq_len,
    BLOCK_SIZE_D=32,
    num_warps=4,
):
    B, total_nodes, D_h = ALL_Q.shape
    assert D_h == H * D

    if not ALL_Q.is_contiguous() or not ALL_K.is_contiguous():
        ALL_Q = ALL_Q.contiguous()
        ALL_K = ALL_K.contiguous()

    stride_b_q, stride_node_q, stride_d_q = ALL_Q.stride()
    
    M = total_nodes - seq_len
    RES_CROSS = torch.empty((B, M, H), device=ALL_Q.device, dtype=torch.float32)
    stride_b_r_cross, stride_m_r_cross, stride_h_r_cross = RES_CROSS.stride()

    # MODIFIED: Grid is now 1D, batch dimension is removed
    grid_cross = (M,)
    hierarchical_cross_logits_kernel_opt[grid_cross](
        ALL_Q, ALL_K, RES_CROSS,
        stride_b_q, stride_node_q, stride_d_q,
        stride_b_r_cross, stride_m_r_cross, stride_h_r_cross,
        # MODIFIED: Pass B as a constexpr
        B=B, H=H, D=D,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps
    )

    RES_SELF = torch.empty((B, seq_len, H), device=ALL_Q.device, dtype=torch.float32)
    stride_b_r_self, stride_seq_r_self, stride_h_r_self = RES_SELF.stride()

    # MODIFIED: Grid is now 1D, batch dimension is removed
    grid_self = (seq_len,)
    hierarchical_self_logits_kernel_opt[grid_self](
        ALL_Q, ALL_K, RES_SELF,
        stride_b_q, stride_node_q, stride_d_q,
        stride_b_r_self, stride_seq_r_self, stride_h_r_self,
        # MODIFIED: Pass B as a constexpr
        B=B, H=H, D=D,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps
    )

    return RES_SELF, RES_CROSS



# ------------------------------------------------------------------
# -------------------- Standard Attention (PyTorch) ----------------
# ------------------------------------------------------------------

def einsum_attention_logits(q, k):
    """
    Calculates standard full attention logits using torch.einsum.
    This is a highly optimized way to perform batched matrix multiplication.

    Args:
        q: Query tensor of shape [B, H, N, D]
        k: Key tensor of shape [B, H, N, D]

    Returns:
        Logits tensor of shape [B, H, N, N]
    """
    # b: batch size, h: number of heads, n: query sequence length,
    # m: key sequence length, d: head dimension
    # einsum calculates the dot product between q and k for each head and batch item.
    scale = 1.0 / math.sqrt(k.size(-1))
    return torch.einsum('bhnd,bhmd->bhnm', q, k) * scale



# ------------------------------------------------------------------
# -------------------- Main Execution & Benchmark ------------------
# ------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # --- Parameters ---
    B, H = 8, 8
    N =  256    # Number of tokens
    D = 32       # Dimension per head
    num_trials = 50

    assert (N & (N - 1)) == 0, f"N must be a power of 2, got N={N}"
    D_h = D * H
    print(f"Benchmark Parameters: B={B}, N={N}, H={H}, D={D}, D_h={D_h}")

    # --- Random input tensors ---
    tok_q = torch.randn(B, N, D_h, device="cuda", dtype=torch.float16)
    tok_k = torch.randn_like(tok_q)
    tok_v = torch.randn_like(tok_q)

    # --- Precompute hierarchy levels for Triton method ---
    level_sizes = []
    cur = N
    while cur // 2 > 0:
        cur //= 2
        level_sizes.append(cur)
    total_nodes = N + sum(level_sizes)

    # --- Allocate Triton buffers ---
    ALL_Q = torch.zeros((B, total_nodes, D_h), device="cuda", dtype=torch.float16)
    ALL_K = torch.zeros_like(ALL_Q)
    ALL_V = torch.zeros_like(ALL_Q)

    #build_hierarchical_span_nodes(
    #    tok_q, tok_k,tok_v,
    #    ALL_Q, ALL_K,ALL_V,
    #    BLOCK_SIZE=256, num_warps=4
    #)

    # --- Benchmark Hierarchical Span Attention (Triton) ---
    print("\n--- Benchmarking Hierarchical Span Attention (Triton) ---")
    torch.cuda.synchronize()
    t0_triton = time.time()
    for _ in range(num_trials):
        #build_hierarchical_span_nodes(
        #    tok_q, tok_k, tok_v,
        #    ALL_Q, ALL_K, ALL_V,
        #    BLOCK_SIZE=256, num_warps=4
        #)
        RES_SELF, RES_CROSS = build_hierarchical_logits(
            ALL_Q, ALL_K,
            H, D,
            seq_len=N,
            BLOCK_SIZE_D=D,   # try 64, 128, 256 for tuning
            num_warps=H       # 2, 4, 8 depending on your GPU
        )
    torch.cuda.synchronize()
    total_time_triton = time.time() - t0_triton
    avg_time_triton = total_time_triton / num_trials
    print(f"Hierarchical Triton total runtime: {total_time_triton:.6f} sec")
    print(f"Hierarchical Triton avg runtime:   {avg_time_triton:.6f} sec")
    print(f"  - RES_SELF shape:  {RES_SELF.shape} (expected [B, H, N])")
    print(f"  - RES_CROSS shape: {RES_CROSS.shape} (expected [B, H, total_nodes-1])")

    # --- Benchmark Standard Attention (PyTorch) ---
    print("\n--- Benchmarking Standard Full Attention (PyTorch) ---")
    
    # Reshape tensors to the standard [B, H, N, D] format for PyTorch attention
    q_std = tok_q.reshape(B, N, H, D).permute(0, 2, 1, 3).contiguous()
    k_std = tok_k.reshape(B, N, H, D).permute(0, 2, 1, 3).contiguous()
    
    # Warmup GPU
    _ = einsum_attention_logits(q_std, k_std)
    
    torch.cuda.synchronize()
    t0_std = time.time()
    for _ in range(num_trials):
        attn_logits = einsum_attention_logits(q_std, k_std)
    torch.cuda.synchronize()
    total_time_std = time.time() - t0_std
    avg_time_std = total_time_std / num_trials
    
    print(f"Standard PyTorch total runtime: {total_time_std:.6f} sec")
    print(f"Standard PyTorch avg runtime:   {avg_time_std:.6f} sec")
    print(f"  - Output shape: {attn_logits.shape} (expected [B, H, N, N])")

    # --- Performance Comparison ---
    print("\n--- Performance Comparison ---")
    if avg_time_triton > 0:
        ratio = avg_time_std / avg_time_triton
        print(f"Average time ratio (Standard / Hierarchical): {ratio:.2f}x")
        if ratio > 1:
            print(f"The hierarchical Triton implementation is {ratio:.2f} times FASTER than standard PyTorch attention.")
        else:
            print(f"The hierarchical Triton implementation is {1/ratio:.2f} times SLOWER than standard PyTorch attention.")
    else:
        print("Could not compute ratio due to zero Triton execution time.")
