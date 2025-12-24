import math
import torch
import triton
import triton.language as tl

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
        #v_span = w0 * v0 + w1 * v1
        v_span = v0 + v1 # Unweighted sum for V

        # Store results back to global memory
        tl.store(ALL_Q + ptr_out, q_span.to(ALL_Q.dtype.element_ty), mask=mask_cols)
        tl.store(ALL_K + ptr_out, k_span.to(ALL_K.dtype.element_ty), mask=mask_cols)
        tl.store(ALL_V + ptr_out, v_span.to(ALL_V.dtype.element_ty), mask=mask_cols)


def build_hierarchical_span_nodes(
    tok_q, tok_k, tok_v,
    ALL_Q, ALL_K, ALL_V,
    dropout_p=0.1,        # <-- Add dropout probability
    is_training=True,    # <-- Add training status flag
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



# ---------------- Fully-batched PyTorch reference (accepts preallocated ALL_*) ----------------
def build_hierarchical_spans_ref(
    tok_q, tok_k, tok_v,
    ALL_Q, ALL_K, ALL_V
):
    """
    Fully-batched PyTorch reference for BD layout.

    tok_q/k/v: [B, N, D_h]
    ALL_Q/K/V: [B, total_nodes, D_h] preallocated (will be written in-place)
    Returns: level_node_starts, level_sizes (for checking)
    """
    B, N, D_h = tok_q.shape

    # compute level sizes
    level_sizes = []
    cur = N
    while cur // 2 > 0:
        cur //= 2
        level_sizes.append(cur)

    level_node_starts = [N]
    offset = N
    for s in level_sizes[:-1]:
        offset += s
        level_node_starts.append(offset)
    # Note: level_node_starts length == len(level_sizes)

    # copy leaves into ALL_*
    ALL_Q[:, :N, :].copy_(tok_q)
    ALL_K[:, :N, :].copy_(tok_k)
    ALL_V[:, :N, :].copy_(tok_v)

    scale = 1.0 / math.sqrt(D_h)

    # process each level (vectorized across batch and parents)
    for lvl_idx, n_parents in enumerate(level_sizes):
        child_base = 0 if lvl_idx == 0 else level_node_starts[lvl_idx - 1]
        out_base = level_node_starts[lvl_idx]

        # extract children blocks: shape [B, n_parents, D_h]
        c0 = ALL_Q[:, child_base : child_base + 2 * n_parents : 2, :]   # first child of each pair
        c1 = ALL_Q[:, child_base + 1 : child_base + 2 * n_parents : 2, :]  # second child
        k0 = ALL_K[:, child_base : child_base + 2 * n_parents : 2, :]
        k1 = ALL_K[:, child_base + 1 : child_base + 2 * n_parents : 2, :]
        v0 = ALL_V[:, child_base : child_base + 2 * n_parents : 2, :]
        v1 = ALL_V[:, child_base + 1 : child_base + 2 * n_parents : 2, :]

        # parent query slots (read from parent slots same as Triton): [B, n_parents, D_h]
        q_parent = ALL_Q[:, out_base : out_base + n_parents, :].to(torch.float32)

        # compute logits and softmax per (B, parent)
        l0 = (q_parent * k0).sum(dim=-1) * scale   # [B, n_parents]
        l1 = (q_parent * k1).sum(dim=-1) * scale   # [B, n_parents]
        m = torch.maximum(l0, l1)
        e0 = torch.exp(l0 - m)
        e1 = torch.exp(l1 - m)
        denom = e0 + e1 + 1e-6
        w0 = (e0 / denom)[:, :, None]   # [B, n_parents, 1]
        w1 = (e1 / denom)[:, :, None]

        # weighted sums (broadcasted)
        q_span = w0 * c0 + w1 * c1      # [B, n_parents, D_h]
        k_span = w0 * k0 + w1 * k1
        v_span = w0 * v0 + w1 * v1

        # write into parent slots (preserve dtype)
        ALL_Q[:, out_base : out_base + n_parents, :].copy_(q_span.to(ALL_Q.dtype))
        ALL_K[:, out_base : out_base + n_parents, :].copy_(k_span.to(ALL_K.dtype))
        ALL_V[:, out_base : out_base + n_parents, :].copy_(v_span.to(ALL_V.dtype))

    return level_node_starts, level_sizes



import time

if __name__ == "__main__":
    torch.manual_seed(42)

    # -------------------------
    # Parameters
    # -------------------------
    B, H = 64, 16
    N = 256          # number of tokens
    D = 32
    num_trials = 50

    D_h = D * H   # combined head dimension
    print(f"Testing hierarchical span nodes: B={B}, N={N}, H={H}, D={D}, D_h={D_h}")

    # -------------------------
    # Random input tensors (BD layout)
    # -------------------------
    tok_q = torch.randn(B, N, D_h, device="cuda", dtype=torch.float16)
    tok_k = torch.randn_like(tok_q)
    tok_v = torch.randn_like(tok_q)

    # -------------------------
    # Precompute hierarchy levels
    # -------------------------
    level_sizes = []
    cur = N
    while cur // 2 > 0:
        cur //= 2
        level_sizes.append(cur)

    level_node_starts = [N]
    offset = N
    for s in level_sizes[:-1]:
        offset += s
        level_node_starts.append(offset)

    total_nodes = N + sum(level_sizes)
    print(f"Level sizes: {level_sizes}")
    print(f"Level node starts: {level_node_starts}")
    print(f"Total nodes: {total_nodes}")

    # -------------------------
    # Allocate Triton buffers
    # -------------------------
    ALL_Q_triton = torch.zeros((B, total_nodes, D_h), device="cuda", dtype=torch.float16)
    ALL_K_triton = torch.zeros_like(ALL_Q_triton)
    ALL_V_triton = torch.zeros_like(ALL_Q_triton)

    # -------------------------
    # Run Triton kernel
    # -------------------------
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_trials):
        build_hierarchical_span_nodes(
            tok_q, tok_k, tok_v,
            ALL_Q_triton, ALL_K_triton, ALL_V_triton,
            BLOCK_SIZE=256,
            num_warps=4
        )
    torch.cuda.synchronize()
    triton_avg = (time.time() - t0) / num_trials
    triton_total = time.time() - t0
    print(f"Triton kernel avg runtime: {triton_avg:.6f} sec")
    print(f"Triton kernel total runtime: {triton_total:.6f} sec")

    # -------------------------
    # Allocate reference buffers
    # -------------------------
    ALL_Q_ref = torch.zeros((B, total_nodes, D_h), device="cuda", dtype=torch.float16)
    ALL_K_ref = torch.zeros_like(ALL_Q_ref)
    ALL_V_ref = torch.zeros_like(ALL_Q_ref)

    # -------------------------
    # Run PyTorch reference
    # -------------------------
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(num_trials):
        level_node_starts_ref, level_sizes_ref = build_hierarchical_spans_ref(
            tok_q, tok_k, tok_v,
            ALL_Q_ref, ALL_K_ref, ALL_V_ref
        )
    torch.cuda.synchronize()
    ref_avg = (time.time() - t1) / num_trials
    ref_total = time.time() - t1
    print(f"PyTorch reference avg runtime: {ref_avg:.6f} sec")
    print(f"PyTorch reference total runtime: {ref_total:.6f} sec")

    # -------------------------
    # Performance summary
    # -------------------------
    print(f"Speedup (ref / triton): {ref_avg / triton_avg:.2f}x")

    # -------------------------
    # Sanity checks
    # -------------------------
    max_err = 0.0
    for lvl_idx, start in enumerate(level_node_starts):
        n_parents = level_sizes[lvl_idx]

        # Triton outputs
        q_triton = ALL_Q_triton[:, start:start+n_parents, :].to(torch.float32)
        k_triton = ALL_K_triton[:, start:start+n_parents, :].to(torch.float32)
        v_triton = ALL_V_triton[:, start:start+n_parents, :].to(torch.float32)

        # Reference outputs
        q_ref = ALL_Q_ref[:, start:start+n_parents, :].to(torch.float32)
        k_ref = ALL_K_ref[:, start:start+n_parents, :].to(torch.float32)
        v_ref = ALL_V_ref[:, start:start+n_parents, :].to(torch.float32)

        err_q = (q_triton - q_ref).abs().max().item()
        err_k = (k_triton - k_ref).abs().max().item()
        err_v = (v_triton - v_ref).abs().max().item()
        print(f"Level {lvl_idx}: n_parents={n_parents}, "
              f"max_err q/k/v = {err_q:.3e} / {err_k:.3e} / {err_v:.3e}")
        max_err = max(max_err, err_q, err_k, err_v)

    print(f"\nFinal max error across all levels: {max_err:.6f}")
    assert max_err < 1.5e-2, "Sanity check failed!"
    print("Sanity check passed.")





























@triton.jit
def hierarchical_self_logits_kernel(
    ALL_Q,            # [B, total_nodes, H*D]
    ALL_K,            # [B, total_nodes, H*D]
    B, H, D,          # scalars
    stride_b, stride_node, stride_d,   # strides for Q/K
    RES,              # [B, H, seq_len] (float32)
    stride_b_r, stride_h_r, stride_s_r,
    BLOCK_SIZE: tl.constexpr
):
    """
    Each program computes one self-logit:
      grid = (seq_len, B, H)
      program_id(0) -> token_idx
      program_id(1) -> b_idx
      program_id(2) -> h_idx

    Rule: q_node = k_node = token_idx (self dot-product)
    """

    token_idx = tl.program_id(0)
    b_idx     = tl.program_id(1)
    h_idx     = tl.program_id(2)

    # compute offsets for Q/K
    b_off = b_idx * stride_b
    node_off = token_idx * stride_node
    head_off = h_idx * D * stride_d

    ptr_q_base = b_off + node_off + head_off
    ptr_k_base = b_off + node_off + head_off

    acc = tl.zeros([1], dtype=tl.float32)

    # blocked dot-product over D
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        q_ptr = ALL_Q + ptr_q_base + cols * stride_d
        k_ptr = ALL_K + ptr_k_base + cols * stride_d

        qv = tl.load(q_ptr, mask=mask, other=0.).to(tl.float32)
        kv = tl.load(k_ptr, mask=mask, other=0.).to(tl.float32)

        acc += tl.sum(qv * kv)

    # scale like attention
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    score = acc * scale

    # store into RES[b, h, token_idx]
    res_ptr = b_idx * stride_b_r + h_idx * stride_h_r + token_idx * stride_s_r
    tl.store(RES + res_ptr, score)





@triton.jit
def hierarchical_cross_logits_kernel(
    ALL_Q,            # [B, total_nodes, D_h]
    ALL_K,            # [B, total_nodes, D_h]
    M,                # int32 scalar: number of nodes to process (sum of all levels except last)
    B, H, D,          # scalars
    stride_b, stride_node, stride_d,   # shared strides for Q/K
    RES,              # [B, H, M] (float32)
    stride_b_r, stride_h_r, stride_s_r,
    BLOCK_SIZE: tl.constexpr
):
    """
    Each instance computes one scalar score:
      grid = (M, B, H)
      program_id(0) -> node_idx (0..M-1)
      program_id(1) -> b_idx
      program_id(2) -> h_idx
    Pair rule: q_node = node_idx, k_node = node_idx ^ 1
    """

    node_idx = tl.program_id(0)   # 0..M-1
    b_idx = tl.program_id(1)
    h_idx = tl.program_id(2)

    ## quick bounds check (shouldn't be necessary if grid is exact)
    #if node_idx >= M:
    #    return

    # If node_idx is even (...0 in binary), then node_idx ^ 1 = node_idx + 1.
    # If node_idx is odd (...1 in binary), then node_idx ^ 1 = node_idx - 1.
    q_node = node_idx
    k_node = node_idx ^ 1  # flip low bit -> partner

    # compute base element offsets
    b_off   = b_idx * stride_b

    node_off_q = q_node * stride_node
    node_off_k = k_node * stride_node

    head_off = h_idx * D * stride_d

    ptr_q_base = b_off + node_off_q + head_off
    ptr_k_base = b_off + node_off_k + head_off

    acc = tl.zeros([1], dtype=tl.float32)
    # blocked dot-product over per-head dimension D
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        q_ptr = ALL_Q + ptr_q_base + cols * stride_d
        k_ptr = ALL_K + ptr_k_base + cols * stride_d

        qv = tl.load(q_ptr, mask=mask, other=0.).to(tl.float32)
        kv = tl.load(k_ptr, mask=mask, other=0.).to(tl.float32)

        acc += tl.sum(qv * kv)

    # scale (standard attention scaling)
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    score = acc * scale

    # store into RES[b, h, node_idx]
    res_ptr = b_idx * stride_b_r + h_idx * stride_h_r + node_idx * stride_s_r
    tl.store(RES + res_ptr, score)


# ---------------- Python launcher ----------------
def build_hierarchical_logits(
    ALL_Q, ALL_K,
    H, D,            # explicit heads and per-head dim
    seq_len,         # number of token nodes at level 0
    BLOCK_SIZE=64,
    num_warps=2
):
    """
    Launches both self-logits (level-0 tokens) and cross-logits (pairwise spans).
    Returns:
        RES_SELF  [B, H, seq_len]  (float32)
        RES_CROSS [B, H, M]        (float32), where M = total_nodes - 1
    """

    assert ALL_Q.is_cuda and ALL_K.is_cuda
    B, total_nodes, D_h = ALL_Q.shape
    assert D_h == H * D, "D_h must equal H * D"

    # ---- Cross logits (all nodes except root) ----
    M = total_nodes - 1
    assert M % 2 == 0, f"M must be even for pairing rule; got M={M}"

    RES_CROSS = torch.zeros((B, H, M), device=ALL_Q.device, dtype=torch.float32)

    stride_b, stride_node, stride_d = ALL_Q.stride()
    stride_b_r, stride_h_r, stride_s_r = RES_CROSS.stride()

    grid_cross = (M, B, H)

    hierarchical_cross_logits_kernel[grid_cross](
        ALL_Q, ALL_K,
        M,
        B, H, D,
        stride_b, stride_node, stride_d,
        RES_CROSS,
        stride_b_r, stride_h_r, stride_s_r,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    # ---- Self logits (level-0 tokens only) ----
    RES_SELF = torch.zeros((B, H, seq_len), device=ALL_Q.device, dtype=torch.float32)

    stride_b_r, stride_h_r, stride_s_r = RES_SELF.stride()
    grid_self = (seq_len, B, H)

    hierarchical_self_logits_kernel[grid_self](
        ALL_Q, ALL_K,
        B, H, D,
        stride_b, stride_node, stride_d,
        RES_SELF,
        stride_b_r, stride_h_r, stride_s_r,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return RES_SELF, RES_CROSS
