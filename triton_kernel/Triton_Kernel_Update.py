import math
import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------
# Triton kernel: update all parents (span nodes) in parallel using previous-turn buffers
# --------------------------------------------------------------------------
@triton.jit
def update_all_span_nodes_kernel(
    PREV_Q, PREV_K, PREV_V,
    NEXT_Q, NEXT_K, NEXT_V,
    CHILD_BASE_NODES,
    OUT_NODE_IDX,
    n_parents,
    n_instances,
    stride_outer, stride_d,
    D,
    BLOCK_SIZE: tl.constexpr
):
    parent_gid = tl.program_id(0)
    instance_idx = tl.program_id(1)

    if parent_gid >= n_parents:
        return

    child0_node = tl.load(CHILD_BASE_NODES + parent_gid)
    out_node = tl.load(OUT_NODE_IDX + parent_gid)

    c0_flat = child0_node * n_instances + instance_idx
    c1_flat = c0_flat + n_instances
    out_flat = out_node * n_instances + instance_idx
    parent_prev_flat = out_node * n_instances + instance_idx

    dot0 = tl.zeros([1], dtype=tl.float32)
    dot1 = tl.zeros([1], dtype=tl.float32)

    # First pass: dot products
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        base_parent = parent_prev_flat * stride_outer + cols * stride_d
        base_c0 = c0_flat * stride_outer + cols * stride_d
        base_c1 = c1_flat * stride_outer + cols * stride_d

        q_parent_blk = tl.load(PREV_Q + base_parent, mask=mask, other=0.).to(tl.float32)
        k0_blk = tl.load(PREV_K + base_c0, mask=mask, other=0.).to(tl.float32)
        k1_blk = tl.load(PREV_K + base_c1, mask=mask, other=0.).to(tl.float32)

        dot0 += tl.sum(q_parent_blk * k0_blk, axis=0)
        dot1 += tl.sum(q_parent_blk * k1_blk, axis=0)

    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    l0 = dot0 * scale
    l1 = dot1 * scale
    m = tl.maximum(l0, l1)
    e0 = tl.exp(l0 - m)
    e1 = tl.exp(l1 - m)
    denom = e0 + e1 + 1e-6
    w0 = e0 / denom
    w1 = e1 / denom

    w0_b = tl.cast(w0, tl.float32)
    w1_b = tl.cast(w1, tl.float32)

    # Second pass: weighted sums
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        base_c0 = c0_flat * stride_outer + cols * stride_d
        base_c1 = c1_flat * stride_outer + cols * stride_d
        base_out = out_flat * stride_outer + cols * stride_d

        q0_blk = tl.load(PREV_Q + base_c0, mask=mask, other=0.).to(tl.float32)
        k0_blk = tl.load(PREV_K + base_c0, mask=mask, other=0.).to(tl.float32)
        v0_blk = tl.load(PREV_V + base_c0, mask=mask, other=0.).to(tl.float32)

        q1_blk = tl.load(PREV_Q + base_c1, mask=mask, other=0.).to(tl.float32)
        k1_blk = tl.load(PREV_K + base_c1, mask=mask, other=0.).to(tl.float32)
        v1_blk = tl.load(PREV_V + base_c1, mask=mask, other=0.).to(tl.float32)

        span_q_blk = w0_b * q0_blk + w1_b * q1_blk
        span_k_blk = w0_b * k0_blk + w1_b * k1_blk
        span_v_blk = w0_b * v0_blk + w1_b * v1_blk

        tl.store(NEXT_Q + base_out, tl.cast(span_q_blk, PREV_Q.dtype.element_ty), mask=mask)
        tl.store(NEXT_K + base_out, tl.cast(span_k_blk, PREV_K.dtype.element_ty), mask=mask)
        tl.store(NEXT_V + base_out, tl.cast(span_v_blk, PREV_V.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------
def build_tree_child_and_out_indices(N):
    cur_nodes = list(range(N))
    next_free = N
    child_bases = []
    out_nodes = []
    level_sizes = []
    level_starts = []

    while len(cur_nodes) // 2 > 0:
        n_parents = len(cur_nodes) // 2
        level_starts.append(next_free)
        level_sizes.append(n_parents)

        parents = []
        for i in range(n_parents):
            c0 = cur_nodes[2*i]
            child_bases.append(c0)
            out_nodes.append(next_free + i)
            parents.append(next_free + i)
        next_free += n_parents
        cur_nodes = parents

    return (
        torch.tensor(child_bases, dtype=torch.int32),
        torch.tensor(out_nodes, dtype=torch.int32),
        level_starts,
        level_sizes
    )


# --------------------------------------------------------------------------
def prepare_and_launch(tok_q, tok_k, tok_v, BLOCK_SIZE=64, num_warps=2, device=None):
    assert tok_q.is_cuda and tok_k.is_cuda and tok_v.is_cuda
    B, H, N, D = tok_q.shape
    n_instances = B * H
    device = tok_q.device if device is None else device

    tok_q_flat = tok_q.permute(2, 0, 1, 3).reshape(N * n_instances, D).contiguous()
    tok_k_flat = tok_k.permute(2, 0, 1, 3).reshape(N * n_instances, D).contiguous()
    tok_v_flat = tok_v.permute(2, 0, 1, 3).reshape(N * n_instances, D).contiguous()

    child_base_nodes, out_node_idx, level_starts, level_sizes = build_tree_child_and_out_indices(N)
    n_parents = child_base_nodes.numel()

    total_nodes = (N + n_parents) * n_instances  # FIXED: multiply by n_instances

    dtype = tok_q_flat.dtype
    ALL_prev_Q = torch.zeros((total_nodes, D), device=device, dtype=dtype)
    ALL_prev_K = torch.zeros_like(ALL_prev_Q)
    ALL_prev_V = torch.zeros_like(ALL_prev_Q)
    ALL_next_Q = torch.zeros_like(ALL_prev_Q)
    ALL_next_K = torch.zeros_like(ALL_prev_Q)
    ALL_next_V = torch.zeros_like(ALL_prev_Q)

    ALL_prev_Q[: N * n_instances].copy_(tok_q_flat)
    ALL_prev_K[: N * n_instances].copy_(tok_k_flat)
    ALL_prev_V[: N * n_instances].copy_(tok_v_flat)

    child_base_nodes = child_base_nodes.to(device)
    out_node_idx = out_node_idx.to(device)

    s_outer, s_d = ALL_prev_Q.stride()

    def step_once(prev_Q, prev_K, prev_V, next_Q, next_K, next_V):
        grid = (n_parents, n_instances)
        update_all_span_nodes_kernel[grid](
            prev_Q, prev_K, prev_V,
            next_Q, next_K, next_V,
            child_base_nodes, out_node_idx,
            n_parents,
            n_instances,
            s_outer, s_d, D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        next_Q[: N * n_instances].copy_(prev_Q[: N * n_instances])
        next_K[: N * n_instances].copy_(prev_K[: N * n_instances])
        next_V[: N * n_instances].copy_(prev_V[: N * n_instances])

    return (
        ALL_prev_Q, ALL_prev_K, ALL_prev_V,
        ALL_next_Q, ALL_next_K, ALL_next_V,
        child_base_nodes, out_node_idx,
        step_once
    )


# --------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    B, H = 1, 8
    N = 1024 * 16
    D = 64
    num_trials = 40

    print(f"Testing hierarchical span nodes with B={B}, H={H}, N={N}, D={D}, trials={num_trials}")

    tok_q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    tok_k = torch.randn_like(tok_q)
    tok_v = torch.randn_like(tok_q)

    import time
    torch.cuda.synchronize()
    start_time = time.time()

    (ALL_prev_Q, ALL_prev_K, ALL_prev_V,
     ALL_next_Q, ALL_next_K, ALL_next_V,
     child_base_nodes, out_node_idx,
     step_once) = prepare_and_launch(tok_q, tok_k, tok_v)

    for _ in range(num_trials):
        step_once(ALL_prev_Q, ALL_prev_K, ALL_prev_V,
                  ALL_next_Q, ALL_next_K, ALL_next_V)
        ALL_prev_Q, ALL_next_Q = ALL_next_Q, ALL_prev_Q
        ALL_prev_K, ALL_next_K = ALL_next_K, ALL_prev_K
        ALL_prev_V, ALL_next_V = ALL_next_V, ALL_prev_V

    torch.cuda.synchronize()
    total_triton_time = time.time() - start_time
    triton_time = total_triton_time / num_trials

    print(f"Triton kernel average runtime: {triton_time:.6f} sec")
    print(f"Triton kernel total runtime: {total_triton_time:.6f} sec")
































































import triton
import triton.language as tl

@triton.jit
def build_span_nodes_level_kernel(
    ALL_Q, ALL_K, ALL_V,        # [B, total_nodes, D_h]
    CHILD_BASE_NODE, OUT_BASE_NODE,
    n_parents, D_h,
    stride_b, stride_node, stride_d,   # strides for [B, node, D_h]
    BLOCK_SIZE: tl.constexpr
):
    parent_idx = tl.program_id(0)   # 0..n_parents-1
    b_idx = tl.program_id(1)        # 0..B-1

    c0_node = CHILD_BASE_NODE + 2 * parent_idx
    c1_node = c0_node + 1
    out_node = OUT_BASE_NODE + parent_idx

    dot0 = tl.zeros([1], dtype=tl.float32)
    dot1 = tl.zeros([1], dtype=tl.float32)

    # --- First pass: dot products ---
    for off in range(0, D_h, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < D_h

        base_c0 = b_idx*stride_b + c0_node*stride_node + cols*stride_d
        base_c1 = b_idx*stride_b + c1_node*stride_node + cols*stride_d
        base_out = b_idx*stride_b + out_node*stride_node + cols*stride_d

        # **Parent query from parent level**
        q_parent = tl.load(ALL_Q + base_out, mask=mask_cols, other=0.).to(tl.float32)

        k0 = tl.load(ALL_K + base_c0, mask=mask_cols, other=0.).to(tl.float32)
        k1 = tl.load(ALL_K + base_c1, mask=mask_cols, other=0.).to(tl.float32)

        dot0 += tl.sum(q_parent * k0, axis=0)
        dot1 += tl.sum(q_parent * k1, axis=0)

    # --- Softmax weights ---
    scale = 1.0 / tl.sqrt(tl.cast(D_h, tl.float32))
    l0, l1 = dot0*scale, dot1*scale
    m = tl.maximum(l0, l1)
    e0, e1 = tl.exp(l0-m), tl.exp(l1-m)
    denom = e0+e1+1e-6
    w0, w1 = e0/denom, e1/denom

    # --- Second pass: weighted sums ---
    for off in range(0, D_h, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < D_h

        base_c0 = b_idx*stride_b + c0_node*stride_node + cols*stride_d
        base_c1 = b_idx*stride_b + c1_node*stride_node + cols*stride_d
        base_out = b_idx*stride_b + out_node*stride_node + cols*stride_d

        q0 = tl.load(ALL_Q + base_c0, mask=mask_cols, other=0.).to(tl.float32)
        q1 = tl.load(ALL_Q + base_c1, mask=mask_cols, other=0.).to(tl.float32)
        k0 = tl.load(ALL_K + base_c0, mask=mask_cols, other=0.).to(tl.float32)
        k1 = tl.load(ALL_K + base_c1, mask=mask_cols, other=0.).to(tl.float32)
        v0 = tl.load(ALL_V + base_c0, mask=mask_cols, other=0.).to(tl.float32)
        v1 = tl.load(ALL_V + base_c1, mask=mask_cols, other=0.).to(tl.float32)

        q_span = w0*q0 + w1*q1
        k_span = w0*k0 + w1*k1
        v_span = w0*v0 + w1*v1

        tl.store(ALL_Q + base_out, tl.cast(q_span, ALL_Q.dtype.element_ty), mask=mask_cols)
        tl.store(ALL_K + base_out, tl.cast(k_span, ALL_K.dtype.element_ty), mask=mask_cols)
        tl.store(ALL_V + base_out, tl.cast(v_span, ALL_V.dtype.element_ty), mask=mask_cols)







# test_bd_layout.py
import time
import math
import torch

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
import torch

if __name__ == "__main__":
    torch.manual_seed(42)

    # -------------------------
    # Parameters
    # -------------------------
    B, H = 1, 16
    N = 1024 * 16           # number of tokens
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
        build_hierarchical_spans_blocked_bd(
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
    assert max_err < 1e-2, "Sanity check failed!"
    print("Sanity check passed.")



