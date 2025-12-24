import math
import torch
import triton
import triton.language as tl


#AUTOTUNE_CONFIGS = [
#    # --- Small block sizes ---
#    triton.Config({'BLOCK_SIZE': 32,  'num_warps': 2}, num_stages=2),
#    triton.Config({'BLOCK_SIZE': 32,  'num_warps': 4}, num_stages=2),
#    triton.Config({'BLOCK_SIZE': 32,  'num_warps': 8}, num_stages=2),
#
#    # --- Medium block sizes ---
#    triton.Config({'BLOCK_SIZE': 64,  'num_warps': 2}, num_stages=2),
#    triton.Config({'BLOCK_SIZE': 64,  'num_warps': 4}, num_stages=2),
#    triton.Config({'BLOCK_SIZE': 64,  'num_warps': 8}, num_stages=2),
#    triton.Config({'BLOCK_SIZE': 64,  'num_warps': 4}, num_stages=3),
#
#    # --- Larger block sizes ---
#    triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}, num_stages=2),
#    triton.Config({'BLOCK_SIZE': 128, 'num_warps': 8}, num_stages=2),
#    triton.Config({'BLOCK_SIZE': 128, 'num_warps': 8}, num_stages=3),
#
#    # --- Very large block sizes ---
#    triton.Config({'BLOCK_SIZE': 256, 'num_warps': 4}, num_stages=2),
#    triton.Config({'BLOCK_SIZE': 256, 'num_warps': 8}, num_stages=2),
#    triton.Config({'BLOCK_SIZE': 256, 'num_warps': 8}, num_stages=3),
#
#    # --- Extreme (only useful on big GPUs like A100/H100) ---
#    triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}, num_stages=2),
#    triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}, num_stages=3),
#]
#
#
#@triton.autotune(
#    configs=AUTOTUNE_CONFIGS,
#    key=['D', 'n_instances'],  # adapt to what matters for perf
#)
@triton.jit
def build_span_nodes_level_kernel(
    ALL_Q, ALL_K, ALL_V,       # [ALL_NODES_FLAT, D]
    CHILD_BASE, OUT_BASE,      # scalar offsets (in nodes)
    n_parents, D,              # sizes
    n_instances,               # number of instances = B * H (host must pass)
    stride_outer, stride_d,    # strides for [nodes, D] layout
    BLOCK_SIZE: tl.constexpr
):
    parent_idx   = tl.program_id(0)  # parent index within a sequence level (0..n_parents-1)
    instance_idx = tl.program_id(1)  # instance index (0..n_instances-1)

    c0 = CHILD_BASE + (parent_idx * 2) * n_instances + instance_idx
    c1 = c0 + n_instances

    out_idx = OUT_BASE + parent_idx * n_instances + instance_idx

    dot0 = tl.zeros([1], dtype=tl.float32)
    dot1 = tl.zeros([1], dtype=tl.float32)

    # --- First pass: dot products ---
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < D

        base0 = c0 * stride_outer + cols * stride_d
        base1 = c1 * stride_outer + cols * stride_d
        out_base = out_idx * stride_outer + cols * stride_d

        # parent query block
        q_parent_blk = tl.load(ALL_Q + out_base, mask=mask_cols, other=0.).to(tl.float32)

        # child key blocks
        k0_blk = tl.load(ALL_K + base0, mask=mask_cols, other=0.).to(tl.float32)
        k1_blk = tl.load(ALL_K + base1, mask=mask_cols, other=0.).to(tl.float32)

        dot0 += tl.sum(q_parent_blk * k0_blk, axis=0)
        dot1 += tl.sum(q_parent_blk * k1_blk, axis=0)

    # --- Softmax weights ---
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    l0 = dot0 * scale
    l1 = dot1 * scale
    m = tl.maximum(l0, l1)
    e0 = tl.exp(l0 - m)
    e1 = tl.exp(l1 - m)
    denom = e0 + e1 + 1e-6
    w0 = e0 / denom
    w1 = e1 / denom

    # --- Second pass: weighted sums ---
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < D

        base0 = c0 * stride_outer + cols * stride_d
        base1 = c1 * stride_outer + cols * stride_d
        out_base = out_idx * stride_outer + cols * stride_d

        q0_blk = tl.load(ALL_Q + base0, mask=mask_cols, other=0.).to(tl.float32)
        q1_blk = tl.load(ALL_Q + base1, mask=mask_cols, other=0.).to(tl.float32)
        k0_blk = tl.load(ALL_K + base0, mask=mask_cols, other=0.).to(tl.float32)
        k1_blk = tl.load(ALL_K + base1, mask=mask_cols, other=0.).to(tl.float32)
        v0_blk = tl.load(ALL_V + base0, mask=mask_cols, other=0.).to(tl.float32)
        v1_blk = tl.load(ALL_V + base1, mask=mask_cols, other=0.).to(tl.float32)

        w0_b = tl.cast(w0, tl.float32)
        w1_b = tl.cast(w1, tl.float32)

        span_q_blk = w0_b * q0_blk + w1_b * q1_blk
        span_k_blk = w0_b * k0_blk + w1_b * k1_blk
        span_v_blk = w0_b * v0_blk + w1_b * v1_blk

        tl.store(ALL_Q + out_base, tl.cast(span_q_blk, ALL_Q.dtype.element_ty), mask=mask_cols)
        tl.store(ALL_K + out_base, tl.cast(span_k_blk, ALL_K.dtype.element_ty), mask=mask_cols)
        tl.store(ALL_V + out_base, tl.cast(span_v_blk, ALL_V.dtype.element_ty), mask=mask_cols)


def build_hierarchical_spans_blocked_flat_2d(
    tok_q, tok_k, tok_v,
    ALL_Q, ALL_K, ALL_V,
    level_starts=None,
    BLOCK_SIZE=64, num_warps=2
):
    """
    Host wrapper to build hierarchical spans with preallocated buffers.

    tok_q/k/v: [B, H, N, D]
    ALL_Q/K/V: preallocated buffers containing leaf + span nodes
    level_starts: optional list of offsets for span levels
    """
    assert tok_q.is_cuda
    B, H, N, D = tok_q.shape

    tok_q_flat = tok_q.permute(2, 0, 1, 3).reshape(N * B * H, D).contiguous()
    tok_k_flat = tok_k.permute(2, 0, 1, 3).reshape(N * B * H, D).contiguous()
    tok_v_flat = tok_v.permute(2, 0, 1, 3).reshape(N * B * H, D).contiguous()

    level_sizes = []
    cur = N
    while cur // 2 > 0:
        cur = cur // 2
        level_sizes.append(cur)

    if level_starts is None:
        level_starts = []
        offset = N * B * H
        for s in level_sizes:
            level_starts.append(offset)
            offset += s * B * H

    # copy leaf tokens into ALL buffers
    ALL_Q[: N * B * H].copy_(tok_q_flat)
    ALL_K[: N * B * H].copy_(tok_k_flat)
    ALL_V[: N * B * H].copy_(tok_v_flat)

    s_outer, s_d = ALL_Q.stride()
    n_instances = B * H

    for lvl_idx, n_parents in enumerate(level_sizes):
        child_base = 0 if lvl_idx == 0 else level_starts[lvl_idx - 1]
        out_base = level_starts[lvl_idx]

        grid = (n_parents, n_instances)
        build_span_nodes_level_kernel[grid](
            ALL_Q, ALL_K, ALL_V,
            child_base, out_base,
            n_parents, D,
            n_instances,
            s_outer, s_d,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )

    return level_starts, level_sizes


import torch
import math

def build_hierarchical_spans_blocked_torch_reference_flat(
    tok_q, tok_k, tok_v,
    ALL_Q, ALL_K, ALL_V,
    level_starts, level_sizes
):
    """
    PyTorch reference mirroring the Triton kernel.

    Args:
        tok_q, tok_k, tok_v : [B, H, N, D] (tokens, updated each run)
        ALL_Q, ALL_K, ALL_V : [total_nodes, D] flat buffers (persist across runs)
        level_starts        : list of start offsets for each level
        level_sizes         : list of n_parents per level

    Returns:
        ALL_Q, ALL_K, ALL_V updated in-place
    """
    B, H, N, D = tok_q.shape
    n_instances = B * H
    scale = 1.0 / math.sqrt(D)

    # -------------------------------------------------------------------------
    # Flatten the token-level inputs into the first N*B*H rows of the flat arrays
    # Layout: [N, B, H, D] -> [N*B*H, D] contiguous
    # -------------------------------------------------------------------------
    tok_q_flat = tok_q.permute(2, 0, 1, 3).reshape(N * n_instances, D).contiguous()
    tok_k_flat = tok_k.permute(2, 0, 1, 3).reshape(N * n_instances, D).contiguous()
    tok_v_flat = tok_v.permute(2, 0, 1, 3).reshape(N * n_instances, D).contiguous()

    ALL_Q[: N * n_instances].copy_(tok_q_flat)
    ALL_K[: N * n_instances].copy_(tok_k_flat)
    ALL_V[: N * n_instances].copy_(tok_v_flat)

    # -------------------------------------------------------------------------
    # Process levels sequentially (deterministic order, same as kernel)
    # -------------------------------------------------------------------------
    for lvl_idx, n_parents in enumerate(level_sizes):
        child_base = 0 if lvl_idx == 0 else level_starts[lvl_idx - 1]
        out_base   = level_starts[lvl_idx]

        for parent_idx in range(n_parents):
            for inst in range(n_instances):
                # child indices in flat layout
                c0 = child_base + (2 * parent_idx) * n_instances + inst
                c1 = c0 + n_instances
                out_idx = out_base + parent_idx * n_instances + inst
                
                # BUG FIX 1: Match Triton kernel by using the (initially zero)
                # output location as the parent query for attention calculation.
                q_parent = ALL_Q[out_idx]
                
                # Load children's Q, K, V
                q0, q1 = ALL_Q[c0], ALL_Q[c1]
                k0, k1 = ALL_K[c0], ALL_K[c1]
                v0, v1 = ALL_V[c0], ALL_V[c1]

                # logits (scalar per instance)
                l0 = torch.dot(q_parent, k0) * scale
                l1 = torch.dot(q_parent, k1) * scale
                m = max(l0, l1)

                e0 = torch.exp(l0 - m)
                e1 = torch.exp(l1 - m)
                denom = e0 + e1 + 1e-6
                w0, w1 = e0 / denom, e1 / denom

                # write outputs
                # BUG FIX 2: Match Triton kernel by creating the new parent Q, K, V
                # as a weighted sum of their respective children.
                ALL_Q[out_idx].copy_(w0 * q0 + w1 * q1)   # weighted queries
                ALL_K[out_idx].copy_(w0 * k0 + w1 * k1)   # weighted keys
                ALL_V[out_idx].copy_(w0 * v0 + w1 * v1)   # weighted values

    return ALL_Q, ALL_K, ALL_V



# -----------------------------------------------------------------------------
# Test Triton kernel vs PyTorch reference with repeated trials
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time, math
    torch.manual_seed(42)  # fixed seed for reproducibility

    #B, H = 32, 8
    #B, H = 16, 8
    B, H = 1, 16
    N = 1024 * 32
    D = 32
    num_trials = 50  # repeat each kernel call

    print(f"Testing hierarchical span nodes with B={B}, H={H}, N={N}, D={D}, trials={num_trials}")

    tok_q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    tok_k = torch.randn_like(tok_q)
    tok_v = torch.randn_like(tok_q)

    # ---------- Precompute level sizes and starts ----------
    level_sizes = []
    cur = N
    while cur // 2 > 0:
        cur //= 2
        level_sizes.append(cur)

    level_starts = []
    offset = N * B * H
    for s in level_sizes:
        level_starts.append(offset)
        offset += s * B * H

    total_nodes = offset

    # Allocate two independent sets of buffers
    ALL_Q_triton = torch.zeros((total_nodes, D), device="cuda", dtype=torch.float16)
    ALL_K_triton = torch.zeros_like(ALL_Q_triton)
    ALL_V_triton = torch.zeros_like(ALL_Q_triton)

    ALL_Q_ref = torch.zeros_like(ALL_Q_triton)
    ALL_K_ref = torch.zeros_like(ALL_Q_triton)
    ALL_V_ref = torch.zeros_like(ALL_Q_triton)

    # ---------- Triton kernel timing ----------
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_trials):
        level_starts_out, level_sizes_out = build_hierarchical_spans_blocked_flat_2d(
            tok_q, tok_k, tok_v,
            ALL_Q_triton, ALL_K_triton, ALL_V_triton,
            level_starts=level_starts,
            BLOCK_SIZE=64, num_warps=2
        )
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_trials
    total_triton_time = time.time() - start_time

    print(f"Triton kernel average runtime: {triton_time:.6f} sec")
    print(f"Triton kernel total runtime:    {total_triton_time:.6f} sec")

    # ---------- PyTorch reference timing ----------
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_trials):
        ALL_Q_ref, ALL_K_ref, ALL_V_ref = build_hierarchical_spans_blocked_torch_reference_flat(
            tok_q, tok_k, tok_v,
            ALL_Q_ref, ALL_K_ref, ALL_V_ref,
            level_starts, level_sizes
        )
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_trials
    total_pytorch_time = time.time() - start_time
    print(f"PyTorch reference average runtime: {pytorch_time:.6f} sec")
    print(f"PyTorch reference total runtime:   {total_pytorch_time:.6f} sec")

    print(f"Performance speedup: {pytorch_time / triton_time:.2f}x")

    # ---------- Correctness check ----------
    max_err_q = max_err_k = max_err_v = 0.0
    n_instances = B * H
    for lvl_idx, start in enumerate(level_starts):
        n_parents = level_sizes[lvl_idx]
        num_nodes_in_level = n_parents * n_instances

        # Triton flat outputs
        q_triton = ALL_Q_triton[start : start + num_nodes_in_level].to(torch.float32)
        k_triton = ALL_K_triton[start : start + num_nodes_in_level].to(torch.float32)
        v_triton = ALL_V_triton[start : start + num_nodes_in_level].to(torch.float32)

        # Reference flat outputs
        q_ref = ALL_Q_ref[start : start + num_nodes_in_level].to(torch.float32)
        k_ref = ALL_K_ref[start : start + num_nodes_in_level].to(torch.float32)
        v_ref = ALL_V_ref[start : start + num_nodes_in_level].to(torch.float32)

        # Reshape both into [N_level, B, H, D] then permute -> [B, H, N_level, D]
        q_triton = q_triton.view(n_parents, B, H, D).permute(1, 2, 0, 3)
        k_triton = k_triton.view(n_parents, B, H, D).permute(1, 2, 0, 3)
        v_triton = v_triton.view(n_parents, B, H, D).permute(1, 2, 0, 3)

        q_ref = q_ref.view(n_parents, B, H, D).permute(1, 2, 0, 3)
        k_ref = k_ref.view(n_parents, B, H, D).permute(1, 2, 0, 3)
        v_ref = v_ref.view(n_parents, B, H, D).permute(1, 2, 0, 3)

        # Errors
        err_q = (q_triton - q_ref).abs().max().item()
        err_k = (k_triton - k_ref).abs().max().item()
        err_v = (v_triton - v_ref).abs().max().item()

        print(f"Level {lvl_idx}: N_level={n_parents}, max_err q/k/v = {err_q:.3e} / {err_k:.3e} / {err_v:.3e}")

        max_err_q = max(max_err_q, err_q)
        max_err_k = max(max_err_k, err_k)
        max_err_v = max(max_err_v, err_v)

    print(f"\nFinal Max Error: {max(max_err_q, max_err_k, max_err_v):.6f}")
    assert max(max_err_q, max_err_k, max_err_v) < 1e-2, "Sanity check failed!"
    print("Sanity check passed.")
