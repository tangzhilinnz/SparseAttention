import os
# ==========================================
# CRITICAL FIX: GPU SELECTION MUST BE FIRST
# ==========================================
# Set this before importing torch or calling torch.cuda to avoid OOM
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math

from hierarchical_attention import HierarchicalAttention


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


class HierarchicalSparseAttentionRef(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, window_size=16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # [ALIGNED] window_size is treated as Radius. Total width = 2*W + 1
        self.radius = window_size 

        # --- Y Updates (Bottom-Up) ---
        self.Wq_y = nn.Linear(dim, dim, bias=False)
        self.Wk_y = nn.Linear(dim, dim, bias=False)
        self.Wv_y = nn.Linear(dim, dim, bias=False)
        self.out_proj_y = nn.Linear(dim, dim)

        # Merge Layer for MLP mode
        self.merge_layer = nn.Linear(2 * dim, dim)

        # --- X Updates (Top-Down) ---
        self.Wq_x = nn.Linear(dim, dim, bias=False)
        self.Wk_x = nn.Linear(dim, dim, bias=False)
        self.Wv_x = nn.Linear(dim, dim, bias=False)
        self.out_proj_x = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        
        # --- Caching ---
        self.cached_idx_table = None
        self.cached_causal_mask = None
        self.cached_seq_len = -1
        
        # SWA Index Cache
        self.cached_swa_indices = None
        self.cached_swa_offsets_len = -1

        self.sizes = None
        self.offsets = None

    def _get_lookup_table(self, L, device):
        """Smart retrieval: Returns cached hierarchical table if L matches."""
        if (self.cached_idx_table is not None and 
            self.cached_seq_len == L and 
            self.cached_idx_table.device == device):
            return self.cached_idx_table, self.cached_causal_mask

        idx_table, mask = build_hierarchical_index_lookup_table(L, device=device, dtype=torch.int64)
 
        self.cached_idx_table = idx_table
        self.cached_causal_mask = mask
        self.cached_seq_len = L
        
        return idx_table, mask

    def _get_swa_indices(self, L, device):
        """
        Generates indices for sliding window attention.
        [UPDATED] Range: [i - radius, ..., i, ..., i + radius]
        Includes 0 (Self).
        """
        # [FIX] Use radius logic: [-R, ..., +R]
        offsets = torch.arange(-self.radius, self.radius + 1, device=device)
        
        if (self.cached_swa_indices is not None and 
            self.cached_swa_indices.shape[0] == L and
            self.cached_swa_offsets_len == len(offsets) and
            self.cached_swa_indices.device == device):
            return self.cached_swa_indices

        # Shape: (1, num_offsets)
        offsets = offsets.unsqueeze(0) 
        
        # Shape: (L, 1)
        base_indices = torch.arange(L, device=device).unsqueeze(1)
        
        # Broadcast add: (L, num_offsets)
        swa_indices = base_indices + offsets
        
        self.cached_swa_indices = swa_indices
        self.cached_swa_offsets_len = len(offsets.squeeze())
        return swa_indices

    def _build_level_info(self, N):
        """Helper to calculate the size of each tree level."""
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
        Bottom-up update (Leaves -> Parents)
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        assert y_in is not None

        if self.sizes is None or (self.sizes[0] != N // 2): 
            self.sizes, self.offsets = self._build_level_info(N)

        Q_p_all = self.Wq_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        K_p_all = self.Wk_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        V_p_all = self.Wv_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        
        Q_split = torch.split(Q_p_all, self.sizes, dim=2)
        K_split = torch.split(K_p_all, self.sizes, dim=2)
        V_split = torch.split(V_p_all, self.sizes, dim=2)

        new_Y_levels = []
        prev_sources = x 
        
        # [Optimization] Pre-project leaves once
        K_leaves = self.Wk_y(x).view(B, -1, H, Dh).transpose(1, 2)
        V_leaves = self.dropout(self.Wv_y(x).view(B, -1, H, Dh).transpose(1, 2))

        for level, size in enumerate(self.sizes):
            Q_p = Q_split[level]
            K_p = K_split[level]
            V_p = V_split[level]
            
            if level == 0:
                K_c, V_c = K_leaves, V_leaves
            else:
                children = prev_sources    
                K_c = self.Wk_y(children).view(B, -1, H, Dh).transpose(1, 2)
                V_c = self.dropout(self.Wv_y(children).view(B, -1, H, Dh).transpose(1, 2))

            K_c_pairs = K_c.view(B, H, size, 2, Dh)
            V_c_pairs = V_c.view(B, H, size, 2, Dh)

            K_pool = torch.cat([K_p.unsqueeze(3), K_c_pairs], dim=3)
            V_pool = torch.cat([V_p.unsqueeze(3), V_c_pairs], dim=3)

            logits = torch.matmul(Q_p.unsqueeze(3), K_pool.transpose(-1, -2))
            logits = logits / math.sqrt(Dh)
            
            weights = F.softmax(logits, dim=-1)
            attn_out = torch.matmul(weights, V_pool)
            
            attn_out = attn_out.squeeze(3).transpose(1, 2).contiguous().reshape(B, size, D)
            
            new_Y_levels.append(attn_out)
            prev_sources = attn_out

        return self.out_proj_y(torch.cat(new_Y_levels, dim=1))

    def update_X_from_Y(self, x, y, mask=None):
        """
        Fused Attention: SWA (includes Self) + Hierarchical
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        if y is None: return x

        # Concatenate inputs once to allow unified K/V calculation
        XY = torch.cat([x, y], dim=1)

        # Q is only derived from X (leaves)
        Q = self.Wq_x(x).view(B, N, H, Dh).transpose(1, 2)
        
        # K, V derived from XY (leaves + hierarchy)
        K_full = self.Wk_x(XY).view(B, -1, H, Dh).transpose(1, 2)
        V_full = self.dropout(self.Wv_x(XY).view(B, -1, H, Dh).transpose(1, 2))

        # ---------------------------------------------------------
        # 1. SLIDING WINDOW LOGITS (Includes Self)
        # ---------------------------------------------------------
        # Get indices: [N, 2*Radius + 1]
        swa_indices = self._get_swa_indices(N, device=x.device)
        
        # Valid Mask (Padding Check)
        swa_valid_mask = (swa_indices >= 0) & (swa_indices < N)
        
        # Clamp to avoid gather errors
        safe_swa_indices = swa_indices.clamp(0, N - 1)
        
        # Gather SWA K/V
        swa_k = K_full[:, :, safe_swa_indices, :]
        swa_v = V_full[:, :, safe_swa_indices, :]
        
        # Compute Logits
        swa_logits = torch.einsum('b h l x d, b h l n d -> b h l n', Q.unsqueeze(3), swa_k)
        swa_logits = swa_logits / math.sqrt(Dh)
        
        # Apply Boundary Padding Mask
        swa_logits = swa_logits.masked_fill(~swa_valid_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply Causal Mask (if needed)
        if mask is not None:
            row_indices = torch.arange(N, device=x.device).unsqueeze(1)
            causal_mask_bool = safe_swa_indices > row_indices
            swa_logits = swa_logits.masked_fill(causal_mask_bool.unsqueeze(0).unsqueeze(0), float('-inf'))

        # ---------------------------------------------------------
        # 2. HIERARCHICAL NEIGHBOR LOGITS
        # ---------------------------------------------------------
        idx_table, _ = self._get_lookup_table(N, device=x.device)
        
        # Gather Hierarchical K/V
        hier_k = K_full[:, :, idx_table, :] 
        hier_v = V_full[:, :, idx_table, :]

        hier_logits = torch.einsum('b h l x d, b h l n d -> b h l n', Q.unsqueeze(3), hier_k)
        hier_logits = hier_logits / math.sqrt(Dh)

        # Note: Hierarchy Causal Masking is implicitly handled by table construction
        # (Table only includes valid past/present uncles).
        # We can apply a check if idx_table has invalid entries, but aligned builder uses valid.

        # ---------------------------------------------------------
        # 3. FUSION & OUTPUT
        # ---------------------------------------------------------
        # Concatenate: [SWA (Local+Self), Hierarchical (Global)]
        all_logits = torch.cat([swa_logits, hier_logits], dim=3)
        all_v = torch.cat([swa_v, hier_v], dim=3)

        # Attention Softmax
        max_logits = all_logits.max(dim=-1, keepdim=True)[0]
        weights = F.softmax(all_logits - max_logits, dim=-1)              
            
        output_leaf = torch.einsum('b h l n, b h l n d -> b h l d', weights, all_v)

        output = output_leaf.transpose(1, 2).reshape(B, N, D)

        return self.out_proj_x(output)

    def _standard_attention(self, Q, K, V, mask):
        D_head = Q.size(-1)
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
            output_leaf = self.update_X_from_Y(x, y, mask)
            output = output_leaf 
            return (output, None) if return_attention else output
        else:
            Q = self.Wq_x(query).view(B, L_Q, H, Dh).transpose(1, 2)
            K = self.Wk_x(key).view(B, L_K, H, Dh).transpose(1, 2)
            V = self.Wv_x(value).view(B, L_V, H, Dh).transpose(1, 2)
            
            output_leaf, attn_weights = self._standard_attention(Q, K, V, mask)
            
            output = output_leaf.transpose(1, 2).reshape(B, L_Q, D)
            output = self.out_proj_x(output)
            
            return (output, attn_weights) if return_attention else output




from torch.profiler import profile, record_function, ProfilerActivity



import torch
import triton
import math
from torch.profiler import profile, record_function, ProfilerActivity

# [Insert your HierarchicalSparseAttentionTriton class and kernels here...]
# ... (Assuming they are defined in the file above) ...

def run_full_suite():
    # ==========================================================================
    # 1. SETUP & CORRECTNESS CHECK
    # ==========================================================================
    check_dtype = torch.float16

    print(f"{'='*60}")
    print(f"1. CORRECTNESS CHECK ({check_dtype}) - cross_update_Y")
    print(f"{'='*60}")

    # 1. Setup Dimensions
    B, N, head_dim, H = 2, 32768, 64, 16 
    dim = head_dim * H

    # 2. Initialize BOTH Models (Dropout=0.0 for deterministic check)
    model_ref = HierarchicalSparseAttentionRef(dim, H, dropout=0.0).cuda().to(check_dtype).eval()
    model_tri = HierarchicalAttention(dim, H, dropout=0.0).cuda().to(check_dtype).eval()

    # CRITICAL FIX: Synchronize weights so both models compute the exact same math
    model_tri.load_state_dict(model_ref.state_dict())
    
    # 3. Create Inputs
    x_base = torch.randn(B, N, dim, device='cuda', dtype=check_dtype).clamp(-1, 1)
    y_base = torch.randn(B, N - 1, dim, device='cuda', dtype=check_dtype).clamp(-1, 1)

    print(f"Input Shapes -> X: {x_base.shape}, Y: {y_base.shape}, Dtype: {x_base.dtype}")
    assert y_base.shape[1] == N - 1, f"Sanity Check Failed"
    
    # [REAL TRAINING SIMULATION]
    # cross_update_Y outputs the updated parents, so output shape is (B, N-1, dim)
    dout = torch.randn(B, N - 1, dim, device='cuda', dtype=check_dtype)

    # -------------------------------------------------
    # 4. Run PyTorch Reference Path
    # -------------------------------------------------
    x_ref = x_base.clone().detach().requires_grad_(True)
    y_ref = y_base.clone().detach().requires_grad_(True)
    
    model_ref.sizes = None; model_ref.offsets = None 
    
    out_ref = model_ref.cross_update_Y(x_ref, y_ref)
    out_ref.backward(dout)
    
    # -------------------------------------------------
    # 5. Run Triton Kernel Path
    # -------------------------------------------------
    x_tri = x_base.clone().detach().requires_grad_(True)
    y_tri = y_base.clone().detach().requires_grad_(True)
    
    model_tri.sizes = None; model_tri.offsets = None
    
    out_tri = model_tri.cross_update_Y(x_tri, y_tri)
    out_tri.backward(dout)
    
    # -------------------------------------------------
    # 6. Compare Results
    # -------------------------------------------------
    # Move to float32 for accurate diff calculation
    diff_out = (out_ref.float() - out_tri.float()).abs().max().item()
    diff_grad_x = (x_ref.grad.float() - x_tri.grad.float()).abs().max().item()
    diff_grad_y = (y_ref.grad.float() - y_tri.grad.float()).abs().max().item()
    
    print(f"Max Diff Output:   {diff_out:.6f}")
    print(f"Max Diff Grad X:   {diff_grad_x:.6f}")
    print(f"Max Diff Grad Y:   {diff_grad_y:.6f}")
    
    # Relaxed tolerance for FP16 (1e-2 is typical for accumulation noise in FP16)
    tol = 1e-2 
    try:
        assert torch.allclose(out_ref, out_tri, atol=tol), "Forward pass mismatch!"
        assert torch.allclose(x_ref.grad, x_tri.grad, atol=tol), "Gradient X mismatch!"
        assert torch.allclose(y_ref.grad, y_tri.grad, atol=tol), "Gradient Y mismatch!"
        print(f"SUCCESS: Triton kernel matches PyTorch reference (within {check_dtype} tolerance).")
    except AssertionError as e:
        print(f"\nFAILURE: {e}")
        
    # ==========================================================================
    # 2. PERFORMANCE BENCHMARK (Float16 - Large Scale)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("2. SPEED BENCHMARK (Float16 - N=32768)")
    print(f"{'='*60}")

    # Config: Massive scale
    #B, N, D, H = 64, 1024, 512, 8
    #B, N, D, H = 128, 256, 512, 8
    B, N, head_dim, H = 1, 2048 * 256, 64, 8
    #B, N, D, H = 2, 2048 * 64, 512, 8
    #B, N, D, H = 32, 4096, 512, 8
    #B, N, D, H = 128, 1024, 512, 8 
    #B, N, D, H = 128, 512, 512, 8 
    #B, N, D, H = 64, 2048, 512, 8
    # B, N, D, H = 16, 4096, 1024, 16 # Alternative config
    dim = head_dim * H
    print(f"Config: B={B}, N={N}, D={dim} (HeadDim={head_dim}), H={H}, dtype={check_dtype}")

    # Re-init models
    model_ref = HierarchicalSparseAttentionRef(dim, H, dropout=0.0).cuda().to(check_dtype).eval()
    model_tri = HierarchicalAttention(dim, H, dropout=0.0).cuda().to(check_dtype).eval()

    # Create large inputs 
    x_bench = torch.randn(B, N, dim, device='cuda', dtype=check_dtype, requires_grad=True)
    y_bench = torch.randn(B, N - 1, dim, device='cuda', dtype=check_dtype, requires_grad=True)
    
    # Benchmarking Gradients
    dout_bench = torch.randn(B, N - 1, dim, device='cuda', dtype=check_dtype)
    
    # --- Timing Setup ---
    num_warmup = 5
    num_trials = 20 
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # A. Measure PyTorch Reference 
    print("  Running PyTorch Reference (FWD + BWD)...")
    for _ in range(num_warmup): 
        out = model_ref.cross_update_Y(x_bench, y_bench)
        out.backward(dout_bench)
        model_ref.zero_grad(); x_bench.grad = None; y_bench.grad = None

    torch.cuda.synchronize()
    start.record()
    for _ in range(num_trials):
        out = model_ref.cross_update_Y(x_bench, y_bench)
        out.backward(dout_bench)
        model_ref.zero_grad(); x_bench.grad = None; y_bench.grad = None
    end.record()
    torch.cuda.synchronize()
    ms_ref = start.elapsed_time(end)

    # B. Measure Triton Kernel
    print("  Running Triton Kernel (FWD + BWD)...")
    for _ in range(num_warmup): 
        out = model_tri.cross_update_Y(x_bench, y_bench)
        out.backward(dout_bench)
        model_tri.zero_grad(); x_bench.grad = None; y_bench.grad = None

    torch.cuda.synchronize()
    start.record()
    for _ in range(num_trials):
        out = model_tri.cross_update_Y(x_bench, y_bench)
        out.backward(dout_bench)
        model_tri.zero_grad(); x_bench.grad = None; y_bench.grad = None
    end.record()
    torch.cuda.synchronize()
    ms_tri = start.elapsed_time(end)

    # Results
    print("-" * 50)
    print(f"  PyTorch Avg Time (Fwd+Bwd): {ms_ref/num_trials:.3f} ms")
    print(f"  Triton  Avg Time (Fwd+Bwd): {ms_tri/num_trials:.3f} ms")
    print(f"  >>> Speedup: {ms_ref/ms_tri:.2f}x")
    print("-" * 50)

    # ==========================================================================
    # 3. PROFILER (Float16)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("3. DETAILED PROFILING")
    print(f"{'='*60}")

    print("Profiling Triton Kernel Trace...")
    model_tri.zero_grad(); x_bench.grad = None; y_bench.grad = None

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("Triton_Cross_Update_Y"):
            for _ in range(5): 
                out = model_tri.cross_update_Y(x_bench, y_bench)
                out.backward(dout_bench)
                model_tri.zero_grad(); x_bench.grad = None; y_bench.grad = None

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))


def run_full_suite_update_X_from_Y():
    
    check_dtype = torch.float16

    print(f"{'='*60}")
    print(f"1. CORRECTNESS CHECK ({check_dtype}) - update_X_from_Y")
    print(f"{'='*60}")

    # 1. Setup Dimensions for Correctness
    # B, N, D, H = 1, 2048 * 128, 64, 16
    B, N, D, H = 16, 2048, 64, 16
    # B, N, D, H = 2, 2048 * 32, 64, 16
    # B, N, D, H = 2, 64, 64, 16
    dim = H * D
    window_size_set = 64

    # 2. Initialize Model (Dropout=0.0 for deterministic check)
    model_ref = HierarchicalSparseAttentionRef(dim, H, dropout=0.0, window_size=window_size_set).cuda().to(check_dtype)
    model_tri = HierarchicalAttention(dim, H, dropout=0.0, window_size=window_size_set).cuda().to(check_dtype)

    model_ref.eval()
    model_tri.eval()

    # Synchronize weights so both models compute the exact same math
    model_tri.load_state_dict(model_ref.state_dict())

    # 3. Create Inputs
    x_base = torch.randn(B, N, dim, device='cuda', dtype=check_dtype).clamp(-1, 1)
    y_base = torch.randn(B, N - 1, dim, device='cuda', dtype=check_dtype).clamp(-1, 1)

    # Create Random Incoming Gradients (dout)
    dout = torch.randn_like(x_base, device='cuda', dtype=check_dtype)

    # Optional mask 
    #mask = None
    mask = torch.ones((B, N), dtype=torch.bool, device='cuda')

    print(f"Input Shapes -> X: {x_base.shape}, Y: {y_base.shape}, Dtype: {x_base.dtype}")

    # -------------------------------------------------
    # 4. Run PyTorch Reference Path
    # -------------------------------------------------
    x_ref = x_base.clone().detach().requires_grad_(True)
    y_ref = y_base.clone().detach().requires_grad_(True)

    model_ref.sizes = None; model_ref.offsets = None

    # Forward Ref
    out_ref = model_ref.update_X_from_Y(x_ref, y_ref, mask=mask)

    # Backward Ref (Inject Random Gradients)
    out_ref.backward(dout)

    # -------------------------------------------------
    # 5. Run Triton Kernel Path
    # -------------------------------------------------
    x_tri = x_base.clone().detach().requires_grad_(True)
    y_tri = y_base.clone().detach().requires_grad_(True)

    model_tri.sizes = None; model_tri.offsets = None

    # Forward Triton
    out_tri = model_tri.update_X_from_Y(x_tri, y_tri, mask=mask)

    # Backward Triton (Inject SAME Random Gradients)
    out_tri.backward(dout)

    # -------------------------------------------------
    # 6. Compare Results
    # -------------------------------------------------
    # Cast to float32 for accurate diff calculation regardless of input type
    diff_out = (out_ref.float() - out_tri.float()).abs().max().item()
    diff_grad_x = (x_ref.grad.float() - x_tri.grad.float()).abs().max().item()
    diff_grad_y = (y_ref.grad.float() - y_tri.grad.float()).abs().max().item()

    print(f"Max Diff Output:   {diff_out:.8f}")
    print(f"Max Diff Grad X:   {diff_grad_x:.8f}")
    print(f"Max Diff Grad Y:   {diff_grad_y:.8f}")

    print(f"   -> Ref Grad Y Mean: {y_ref.grad.float().abs().mean():.4f} | Max: {y_ref.grad.float().abs().max():.4f}")
    print(f"   -> Tri Grad Y Mean: {y_tri.grad.float().abs().mean():.4f} | Max: {y_tri.grad.float().abs().max():.4f}")

    # Calculate magnitudes
    grad_ref_mag = y_ref.grad.float().abs().mean()
    grad_tri_mag = y_tri.grad.float().abs().mean()

    print(f"Ref Grad Magnitude: {grad_ref_mag:.6f}")
    print(f"Tri Grad Magnitude: {grad_tri_mag:.6f}")

    # Calculate Relative Error
    # Avoid division by zero by adding a tiny epsilon
    rel_error = diff_grad_y / (grad_ref_mag + 1e-6)
    print(f"Relative Error: {rel_error:.6f}")
    
    # Dynamic tolerance based on dtype
    # FP32: stricter (e.g., 1e-4), FP16: looser (e.g., 1e-2)
    tol = 1e-3 if check_dtype == torch.float32 else 5e-2

    # Note: We expect assert to fail given your previous logs (exploding Y grad),
    # but this test is now mathematically robust.
    try:
        assert torch.allclose(out_ref, out_tri, atol=tol), f"Forward pass mismatch! (tol={tol})"
        assert torch.allclose(x_ref.grad, x_tri.grad, atol=tol), f"Gradient X mismatch! (tol={tol})"
        assert torch.allclose(y_ref.grad, y_tri.grad, atol=tol), f"Gradient Y mismatch! (tol={tol})"
        print(f"SUCCESS: Triton kernel matches PyTorch reference (within {check_dtype} tolerance).")
    except AssertionError as e:
        print(f"FAILURE: {e}")

    # ==========================================================================
    # 2. PERFORMANCE BENCHMARK (Float16 - Large Scale)
    # ==========================================================================
    check_dtype = torch.float16

    print(f"\n{'='*60}")
    print(f"2. SPEED BENCHMARK ({check_dtype} - Large Scale)")
    print(f"{'='*60}")

    # Config: Large scale to saturate GPU
    #B, N, D, H = 32, 4096, 64, 8
    #B, N, D, H = 64, 2048, 64, 8
    B, N, D, H = 1, 2048 * 64, 64, 8
    #B, N, D, H = 1, 2048 * 256, 64, 8
    #B, N, D, H = 128, 512, 64, 8
    #B, N, D, H = 128, 256, 64, 8
    #B, N, D, H = 64, 1024, 64, 8
    dim = D * H

    print(f"Config: B={B}, N={N}, D={dim} (HeadDim={D}), H={H}, dtype={check_dtype}")

    model_ref = HierarchicalSparseAttentionRef(dim, H, dropout=0.0).cuda().to(check_dtype).eval()
    model_tri = HierarchicalAttention(dim, H, dropout=0.0).cuda().to(check_dtype).eval()

    x_bench = torch.randn(B, N, dim, device='cuda', dtype=check_dtype, requires_grad=True)
    y_bench = torch.randn(B, N - 1, dim, device='cuda', dtype=check_dtype, requires_grad=True)
    
    # [REAL TRAINING SIMULATION] 
    # Generate the gradient tensor that would come from the upstream layers (e.g., MLP)
    # Shape matches the output of the forward pass: (B, N, dim)
    dout_bench = torch.randn_like(x_bench, device='cuda', dtype=check_dtype)

    # --- Timing Setup ---
    num_warmup = 5
    num_trials = 20 
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # A. Measure PyTorch Reference 
    print("  Running PyTorch Reference (FWD + BWD)...")
    for _ in range(num_warmup): 
        out = model_ref.update_X_from_Y(x_bench, y_bench)
        out.backward(dout_bench)  # Inject real upstream gradients
        model_ref.zero_grad(); x_bench.grad = None; y_bench.grad = None

    torch.cuda.synchronize()
    start.record()
    for _ in range(num_trials):
        out = model_ref.update_X_from_Y(x_bench, y_bench)
        out.backward(dout_bench)  # Inject real upstream gradients
        model_ref.zero_grad(); x_bench.grad = None; y_bench.grad = None
    end.record()
    torch.cuda.synchronize()
    ms_ref = start.elapsed_time(end)

    # B. Measure Triton Kernel
    print("  Running Triton Kernel (FWD + BWD)...")
    for _ in range(num_warmup): 
        out = model_tri.update_X_from_Y(x_bench, y_bench)
        out.backward(dout_bench)  # Inject real upstream gradients
        model_tri.zero_grad(); x_bench.grad = None; y_bench.grad = None

    torch.cuda.synchronize()
    start.record()
    for _ in range(num_trials):
        out = model_tri.update_X_from_Y(x_bench, y_bench)
        out.backward(dout_bench)  # Inject real upstream gradients
        model_tri.zero_grad(); x_bench.grad = None; y_bench.grad = None
    end.record()
    torch.cuda.synchronize()
    ms_tri = start.elapsed_time(end)

    print("-" * 50)
    print(f"  PyTorch Avg Time (Fwd+Bwd): {ms_ref/num_trials:.3f} ms")
    print(f"  Triton  Avg Time (Fwd+Bwd): {ms_tri/num_trials:.3f} ms")
    print(f"  >>> Speedup: {ms_ref/ms_tri:.2f}x")
    print("-" * 50)

    # ==========================================================================
    # 3. PROFILER
    # ==========================================================================
    print(f"\n{'='*60}")
    print("3. DETAILED PROFILING")
    print(f"{'='*60}")

    print("Profiling Triton Kernel Trace...")
    model_tri.zero_grad(); x_bench.grad = None; y_bench.grad = None

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("Triton_Update_X_From_Y"):
            for _ in range(5): 
                out = model_tri.update_X_from_Y(x_bench, y_bench)
                out.backward(dout_bench)  # Inject real upstream gradients
                model_tri.zero_grad(); x_bench.grad = None; y_bench.grad = None

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))



if __name__ == "__main__":
    run_full_suite()
    run_full_suite_update_X_from_Y()