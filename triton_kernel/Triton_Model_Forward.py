import os
# 1. Select Physical GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 2. Tell PyTorch to use the "first and only" visible GPU
# (Do NOT use "cuda:2" or "cuda:3" here, because Python re-indexes it to 0)
device = "cuda:0"

import torch
import triton
import torch.nn as nn
import torch.nn.functional as F
import triton.language as tl
from torch import einsum
import math

import torch
import triton
import triton.language as tl
import math

# ------------------------------------------------------------------
#                    Triton Kernel
# ------------------------------------------------------------------
@triton.jit
def build_parent_self_attention_kernel(
    Q_ptr,              # Parent Query [B, P, H, D]
    Kp_ptr, Vp_ptr,     # Parent Key/Value (Self) [B, P, H, D]
    Kc_ptr, Vc_ptr,     # Child Key/Value (Children) [B, 2*P, H, D]
    Out_ptr,            # Output [B, P, H, D]
    # Strides (Q - Parent Query)
    sq_b, sq_n, sq_h, sq_d,
    # Strides (Kp/Vp - Parent Self)
    skp_b, skp_n, skp_h, skp_d,
    # Strides (Kc/Vc - Children)
    skc_b, skc_n, skc_h, skc_d,
    # Strides (Out)
    so_b, so_n, so_h, so_d,
    # Constants
    sm_scale,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr, 
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    # -----------------------------------------------------------
    # 1. Grid & Indices
    # -----------------------------------------------------------
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # -----------------------------------------------------------
    # 2. Base Pointers Setup
    # -----------------------------------------------------------
    # Parent (Self) pointers
    q_base_ptr = Q_ptr + (b_idx * sq_b) + (node_idx * sq_n)
    kp_base_ptr = Kp_ptr + (b_idx * skp_b) + (node_idx * skp_n)
    
    # Child pointers (2 children per parent)
    child0_idx = 2 * node_idx
    child1_idx = 2 * node_idx + 1
    
    kc0_base_ptr = Kc_ptr + (b_idx * skc_b) + (child0_idx * skc_n)
    kc1_base_ptr = Kc_ptr + (b_idx * skc_b) + (child1_idx * skc_n)
    
    # -----------------------------------------------------------
    # 3. Compute Scores (3-Way: Self, Left, Right)
    # -----------------------------------------------------------
    # Accumulators
    score_self = tl.zeros([BLOCK_H], dtype=tl.float32)
    score_c0   = tl.zeros([BLOCK_H], dtype=tl.float32)
    score_c1   = tl.zeros([BLOCK_H], dtype=tl.float32)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        
        mask_load = mask_h[:, None] & mask_d[None, :]
        
        # Load Q
        ptr_q = q_base_ptr + (offs_h[:, None] * sq_h) + (offs_d[None, :] * sq_d)
        q = tl.load(ptr_q, mask=mask_load, other=0.0)
        
        # Load K (Self)
        ptr_kp = kp_base_ptr + (offs_h[:, None] * skp_h) + (offs_d[None, :] * skp_d)
        kp = tl.load(ptr_kp, mask=mask_load, other=0.0)

        # Load K (Children)
        ptr_kc0 = kc0_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d)
        ptr_kc1 = kc1_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d)
        kc0 = tl.load(ptr_kc0, mask=mask_load, other=0.0)
        kc1 = tl.load(ptr_kc1, mask=mask_load, other=0.0)
        
        # Accumulate Dot Products
        score_self += tl.sum(q * kp, axis=1)
        score_c0   += tl.sum(q * kc0, axis=1)
        score_c1   += tl.sum(q * kc1, axis=1)

    # -----------------------------------------------------------
    # 4. Softmax (over 3 elements)
    # -----------------------------------------------------------
    score_self = score_self * sm_scale
    score_c0   = score_c0 * sm_scale
    score_c1   = score_c1 * sm_scale

    # Find Max for numerical stability
    max_score = tl.maximum(score_self, tl.maximum(score_c0, score_c1))
    
    exp_self = tl.exp(score_self - max_score)
    exp_c0   = tl.exp(score_c0 - max_score)
    exp_c1   = tl.exp(score_c1 - max_score)
    
    denom = exp_self + exp_c0 + exp_c1 + 1e-9
    
    w_self = exp_self / denom
    w_c0   = exp_c0 / denom
    w_c1   = exp_c1 / denom

    # -----------------------------------------------------------
    # 5. Weighted Sum & Store
    # -----------------------------------------------------------
    out_base_ptr = Out_ptr + (b_idx * so_b) + (node_idx * so_n)
    vp_base_ptr = Vp_ptr + (b_idx * skp_b) + (node_idx * skp_n)
    
    # Values for children need separate base pointers (re-using child indices)
    # Assuming Vc follows same indexing logic as Kc
    vc0_base_ptr = Vc_ptr + (b_idx * skc_b) + (child0_idx * skc_n)
    vc1_base_ptr = Vc_ptr + (b_idx * skc_b) + (child1_idx * skc_n)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        mask_op = mask_h[:, None] & mask_d[None, :]
        
        # Calculate Pointers for V
        ptr_vp  = vp_base_ptr  + (offs_h[:, None] * skp_h) + (offs_d[None, :] * skp_d)
        ptr_vc0 = vc0_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d)
        ptr_vc1 = vc1_base_ptr + (offs_h[:, None] * skc_h) + (offs_d[None, :] * skc_d)
        ptr_out = out_base_ptr + (offs_h[:, None] * so_h) + (offs_d[None, :] * so_d)
        
        vp  = tl.load(ptr_vp, mask=mask_op, other=0.0)
        vc0 = tl.load(ptr_vc0, mask=mask_op, other=0.0)
        vc1 = tl.load(ptr_vc1, mask=mask_op, other=0.0)
        
        # Weighted Sum: w_self*Vp + w_c0*Vc0 + w_c1*Vc1
        out_val = (w_self[:, None] * vp) + (w_c0[:, None] * vc0) + (w_c1[:, None] * vc1)
        
        tl.store(ptr_out, out_val, mask=mask_op)

# ------------------------------------------------------------------
#                    Python Wrapper
# ------------------------------------------------------------------
def build_parent_nodes(Q_p, K_p, V_p, K_c, V_c):
    """
    Computes Parent-Self Attention using Triton.
    
    Inputs:
      Q_p, K_p, V_p: [B, Parent_Count, H, D] (The parents)
      K_c, V_c:      [B, Child_Count, H, D]  (The children)
      
    Note: Child_Count must equal 2 * Parent_Count
    """
    B, P, H, D = Q_p.shape
    C = K_c.shape[1]
    assert C == 2 * P, f"Child count {C} must be 2x Parent count {P}"
    
    Out = torch.empty_like(Q_p)
    
    grid = (P, B)
    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_SIZE = triton.next_power_of_2(D)

    # We assume V has same strides as K for simplicity in python args, 
    # but pass them explicitly if your memory layout is weird.
    
    build_parent_self_attention_kernel[grid](
        # Pointers
        Q_p, 
        K_p, V_p, 
        K_c, V_c, 
        Out,
        # Strides Q
        Q_p.stride(0), Q_p.stride(1), Q_p.stride(2), Q_p.stride(3),
        # Strides Parent K/V (Assuming K_p and V_p share layout or are contiguous)
        K_p.stride(0), K_p.stride(1), K_p.stride(2), K_p.stride(3),
        # Strides Child K/V
        K_c.stride(0), K_c.stride(1), K_c.stride(2), K_c.stride(3),
        # Strides Out
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        # Constants
        sm_scale=1.0 / math.sqrt(D),
        H=H,
        BLOCK_H=BLOCK_H,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    
    return Out


@triton.jit
def hierarchical_attention_forward_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, 
    Lookup_ptr,        # [CHANGED] Fused Index Table (contains signs)
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
    BLOCK_LEVELS: tl.constexpr
    # [REMOVED] HAS_MASK, Mask_ptr
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    # 1. Setup
    h_idx = tl.arange(0, BLOCK_H)
    mask_h = h_idx < H
    offs_d = tl.arange(0, BLOCK_D)
    
    # The levels dimension now includes Self + Neighbors
    offs_lvl = tl.arange(0, BLOCK_LEVELS)
    mask_lvl = offs_lvl < LEVELS

    # 2. Load Topology (Fused)
    off_lookup = node_idx * sl_n + offs_lvl * sl_lvl
    
    # Load raw encoded values. 
    # 'other=1' ensures padding defaults to (Index 0, Visible) to prevent pointer crashes
    raw_encoded = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl, other=1)

    # --- DECODE ---
    # Real Index = abs(raw) - 1
    neighbor_indices = tl.abs(raw_encoded) - 1
    
    # Mask Status: If raw is negative, it's masked (Future)
    is_masked_causal = (raw_encoded < 0)

    # 3. Base Pointers
    # We calculate the base offsets for K and V batches
    k_batch_base = K_ptr + b_idx * sk_b
    v_batch_base = V_ptr + b_idx * sv_b
    
    # Q Pointer setup
    q_ptr = Q_ptr + (b_idx * sq_b) + (node_idx * sq_n) + \
            (h_idx[:, None] * sq_h) + (offs_d[None, :] * sq_d)

    # Unified Accumulator (Covers Self + Cross)
    acc = tl.zeros([BLOCK_H, BLOCK_LEVELS], dtype=tl.float32)

    # 4. Score Loop (Compute Q * K^T)
    for off_d_start in range(0, D, BLOCK_D):
        cur_offs_d = off_d_start + offs_d
        d_mask = cur_offs_d < D
        mask_q = mask_h[:, None] & d_mask[None, :]
        
        q = tl.load(q_ptr, mask=mask_q, other=0.0)
        
        # --- UNIFIED K LOAD ---
        # Calculate offsets for ALL items in the topology list at once
        off_node_all = neighbor_indices * sk_n
        
        ptr_k = k_batch_base + \
                off_node_all[None, :, None] + \
                (h_idx[:, None, None] * sk_h) + \
                (cur_offs_d[None, None, :] * sk_d)
        
        mask_k = mask_h[:, None, None] & mask_lvl[None, :, None] & d_mask[None, None, :]
        k = tl.load(ptr_k, mask=mask_k, other=0.0)
        
        # Dot Product
        # Q: [BLOCK_H, BLOCK_D]
        # K: [BLOCK_H, BLOCK_LEVELS, BLOCK_D]
        # Result: [BLOCK_H, BLOCK_LEVELS]
        acc += tl.sum(q[:, None, :] * k, axis=2)
        
        # Advance Q pointer
        q_ptr += BLOCK_D * sq_d

    # 5. Softmax
    acc = acc * sm_scale
    
    # --- APPLY MASK ---
    # Combine structural padding (offs_lvl >= LEVELS) AND our fused causal mask
    mask_final = (offs_lvl >= LEVELS) | is_masked_causal
    
    acc = tl.where(mask_final[None, :], -float('inf'), acc)
    
    # Standard Softmax Logic
    max_val = tl.max(acc, axis=1)
    exp_val = tl.exp(acc - max_val[:, None])
    denom = tl.sum(exp_val, axis=1)
    weights = exp_val / denom[:, None]

    # Save Weights
    w_base_ptr = W_ptr + (b_idx * sw_b) + (node_idx * sw_n) + (h_idx * sw_h)
    w_ptr_offsets = w_base_ptr[:, None] + (offs_lvl[None, :] * sw_lvl)
    tl.store(w_ptr_offsets, weights, mask=mask_h[:, None] & mask_lvl[None, :])

    # 6. Weighted Sum Loop (Compute Weights * V)
    out_base_ptr = Out_ptr + (b_idx * so_b) + (node_idx * so_n) + \
                   (h_idx[:, None] * so_h) + (offs_d[None, :] * so_d)

    for off_d_start in range(0, D, BLOCK_D):
        cur_offs_d = off_d_start + offs_d
        d_mask = cur_offs_d < D
        mask_op = mask_h[:, None] & d_mask[None, :]

        # --- UNIFIED V LOAD ---
        # Reuse decoded neighbor_indices, but apply V strides
        off_node_all = neighbor_indices * sv_n 
        
        ptr_v = v_batch_base + \
                off_node_all[None, :, None] + \
                (h_idx[:, None, None] * sv_h) + \
                (cur_offs_d[None, None, :] * sv_d)
                      
        mask_v = mask_h[:, None, None] & mask_lvl[None, :, None] & d_mask[None, None, :]
        v = tl.load(ptr_v, mask=mask_v, other=0.0)
        
        # Weighted Sum
        # Weights: [BLOCK_H, BLOCK_LEVELS]
        # V:       [BLOCK_H, BLOCK_LEVELS, BLOCK_D]
        # Out:     [BLOCK_H, BLOCK_D]
        out_acc = tl.sum(weights[:, :, None] * v, axis=1)
        
        tl.store(out_base_ptr, out_acc.to(Out_ptr.dtype.element_ty), mask=mask_op)
        out_base_ptr += BLOCK_D * so_d

def hierarchical_fused_attention(Q, K, V, idx_table):
    """
    Args:
        Q: [B, N, H, D]
        K: [B, Total_Nodes, H, D]
        V: [B, Total_Nodes, H, D]
        idx_table: [N, LEVELS] (Int32) - Fused Table (Self + Neighbors)
                   Contains sign-bit encoded masking info.
    """
    B, N, H, Dh = Q.shape
    
    # The table width defines the attention span (Self + Hierarchy)
    LEVELS = idx_table.shape[1] 
    
    Out = torch.empty_like(Q)

    # Weights buffer shape matches table width exactly
    Weights = torch.empty((B, N, H, LEVELS), device=Q.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # ALIGNMENT CHECKS
    # ------------------------------------------------------------------
    assert Q.data_ptr() % 16 == 0, "Q ptr not 16-byte aligned"
    assert K.data_ptr() % 16 == 0, "K ptr not 16-byte aligned"
    assert V.data_ptr() % 16 == 0, "V ptr not 16-byte aligned"
    assert Out.data_ptr() % 16 == 0, "Out ptr not 16-byte aligned"

    grid = (N, B)

    BLOCK_LEVELS = triton.next_power_of_2(LEVELS)
    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_D = triton.next_power_of_2(Dh)

    hierarchical_attention_forward_kernel[grid](
        Q, K, V, 
        idx_table, # No mask table needed
        Out,
        Weights,
        # Strides
        *Q.stride(),
        *K.stride(), 
        *V.stride(),
        idx_table.stride(0), idx_table.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        *Weights.stride(),
        # Constants
        sm_scale=1.0 / math.sqrt(Dh),
        H=H, 
        BLOCK_H=BLOCK_H, 
        D=Dh, 
        LEVELS=LEVELS,
        BLOCK_D=BLOCK_D,
        BLOCK_LEVELS=BLOCK_LEVELS,
        num_warps=4 
    )
    return Out



def build_hierarchical_index_lookup_table_bak(seq_len, device="cuda", dtype=torch.int32):
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
                #assert False, "Should not be here!!!"
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


def build_hierarchical_index_lookup_table(seq_len, is_causal=True, device="cuda", dtype=torch.int32):
    """
    Builds a FUSED index table with optional Causal Masking.
    
    Args:
        seq_len (int): Sequence length (must be power of 2).
        is_causal (bool): 
            - If True: Future neighbors are encoded as Negative (Masked).
            - If False: All valid neighbors are Positive (Visible/Bidirectional).
            
    Returns: 
        idx_map: [seq_len, level_num + 1]
    """
    assert (seq_len & (seq_len - 1)) == 0, "seq_len must be a power of 2"

    total_nodes = 2 * seq_len - 1
    max_valid = total_nodes - 2
    level_num = int(math.log2(seq_len))
    total_cols = level_num + 1

    # Initialize with 1 (points to Node 0, Visible) to be safe for padding
    idx_map = torch.ones((seq_len, total_cols), dtype=dtype, device=device)

    # 1. Setup Column 0 (Self Node) -> Always Visible (Positive)
    # Stores (Index + 1)
    n_cur = torch.arange(seq_len, device=device, dtype=torch.int64)
    idx_map[:, 0] = (n_cur + 1).to(dtype)

    # 2. Hierarchical Traversal
    for lvl in range(1, total_cols):
        if lvl == 1:
            n_next = n_cur ^ 1
            pair = n_cur
        else:
            n_next = (n_cur // 2 + seq_len) ^ 1
            pair = (n_cur // 2 + seq_len)

        valid_mask = n_next <= max_valid
        
        # Base value to store: (Index + 1)
        val_to_store = n_next + 1
        
        # [CHANGE] Apply Causal Logic only if requested
        if is_causal:
            # Check if neighbor is in the "future" (pair < n_next)
            should_mask = (pair < n_next) & valid_mask
            # Flip sign to Negative if masked
            val_to_store = torch.where(should_mask, -val_to_store, val_to_store)

        # Update Table: Only overwrite where valid
        idx_map[:, lvl] = torch.where(valid_mask, val_to_store.to(dtype), idx_map[:, lvl])

        n_cur = n_next

    return idx_map

class HierarchicalSparseAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # --- Y Updates (Bottom-Up) ---
        self.Wq_y = nn.Linear(dim, dim, bias=False)
        self.Wk_y = nn.Linear(dim, dim, bias=False)
        self.Wv_y = nn.Linear(dim, dim, bias=False)
        self.out_proj_y = nn.Linear(dim, dim)

        # --- X Updates (Top-Down) ---
        self.Wq_x = nn.Linear(dim, dim, bias=False)
        self.Wk_x = nn.Linear(dim, dim, bias=False)
        self.Wv_x = nn.Linear(dim, dim, bias=False)
        self.out_proj_x = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        self.cached_idx_table = None
        self.cached_seq_len = -1
        self.cached_is_causal = None # Add this

        self.sizes = None
        self.offsets = None

    def _get_lookup_table(self, L, is_causal, device):
        """
        Smart retrieval: Returns cached fused table if (L, is_causal, device) match.
        """
        # Check if we can reuse the cache
        # We must ensure the cached table matches the requested causality (is_causal)
        if (self.cached_idx_table is not None and 
            self.cached_seq_len == L and 
            self.cached_idx_table.device == device and
            getattr(self, 'cached_is_causal', None) == is_causal):
            return self.cached_idx_table

        # If not, recompute and update cache
        # [CHANGE] Now returns only idx_table (mask is inside it)
        idx_table = build_hierarchical_index_lookup_table(
            L, 
            is_causal=is_causal, 
            device=device, 
            dtype=torch.int32
        )
    
        self.cached_idx_table = idx_table
        self.cached_seq_len = L
        self.cached_is_causal = is_causal # [CHANGE] Store causality state
    
        return idx_table

    @staticmethod
    def generate_span_input_Y(x):
        """Initializes Y by averaging adjacent pairs of X."""
        B, N, D = x.shape
        Y_levels = []
        curr = x
        while curr.size(1) > 1:
            L = curr.size(1)
            even = L - (L % 2)
            curr_pairs = curr[:, :even, :].reshape(B, even // 2, 2, D)
            parents = 0.5 * curr_pairs[:, :, 0, :] + 0.5 * curr_pairs[:, :, 1, :]
            Y_levels.append(parents)
            curr = parents
        
        if not Y_levels:
            return None
            
        return torch.cat(Y_levels, dim=1)

    @staticmethod
    def build_level_info(N):
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
        Architecture: Recursive Parent-Self Attention.
        
        Refactored Layout: (Batch, Heads, Sequence, Head_Dim) aka (B, H, N, Dh).
        This matches standard PyTorch MultiheadAttention implementation details.
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        assert y_in is not None

        if self.sizes is None: self.sizes, self.offsets = self.build_level_info(N)

        # -------------------------------------------------------
        # OPTIMIZATION: Global Parent Projection (Standard Layout)
        # -------------------------------------------------------
        # Input: (B, Total_Parents, D)
        # 1. Project -> (B, Total_Parents, H, Dh)
        # 2. Transpose -> (B, H, Total_Parents, Dh)
        Q_p_all = self.Wq_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        K_p_all = self.Wk_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        V_p_all = self.Wv_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        
        new_Y_levels = []
        prev_sources = x 

        for level, parent_count in enumerate(self.sizes):
            # ---------------------------------------------------
            # 1. Prepare Children
            # ---------------------------------------------------
            useful_len = parent_count * 2
            children = prev_sources[:, :useful_len, :] 
            
            # Project Children -> (B, H, 2*P, Dh)
            K_c = self.Wk_y(children).view(B, -1, H, Dh).transpose(1, 2)
            V_c = self.Wv_y(children).view(B, -1, H, Dh).transpose(1, 2)
            V_c = self.dropout(V_c)

            # ---------------------------------------------------
            # 2. Reshape Children to Pairs
            # ---------------------------------------------------
            # Current: (B, H, 2*P, Dh)
            # Target:  (B, H, P, 2, Dh)  <-- Group pairs together
            K_c_pairs = K_c.view(B, H, parent_count, 2, Dh)
            V_c_pairs = V_c.view(B, H, parent_count, 2, Dh)

            # ---------------------------------------------------
            # 3. Slice Parents
            # ---------------------------------------------------
            # Current: (B, H, Total_Parents, Dh)
            # Target:  (B, H, P, Dh)
            offset = self.offsets[level]
            Q_p = Q_p_all[:, :, offset : offset + parent_count, :]
            K_p = K_p_all[:, :, offset : offset + parent_count, :]
            V_p = V_p_all[:, :, offset : offset + parent_count, :]

            # ---------------------------------------------------
            # 4. Form Attention Pool
            # ---------------------------------------------------
            # Parent (Self) needs to be unsqueezed to match Child pairs structure
            # K_p: (B, H, P, Dh) -> (B, H, P, 1, Dh)
            # Pool: Concat along the 'pair' dim -> (B, H, P, 3, Dh)
            # Pool Order: [Parent(Self), Child_Left, Child_Right]
            K_pool = torch.cat([K_p.unsqueeze(3), K_c_pairs], dim=3)
            V_pool = torch.cat([V_p.unsqueeze(3), V_c_pairs], dim=3)

            # ---------------------------------------------------
            # 5. Attention Scores
            # ---------------------------------------------------
            # Q: (B, H, P, Dh) -> (B, H, P, 1, Dh)
            # K_pool: (B, H, P, 3, Dh) -> Transpose to (B, H, P, Dh, 3)
            
            # Matmul: (B,H,P,1,Dh) @ (B,H,P,Dh,3) -> (B, H, P, 1, 3)
            # This computes 3 scores per parent: Self, Left, Right
            logits = torch.matmul(Q_p.unsqueeze(3), K_pool.transpose(-1, -2))
            logits = logits / math.sqrt(Dh)
            
            weights = F.softmax(logits, dim=-1) # (B, H, P, 1, 3)

            # ---------------------------------------------------
            # 6. Weighted Sum
            # ---------------------------------------------------
            # V_pool: (B, H, P, 3, Dh)
            # Matmul: (B,H,P,1,3) @ (B,H,P,3,Dh) -> (B, H, P, 1, Dh)
            attn_out = torch.matmul(weights, V_pool)
            
            # ---------------------------------------------------
            # 7. Merge Heads
            # ---------------------------------------------------
            # 1. Squeeze the '1' dim: (B, H, P, Dh)
            # 2. Transpose back to Sequence First: (B, P, H, Dh)
            # 3. Flatten/Reshape: (B, P, D)
            attn_out = attn_out.squeeze(3).transpose(1, 2).contiguous().reshape(B, parent_count, D)
            
            new_Y_levels.append(attn_out)
            prev_sources = attn_out

        return self.out_proj_y(torch.cat(new_Y_levels, dim=1))

    def update_X_from_Y(self, x, y, mask=None):

        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        if y is None: return x

        # Concatenate inputs once to allow unified K/V calculation
        XY = torch.cat([x, y], dim=1)

        # --- Inline Split Heads ---
        Q = self.Wq_x(x).view(B, N, H, Dh).transpose(1, 2)
        
        # Calculate K, V on combined input (Efficiency: one large matmul instead of two)
        kv_input = self.Wk_x(XY).view(B, -1, H, Dh).transpose(1, 2)
        v_input = self.Wv_x(XY).view(B, -1, H, Dh).transpose(1, 2)

        K_full = kv_input
        V_full = self.dropout(v_input) # Apply dropout to the projected values

        idx_table, neighbor_causal_mask = self._get_lookup_table(N, device=x.device)
            
        # Assert table validity
        # assert (idx_table != -1).all(), "Index table contains invalid entries."

        # Self: The leaves attending to themselves (indices 0 to N-1)
        K_self = K_full[:, :, :N, :]                 # [B, H, N, D]
        V_self = V_full[:, :, :N, :]                 # [B, H, N, D]

        # Gather neighbors using index table
        # idx_table: [N, Levels]
        gather_indices = idx_table
            
        # neighbors_k: [B, H, L, Levels, D]
        neighbors_k = K_full[:, :, gather_indices, :] 
        neighbors_v = V_full[:, :, gather_indices, :]

        # Compute Self Logits (Leaf Q * Leaf K)
        # [B, H, L, D] * [B, H, L, D] -> [B, H, L]
        self_logits = torch.einsum('b h n d, b h n d -> b h n', Q, K_self) / math.sqrt(Dh)

        # Compute Neighbor Logits (Leaf Q * Neighbor K)
        # Q: [B, H, N, D] -> unsqueeze -> [B, H, N, 1, D]
        # neighbors_k: [B, H, N, Levels, D]
        # Result: [B, H, N, Levels]
        neighbors_logits = torch.einsum('b h l x d, b h l n d -> b h l n', Q.unsqueeze(3), neighbors_k)
        neighbors_logits = neighbors_logits / math.sqrt(Dh)

        # Apply Causal Mask
        if mask is not None:
            # neighbor_causal_mask is [L, Levels]. 
            # True means "future" (masked).
            neighbors_logits = neighbors_logits.masked_fill(neighbor_causal_mask, float('-inf'))

        # Concatenate (Self + Neighbors)
        # V: [B, H, L, D] -> [B, H, L, 1, D]
        # self_logits: [B, H, L] -> [B, H, L, 1]  
        # Corrected: Use 'V_self' instead of 'V' (which is undefined in this scope)
        all_v = torch.cat([V_self.unsqueeze(3), neighbors_v], dim=3)             # [B, H, L, Levels+1, Dh]
        all_logits = torch.cat([self_logits.unsqueeze(3), neighbors_logits], dim=3) # [B, H, L, Levels+1]

        # Attention Softmax & Weighted Sum
        max_logits = all_logits.max(dim=-1, keepdim=True)[0]
        weights = F.softmax(all_logits - max_logits, dim=-1)             # [B, H, L, Levels+1]
            
        output_leaf = torch.einsum('b h l n, b h l n d -> b h l d', weights, all_v)

        # --- Inline Merge Heads ---
        return output_leaf.transpose(1, 2).reshape(B, N, D)

    def _standard_attention(self, Q, K, V, mask):
        # Q, K, V are already [B, H, N, Dh]
        D_head = Q.size(-1)
        
        # Optimization: removed explicit .contiguous() calls.
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
            # Hierarchical Update
            output_leaf = self.update_X_from_Y(x, y, mask)
            output = self.out_proj_x(output_leaf)
            return (output, None) if return_attention else output
        else:
            # Standard Attention Fallback
            # --- Inline Split ---
            Q = self.Wq_x(query).view(B, L_Q, H, Dh).transpose(1, 2)
            K = self.Wk_x(key).view(B, L_K, H, Dh).transpose(1, 2)
            V = self.Wv_x(value).view(B, L_V, H, Dh).transpose(1, 2)
            
            output_leaf, attn_weights = self._standard_attention(Q, K, V, mask)
            
            # --- Inline Merge ---
            output = output_leaf.transpose(1, 2).reshape(B, L_Q, D)
            output = self.out_proj_x(output)
            
            return (output, attn_weights) if return_attention else output


class HierarchicalSparseAttentionTriton(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # --- Y Updates (Bottom-Up) ---
        self.Wq_y = nn.Linear(dim, dim, bias=False)
        self.Wk_y = nn.Linear(dim, dim, bias=False)
        self.Wv_y = nn.Linear(dim, dim, bias=False)
        self.out_proj_y = nn.Linear(dim, dim)

        # --- X Updates (Top-Down) ---
        self.Wq_x = nn.Linear(dim, dim, bias=False)
        self.Wk_x = nn.Linear(dim, dim, bias=False)
        self.Wv_x = nn.Linear(dim, dim, bias=False)
        self.out_proj_x = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        
        self.cached_idx_table = None
        self.cached_seq_len = -1
        self.cached_is_causal = None

        self.sizes = None
        self.offsets = None

    def _get_lookup_table(self, L, is_causal, device):
        """
        Smart retrieval: Returns cached fused table if (L, is_causal, device) match.
        """
        # Check if we can reuse the cache
        # We must ensure the cached table matches the requested causality (is_causal)
        if (self.cached_idx_table is not None and 
            self.cached_seq_len == L and 
            self.cached_idx_table.device == device and
            getattr(self, 'cached_is_causal', None) == is_causal):
            return self.cached_idx_table

        # If not, recompute and update cache
        # [CHANGE] Now returns only idx_table (mask is inside it)
        idx_table = build_hierarchical_index_lookup_table(
            L, 
            is_causal=is_causal, 
            device=device, 
            dtype=torch.int32
        )
    
        self.cached_idx_table = idx_table
        self.cached_seq_len = L
        self.cached_is_causal = is_causal # [CHANGE] Store causality state
    
        return idx_table

    @staticmethod
    def generate_span_input_Y(x):
        """Initializes Y by averaging adjacent pairs of X."""
        B, N, D = x.shape
        Y_levels = []
        curr = x
        while curr.size(1) > 1:
            L = curr.size(1)
            even = L - (L % 2)
            curr_pairs = curr[:, :even, :].reshape(B, even // 2, 2, D)
            parents = 0.5 * curr_pairs[:, :, 0, :] + 0.5 * curr_pairs[:, :, 1, :]
            Y_levels.append(parents)
            curr = parents
        
        if not Y_levels:
            return None
            
        return torch.cat(Y_levels, dim=1)

    @staticmethod
    def build_level_info(N):
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
        Optimized bottom-up update using Triton Kernel with Parent-Self Attention.
        
        Logic:
          - Pool = [Parent (Self), Child_Left, Child_Right]
          - Q = Parent
          - K, V = Pool
          
        Key Optimization:
          - Global projection of all Parents (Q, K, V) at once.
          - Single fused kernel handles the 3-way attention and reduction.
          - Maintains [Batch, Node, Head, Dim] layout to avoid transposes.
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        
        assert y_in is not None, "y_in cannot be None"
        assert y_in.size(1) > 0, "y_in must have valid tokens"

        if self.sizes is None: self.sizes, self.offsets = self.build_level_info(N)

        # -------------------------------------------------------
        # OPTIMIZATION 1: Global Parent Projection
        # -------------------------------------------------------
        # We project ALL parents at once for Q, K, and V.
        # This allows the parent to attend to itself (Parent-Self Attention).
        # Layout: [B, Total_Parents, H, Dh] (Native Linear output, no transpose)
        Q_p_all = self.Wq_y(y_in).view(B, -1, H, Dh)
        K_p_all = self.Wk_y(y_in).view(B, -1, H, Dh)
        V_p_all = self.Wv_y(y_in).view(B, -1, H, Dh)
        
        new_Y_levels = []
        prev_sources = x # Starts as the leaves (first layer children)
        
        for level, parent_count in enumerate(self.sizes):
            offset = self.offsets[level]

            # ---------------------------------------------------
            # 1. Prepare Parents (Slicing)
            # ---------------------------------------------------
            # Slice Q, K, V for this specific level's parents
            # Shape: [B, Parent_Count, H, Dh]
            Q_p = Q_p_all[:, offset : offset + parent_count, :, :]
            K_p = K_p_all[:, offset : offset + parent_count, :, :]
            V_p = V_p_all[:, offset : offset + parent_count, :, :]
            
            # ---------------------------------------------------
            # 2. Prepare Children (Projection)
            # ---------------------------------------------------
            useful_len = parent_count * 2
            # Slice strictly to useful length to ensure even pairs
            children_in = prev_sources[:, :useful_len, :]
            
            # Project Children Keys/Values (Must be done in loop as prev_sources changes)
            # Shape: [B, 2*P, H, Dh]
            K_c = self.Wk_y(children_in).view(B, -1, H, Dh)
            V_c = self.Wv_y(children_in).view(B, -1, H, Dh)
            V_c = self.dropout(V_c)

            # ---------------------------------------------------
            # 3. Triton Kernel (3-Way Attention)
            # ---------------------------------------------------
            # Replaces: Reshape -> Cat -> Dot -> Softmax -> Weighted Sum
            # Input:  Parents (Q,K,V) and Children (K,V)
            # Output: Updated Parents [B, P, H, Dh]
            updated_heads = build_parent_nodes(Q_p, K_p, V_p, K_c, V_c)

            # ---------------------------------------------------
            # 4. Merge Heads
            # ---------------------------------------------------
            # [B, P, H, Dh] -> [B, P, D]
            updated_merged = updated_heads.reshape(B, parent_count, D)
            
            new_Y_levels.append(updated_merged)
            prev_sources = updated_merged

        # Concatenate and Project
        Y_new = torch.cat(new_Y_levels, dim=1)
        Y_new = self.out_proj_y(Y_new)
        
        return Y_new

    def update_X_from_Y(self, x, y, mask=None):
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # [Check] Enforce strict existence
        assert y is not None, "y (parents) cannot be None during top-down update"

        # 1. Combine Input (Leaves + Parents)
        XY = torch.cat([x, y], dim=1)
        
        # 2. Separate Projections (No Fused Wkv)
        # Q: [B, N, H, Dh]
        Q = self.Wq_x(x).view(B, N, H, Dh)
        
        # K, V: [B, Total_Nodes, H, Dh]
        # We project K and V separately using your existing layers
        K_full = self.Wk_x(XY).view(B, -1, H, Dh)
        V_full = self.Wv_x(XY).view(B, -1, H, Dh)
        
        # Apply dropout to V
        V_full = self.dropout(V_full)

        # 3. Ensure Contiguity
        # The Triton kernel does pointer arithmetic. While Linear() outputs are usually 
        # contiguous, operations like Dropout or View can sometimes create strides 
        # that break optimized kernels. We enforce it here to be safe.
        if not K_full.is_contiguous(): K_full = K_full.contiguous()
        if not V_full.is_contiguous(): V_full = V_full.contiguous()

        is_causal = True if mask is not None else False

        # Get Topology
        idx_table = self._get_lookup_table(N, is_causal=is_causal, device=x.device)

        output_leaf_heads = hierarchical_fused_attention( Q, K_full, V_full, idx_table)
    
        return output_leaf_heads.view(B, N, D)

    def _standard_attention(self, Q, K, V, mask):
        # Q, K, V are already [B, H, N, Dh]
        D_head = Q.size(-1)
        
        # Optimization: removed explicit .contiguous() calls.
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
            # Hierarchical Update
            output_leaf = self.update_X_from_Y(x, y, mask)
            output = self.out_proj_x(output_leaf)
            return (output, None) if return_attention else output
        else:
            # Standard Attention Fallback
            # --- Inline Split ---
            Q = self.Wq_x(query).view(B, L_Q, H, Dh).transpose(1, 2)
            K = self.Wk_x(key).view(B, L_K, H, Dh).transpose(1, 2)
            V = self.Wv_x(value).view(B, L_V, H, Dh).transpose(1, 2)
            
            output_leaf, attn_weights = self._standard_attention(Q, K, V, mask)
            
            # --- Inline Merge ---
            output = output_leaf.transpose(1, 2).reshape(B, L_Q, D)
            output = self.out_proj_x(output)
            
            return (output, attn_weights) if return_attention else output



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None, return_attention=False):
        batch_size = query.shape[0]
        
        # 1. Project and Reshape
        # Resulting shape: [Batch, Heads, SeqLen, HeadDim]
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. FlashAttention Implementation
        if not return_attention:
            # OPTIMIZATION:
            # If mask is present, we assume it's the standard causal mask.
            # We set is_causal=True and attn_mask=None to force the FlashAttention-2 kernel.
            is_causal = (mask is not None)
            
            output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None,           # <--- Pass None here
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal       # <--- Pass True here
            )
            attn_weights = None
            
        else:
            # 3. Fallback for Visualization (Standard Implementation)
            # FlashAttention cannot return weights because it never calculates the full NxN matrix.
            # We revert to standard logic only if weights are explicitly requested.
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
            
            if mask is not None:
                min_value = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(mask == 0, min_value)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, V)

        # 4. Reassemble Heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_weights
        
        return output



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=80000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (part of state_dict but not a trained parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        # self.pe: [1, Max_Len, Dim] -> Sliced to [1, Seq_Len, Dim]
        # Broadcasting adds pe to every item in the batch automatically
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- IMPROVED DECODER LAYER (Pre-LN Architecture) ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, triton=False, dropout=0.1):
        super().__init__()
        # Note: dim=d_model to match your previous code hyperparameters
        if triton:
            self.self_attn = HierarchicalSparseAttentionTriton(d_model, num_heads, dropout)
        else:
            self.self_attn = HierarchicalSparseAttention(d_model, num_heads, dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # --- NEW: Norm for Y ---
        self.norm_y = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y, mask=None, return_attention=False):
 
        # 1. Update Y (Hierarchy) with Residual + Norm
        # We use 'norm_y(y)' as input to be safe, similar to Pre-LN for x
        y_norm = self.norm_y(y)
        
        # Calculate the update (delta)
        y_delta = self.self_attn.cross_update_Y(x, y_in=y_norm)
        
        # Apply Residual Connection to Y
        y_next = y + self.dropout(y_delta)

    
        # PRE-LAYER NORMALIZATION (Apply Norm BEFORE Attention)
        # This significantly improves stability and convergence speed
        
        # Norm -> Attention -> Add
        norm_x = self.norm1(x)

        if return_attention:
            attn_output, self_attn_weights = self.self_attn(norm_x, norm_x, norm_x, y=y_next, mask=mask, return_attention=True)
        else:
            attn_output = self.self_attn(norm_x, norm_x, norm_x, y=y_next, mask=mask)
            
        x = x + self.dropout(attn_output) # Residual
        
        # FF
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        if return_attention:
            return x, y_next, self_attn_weights
        return x, y_next

# --- IMPROVED MODEL CLASS (Added Initialization) ---
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, triton=False, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, triton, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Weight Tying (Optional but good for PPL)
        self.fc_out.weight = self.embedding.weight
        self.d_model = d_model
        
        # --- Apply Initialization ---
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        # Initialize weights with small std (0.02) to prevent high starting loss
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def make_causal_mask(self, x):
        seq_len = x.shape[1]
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, trg, return_attention=False):
        x = self.embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Initialize Y from the embeddings using the static method
        y = HierarchicalSparseAttention.generate_span_input_Y(x)
        
        trg_mask = self.make_causal_mask(trg)

        attentions = [] if return_attention else None
        
        for layer in self.layers:
            if return_attention:
                x, y, attention = layer(x, y, mask=None, return_attention=True)
                attentions.append(attention)
            else:
                x, y = layer(x, y, mask=trg_mask)
        
        output = self.fc_out(x)
        return output


class DecoderLayerStandard(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, trg_mask=None, return_attention=False):
        # PRE-LAYER NORMALIZATION (Apply Norm BEFORE Attention)
        # This significantly improves stability and convergence speed
        
        # 1. Norm -> Attention -> Add
        norm_x = self.norm1(x)
        if return_attention:
            attn_output, self_attn_weights = self.self_attn(norm_x, norm_x, norm_x, mask=trg_mask, return_attention=True)
        else:
            attn_output = self.self_attn(norm_x, norm_x, norm_x, mask=trg_mask)
            
        x = x + self.dropout(attn_output) # Residual
        
        # 2. Norm -> FeedForward -> Add
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output) # Residual
        
        if return_attention:
            return x, self_attn_weights
        return x


class TransformerLMStandard(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayerStandard(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Weight Tying (Optional but good for PPL)
        self.fc_out.weight = self.embedding.weight
        self.d_model = d_model
        
        # --- Apply Initialization ---
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        # Initialize weights with small std (0.02) to prevent high starting loss
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def make_causal_mask(self, x):
        seq_len = x.shape[1]
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, trg, return_attention=False):
        trg_mask = self.make_causal_mask(trg)
        
        x = self.embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        attentions = [] if return_attention else None
        
        for layer in self.layers:
            if return_attention:
                x, attention = layer(x, trg_mask=None, return_attention=True)
                attentions.append(attention)
            else:
                x = layer(x, trg_mask=trg_mask)
        
        output = self.fc_out(x)
        return output



import time
from torch.profiler import profile, record_function, ProfilerActivity

def run_transformer_benchmark():
    # --------------------------------------------------------------------------
    # 0. CONFIGURATION
    # --------------------------------------------------------------------------
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    torch.cuda.set_device(device)

    # Hyperparameters from request
    VOCAB_SIZE = 50000
    D_MODEL = 768
    NUM_HEADS = 12
    D_FF = 3072
    NUM_LAYERS = 12
    DROPOUT = 0.0 # As requested
    
    # Batch config
    B = 1   # Reduced slightly from 16 to ensure safety on standard VRAM with 12 layers
    SEQ_LEN = 2048 * 32
    
    print(f"\n{'='*60}")
    print(f" TRANSFORMER LM BENCHMARK (B={B}, L={SEQ_LEN}, Layers={NUM_LAYERS})")
    print(f" D_Model={D_MODEL}, Heads={NUM_HEADS}, Vocab={VOCAB_SIZE}")
    print(f"{'='*60}")

    # --------------------------------------------------------------------------
    # 1. SETUP & WEIGHT SYNCHRONIZATION
    # --------------------------------------------------------------------------
    print("Initializing models...")
    
    # A. Standard PyTorch Model
    model_ref = TransformerLM(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS, 
        d_ff=D_FF, num_layers=NUM_LAYERS, triton=False, dropout=DROPOUT
    ).to(device).half() # Float16
    
    # B. Triton Optimized Model
    model_opt = TransformerLM(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS, 
        d_ff=D_FF, num_layers=NUM_LAYERS, triton=True, dropout=DROPOUT
    ).to(device).half() # Float16
    
    # C. Flash Attention Model
    model_flash = TransformerLMStandard(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS, 
        d_ff=D_FF, num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device).half() # Float16

    # CRITICAL: Load weights from Ref to Opt to ensure outputs are comparable
    print("Synchronizing weights (Ref -> Opt)...")
    model_opt.load_state_dict(model_ref.state_dict())

    model_ref.eval()
    model_opt.eval()
    model_flash.eval()

    # Generate Dummy Input
    # Integers for Embedding layer
    x_input = torch.randint(0, VOCAB_SIZE, (B, SEQ_LEN), device=device)

    print("Pre-building lookup tables...")
    # Access the first layer -> self_attn -> call _get_lookup_table directly
    # We don't need update_X_from_Y, we just need to trigger the table build
    model_opt.layers[0].self_attn._get_lookup_table(SEQ_LEN, is_causal=True, device=device)
    model_ref.layers[0].self_attn._get_lookup_table(SEQ_LEN, is_causal=True, device=device)
    print("Tables built.")

    # In run_transformer_benchmark, before the sanity check loop:

    # --------------------------------------------------------------------------
    # 2. SANITY CHECK (Correctness)
    # --------------------------------------------------------------------------
    print("\n[1] SANITY CHECK")
    print("-" * 30)
    
    with torch.no_grad():
        # Standard Forward
        out_ref = model_ref(x_input)
        # Triton Forward
        out_opt = model_opt(x_input)
    
    # Compare Logits
    # Note: In deep FP16 networks, accumulated error can be around 1e-2 to 1e-1
    diff = (out_ref - out_opt).abs().max().item()
    print(f"Max Logit Difference: {diff:.6f}")
    
    if diff < 0.1:
        print(">>> STATUS: PASS (Outputs match within FP16 tolerance)")
    else:
        print(">>> STATUS: WARNING (Difference is high, check kernels or sync)")

    # --------------------------------------------------------------------------
    # 3. SPEED BENCHMARK
    # --------------------------------------------------------------------------
    print("\n[2] SPEED BENCHMARK (End-to-End Forward Pass)")
    print("-" * 30)
    
    num_warmup = 5
    num_trials = 20

    ## --- Benchmark PyTorch ---
    #print(f"Benchmarking PyTorch (Ref)...")
    #with torch.no_grad():
    #    for _ in range(num_warmup):
    #        _ = model_ref(x_input)
    #    
    #    torch.cuda.synchronize()
    #    start_event = torch.cuda.Event(enable_timing=True)
    #    end_event = torch.cuda.Event(enable_timing=True)
    #    
    #    start_event.record()
    #    for _ in range(num_trials):
    #        _ = model_ref(x_input)
    #    end_event.record()
    #    torch.cuda.synchronize()
    #    
    #    ms_ref = start_event.elapsed_time(end_event) / num_trials

    # --- Benchmark Triton ---
    print(f"Benchmarking Triton (Opt)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_opt(x_input)
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_trials):
            _ = model_opt(x_input)
        end_event.record()
        torch.cuda.synchronize()
        
        ms_opt = start_event.elapsed_time(end_event) / num_trials


    # --- Benchmark FlashAttention ---
    print(f"Benchmarking FlashAttention (Ref)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_flash(x_input)
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_trials):
            _ = model_flash(x_input)
        end_event.record()
        torch.cuda.synchronize()
        
        ms_flash = start_event.elapsed_time(end_event) / num_trials

    print(f"\nRESULTS:")
    #print(f"  PyTorch Avg Latency: {ms_ref:.2f} ms")
    print(f"  FlashAttention Avg Latency: {ms_flash:.2f} ms")
    print(f"  Triton  Avg Latency: {ms_opt:.2f} ms")
    #print(f"  >>> Speedup: {ms_ref / ms_opt:.2f}x")
    print(f"  >>> Speedup: {ms_flash / ms_opt:.2f}x")

    # --------------------------------------------------------------------------
    # 4. PROFILING
    # --------------------------------------------------------------------------
    print("\n[3] TRITON PROFILING")
    print("-" * 30)
    print("Running profiler on Triton model (1 iteration)...")
    
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False
        ) as prof:
            # Run the forward pass 5 times
            for i in range(5):
                with record_function(f"model_forward_triton"):
                    model_opt(x_input)
                    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))


    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False
        ) as prof:
            # Run the forward pass 5 times
            for i in range(5):
                with record_function(f"model_forward_flash"):
                    model_flash(x_input)
                    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

if __name__ == "__main__":
    run_transformer_benchmark()