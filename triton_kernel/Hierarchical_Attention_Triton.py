import torch
import triton
import torch.nn as nn
import torch.nn.functional as F
import triton.language as tl
from torch import einsum
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
def hierarchical_fused_attention_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, 
    Lookup_ptr, Mask_ptr, 
    Out_ptr,
    
    # Strides
    sq_b, sq_n, sq_h, sq_d,
    sk_b, sk_n, sk_h, sk_d,
    sl_n, sl_lvl,
    so_b, so_n, so_h, so_d,
    
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

    # -----------------------------------------------------------
    # 1. Setup & Alignment
    # -----------------------------------------------------------
    # FIXED: Padded H to power of 2
    h_idx = tl.arange(0, BLOCK_H)
    mask_h = h_idx < H

    offs_d = tl.arange(0, BLOCK_D)
    offs_lvl = tl.arange(0, BLOCK_LEVELS)
    
    # Pre-calculate Mask for Levels
    mask_lvl = offs_lvl < LEVELS

    # -----------------------------------------------------------
    # 2. Load Topology (The "Random Access" Part)
    # -----------------------------------------------------------
    off_lookup = node_idx * sl_n + offs_lvl * sl_lvl
    
    # Load Indices
    neighbor_indices = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl, other=0)
    
    # Load Causal Mask (Optimization: Skip if HAS_MASK is False at compile time)
    neighbor_mask_val = tl.zeros([BLOCK_LEVELS], dtype=tl.int1)
    if HAS_MASK:
        val_int8 = tl.load(Mask_ptr + off_lookup, mask=mask_lvl, other=1).to(tl.int8)
        neighbor_mask_val = (val_int8 != 0)

    # -----------------------------------------------------------
    # 3. Base Pointers & Offset Pre-calculation
    # -----------------------------------------------------------
    # Optimization: Calculate the "Base Address" for every neighbor ONCE.
    # K Base for this batch
    k_batch_base = K_ptr + b_idx * sk_b
    v_batch_base = V_ptr + b_idx * sk_b
    
    # Compute the fixed offset for "Self" and "Neighbors" (Node-level)
    # Self Offset: [1]
    off_node_self = node_idx * sk_n
    
    # Neighbor Offsets: [BLOCK_LEVELS]
    # We pre-multiply the neighbor index by the stride. 
    # This vector stays in registers and is reused in the loop.
    off_node_cross = neighbor_indices * sk_n 

    # Prepare Q Pointer
    q_ptr = Q_ptr + (b_idx * sq_b) + (node_idx * sq_n) + \
            (h_idx[:, None] * sq_h) + (offs_d[None, :] * sq_d)

    # Accumulators
    acc_self = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_cross = tl.zeros([BLOCK_H, BLOCK_LEVELS], dtype=tl.float32)

    # -----------------------------------------------------------
    # 4. Score Loop
    # -----------------------------------------------------------
    for off_d_start in range(0, D, BLOCK_D):
        # Update D offsets
        # If D is Power of 2 (e.g. 64, 128), we can skip the mask check 
        # But we keep it generic here.
        cur_offs_d = off_d_start + offs_d
        d_mask = cur_offs_d < D
        
        # Load Q
        # Pointer arithmetic is minimal here (just adding off_d_start)
        mask_q = mask_h[:, None] & d_mask[None, :]
        q = tl.load(q_ptr, mask=mask_q, other=0.0)
        
        # --- K SELF ---
        # Reconstruct pointer using pre-calculated offsets
        # Base + Node_Offset + Head_Offset + Dim_Offset
        ptr_k_self = k_batch_base + off_node_self + \
                     (h_idx[:, None] * sk_h) + (cur_offs_d[None, :] * sk_d)
        
        k_self = tl.load(ptr_k_self, mask=mask_q, other=0.0)
        acc_self += tl.sum(q * k_self, axis=1)

        # --- K CROSS (Neighbors) ---
        # Broadcast magic:
        # off_node_cross is [BLOCK_LEVELS] -> broadcast to [1, LEVELS, 1]
        # h_idx is [H] -> broadcast to [H, 1, 1]
        # cur_offs_d is [BLOCK_D] -> broadcast to [1, 1, BLOCK_D]
        
        ptr_k_cross = k_batch_base + \
                      off_node_cross[None, :, None] + \
                      (h_idx[:, None, None] * sk_h) + \
                      (cur_offs_d[None, None, :] * sk_d)
        
        # Mask: H & Levels & Dim
        mask_k = mask_h[:, None, None] & mask_lvl[None, :, None] & d_mask[None, None, :]
        k_cross = tl.load(ptr_k_cross, mask=mask_k, other=0.0)
        
        # Math: [H, 1, D] * [H, LEVELS, D]
        acc_cross += tl.sum(q[:, None, :] * k_cross, axis=2)
        
        # Advance Q ptr for next block
        q_ptr += BLOCK_D * sq_d

    # -----------------------------------------------------------
    # 5. Softmax
    # -----------------------------------------------------------
    acc_self = acc_self * sm_scale
    acc_cross = acc_cross * sm_scale
    
    mask_broadcast = (offs_lvl >= LEVELS)
    if HAS_MASK:
        mask_broadcast = mask_broadcast | neighbor_mask_val
        
    acc_cross = tl.where(mask_broadcast[None, :], -float('inf'), acc_cross)
    
    max_cross = tl.max(acc_cross, axis=1)
    max_all = tl.maximum(acc_self, max_cross)
    
    # Fast math exp
    exp_self = tl.exp(acc_self - max_all)
    exp_cross = tl.exp(acc_cross - max_all[:, None])
    
    denom = exp_self + tl.sum(exp_cross, axis=1)
    w_self = exp_self / denom 
    w_cross = exp_cross / denom[:, None]

    # -----------------------------------------------------------
    # 6. Weighted Sum Loop
    # -----------------------------------------------------------
    # Output Pointer Setup
    out_base_ptr = Out_ptr + (b_idx * so_b) + (node_idx * so_n) + \
                   (h_idx[:, None] * so_h) + (offs_d[None, :] * so_d)

    for off_d_start in range(0, D, BLOCK_D):
        cur_offs_d = off_d_start + offs_d
        d_mask = cur_offs_d < D
        
        mask_op = mask_h[:, None] & d_mask[None, :]

        # --- V SELF ---
        ptr_v_self = v_batch_base + off_node_self + \
                     (h_idx[:, None] * sk_h) + (cur_offs_d[None, :] * sk_d)
                      
        v_self = tl.load(ptr_v_self, mask=mask_op, other=0.0)
        out_acc = w_self[:, None] * v_self
        
        # --- V CROSS ---
        ptr_v_cross = v_batch_base + \
                      off_node_cross[None, :, None] + \
                      (h_idx[:, None, None] * sk_h) + \
                      (cur_offs_d[None, None, :] * sk_d)
                      
        mask_v = mask_h[:, None, None] & mask_lvl[None, :, None] & d_mask[None, None, :]
        v_cross = tl.load(ptr_v_cross, mask=mask_v, other=0.0)
        
        # Accumulate
        out_acc += tl.sum(w_cross[:, :, None] * v_cross, axis=1)
        
        # Store
        # ptr matches out_base_ptr structure
        # FIXED: Mask store by H
        tl.store(out_base_ptr, out_acc.to(Out_ptr.dtype.element_ty), mask=mask_op)
        
        # Advance Output Pointer
        out_base_ptr += BLOCK_D * so_d

def hierarchical_fused_attention(Q, K, V, idx_table, mask_table):
    B, N, H, Dh = Q.shape
    LEVELS = idx_table.shape[1]
    Out = torch.empty_like(Q)

    HAS_MASK = (mask_table is not None)
    mask_ptr_safe = mask_table if HAS_MASK else Q 

    # ------------------------------------------------------------------
    # ALIGNMENT CHECKS (Safety for Vectorized Loads)
    # ------------------------------------------------------------------
    # 1. Check Base Pointers: Must be divisible by 16 bytes
    #    This allows tl.multiple_of(ptr, 16) in the kernel.
    assert Q.data_ptr() % 16 == 0, "Q ptr not 16-byte aligned"
    assert K.data_ptr() % 16 == 0, "K ptr not 16-byte aligned"
    assert V.data_ptr() % 16 == 0, "V ptr not 16-byte aligned"
    assert Out.data_ptr() % 16 == 0, "Out ptr not 16-byte aligned"

    # 2. Check Strides (Optional but recommended for max perf)
    #    If the last dim stride is 1 (contiguous), we want the row stride 
    #    (stride(2) for Head, or stride(3) for Dim) to be a multiple of 8 elements 
    #    (assuming fp16: 8 elements * 2 bytes = 16 bytes).
    #    This ensures every new row starts on an aligned address.
    #    Since D=Dh (Head Dim), usually 64 or 128, this is naturally true.
    
    grid = (N, B)

    BLOCK_LEVELS = triton.next_power_of_2(LEVELS)
    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_D = triton.next_power_of_2(Dh)

    hierarchical_fused_attention_kernel[grid](
        Q, K, V, 
        idx_table, mask_ptr_safe, 
        Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        idx_table.stride(0), idx_table.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        sm_scale=1.0 / math.sqrt(Dh),
        H=H,
        BLOCK_H=BLOCK_H, # Pass padded H
        D=Dh,
        LEVELS=LEVELS,
        BLOCK_D=BLOCK_D,
        BLOCK_LEVELS=BLOCK_LEVELS, 
        HAS_MASK=HAS_MASK,
        num_warps=4 # 4 is usually better than 8 for smaller workloads unless D > 128
    )
    return Out

def build_hierarchical_index_lookup_table(seq_len, device="cuda", dtype=torch.int32):
    """
    Vectorized version: Builds index table without Python loops over seq_len.
    Drastically faster for large sequences.

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

    # Initialize Tensors
    causal_mask = torch.zeros((seq_len, level_num), dtype=torch.bool, device=device)
    idx_map = torch.full((seq_len, level_num), -1, dtype=dtype, device=device)

    # Start with the leaf nodes [0, 1, 2, ..., seq_len-1]
    n_cur = torch.arange(seq_len, device=device, dtype=torch.int64)

    for lvl in range(level_num):
        # Vectorized Logic
        if lvl == 0:
            n_next = n_cur ^ 1
            pair = n_cur
        else:
            # Vectorized formula
            n_next = (n_cur // 2 + seq_len) ^ 1
            pair = (n_cur // 2 + seq_len)

        # Boundary Check (Vectorized)
        # We use a mask to prevent writing invalid indices, 
        # effectively mimicking the 'break' in the loop
        valid_mask = n_next <= max_valid
        
        # Causal Masking Logic (Vectorized)
        # pair < n_next means neighbor is in the "future"
        mask_step = (pair < n_next) & valid_mask

        # Update Tables (Batch assignment)
        # We only update where valid_mask is True
        idx_map[:, lvl] = torch.where(valid_mask, n_next.to(dtype), idx_map[:, lvl])
        causal_mask[:, lvl] = torch.where(valid_mask, mask_step, causal_mask[:, lvl])

        # Climb up via the neighbor for the next iteration
        n_cur = n_next

    return idx_map, causal_mask

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
        self.cached_causal_mask = None
        self.cached_seq_len = -1

        self.sizes = None
        self.offsets = None

    def _get_lookup_table(self, L, device):
        """
        Smart retrieval: Returns cached table if L matches, otherwise recomputes.
        """
        # Check if we can reuse the cache
        # 1. Cache exists
        # 2. Sequence length matches
        # 3. Device matches (crucial for moving model CPU <-> GPU)
        if (self.cached_idx_table is not None and 
            self.cached_seq_len == L and 
            self.cached_idx_table.device == device):
            return self.cached_idx_table, self.cached_causal_mask

        # If not, recompute and update cache
        idx_table, mask = build_hierarchical_index_lookup_table(L, device=device, dtype=torch.int64)
        
        self.cached_idx_table = idx_table
        self.cached_causal_mask = mask
        self.cached_seq_len = L
        
        return idx_table, mask

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

        # Get Topology
        idx_table, neighbor_mask = self._get_lookup_table(N, device=x.device)

        # DECISION LOGIC:
        # If the user passed a mask (indicating causality is on), use neighbor_mask.
        # If mask is None (indicating full/bidirectional), pass None.
        active_mask = neighbor_mask if mask is not None else None

        output_leaf_heads = hierarchical_fused_attention(
            Q, K_full, V_full, 
            idx_table, active_mask
        )
    
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