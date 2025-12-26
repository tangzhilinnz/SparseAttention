import torch
import triton
import torch.nn as nn
import torch.nn.functional as F
import triton.language as tl
from torch import einsum
import math

# ------------------------------------------------------------------
#                   Triton Kernel
# ------------------------------------------------------------------
@triton.jit
def build_parent_nodes_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Strides
    sq_b, sq_n, sq_h, sq_d,
    sk_b, sk_n, sk_h, sk_d,
    so_b, so_n, so_h, so_d,
    # Constants
    sm_scale,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,  # <--- NEW: Power of 2 Size
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    # -----------------------------------------------------------
    # 1. Grid & Indices (FIXED)
    # -----------------------------------------------------------
    # Create range of size 16 (if H=12)
    offs_h = tl.arange(0, BLOCK_H)
    # Create a mask to ignore indices 12, 13, 14, 15
    mask_h = offs_h < H

    # -----------------------------------------------------------
    # 2. Base Pointers
    # -----------------------------------------------------------
    q_base_ptr = Q_ptr + (b_idx * sq_b) + (node_idx * sq_n)
    child0_idx = 2 * node_idx
    child1_idx = 2 * node_idx + 1
    
    k0_base_ptr = K_ptr + (b_idx * sk_b) + (child0_idx * sk_n)
    k1_base_ptr = K_ptr + (b_idx * sk_b) + (child1_idx * sk_n)
    
    # -----------------------------------------------------------
    # 3. Compute Scores
    # -----------------------------------------------------------
    # Accumulators must be size [BLOCK_H]
    score0 = tl.zeros([BLOCK_H], dtype=tl.float32)
    score1 = tl.zeros([BLOCK_H], dtype=tl.float32)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        
        # Combined Mask: (Valid Head) AND (Valid Dim)
        mask_load = mask_h[:, None] & mask_d[None, :]
        
        ptr_q = q_base_ptr + (offs_h[:, None] * sq_h) + (offs_d[None, :] * sq_d)
        ptr_k0 = k0_base_ptr + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
        ptr_k1 = k1_base_ptr + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
        
        q = tl.load(ptr_q, mask=mask_load, other=0.0)
        k0 = tl.load(ptr_k0, mask=mask_load, other=0.0)
        k1 = tl.load(ptr_k1, mask=mask_load, other=0.0)
        
        score0 += tl.sum(q * k0, axis=1)
        score1 += tl.sum(q * k1, axis=1)

    # -----------------------------------------------------------
    # 4. Softmax
    # -----------------------------------------------------------
    score0 = score0 * sm_scale
    score1 = score1 * sm_scale
    
    # Note: We don't mask score calculations because masked values are 0.0,
    # which results in valid but irrelevant softmax outputs. 
    # The crucial part is masking the STORE later.
    
    max_score = tl.maximum(score0, score1)
    exp0 = tl.exp(score0 - max_score)
    exp1 = tl.exp(score1 - max_score)
    denom = exp0 + exp1 + 1e-9
    
    w0 = exp0 / denom 
    w1 = exp1 / denom 

    # -----------------------------------------------------------
    # 5. Weighted Sum & Store
    # -----------------------------------------------------------
    out_base_ptr = Out_ptr + (b_idx * so_b) + (node_idx * so_n)
    v0_base_ptr = V_ptr + (b_idx * sk_b) + (child0_idx * sk_n)
    v1_base_ptr = V_ptr + (b_idx * sk_b) + (child1_idx * sk_n)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        
        # Combined Mask
        mask_op = mask_h[:, None] & mask_d[None, :]
        
        ptr_v0 = v0_base_ptr + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
        ptr_v1 = v1_base_ptr + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
        ptr_out = out_base_ptr + (offs_h[:, None] * so_h) + (offs_d[None, :] * so_d)
        
        v0 = tl.load(ptr_v0, mask=mask_op, other=0.0)
        v1 = tl.load(ptr_v1, mask=mask_op, other=0.0)
        
        out_val = w0[:, None] * v0 + w1[:, None] * v1
        
        # CRITICAL: Mask the store so we don't write to invalid memory
        tl.store(ptr_out, out_val, mask=mask_op)

# ------------------------------------------------------------------
#                   Python Wrapper
# ------------------------------------------------------------------
def build_parent_nodes(Q, K, V):
    """
    Q: [B, Parent_Count, H, D]
    K: [B, Child_Count, H, D]
    V: [B, Child_Count, H, D]
    Returns: [B, Parent_Count, H, D]
    """
    # Check shapes
    B, P, H, D = Q.shape
    Out = torch.empty_like(Q)
    grid = (P, B)

    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_SIZE = triton.next_power_of_2(D)

    build_parent_nodes_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
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
        Optimized bottom-up update.
        1. Projects all Queries at once.
        2. Replaces O(N^2) masked attention with O(N) pair-wise reduction.
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        
        assert y_in is not None, "y_in cannot be None"
        assert y_in.size(1) > 0, "y_in must have valid tokens"

        if self.sizes is None: self.sizes, self.offsets = self.build_level_info(N)

        # OPTIMIZATION 1: Global Query Projection
        # Project all Y tokens at once to avoid small GEMMs inside the loop
        # y_in: [B, Total_Nodes, D] -> Q_all: [B, H, Total_Nodes, Dh]
        Q_all = self.Wq_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        
        new_Y_levels = []
        prev_sources = x # Starts as the leaves
        
        for level, parent_count in enumerate(self.sizes):
            offset = self.offsets[level]

            # 1. Get Queries for this level (Slicing is free)
            # [B, H, Parent_Count, Dh]
            Q = Q_all[:, :, offset : offset + parent_count, :]
            
            # 2. Prepare Children (Keys/Values)
            # The topology is strictly binary: Parent i connects to Child 2i and 2i+1.
            # If prev_sources has odd length, the last token is ignored (per your original mask logic).
            # We truncate to ensure strictly even pairs.
            useful_len = parent_count * 2
            children_in = prev_sources[:, :useful_len, :]
            
            # Project K and V
            # Note: We must project here because prev_sources changes every iteration
            # [B, Useful_Len, H, Dh] -> [B, H, Useful_Len, Dh]
            K = self.Wk_y(children_in).view(B, -1, H, Dh).transpose(1, 2)
            V = self.Wv_y(children_in).view(B, -1, H, Dh).transpose(1, 2)
            V = self.dropout(V)

            # OPTIMIZATION 2: Reshape & Reduce (The "Tree Attention")
            # Instead of Attn(Parent, Child) with a mask, we reshape Child to (Parent, 2)
            # Shape changes: [B, H, 2*P, D] -> [B, H, P, 2, D]
            K_pairs = K.view(B, H, parent_count, 2, Dh)
            V_pairs = V.view(B, H, parent_count, 2, Dh)
            
            # Q shape: [B, H, P, D] -> [B, H, P, 1, D] for broadcast
            Q_expanded = Q.unsqueeze(3)

            # Compute Scores: Dot product between Parent and its 2 Children
            # [B, H, P, 1, D] * [B, H, P, 2, D] -> sum(D) -> [B, H, P, 2]
            attn_logits = torch.sum(Q_expanded * K_pairs, dim=-1) / math.sqrt(Dh)
            
            # Softmax over the 2 children
            # attn_weights = F.softmax(attn_logits, dim=-1) # [B, H, P, 2]

            # ---- SAFE SOFTMAX (size=2) ----
            max_logits = attn_logits.max(dim=-1, keepdim=True).values
            exp_logits = torch.exp(attn_logits - max_logits)
            denom = exp_logits.sum(dim=-1, keepdim=True) + 1e-9
            attn_weights = exp_logits / denom  # [B, H, P, 2]
     
            # Weighted Sum
            # Weights: [B, H, P, 2, 1] (unsqueeze for broadcast)
            # V_pairs: [B, H, P, 2, D]
            # Result:  sum(dim 2) -> [B, H, P, D]
            updated = torch.sum(attn_weights.unsqueeze(-1) * V_pairs, dim=3)

            # Merge Heads
            updated_merged = updated.transpose(1, 2).reshape(B, parent_count, D)
            
            new_Y_levels.append(updated_merged)
            prev_sources = updated_merged

        # Concatenate and Project
        Y_new = torch.cat(new_Y_levels, dim=1)
        Y_new = self.out_proj_y(Y_new)
        
        return Y_new

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
        Optimized bottom-up update using Triton Kernel.
        
        Key Optimization:
        - Uses [Batch, Node, Head, Dim] layout to avoid transposes.
        - Replaces the O(N) loop of reshaping/masking with a single fused kernel.
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        
        assert y_in is not None, "y_in cannot be None"
        assert y_in.size(1) > 0, "y_in must have valid tokens"

        if self.sizes is None: self.sizes, self.offsets = self.build_level_info(N)

        # -------------------------------------------------------
        # OPTIMIZATION 1: Global Query Projection (Layout Adjusted)
        # -------------------------------------------------------
        # We project all Y tokens at once. 
        # CRITICAL: We DO NOT transpose to [B, H, N, D]. 
        # We keep it as [B, N, H, D] which is the native Linear output.
        # This matches the stride logic of our new Triton kernel perfectly.
        Q_all = self.Wq_y(y_in).view(B, -1, H, Dh) 
        
        new_Y_levels = []
        prev_sources = x # Starts as the leaves
        
        for level, parent_count in enumerate(self.sizes):
            offset = self.offsets[level]

            # ---------------------------------------------------
            # 1. Prepare Inputs (Zero-Copy Slicing)
            # ---------------------------------------------------
            # Slice Q for this level. Shape: [B, Parent_Count, H, Dh]
            Q = Q_all[:, offset : offset + parent_count, :, :]
            
            # Prepare Children (Keys/Values)
            useful_len = parent_count * 2
            # Slice strictly to useful length to ensure even pairs
            children_in = prev_sources[:, :useful_len, :]
            
            # Project K and V
            # Shape: [B, Child_Count, H, Dh]
            # Again, NO transpose. We keep the native [B, N, H, D] layout.
            K = self.Wk_y(children_in).view(B, -1, H, Dh)
            V = self.Wv_y(children_in).view(B, -1, H, Dh)
            V = self.dropout(V)

            # ---------------------------------------------------
            # 2. Triton Kernel (Fused Reduction)
            # ---------------------------------------------------
            # Replaces: Reshape -> Dot -> Softmax -> Weighted Sum -> Reshape
            # The kernel natively handles the tree structure (2 children -> 1 parent)
            # Input:  [B, 2*P, H, D] (Children)
            # Output: [B, P, H, D]   (Parents)
            updated_heads = build_parent_nodes(Q, K, V)

            # ---------------------------------------------------
            # 3. Merge Heads
            # ---------------------------------------------------
            # [B, P, H, Dh] -> [B, P, H*Dh] (i.e., [B, P, D])
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







class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
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
        ## Update Y (Hierarchy)
        ## Using the specific cross_update_Y method from your Attention class
        #y_next = self.self_attn.cross_update_Y(x, y_in=y)
        
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
                x, y, attention = layer(x, y, mask=trg_mask, return_attention=True)
                attentions.append(attention)
            else:
                x, y = layer(x, y, mask=trg_mask)
        
        output = self.fc_out(x)
        return output
    
    def generate(self, src, start_token=2, max_len=50, temperature=1.0):
        self.eval()
        device = next(self.parameters()).device
        
        # Src is treated as prompt in Decoder-Only
        current_tokens = src.to(device)
        if current_tokens.dim() == 1:
            current_tokens = current_tokens.unsqueeze(0)
            
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(current_tokens)
                last_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(last_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                
                if next_token.item() == 3: # EOS token ID
                    break
        
        return current_tokens


import time
from torch.profiler import profile, record_function, ProfilerActivity

def run_transformer_benchmark():
    # --------------------------------------------------------------------------
    # 0. CONFIGURATION
    # --------------------------------------------------------------------------
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Error: Triton requires a GPU. Exiting.")
        return

    # Hyperparameters from request
    VOCAB_SIZE = 50000
    D_MODEL = 768
    NUM_HEADS = 12
    D_FF = 3072
    NUM_LAYERS = 12
    DROPOUT = 0.0 # As requested
    
    # Batch config
    B = 2   # Reduced slightly from 16 to ensure safety on standard VRAM with 12 layers
    SEQ_LEN = 4096
    
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

    # CRITICAL: Load weights from Ref to Opt to ensure outputs are comparable
    print("Synchronizing weights (Ref -> Opt)...")
    model_opt.load_state_dict(model_ref.state_dict())

    model_ref.eval()
    model_opt.eval()

    # Generate Dummy Input
    # Integers for Embedding layer
    x_input = torch.randint(0, VOCAB_SIZE, (B, SEQ_LEN), device=device)

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

    # --- Benchmark PyTorch ---
    print(f"Benchmarking PyTorch (Ref)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_ref(x_input)
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_trials):
            _ = model_ref(x_input)
        end_event.record()
        torch.cuda.synchronize()
        
        ms_ref = start_event.elapsed_time(end_event) / num_trials

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

    print(f"\nRESULTS:")
    print(f"  PyTorch Avg Latency: {ms_ref:.2f} ms")
    print(f"  Triton  Avg Latency: {ms_opt:.2f} ms")
    print(f"  >>> Speedup: {ms_ref / ms_opt:.2f}x")

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

if __name__ == "__main__":
    run_transformer_benchmark()