import torch
import triton
import triton.language as tl
import math

# ------------------------------------------------------------------
#                   Triton Kernel
# ------------------------------------------------------------------
@triton.jit
def build_parent_nodes_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Strides for Q: [Batch, Node, Head, Dim]
    sq_b, sq_n, sq_h, sq_d,
    # Strides for K/V: [Batch, Node, Head, Dim]
    sk_b, sk_n, sk_h, sk_d,
    # Strides for Out: [Batch, Node, Head, Dim]
    so_b, so_n, so_h, so_d,
    # Constants
    sm_scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # -----------------------------------------------------------
    # 1. Grid & Indices
    # Matches your reference: Grid is (Node, Batch)
    # -----------------------------------------------------------
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    
    # We process ALL heads [0..H] in this single thread block
    offs_h = tl.arange(0, H)

    # -----------------------------------------------------------
    # 2. Base Pointers (Batch + Node)
    # -----------------------------------------------------------
    # Q Base: [Batch, Node, :, :]
    q_base_ptr = Q_ptr + (b_idx * sq_b) + (node_idx * sq_n)
    
    # K/V Child Indices
    child0_idx = 2 * node_idx
    child1_idx = 2 * node_idx + 1
    
    # K Base: [Batch, Child, :, :]
    k0_base_ptr = K_ptr + (b_idx * sk_b) + (child0_idx * sk_n)
    k1_base_ptr = K_ptr + (b_idx * sk_b) + (child1_idx * sk_n)
    
    # -----------------------------------------------------------
    # 3. Compute Scores (Vectorized over H)
    # -----------------------------------------------------------
    # Accumulators shape: [H]
    score0 = tl.zeros([H], dtype=tl.float32)
    score1 = tl.zeros([H], dtype=tl.float32)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        
        # Pointer Arithmetic for [H, BLOCK_SIZE] block
        # ptr = base + (h * stride_h) + (d * stride_d)
        # Broadcasting: offs_h[:, None] ensures shape [H, 1]
        # Broadcasting: offs_d[None, :] ensures shape [1, BLOCK]
        
        ptr_q = q_base_ptr + (offs_h[:, None] * sq_h) + (offs_d[None, :] * sq_d)
        ptr_k0 = k0_base_ptr + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
        ptr_k1 = k1_base_ptr + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
        
        # Load Data [H, BLOCK_SIZE]
        # We mask strictly on the D dimension. H is assumed to fit (compile-time constant)
        q = tl.load(ptr_q, mask=mask_d[None, :], other=0.0)
        k0 = tl.load(ptr_k0, mask=mask_d[None, :], other=0.0)
        k1 = tl.load(ptr_k1, mask=mask_d[None, :], other=0.0)
        
        # Dot Product along D (axis=1) -> Result [H]
        score0 += tl.sum(q * k0, axis=1)
        score1 += tl.sum(q * k1, axis=1)

    # -----------------------------------------------------------
    # 4. Softmax (Vectorized over H)
    # -----------------------------------------------------------
    score0 = score0 * sm_scale
    score1 = score1 * sm_scale
    
    # Element-wise operations on [H] vector
    max_score = tl.maximum(score0, score1)
    exp0 = tl.exp(score0 - max_score)
    exp1 = tl.exp(score1 - max_score)
    denom = exp0 + exp1 + 1e-9
    
    w0 = exp0 / denom # [H]
    w1 = exp1 / denom # [H]

    # -----------------------------------------------------------
    # 5. Weighted Sum & Store
    # -----------------------------------------------------------
    # Output Base Pointer
    out_base_ptr = Out_ptr + (b_idx * so_b) + (node_idx * so_n)

    # Reuse V offsets (assuming V strides == K strides)
    # If V is separate, we'd need sv_b, sv_n, etc.
    v0_base_ptr = V_ptr + (b_idx * sk_b) + (child0_idx * sk_n)
    v1_base_ptr = V_ptr + (b_idx * sk_b) + (child1_idx * sk_n)

    for off_d in range(0, D, BLOCK_SIZE):
        offs_d = off_d + tl.arange(0, BLOCK_SIZE)
        mask_d = offs_d < D
        
        # Calculate Pointers [H, BLOCK]
        ptr_v0 = v0_base_ptr + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
        ptr_v1 = v1_base_ptr + (offs_h[:, None] * sk_h) + (offs_d[None, :] * sk_d)
        ptr_out = out_base_ptr + (offs_h[:, None] * so_h) + (offs_d[None, :] * so_d)
        
        # Load V
        v0 = tl.load(ptr_v0, mask=mask_d[None, :], other=0.0)
        v1 = tl.load(ptr_v1, mask=mask_d[None, :], other=0.0)
        
        # Weighted Sum
        # w0 is [H], v0 is [H, BLOCK]. 
        # w0[:, None] broadcasts to [H, 1] for multiplication
        out_val = w0[:, None] * v0 + w1[:, None] * v1
        
        # Store
        tl.store(ptr_out, out_val, mask=mask_d[None, :])


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
    
    # 2D Grid: (Parent_Nodes, Batch)
    # We let the internal loop handle H
    grid = (P, B)

    #print(f" BLOCK_SIZE (next_power_of_2)={triton.next_power_of_2(D)}")
    
    build_parent_nodes_kernel[grid](
        Q, K, V, Out,
        # Q Strides
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        # K Strides
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        # Out Strides
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        # Constants
        sm_scale=1.0 / math.sqrt(D),
        H=H,
        D=D,
        BLOCK_SIZE=triton.next_power_of_2(D)
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
    D: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_LEVELS: tl.constexpr, # <--- NEW CONSTANT (Power of 2)
    HAS_MASK: tl.constexpr 
):
    node_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    # 1. Setup
    h_idx = tl.arange(0, H)
    offs_d = tl.arange(0, BLOCK_D)
    
    # --- FIX: Use BLOCK_LEVELS for arange (Must be Power of 2) ---
    offs_lvl = tl.arange(0, BLOCK_LEVELS)
    mask_lvl = offs_lvl < LEVELS
    
    # Load Topology Indices
    # We use 'offs_lvl' for offsets, but apply 'mask_lvl' to ensure we don't read garbage
    off_lookup = node_idx * sl_n + offs_lvl * sl_lvl
    neighbor_indices = tl.load(Lookup_ptr + off_lookup, mask=mask_lvl, other=0)
    
    # Load Causal Mask (Conditional)
    neighbor_mask_val = tl.zeros([BLOCK_LEVELS], dtype=tl.int1)
    if HAS_MASK:
        neighbor_mask_val = tl.load(Mask_ptr + off_lookup, mask=mask_lvl, other=1).to(tl.int1)

    # Base Pointers
    q_base = Q_ptr + b_idx * sq_b + node_idx * sq_n
    k_base = K_ptr + b_idx * sk_b 
    
    acc_self = tl.zeros([H], dtype=tl.float32)
    
    # Accumulator must match the Block Size
    acc_cross = tl.zeros([H, BLOCK_LEVELS], dtype=tl.float32)

    # 2. Score Accumulation (Dot Product)
    for off_d_start in range(0, D, BLOCK_D):
        d_range = off_d_start + offs_d
        d_mask = d_range < D
        
        # Load Q
        ptr_q = q_base + (h_idx[:, None] * sq_h) + (d_range[None, :] * sq_d)
        q = tl.load(ptr_q, mask=d_mask[None, :], other=0.0)
        
        # Load K_Self
        ptr_k_self = k_base + (node_idx * sk_n) + (h_idx[:, None] * sk_h) + (d_range[None, :] * sk_d)
        k_self = tl.load(ptr_k_self, mask=d_mask[None, :], other=0.0)
        acc_self += tl.sum(q * k_self, axis=1)

        # Load K_Neighbors (Indirect)
        # neighbor_indices is [BLOCK_LEVELS]
        ptr_k_cross = k_base + (neighbor_indices[None, :, None] * sk_n) + \
                      (h_idx[:, None, None] * sk_h) + (d_range[None, None, :] * sk_d)
        
        # Mask: valid Levels AND valid Dim
        mask_k_cross = mask_lvl[None, :, None] & d_mask[None, None, :]
        k_cross = tl.load(ptr_k_cross, mask=mask_k_cross, other=0.0)
        
        acc_cross += tl.sum(q[:, None, :] * k_cross, axis=2)

    # 3. Softmax
    acc_self = acc_self * sm_scale
    acc_cross = acc_cross * sm_scale
    
    # Mask invalid levels (Padding or User Mask)
    # 1. Padding levels (offs_lvl >= LEVELS)
    # 2. User Mask (neighbor_mask_val == 1)
    mask_broadcast = (offs_lvl >= LEVELS)
    if HAS_MASK:
        mask_broadcast = mask_broadcast | (neighbor_mask_val == 1)
        
    acc_cross = tl.where(mask_broadcast[None, :], -1.0e9, acc_cross)
    
    max_cross = tl.max(acc_cross, axis=1)
    max_all = tl.maximum(acc_self, max_cross)
    
    exp_self = tl.exp(acc_self - max_all)
    exp_cross = tl.exp(acc_cross - max_all[:, None])
    
    denom = exp_self + tl.sum(exp_cross, axis=1)
    w_self = exp_self / denom 
    w_cross = exp_cross / denom[:, None]

    # 4. Weighted Sum
    v_base = V_ptr + b_idx * sk_b
    out_base = Out_ptr + b_idx * so_b + node_idx * so_n
    
    for off_d_start in range(0, D, BLOCK_D):
        d_range = off_d_start + offs_d
        d_mask = d_range < D
        
        # Load V_Self
        ptr_v_self = v_base + (node_idx * sk_n) + (h_idx[:, None] * sk_h) + (d_range[None, :] * sk_d)
        v_self = tl.load(ptr_v_self, mask=d_mask[None, :], other=0.0)
        out_acc = w_self[:, None] * v_self
        
        # Load V_Neighbors (Indirect)
        ptr_v_cross = v_base + (neighbor_indices[None, :, None] * sk_n) + \
                      (h_idx[:, None, None] * sk_h) + (d_range[None, None, :] * sk_d)
        
        mask_v_cross = mask_lvl[None, :, None] & d_mask[None, None, :]
        v_cross = tl.load(ptr_v_cross, mask=mask_v_cross, other=0.0)
        
        out_acc += tl.sum(w_cross[:, :, None] * v_cross, axis=1)
        
        # Store
        ptr_out = out_base + (h_idx[:, None] * so_h) + (d_range[None, :] * so_d)
        tl.store(ptr_out, out_acc.to(Out_ptr.dtype.element_ty), mask=d_mask[None, :])



def hierarchical_fused_attention(Q, K, V, idx_table, mask_table):
    B, N, H, Dh = Q.shape
    LEVELS = idx_table.shape[1]
    Out = torch.empty_like(Q)

    HAS_MASK = (mask_table is not None)
    mask_ptr_safe = mask_table if HAS_MASK else Q 

    grid = (N, B)

    # --- FIX: Calculate next power of 2 for LEVELS ---
    BLOCK_LEVELS = triton.next_power_of_2(LEVELS)

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
        D=Dh,
        LEVELS=LEVELS,
        BLOCK_D=triton.next_power_of_2(Dh),
        BLOCK_LEVELS=BLOCK_LEVELS, # <--- PASS THIS
        HAS_MASK=HAS_MASK,
        num_warps=4 
    )
    return Out



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math

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

    def cross_update_Y_Triton(self, x, y_in):
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


    def update_X_from_Y_Triton(self, x, y, mask=None):
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
    
        return self.out_proj_x(output_leaf_heads.view(B, N, D))

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


import time

#if __name__ == "__main__":
#    # ------------------------------------------------------------------
#    # Configuration & Setup
#    # ------------------------------------------------------------------
#    torch.manual_seed(42)
#
#    # Check availability of GPU 2
#    target_device_idx = 2
#    if torch.cuda.device_count() <= target_device_idx:
#        print(f"Error: GPU {target_device_idx} not found. Available GPUs: {torch.cuda.device_count()}")
#        print("Switching to 'cuda:0' by default.")
#        device = "cuda:0"
#    else:
#        device = f"cuda:{target_device_idx}"
#    
#    # Large enough size to make the kernel launch overhead negligible compared to math
#    B = 16          # Batch size
#    N = 2048        # Sequence length (Leaf nodes)
#    D = 2048        # Model dimension
#    H = 16          # Number of heads
#    dropout = 0.0   # Disable dropout for deterministic validation checking
#    
#    device = "cuda"
#    dtype = torch.float16 # Test in FP16 for realistic performance metrics
#
#    print(f"--- Benchmarking Configuration ---")
#    print(f"Batch Size (B): {B}")
#    print(f"Seq Len (N)   : {N}")
#    print(f"Dim (D)       : {D}")
#    print(f"Heads (H)     : {H}")
#    print(f"Dtype         : {dtype}")
#    print(f"----------------------------------")
#
#    # 1. Initialize Model
#    model = HierarchicalSparseAttention(dim=D, num_heads=H, dropout=dropout).to(device).to(dtype)
#    model.eval() # Set to eval to disable dropout non-determinism
#
#    # 2. Generate Random Inputs
#    # x: The leaf nodes [B, N, D]
#    x = torch.randn(B, N, D, device=device, dtype=dtype)
#    
#    # y_in: The input for the query projection [B, N, D] 
#    # (Size doesn't strictly matter for the kernel logic, but usually matches N)
#    y_in = torch.randn(B, N, D, device=device, dtype=dtype)
#
#
#    # ------------------------------------------------------------------
#    # Correctness Check (Sanity Check)
#    # ------------------------------------------------------------------
#    print("\n[1] Running Correctness Check...")
#    
#    # Run PyTorch Reference
#    with torch.no_grad():
#        out_ref = model.cross_update_Y(x, y_in)
#    
#    # Run Triton Optimized
#    with torch.no_grad():
#        out_triton = model.cross_update_Y_Triton(x, y_in)
#
#    # Calculate Error
#    # Note: FP16 accumulation can lead to slight deviations vs FP32, 
#    # but since both use similar logic, they should be very close.
#    max_diff = (out_ref - out_triton).abs().max().item()
#    mean_diff = (out_ref - out_triton).abs().mean().item()
#    
#    print(f"Max Absolute Error: {max_diff:.6e}")
#    print(f"Mean Absolute Error: {mean_diff:.6e}")
#    
#    if max_diff < 1e-2: # Relaxed tolerance for FP16
#        print(">>> SUCCESS: Outputs match.")
#    else:
#        print(">>> WARNING: Outputs differ significantly!")
#
#    # ------------------------------------------------------------------
#    # Performance Benchmarking
#    # ------------------------------------------------------------------
#    print("\n[2] Running Performance Benchmark...")
#    
#    num_warmup = 10
#    num_trials = 100
#
#    # A. Benchmark PyTorch Reference
#    # -----------------------------
#    # Warmup
#    for _ in range(num_warmup):
#        _ = model.cross_update_Y(x, y_in)
#    torch.cuda.synchronize()
#    
#    start_event = torch.cuda.Event(enable_timing=True)
#    end_event = torch.cuda.Event(enable_timing=True)
#    
#    start_event.record()
#    for _ in range(num_trials):
#        _ = model.cross_update_Y(x, y_in)
#    end_event.record()
#    torch.cuda.synchronize()
#    
#    ref_latency_ms = start_event.elapsed_time(end_event) / num_trials
#    print(f"PyTorch Reference Avg time: {ref_latency_ms:.4f} ms")
#
#    # --- Benchmark PyTorch Reference ---
#    # Warmup
#    for _ in range(num_warmup):
#        _ = model.cross_update_Y(x, y_in)
#    torch.cuda.synchronize(device=device)
#    
#    start_event = torch.cuda.Event(enable_timing=True)
#    end_event = torch.cuda.Event(enable_timing=True)
#    
#    start_event.record()
#    for _ in range(num_trials):
#        _ = model.cross_update_Y(x, y_in)
#    end_event.record()
#    torch.cuda.synchronize(device=device)
#    
#    ref_total_ms = start_event.elapsed_time(end_event)
#    ref_avg_ms = ref_total_ms / num_trials
#
#    # --- Benchmark Triton Optimized ---
#    # Warmup
#    for _ in range(num_warmup):
#        _ = model.cross_update_Y_Triton(x, y_in)
#    torch.cuda.synchronize(device=device)
#    
#    start_event.record()
#    for _ in range(num_trials):
#        _ = model.cross_update_Y_Triton(x, y_in)
#    end_event.record()
#    torch.cuda.synchronize(device=device)
#    
#    triton_total_ms = start_event.elapsed_time(end_event)
#    triton_avg_ms = triton_total_ms / num_trials
#
#    # ------------------------------------------------------------------
#    # Final Summary
#    # ------------------------------------------------------------------
#    print(f"\n{'='*40}")
#    print(f"Results ({num_trials} trials)")
#    print(f"{'='*40}")
#    
#    print(f"{'Metric':<20} | {'PyTorch (Ref)':<15} | {'Triton (Opt)':<15}")
#    print(f"{'-'*20}-+-{'-'*15}-+-{'-'*15}")
#    print(f"{'Avg Latency':<20} | {ref_avg_ms:.4f} ms      | {triton_avg_ms:.4f} ms")
#    print(f"{'Total Time':<20} | {ref_total_ms:.2f} ms      | {triton_total_ms:.2f} ms")
#    print(f"{'Throughput (est)':<20} | {1000/ref_avg_ms:.1f} iter/s   | {1000/triton_avg_ms:.1f} iter/s")
#    print(f"{'='*40}")
#    
#    print(f"\n>>> Speedup: {ref_avg_ms / triton_avg_ms:.2f}x faster")
#
#
#
#    from torch.profiler import profile, record_function, ProfilerActivity
#
#    # ... setup your model and inputs ...
#
#    print("Profiling...")
#    # Warmup to ensure we don't profile compilation time
#    for _ in range(5):
#        model.cross_update_Y_Triton(x, y_in)
#
#    with profile(
#        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/hierarchical_opt'),
#        record_shapes=True,
#        profile_memory=True,
#        with_stack=True
#    ) as prof:
#        for i in range(5): # run enough iterations for the schedule
#            # Optional: Add labels to specific parts of your code
#            with record_function("Top_Level_Step"):
#                model.cross_update_Y_Triton(x, y_in)
#            prof.step()
#
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))




# ==============================================================================
#                               BENCHMARK SUITE
# ==============================================================================

from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": print("No GPU found."); exit()

    B, N, D, H = 16, 2048, 1024, 16
    dtype = torch.float16
    
    print(f"\n--- CONFIG: B={B}, N={N}, D={D}, H={H} ---")
    model = HierarchicalSparseAttention(dim=D, num_heads=H).to(device).to(dtype)
    model.eval()

    x = torch.randn(B, N, D, device=device, dtype=dtype)
    y_in = torch.randn(B, N-1, D, device=device, dtype=dtype) # N-1 for Bottom-Up
    y_full = torch.randn(B, N, D, device=device, dtype=dtype)   # N for Top-Down (Simulated)

    # --------------------------------------------------------------------------
    # 1. BOTTOM-UP BENCHMARK
    # --------------------------------------------------------------------------
    print("\n[1] TEST: BOTTOM-UP (cross_update_Y)")
    print("-" * 50)
    
    # A. Correctness
    with torch.no_grad():
        out_ref = model.cross_update_Y(x, y_in)
        out_opt = model.cross_update_Y_Triton(x, y_in)
    err = (out_ref - out_opt).abs().max().item()
    print(f"    Correctness Max Diff: {err:.6e} -> {'OK' if err < 1e-2 else 'FAIL'}")

    # B. Speed
    num_warmup, num_trials = 10, 100
    for _ in range(num_warmup): model.cross_update_Y(x, y_in)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_trials): model.cross_update_Y(x, y_in)
    end.record()
    torch.cuda.synchronize()
    ms_ref = start.elapsed_time(end)

    for _ in range(num_warmup): model.cross_update_Y_Triton(x, y_in)
    torch.cuda.synchronize()
    start.record()
    for _ in range(num_trials): model.cross_update_Y_Triton(x, y_in)
    end.record()
    torch.cuda.synchronize()
    ms_opt = start.elapsed_time(end)

    print(f"    PyTorch (Total/Avg): {ms_ref:.1f}ms / {ms_ref/num_trials:.3f}ms")
    print(f"    Triton  (Total/Avg): {ms_opt:.1f}ms / {ms_opt/num_trials:.3f}ms")
    print(f"    >>> Speedup: {ms_ref/ms_opt:.2f}x")

    # C. Profiler (Triton)
    print("    Profiling Triton Kernel...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(5):
             with record_function("Bottom_Up_Triton"):
                model.cross_update_Y_Triton(x, y_in)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # --------------------------------------------------------------------------
    # 2. TOP-DOWN BENCHMARK
    # --------------------------------------------------------------------------
    print("\n[2] TEST: TOP-DOWN (update_X_from_Y)")
    print("-" * 50)

    # A. Correctness
    with torch.no_grad():
        out_ref = model.update_X_from_Y(x, y_full, mask=None)
        out_opt = model.update_X_from_Y_Triton(x, y_full, mask=None)
    err = (out_ref - out_opt).abs().max().item()
    print(f"    Correctness Max Diff: {err:.6e} -> {'OK' if err < 1e-1 else 'FAIL'}")

    # B. Speed
    for _ in range(num_warmup): model.update_X_from_Y(x, y_full, mask=None)
    torch.cuda.synchronize()
    start.record()
    for _ in range(num_trials): model.update_X_from_Y(x, y_full, mask=None)
    end.record()
    torch.cuda.synchronize()
    ms_ref = start.elapsed_time(end)

    for _ in range(num_warmup): model.update_X_from_Y_Triton(x, y_full, mask=None)
    torch.cuda.synchronize()
    start.record()
    for _ in range(num_trials): model.update_X_from_Y_Triton(x, y_full, mask=None)
    end.record()
    torch.cuda.synchronize()
    ms_opt = start.elapsed_time(end)

    print(f"    PyTorch (Total/Avg): {ms_ref:.1f}ms / {ms_ref/num_trials:.3f}ms")
    print(f"    Triton  (Total/Avg): {ms_opt:.1f}ms / {ms_opt/num_trials:.3f}ms")
    print(f"    >>> Speedup: {ms_ref/ms_opt:.2f}x")

    # C. Profiler (Triton)
    print("    Profiling Triton Kernel...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(5):
             with record_function("Top_Down_Triton"):
                model.update_X_from_Y_Triton(x, y_full, mask=None)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))