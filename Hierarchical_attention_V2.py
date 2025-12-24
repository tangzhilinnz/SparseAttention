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