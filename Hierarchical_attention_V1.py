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
    def __init__(self, d_model, num_heads, dropout=0.1, max_levels=18):
        super().__init__()
        assert d_model % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_levels = max_levels

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.cached_idx_table = None
        self.cached_causal_mask = None
        self.cached_seq_len = -1

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

    def _compute_pair_weights(self, q, k):
        """
        Compute pair attention weights for a query node attending to two child nodes.

        Args:
            q: [B, H, M, D]  <- query for each parent node (M = num_parents)
            k: [B, H, M, 2, D] <- keys for each parent's 2 child nodes

        Returns:
            w0, w1: [B, H, M, 1] attention weights for child 0 and child 1
        """
        # Step 1: Unsqueeze q to shape [B,H,M,1,D] to match child dimension
        q_unsqueezed = q.unsqueeze(3)  # Adds singleton dim at child axis

        # Step 2: Compute dot product with keys using einsum
        # Use letters only; 'x' = singleton dim for q, 'c' = child dim, 'd' = head dim
        logits = torch.einsum('b h m x d, b h m c d -> b h m c', q_unsqueezed, k)  # [B,H,M,2]

        # Step 3: Scale by sqrt(head_dim) for stable softmax
        logits = logits / math.sqrt(self.head_dim)

        # Step 4: Numerical stability: subtract max over child dim
        logits = logits - logits.max(dim=-1, keepdim=True)[0]

        # Step 5: Softmax over child dimension to get attention weights
        weights = F.softmax(logits, dim=-1)  # [B,H,M,2]

        # Step 6: Split weights for each child and add singleton last dim
        w0 = weights[..., 0].unsqueeze(-1)  # [B,H,M,1]
        w1 = weights[..., 1].unsqueeze(-1)  # [B,H,M,1]

        return w0, w1

    def _build_hierarchy(self, Q, K, V):
        """
        Build hierarchical Q, K, V representations using true parent-driven design.

        Each parent node starts with its own (zero) query and computes attention
        weights over its two children to form its Q/K/V.

        Q, K, V: [B, H, L, D]
        Returns:
            hierarchy_Q, hierarchy_K, hierarchy_V: lists (from leaf to root)
        """
        B, H, L, D = Q.shape

        hierarchy_Q = [Q]
        hierarchy_K = [K]
        hierarchy_V = [V]

        curr_Q, curr_K, curr_V = Q, K, V
        level = 0

        while curr_Q.size(2) > 1 and level < self.max_levels:
            L_curr = curr_Q.size(2)
            even = L_curr - (L_curr % 2)

            # Reshape into (pairs of children)
            Q_pairs = curr_Q[:, :, :even].view(B, H, even // 2, 2, D)
            K_pairs = curr_K[:, :, :even].view(B, H, even // 2, 2, D)
            V_pairs = curr_V[:, :, :even].view(B, H, even // 2, 2, D)

            num_parents = Q_pairs.size(2)

            # -----------------------------
            # 1 Initialize parent queries to zero
            # -----------------------------
            Q_parent = torch.zeros(B, H, num_parents, D, device=Q.device, dtype=Q.dtype)

            # -----------------------------
            # 2 Parent queries its two children (compute attention weights)
            # -----------------------------
            # Compute pair attention weights per parent
            # Each parent attends to its 2 children’s keys
            w0, w1 = self._compute_pair_weights(Q_parent, K_pairs)
            # w0, w1: [B, H, num_parents, 1]

            # -----------------------------
            # 3 Weighted combination to form parent K/V (and optionally Q)
            # -----------------------------
            Q_parent = w0 * Q_pairs[:, :, :, 0, :] + w1 * Q_pairs[:, :, :, 1, :]
            K_parent = w0 * K_pairs[:, :, :, 0, :] + w1 * K_pairs[:, :, :, 1, :]
            V_parent = V_pairs[:, :, :, 0, :] + V_pairs[:, :, :, 1, :]

            # -----------------------------
            # 4 Handle odd leftover child (if sequence length is odd)
            # -----------------------------
            if L_curr % 2 == 1:
                Q_parent = torch.cat([Q_parent, curr_Q[:, :, -1:, :]], dim=2)
                K_parent = torch.cat([K_parent, curr_K[:, :, -1:, :]], dim=2)
                V_parent = torch.cat([V_parent, curr_V[:, :, -1:, :]], dim=2)

            # -----------------------------
            # 5 Store results and move up hierarchy
            # -----------------------------
            hierarchy_Q.append(Q_parent)
            hierarchy_K.append(K_parent)
            hierarchy_V.append(V_parent)

            curr_Q, curr_K, curr_V = Q_parent, K_parent, V_parent
            level += 1

        return hierarchy_Q, hierarchy_K, hierarchy_V
  
    def _standard_attention(self, Q, K, V, mask):
        D = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(D)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V), attn

    def forward(self, query, key, value, mask=None, return_attention=False):
        B, L_Q, D = query.size()
        Dh = self.head_dim
        L_K = key.size(1)
        L_V = value.size(1)

        # 1. Project
        Q = self.q_proj(query).view(B, L_Q, self.num_heads, Dh).transpose(1, 2) # [B,H,L,D]
        K = self.k_proj(key).view(B, L_K, self.num_heads, Dh).transpose(1, 2)
        V = self.v_proj(value).view(B, L_V, self.num_heads, Dh).transpose(1, 2)

        if L_Q == L_K == L_V:
            L = L_Q
            V = self.dropout(V)

            # 2. Build Hierarchy
            # We still need hierarchy_Q to build the tree, but we won't use it for the final attention.
            _, hierarchy_K, hierarchy_V = self._build_hierarchy(Q, K, V)

            # 2a. Flatten K and V for indexing
            # Note: We exclude the last element of the hierarchy lists if they represent the 
            # single root node that might not be addressable in the table logic, 
            # but usually we just cat everything. 
            # Your original code did `[..., :-1, :]` likely to align with 2*seq_len-1 structure?
            # We will concat normally.
            flat_hierarchy_K = torch.cat(hierarchy_K, dim=2) # [B, H, TotalNodes, D]
            flat_hierarchy_V = torch.cat(hierarchy_V, dim=2) # [B, H, TotalNodes, D]

            # 3. Retrieve Lookup Table
            idx_table, neighbor_causal_mask = self._get_lookup_table(L, device=Q.device)
            
            # Assert table validity
            # assert (idx_table != -1).all(), "Index table contains invalid entries."

            # 4. Gather Neighbors (K and V)
            # idx_table is [L, Levels]. 
            # We gather from dimension 2 (node index).
            gather_indices = idx_table
            
            # neighbors_k: [B, H, L, Levels, D]
            neighbors_k = flat_hierarchy_K[:, :, gather_indices, :] 
            neighbors_v = flat_hierarchy_V[:, :, gather_indices, :]

            # 5. Compute Self Logits (Leaf Q * Leaf K)
            # [B, H, L, D] * [B, H, L, D] -> [B, H, L]
            self_logits = torch.einsum('b h n d, b h n d -> b h n', Q, K) / math.sqrt(Dh)

            # 6. Compute Neighbor Logits (Leaf Q * Neighbor K)
            # Q: [B, H, L, D] -> unsqueeze -> [B, H, L, 1, D]
            # neighbors_k: [B, H, L, Levels, D]
            # Result: [B, H, L, Levels]
            neighbors_logits = torch.einsum('b h l x d, b h l n d -> b h l n', Q.unsqueeze(3), neighbors_k)
            neighbors_logits = neighbors_logits / math.sqrt(Dh)

            # 7. Apply Causal Mask
            if mask is not None:
                # neighbor_causal_mask is [L, Levels]. 
                # True means "future" (masked).
                neighbors_logits = neighbors_logits.masked_fill(neighbor_causal_mask, float('-inf'))

            # 8. Concatenate (Self + Neighbors)
            # V: [B, H, L, D] -> [B, H, L, 1, D]
            # self_logits: [B, H, L] -> [B, H, L, 1]
            
            all_v = torch.cat([V.unsqueeze(3), neighbors_v], dim=3)         # [B, H, L, Levels+1, D]
            all_logits = torch.cat([self_logits.unsqueeze(3), neighbors_logits], dim=3) # [B, H, L, Levels+1]

            # 9. Attention Softmax & Weighted Sum
            max_logits = all_logits.max(dim=-1, keepdim=True)[0]
            weights = F.softmax(all_logits - max_logits, dim=-1)            # [B, H, L, Levels+1]
            
            output_leaf = torch.einsum('b h l n, b h l n d -> b h l d', weights, all_v)
            
            # 10. Output Projection
            output = output_leaf.transpose(1, 2).contiguous().view(B, L, D)
            
            if return_attention:
                return self.out_proj(output), [weights]
            return self.out_proj(output)

        else:
            # Fallback for inference/cross-attention if lengths differ
            output_leaf, attn_weights = self._standard_attention(Q, K, V, mask)
            output = output_leaf.transpose(1, 2).contiguous().view(B, L_Q, D)
            return (self.out_proj(output), attn_weights) if return_attention else self.out_proj(output)