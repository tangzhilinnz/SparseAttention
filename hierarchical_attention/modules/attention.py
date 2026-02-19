import torch
import torch.nn as nn
from typing import Optional, Tuple, List

# Import from the internal ops package
from ..ops.topology import build_tree_topology, generate_span_input_Y
from ..ops.functional import build_parent_nodes, hierarchical_attention


class HierarchicalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, window_size: int = 16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.window_size = window_size

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

        # --- Topology Cache ---
        # We cache the entire dictionary returned by build_tree_topology
        self.cached_tables = None       
        self.cached_seq_len = -1
        self.cached_is_causal = None  # To invalidate cache if switching causal/non-causal

        # Metadata for bottom-up levels
        self.sizes: Optional[List[int]] = None
        self.offsets: Optional[List[int]] = None

    def _get_lookup_table(self, L: int, is_causal: bool, device: torch.device):
        """
        Smart retrieval: Returns cached topology tables if (L, mode, device) match.
        """
        # 1. Check if cache is valid
        if (self.cached_tables is not None and 
            self.cached_seq_len == L and 
            self.cached_is_causal == is_causal and
            self.cached_tables['forward_idx'].device == device):
            return self.cached_tables

        # 2. Recompute using the unified builder
        # This builds Forward Indices, Forward Masks, and Backward Gather Ranges
        tables = build_tree_topology(L, is_causal=is_causal, device=device)
        
        # 3. Update Cache
        self.cached_tables = tables
        self.cached_seq_len = L
        self.cached_is_causal = is_causal
        
        return tables

    def _build_level_info(self, N: int):
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

    def cross_update_Y(self, x: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
        """
        Optimized bottom-up update using 'build_parent_nodes'.
        Merges children information into parent nodes.
        """

        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        
        assert y_in is not None, "y_in cannot be None"
        assert y_in.size(1) == N - 1, f"y_in size {y_in.size(1)} mismatch! Expected N-1 ({N-1}) for N={N} leaves."

        # Initialize level metadata if needed
        if self.sizes is None:
            self.sizes, self.offsets = self._build_level_info(N)

        # -------------------------------------------------------
        # 1. Global Parent Projection
        # -------------------------------------------------------
        # We project ALL parents at once for Q, K, and V.
        # This allows the parent to attend to itself (Parent-Self Attention).
        # Layout: [B, Total_Parents, H, Dh] (Native Linear output, no transpose)
        Q_p_all = self.Wq_y(y_in).view(B, -1, H, Dh)
        K_p_all = self.Wk_y(y_in).view(B, -1, H, Dh)
        V_p_all = self.Wv_y(y_in).view(B, -1, H, Dh)
        
        # [FIX #1] Pre-split tensors to avoid creating hundreds of slice views in the loop
        # This creates 1 SplitBackward node instead of N SliceBackward nodes.
        Q_p_levels = torch.split(Q_p_all, self.sizes, dim=1)
        K_p_levels = torch.split(K_p_all, self.sizes, dim=1)
        V_p_levels = torch.split(V_p_all, self.sizes, dim=1)
        
        new_Y_levels = []
        prev_sources = x # Starts as the leaves (first layer children)
        
        # [OPTIMIZATION] Project Level 0 Leaves ONCE here.
        # This removes the projection overhead for 50% of the total nodes.
        K_leaves = self.Wk_y(prev_sources).view(B, -1, H, Dh)
        V_leaves = self.dropout(self.Wv_y(prev_sources).view(B, -1, H, Dh))

        for level, parent_count in enumerate(self.sizes):
            # ---------------------------------------------------
            # 1. Prepare Parents (Using Pre-Split Tensors)
            # ---------------------------------------------------
            # [FIX #1] Access list instead of slicing tensor
            Q_p = Q_p_levels[level]
            K_p = K_p_levels[level]
            V_p = V_p_levels[level]
            
            # ---------------------------------------------------
            # 2. Prepare Children (Projection)
            # ---------------------------------------------------
            if level == 0:
                # FAST PATH: Use pre-projected leaves
                K_c = K_leaves
                V_c = V_leaves
                V_c = self.dropout(V_c)
            else:
                # STANDARD PATH: Project output of previous level
                # Must slice prev_sources because it is a changing reference
                children_in = prev_sources    
                # Project Children Keys/Values
                K_c = self.Wk_y(children_in).view(B, -1, H, Dh)
                V_c = self.dropout(self.Wv_y(children_in).view(B, -1, H, Dh))

            # ---------------------------------------------------
            # 3. Triton Kernel (3-Way Attention)
            # ---------------------------------------------------
            # Replaces: Reshape -> Cat -> Dot -> Softmax -> Weighted Sum
            updated_heads = build_parent_nodes(Q_p, K_p, V_p, K_c, V_c)

            # ---------------------------------------------------
            # 4. Merge Heads
            # ---------------------------------------------------
            updated_merged = updated_heads.reshape(B, parent_count, D)
            new_Y_levels.append(updated_merged)
            
            # Update pointer for next level
            prev_sources = updated_merged

        # Concatenate and Project
        Y_new = torch.cat(new_Y_levels, dim=1)
        # Concatenate and Project
        return self.out_proj_y(Y_new)


    def update_X_from_Y(self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Optimized top-down update using 'hierarchical_attention'.
        Distributes parent information back to leaves.
        """
    
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
        V_full = self.dropout(self.Wv_x(XY).view(B, -1, H, Dh))

        # 3. Ensure Contiguity
        # The Triton kernel does pointer arithmetic. While Linear() outputs are usually 
        # contiguous, operations like Dropout or View can sometimes create strides 
        # that break optimized kernels. We enforce it here to be safe.
        if not Q.is_contiguous(): Q = Q.contiguous()
        if not K_full.is_contiguous(): K_full = K_full.contiguous()
        if not V_full.is_contiguous(): V_full = V_full.contiguous()

        # 4. Get Topology Tables
        # Determine causality based on user input (mask presence implies causal)
        is_causal = (mask is not None)
        tables = self._get_lookup_table(N, is_causal, device=x.device)
        
        idx_table = tables["forward_idx"]
        neighbor_mask = tables["forward_mask"]
        gather_table = tables["backward_gather"]
        
        # If active_mask is None, kernel assumes full visibility.
        # If is_causal, we pass the pre-computed mask from the table.
        active_mask = neighbor_mask if is_causal else None

        output_leaf_heads = hierarchical_attention(
            Q, K_full, V_full, 
            idx_table,      # Forward Topology
            gather_table,
            active_mask,     # Mask Table
            self.window_size
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
        """
        Args:
            query, key, value: Input tensors [B, N, D]
            y: Pre-computed parent nodes [B, N-1, D] (Required for hierarchical mode)
            mask: Attention mask
        """

        x = query 
        B, L_Q, D = x.size()
        H, Dh = self.num_heads, self.head_dim
        L_K = key.size(1)
        L_V = value.size(1)

        # Check conditions for Hierarchical Mode
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