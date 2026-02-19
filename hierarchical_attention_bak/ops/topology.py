import math
import torch
from typing import Dict, Tuple


def build_tree_topology(seq_len: int, is_causal: bool = True, device: str = "cuda", dtype=torch.int32) -> Dict[str, torch.Tensor]:
    """
    Builds ALL topology tables needed for Hierarchical Attention (Forward & Backward).
    
    Args:
        seq_len (int): Number of leaf tokens (must be power of 2).
        is_causal (bool): Forward/Backward causal masking mode.
        
    Returns:
        dict: {
            "forward_idx": [seq_len, levels] int32 (Who leaves attend to),
            "forward_mask": [seq_len, levels] bool (Causal mask),
            "backward_gather": [total_nodes, 3] int32 (Start, End, Level for Gather)
        }
    """
    assert (seq_len & (seq_len - 1)) == 0, "seq_len must be power of 2"
    
    total_nodes = 2 * seq_len - 1
    level_num = int(math.log2(seq_len))
    max_valid = total_nodes - 2
    
    # ===============================================================
    # 1. Forward Table Construction (Leaf -> Neighbors)
    # ===============================================================
    forward_idx = torch.full((seq_len, level_num), -1, dtype=dtype, device=device)
    forward_mask = torch.zeros((seq_len, level_num), dtype=torch.bool, device=device)
    
    # Vectorized "Climb Up" Logic
    #n_cur = torch.arange(seq_len, device=device, dtype=torch.int64)
    n_cur = torch.arange(seq_len, device=device, dtype=dtype)
    
    for lvl in range(level_num):
        if lvl == 0:
            n_next = n_cur ^ 1
            pair = n_cur
        else:
            # Parent of n is (n // 2 + seq_len)
            parent = (n_cur // 2 + seq_len)
            n_next = parent ^ 1 # Sibling of parent
            pair = parent

        valid_mask = n_next <= max_valid
        
        # Causal Logic: pair < n_next means neighbor is in the Future
        is_future = pair < n_next
        
        # Store Index
        forward_idx[:, lvl] = torch.where(valid_mask, n_next.to(dtype), forward_idx[:, lvl])
        
        # Store Mask (If causal=True, block future connections)
        if is_causal:
            forward_mask[:, lvl] = torch.where(valid_mask, is_future, forward_mask[:, lvl])
        
        n_cur = n_next # Climb

    # ===============================================================
    # 2. Backward Table Construction (Node -> Leaf Range)
    # ===============================================================
    #subtree_ranges = torch.zeros((total_nodes, 2), dtype=torch.int64, device=device)
    subtree_ranges = torch.zeros((total_nodes, 2), dtype=dtype, device=device)
    node_levels = torch.zeros(total_nodes, dtype=dtype, device=device)
    
    # Initialize Leaves (Level 0)
    leaves = torch.arange(seq_len, device=device)
    subtree_ranges[:seq_len, 0] = leaves
    subtree_ranges[:seq_len, 1] = leaves + 1
    # Leaves are effectively Level 0 (Self/Sibling interaction)
    node_levels[:seq_len] = 0 
    
    curr_start = 0
    curr_count = seq_len
    
    for lvl in range(level_num):
        next_start = curr_start + curr_count
        next_count = curr_count // 2
        
        parents = torch.arange(next_start, next_start + next_count, device=device)
        indices = torch.arange(next_count, device=device)
        left_children = curr_start + 2 * indices
        right_children = curr_start + 2 * indices + 1
        
        subtree_ranges[parents, 0] = subtree_ranges[left_children, 0]
        subtree_ranges[parents, 1] = subtree_ranges[right_children, 1]
        
        # --- CRITICAL FIX ---
        # Old: node_levels[parents] = lvl 
        # New: node_levels[parents] = lvl + 1
        # Reason: 
        # Leaves (lvl 0 in loop) handle forward_idx[:, 0] -> stored as Level 0
        # Parents created in lvl=0 loop handle forward_idx[:, 1] -> stored as Level 1
        node_levels[parents] = lvl + 1
        # --------------------
        
        curr_start = next_start
        curr_count = next_count

    # Gather Logic
    all_nodes = torch.arange(total_nodes, device=device)
    siblings = all_nodes ^ 1
    siblings = torch.clamp(siblings, max=total_nodes-1)

    gather_info = torch.full((total_nodes, 3), -1, dtype=dtype, device=device)

    if is_causal:
        should_gather = siblings > all_nodes
    else:
        should_gather = siblings != all_nodes

    # Mask out Root (it has no valid sibling to gather from)
    should_gather = should_gather & (siblings != all_nodes)

    valid_nodes = all_nodes[should_gather]
    valid_siblings = siblings[should_gather]

    gather_info[valid_nodes, 0] = subtree_ranges[valid_siblings, 0].to(dtype)
    gather_info[valid_nodes, 1] = subtree_ranges[valid_siblings, 1].to(dtype)
    gather_info[valid_nodes, 2] = node_levels[valid_nodes] # Now uses the corrected levels


    ## ===============================================================
    ## 3. DEBUG: Print Line by Line
    ## ===============================================================
    #print("\n=== Gather Info Table ===")
    #print(f"{'Node':<6} | {'Start':<6} | {'End':<6} | {'Level':<6}")
    #print("-" * 30)
    #
    ## Convert to CPU list for clean iteration
    #gi_cpu = gather_info.detach().cpu().numpy()
    #
    #for i in range(total_nodes):
    #    s, e, l = gi_cpu[i]
    #    # Only print nodes that actually gather something (optional, remove 'if' to see all)
    #    if s != -1: 
    #        print(f"{i:<6} | {s:<6} | {e:<6} | {l:<6}")
    #    else:
    #        print(f"{i:<6} | {'-1':<6} | {'-1':<6} | {'-1':<6}")
    #        
    #print("=========================\n")


    return {
        "forward_idx": forward_idx,
        "forward_mask": forward_mask,
        "backward_gather": gather_info
    }


def generate_span_input_Y(x: torch.Tensor) -> torch.Tensor:
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