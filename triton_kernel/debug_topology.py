
import torch
import math

def build_tree_topology_ref(seq_len):
    total_nodes = 2 * seq_len - 1
    level_num = int(math.log2(seq_len))
    
    # Re-implement the logic from build_tree_topology to verification
    # ...
    
    # Actually just import the function to see what it generates
    return None

import sys
import os
sys.path.append("d:\\SparseAttention\\triton_kernel")
from Hierarchical_Attention_Triton import build_tree_topology

def inspect_topology():
    N = 32
    print(f"Building Topology for N={N}")
    tables = build_tree_topology(N, is_causal=False, device='cpu')
    
    gather = tables['backward_gather']
    # gather shape: [Total_Nodes, 3] -> [Start, End, Level]
    
    print(f"Gather Table Shape: {gather.shape}")
    
    # Check Level 1 Parents
    # Nodes N .. N + N/2 - 1
    # For N=32, Nodes 32..47.
    # Level should be 1.
    # Width (End - Start) should be 2.
    
    print("\n--- Level 1 Parents (Expect Level=1, Width=2) ---")
    start_node = N
    end_node = N + N//2
    
    for i in range(start_node, min(start_node + 5, end_node)):
        g = gather[i]
        width = g[1] - g[0]
        print(f"Node {i}: Start={g[0]}, End={g[1]}, Level={g[2]}, Width={width}")
        
    # Check Level 2 Parents
    # Nodes N + N/2 .. N + N/2 + N/4 - 1
    # 32+16 = 48. Nodes 48..55.
    print("\n--- Level 2 Parents (Expect Level=2, Width=4) ---")
    start_node = N + N//2
    for i in range(start_node, min(start_node + 5, start_node + N//4)):
        g = gather[i]
        width = g[1] - g[0]
        print(f"Node {i}: Start={g[0]}, End={g[1]}, Level={g[2]}, Width={width}")

    # Check Root
    root = 2*N - 2
    g = gather[root]
    print(f"\n--- Root (Node {root}) ---")
    print(f"Node {root}: Start={g[0]}, End={g[1]}, Level={g[2]}, Width={g[1]-g[0]}")

if __name__ == "__main__":
    inspect_topology()
