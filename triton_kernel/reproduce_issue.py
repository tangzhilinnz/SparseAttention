
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch
import triton
import triton.language as tl
import math
import torch.nn as nn
import torch.nn.functional as F

# Paste the content of Hierarchical_Attention_Triton.py here
# To avoid huge context i will import it if possible, but since i need to modify/run it standalone I will assume I can just import from the file.
# Actually I will just import the module since it exists on disk.

import sys
sys.path.append("d:\\SparseAttention\\triton_kernel")
from Hierarchical_Attention_Triton import HierarchicalSparseAttentionTriton, run_full_suite_update_X_from_Y

if __name__ == "__main__":
    try:
        run_full_suite_update_X_from_Y()
    except Exception as e:
        print(f"Error: {e}")
