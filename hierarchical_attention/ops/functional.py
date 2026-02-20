import torch
import triton
import math
# Import kernels from local package
from .triton import build_parents as tri_parents
from .triton import sparse_attn as tri_attn


# =================================================================
# 1. HELPER FUNCTIONS (Tuning & Heuristics)
# =================================================================

def get_cutoff_level(N: int) -> int:
    """
    Determines the cutoff level for switching between 'low-level' (leaf) 
    and 'high-level' (node) backward kernels based on sequence length.
    Optimized for NVIDIA A100 80G.
    """
    if N <= 1025: return 5
    elif N <= 2048: return 6
    elif N <= 4096: return 8
    return 10


# =================================================================
# 2. AUTOGRAD FUNCTIONS
# =================================================================
class BuildParentNodesFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q_p, K_p, V_p, K_c, V_c):
        # Enforce contiguity
        Q_p = Q_p.contiguous()
        K_p = K_p.contiguous()
        V_p = V_p.contiguous()
        K_c = K_c.contiguous()
        V_c = V_c.contiguous()
        
        B, P, H, D = Q_p.shape
        assert K_c.shape[1] == 2 * P, "Child count mismatch"
        
        Out = torch.empty_like(Q_p)
        Weights = torch.empty((B, P, H, 3), device=Q_p.device, dtype=Q_p.dtype)
        
        grid = (P, B)
        BLOCK_H = triton.next_power_of_2(H)
        
        # [OPTIMIZATION] Cap BLOCK_SIZE to avoid register spilling for large D
        # 128 is a safe sweet spot for shared memory usage.
        BLOCK_SIZE = min(128, triton.next_power_of_2(D))
        sm_scale = 1.0 / math.sqrt(D)

        # FIXED: Added 'tri_parents.' prefix
        tri_parents.build_parent_nodes_forward_kernel[grid](
            Q_p, K_p, V_p, K_c, V_c, Out, Weights,
            *Q_p.stride(), *K_p.stride(), *K_c.stride(),
            *Out.stride(), *Weights.stride(),
            sm_scale, H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2
        )
        
        ctx.save_for_backward(Q_p, K_p, V_p, K_c, V_c, Weights)
        ctx.constants = (sm_scale, H, BLOCK_H, D, BLOCK_SIZE)
        return Out

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Retrieve Tensors
        Q_p, K_p, V_p, K_c, V_c, Weights = ctx.saved_tensors
        sm_scale, H, BLOCK_H, D, BLOCK_SIZE = ctx.constants
        
        # Ensure gradient is contiguous
        grad_output = grad_output.contiguous()
        
        # 2. Allocate Gradients
        # [FIX #2] Use empty_like instead of zeros_like because the kernel 
        # fully overwrites these buffers unconditionally using tl.store.
        # This saves 5x memset operations.
        dQ = torch.empty_like(Q_p)
        dKp = torch.empty_like(K_p)
        dVp = torch.empty_like(V_p)
        dKc = torch.empty_like(K_c)
        dVc = torch.empty_like(V_c)
        
        # 3. Launch Backward
        B, P = Q_p.shape[0], Q_p.shape[1]
        grid = (P, B)
        
        # FIXED: Added 'tri_parents.' prefix
        tri_parents.build_parent_nodes_backward_kernel[grid](
            # Inputs
            grad_output,
            Weights, 
            Q_p, K_p, V_p, K_c, V_c,
            
            # Outputs
            dQ, dKp, dVp, dKc, dVc,
            
            # Strides (Inputs)
            *Q_p.stride(), *K_p.stride(), *K_c.stride(),
            *Weights.stride(),
            
            # Strides (Gradients)
            *grad_output.stride(), *dQ.stride(), *dKp.stride(), *dKc.stride(),

            # Constants
            sm_scale=sm_scale,
            H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2
        )
        
        return dQ, dKp, dVp, dKc, dVc

class HierarchicalAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, idx_table, gather_table, mask_table=None, window_size=16):
        Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous()
        idx_table = idx_table.contiguous()
        if mask_table is not None:
            mask_table = mask_table.contiguous()

        B, N, H, D = Q.shape
        LEVELS = idx_table.shape[1]

        RADIUS = window_size
        WINDOW_TOTAL_WIDTH = 2 * RADIUS + 1
        
        # [NEW] Calculate padded block window size for Triton allocations
        BLOCK_WINDOW = triton.next_power_of_2(WINDOW_TOTAL_WIDTH)
        
        Out = torch.empty_like(Q)
        Weights = torch.empty((B, N, H, WINDOW_TOTAL_WIDTH + LEVELS), device=Q.device, dtype=torch.float32)
        
        HAS_MASK = (mask_table is not None)
        mask_ptr_safe = mask_table if HAS_MASK else Q
        
        grid = (N, B)
        BLOCK_H = triton.next_power_of_2(H)
        BLOCK_LEVELS = triton.next_power_of_2(LEVELS)
        BLOCK_D = min(64, triton.next_power_of_2(D))
        sm_scale = 1.0 / math.sqrt(D)
        
        tri_attn.hierarchical_attention_forward_kernel[grid](
            Q, K, V,
            idx_table, mask_ptr_safe,
            Out, Weights,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *idx_table.stride(),
            *Out.stride(), *Weights.stride(),
            sm_scale=sm_scale,
            H=H, BLOCK_H=BLOCK_H,
            D=D, LEVELS=LEVELS,
            BLOCK_D=BLOCK_D, BLOCK_LEVELS=BLOCK_LEVELS,
            BLOCK_WINDOW=BLOCK_WINDOW, # [NEW] Injecting padded size
            HAS_MASK=HAS_MASK,
            RADIUS=RADIUS,
            N=N,
            num_warps=2
        )
        
        ctx.save_for_backward(Q, K, V, idx_table, gather_table, Weights, mask_table)
        # [NEW] Saving BLOCK_WINDOW to ctx
        ctx.constants = (sm_scale, H, BLOCK_H, D, BLOCK_D, LEVELS, BLOCK_LEVELS, RADIUS, WINDOW_TOTAL_WIDTH, BLOCK_WINDOW)
        return Out

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, idx_table, gather_table, Weights, mask_table = ctx.saved_tensors
        # [NEW] Unpack BLOCK_WINDOW
        sm_scale, H, BLOCK_H, D, BLOCK_D, LEVELS, BLOCK_LEVELS, RADIUS, WINDOW_SIZE, BLOCK_WINDOW = ctx.constants

        grad_output = grad_output.contiguous()
        B, N = Q.shape[0], Q.shape[1]
        grad_output_4d = grad_output.view(B, N, H, D)
        
        DS = torch.empty_like(Weights)
        grid_ds = (N, B)
        HAS_MASK = (mask_table is not None)
        mask_ptr_safe = mask_table if HAS_MASK else Weights
        
        tri_attn.hierarchical_attention_backward_dS_kernel[grid_ds](
            grad_output_4d, Weights, V, idx_table, DS, mask_ptr_safe,
            *grad_output_4d.stride(), *Weights.stride(), *V.stride(), 
            *idx_table.stride(), *DS.stride(),            
            sm_scale, H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_D=32, 
            LEVELS=LEVELS, BLOCK_LEVELS=BLOCK_LEVELS, 
            BLOCK_WINDOW=BLOCK_WINDOW, # [NEW] Injecting padded size
            HAS_MASK=HAS_MASK,
            RADIUS=RADIUS, WINDOW_SIZE=WINDOW_SIZE, N=N,
            num_warps=2
        )

        # --- SETUP PARALLELISM ---
        #dK = torch.zeros_like(K)
        #dV = torch.zeros_like(V)
        dK = torch.zeros_like(K, dtype=torch.float32)
        dV = torch.zeros_like(V, dtype=torch.float32)

        # --- BRANCH 2: dK/dV (Dependent on dS) ---
        
        # Step A: Leaf Kernel (Level 0)
        grid_window = (N, B)
        # FIXED: Added 'tri_attn.' prefix
        tri_attn.hierarchical_attention_backward_dK_dV_window_kernel[grid_window](
            DS, Q, Weights, grad_output_4d,
            dK, dV,
            *DS.stride(), *Q.stride(), *Weights.stride(), 
            *grad_output_4d.stride(), *dK.stride(),
            H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_D=BLOCK_D,
            RADIUS=RADIUS, WINDOW_SIZE=WINDOW_SIZE, N=N,
            num_warps=2
        )

        # --- Dynamic CUTOFF_LEVEL Logic (Heuristic from helper) ---
        CUTOFF_LEVEL = get_cutoff_level(N)
        
        # --- KERNEL A: Low Levels (Split=1) ---
        if LEVELS >= 1:
            limit = min(LEVELS, CUTOFF_LEVEL)
            # Total blocks = N - (N >> limit)
            total_blocks_low = N - (N >> limit)
            
            grid_low = (total_blocks_low, B)
            
            # FIXED: Added 'tri_attn.' prefix
            tri_attn.hierarchical_attention_backward_low_level_kernel[grid_low](
                DS, Q, Weights, grad_output_4d, gather_table,
                dK, dV,
                *DS.stride(), *Q.stride(), *Weights.stride(),
                *grad_output_4d.stride(), *dK.stride(), *gather_table.stride(),
                H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_D=BLOCK_D,
                N=N, 
                MAX_LEVEL=limit, 
                num_warps=4
            )
        
        # --- KERNEL B: High Levels (Split>1) ---
        if LEVELS > CUTOFF_LEVEL:
            num_high_levels = LEVELS - CUTOFF_LEVEL
            
            # Constant blocks per level = N >> (CUTOFF)
            # Actually, logic dictates: N >> (START_LEVEL - 1)
            # If START_LEVEL=9, we need N >> 8.
            blocks_per_lvl = N >> CUTOFF_LEVEL
            
            total_blocks_high = blocks_per_lvl * num_high_levels
            
            grid_high = (total_blocks_high, B)
            
            # FIXED: Added 'tri_attn.' prefix
            tri_attn.hierarchical_attention_backward_high_level_kernel[grid_high](
                DS, Q, Weights, grad_output_4d, gather_table,
                dK, dV,
                *DS.stride(), *Q.stride(), *Weights.stride(),
                *grad_output_4d.stride(), *dK.stride(), *gather_table.stride(),
                H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_D=BLOCK_D,
                N=N,
                START_LEVEL=CUTOFF_LEVEL + 1,
                num_warps=2
            )

        # --- BRANCH 1: dQ (Independent) ---
        dQ = torch.empty_like(Q)
        grid_dq = (N, B)

        tri_attn.hierarchical_attention_backward_dQ_kernel[grid_dq](
            DS, K, idx_table, dQ, mask_ptr_safe,
            *DS.stride(), *K.stride(), *idx_table.stride(), *dQ.stride(),
            H=H, BLOCK_H=BLOCK_H, D=D, BLOCK_D=32, LEVELS=LEVELS,
            HAS_MASK=HAS_MASK, RADIUS=RADIUS, WINDOW_SIZE=WINDOW_SIZE, N=N,
            num_warps=2
        )
            
        return dQ, dK.to(K.dtype), dV.to(V.dtype), None, None, None, None


# =================================================================
# 3. PUBLIC API WRAPPERS
# =================================================================
# Public functional API
def build_parent_nodes(Q_p, K_p, V_p, K_c, V_c):
    return BuildParentNodesFunc.apply(Q_p, K_p, V_p, K_c, V_c)

def hierarchical_attention(Q, K, V, idx_table, gather_table, mask_table=None, window_size=16):
    return HierarchicalAttentionFunc.apply(Q, K, V, idx_table, gather_table, mask_table, window_size)