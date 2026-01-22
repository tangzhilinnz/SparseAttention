
import torch
import torch.nn as nn
import os

# CRITICAL: SET GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Hierarchical_Attention_Triton import HierarchicalSparseAttentionTriton

def run_reproduction():
    # Config from User Report
    # Input Shapes -> X: torch.Size([2, 65536, 1024]), Y: torch.Size([2, 65535, 1024]), Dtype: torch.float16
    B = 2
    N = 65536
    # D is 1024 based on the shape [2, 65536, 1024]
    D = 1024
    H = 8 # Assuming 8 heads for D=1024 (HeadDim=128)
    
    # Check if HeadDim * H == D
    HeadDim = D // H
    
    print(f"Reproduction Config: B={B}, N={N}, D={D}, H={H}, HeadDim={HeadDim}")
    
    dtype = torch.float16
    device = "cuda"

    model = HierarchicalSparseAttentionTriton(dim=D, num_heads=H, dropout=0.0).to(device).to(dtype)
    
    # Create Inputs
    x = torch.randn(B, N, D, device=device, dtype=dtype).clamp(-1, 1).requires_grad_(True)
    y = torch.randn(B, N - 1, D, device=device, dtype=dtype).clamp(-1, 1).requires_grad_(True)
    mask = torch.ones((B, N), dtype=torch.bool, device=device)

    # 1. Reference Run (PyTorch)
    # We need to access the reference implementation. 
    # Based on the file analysis, Hierarchical_Attention_Triton.py contains HierarchicalSparseAttentionTriton
    # which inherits from nn.Module.
    # The reference `update_X_from_Y_Ref` needs to be checked if it exists on the object.
    # If not, we might need to rely on the `cross_update_Y_Ref` if that's what `update_X_from_Y` calls or similar logic.
    # However, `update_X_from_Y` in the user's `Hierarchical_Attention_Triton_Test.py` explicitly calls `update_X_from_Y_Ref`.
    # Let's inspect `Hierarchical_Attention_Triton.py` again to see if `update_X_from_Y_Ref` is defined there.
    # If not, I will implement a quick reference here matching the `update_X_from_Y` logic but with standard attention.
    
    # Re-implementing simplified Ref for reproduction to be self-contained if needed, 
    # but let's try calling it if it exists.
    
    print("Running Triton Kernel...")
    model.sizes = None
    out_tri = model.update_X_from_Y(x, y, mask=mask)
    
    # We need to clear grads before backward to start fresh
    x.grad = None
    y.grad = None
    
    loss_tri = out_tri.sum()
    loss_tri.backward()
    
    grad_x_tri = x.grad.clone()
    grad_y_tri = y.grad.clone()
    
    print(f"Triton Grad Y Mean: {grad_y_tri.float().abs().mean().item()}")
    print(f"Triton Grad Y Max:  {grad_y_tri.float().abs().max().item()}")

    # 2. Reference Run
    # Clearing grads
    x.grad = None
    y.grad = None
    
    # If update_X_from_Y_Ref is not on the instance, we define it roughly:
    if not hasattr(model, 'update_X_from_Y_Ref'):
        print("Note: update_X_from_Y_Ref not found on model. Using local implementation.")
        def ref_impl(x, y, mask):
             # Simplified Reference
             B, N, D = x.shape
             # 1. Concat
             XY = torch.cat([x, y], dim=1)
             # 2. Project
             Q = model.Wq_x(x).view(B, N, H, HeadDim).transpose(1, 2)
             K = model.Wk_x(XY).view(B, -1, H, HeadDim).transpose(1, 2)
             V = model.Wv_x(XY).view(B, -1, H, HeadDim).transpose(1, 2)
             
             # 3. Attn
             scores = torch.matmul(Q, K.transpose(-1, -2)) / (HeadDim ** 0.5)
             # Masking (Causal only if we had strict causal mask, but here use provided mask logic if complex)
             # For update_X_from_Y, typically it's full attention or specific topology. 
             # Let's assume standard attention for the "Reference" check on magnitude.
             probs = torch.softmax(scores, dim=-1)
             out = torch.matmul(probs, V)
             out = out.transpose(1, 2).reshape(B, N, D)
             return model.out_proj_x(out)
        out_ref = ref_impl(x, y, mask)
    else:
        out_ref = model.update_X_from_Y_Ref(x, y, mask=mask)
        
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    grad_y_ref = y.grad.clone()
    
    print(f"Ref Grad Y Mean:    {grad_y_ref.float().abs().mean().item()}")
    print(f"Ref Grad Y Max:     {grad_y_ref.float().abs().max().item()}")
    
    diff = (grad_y_ref - grad_y_tri).abs().max()
    print(f"Max Diff Grad Y:    {diff.item()}")
    
    if diff > 1.0:
        print("FAILURE: Mismatch Reproduced")
    else:
        print("SUCCESS: No Mismatch")

if __name__ == "__main__":
    run_reproduction()
