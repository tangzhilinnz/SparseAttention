# -*- coding: utf-8 -*-

import os
# ==========================================
# CRITICAL FIX: GPU SELECTION MUST BE FIRST
# ==========================================
# Set this before importing torch or calling torch.cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import datasets
# Essential PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# Import AMP for Mixed Precision
from torch.cuda.amp import autocast, GradScaler
# Import Checkpointing (Critical for memory savings)
from torch.utils.checkpoint import checkpoint

# Dataset and text processing
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

# Data manipulation and utilities
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from datetime import timedelta
from torch.autograd import Variable

import math
import gc

# Set device (Now this will correctly see only GPU 2 as "cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Physical GPU Count Visible: {torch.cuda.device_count()}")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed()

# ==========================================
# 1. DATA PROCESSING (WikiText-2 Adapted for T4*2)
# ==========================================

class VocabBuilder:
    def __init__(self, texts, max_vocab_size=35000):
        # Slightly increased max_vocab_size to cover full WikiText-2 (~33k)
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.build_vocab(texts)
        
    def build_vocab(self, texts):
        print("Counting words...")
        word_counts = Counter()
        for text in texts:
            if len(text.strip()) > 0:
                words = text.lower().split()
                word_counts.update(words)
        
        # Take most common words (-4 for special tokens)
        most_common = word_counts.most_common(self.max_vocab_size - 4)
        for word, count in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        print(f"\n Vocabulary Statistics:")
        print(f"Total unique words found: {len(word_counts)}")
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Sample words: {list(self.word2idx.keys())[:10]}")

class WikiTextDataset(Dataset):
    def __init__(self, texts, vocab_builder, max_len=128):
        self.vocab = vocab_builder
        self.max_len = max_len
        
        # --- IMPROVEMENT START: Continuous Chunking for PPL ---
        # Flatten all text into one long stream instead of isolated lines
        token_list = []
        for text in texts:
            if len(text.strip()) > 0:
                words = text.lower().split()
                # Encode and add EOS
                ids = [self.vocab.word2idx.get(w, self.vocab.word2idx['<UNK>']) for w in words]
                ids.append(self.vocab.word2idx['<EOS>'])
                token_list.extend(ids)
        
        self.data = torch.tensor(token_list, dtype=torch.long)
        # Calculate number of full chunks
        self.num_samples = len(self.data) // (self.max_len + 1)
        # --- IMPROVEMENT END ---
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # --- IMPROVEMENT START: Slice from stream ---
        start_idx = idx * self.max_len
        end_idx = start_idx + self.max_len + 1
        chunk = self.data[start_idx:end_idx]
        
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        
        return {
            'input_ids': input_ids,
            'label': target_ids, 
            'text': "" # Text is not easily available in chunked mode, leaving empty to match structure
        }
        # --- IMPROVEMENT END ---

# Load data
print("\n<> Loading WikiText-2 Dataset...")
dataset = load_dataset("wikitext", "wikitext-2-v1")

train_texts = dataset['train']['text']
valid_texts = dataset['validation']['text']
test_texts = dataset['test']['text']

# Build vocabulary 
print("\n<> Building Vocabulary...")
vocab_builder = VocabBuilder(train_texts, max_vocab_size=35000)

# Create datasets
print("\n<> Creating Datasets...")
# HYPERPARAMETER: max_len
# INCREASED TO 512 (Point 7: Increase context length)
MAX_LEN = 512
train_dataset = WikiTextDataset(train_texts, vocab_builder, max_len=MAX_LEN)
valid_dataset = WikiTextDataset(valid_texts, vocab_builder, max_len=MAX_LEN)
test_dataset = WikiTextDataset(test_texts, vocab_builder, max_len=MAX_LEN)

# Create dataloaders
# ==========================================================
# FIX 1: OPTIMIZED BATCH SIZE
# Set to 16 for d_model=512. We compensate with accumulation_steps=8
# ==========================================================
batch_size = 16
# CHANGED: shuffle=False (Point 3: Maintain continuity)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, pin_memory=True)

# Print detailed information
print("\n<> Dataset Statistics:")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Get a batch and print its information
sample_batch = next(iter(train_loader))
print("\n<> Sample Batch Information:")
print(f"Input shape: {sample_batch['input_ids'].shape}")
print(f"Label shape: {sample_batch['label'].shape}")


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math


def build_hierarchical_index_lookup_table(seq_len, device="cuda", dtype=torch.int32):
    """
    Build a hierarchical index lookup table storing only neighbor nodes
    across different levels (excluding the leaf itself).
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
        if (self.cached_idx_table is not None and 
            self.cached_seq_len == L and 
            self.cached_idx_table.device == device):
            return self.cached_idx_table, self.cached_causal_mask

        idx_table, mask = build_hierarchical_index_lookup_table(L, device=device, dtype=torch.int64)
        
        self.cached_idx_table = idx_table
        self.cached_causal_mask = mask
        self.cached_seq_len = L
        
        return idx_table, mask

    def _compute_pair_weights(self, q, k):
        """
        Compute pair attention weights for a query node attending to two child nodes.
        """
        # Step 1: Unsqueeze q to shape [B,H,M,1,D] to match child dimension
        q_unsqueezed = q.unsqueeze(3)  # Adds singleton dim at child axis

        # Step 2: Compute dot product with keys using einsum
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
            w0, w1 = self._compute_pair_weights(Q_parent, K_pairs)

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
        # Optimization: Use Flash Attention for fallback
        output = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=mask, 
            dropout_p=0.1 if self.training else 0.0
        )
        return output, None 

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
            _, hierarchy_K, hierarchy_V = self._build_hierarchy(Q, K, V)

            # 2a. Flatten K and V for indexing
            flat_hierarchy_K = torch.cat(hierarchy_K, dim=2) # [B, H, TotalNodes, D]
            flat_hierarchy_V = torch.cat(hierarchy_V, dim=2) # [B, H, TotalNodes, D]

            # 3. Retrieve Lookup Table
            idx_table, neighbor_causal_mask = self._get_lookup_table(L, device=Q.device)
            
            # 4. Gather Neighbors (K and V)
            gather_indices = idx_table
            
            # neighbors_k: [B, H, L, Levels, D]
            neighbors_k = flat_hierarchy_K[:, :, gather_indices, :] 
            neighbors_v = flat_hierarchy_V[:, :, gather_indices, :]

            # 5. Compute Self Logits (Leaf Q * Leaf K)
            self_logits = torch.einsum('b h n d, b h n d -> b h n', Q, K) / math.sqrt(Dh)

            # 6. Compute Neighbor Logits (Leaf Q * Neighbor K)
            neighbors_logits = torch.einsum('b h l x d, b h l n d -> b h l n', Q.unsqueeze(3), neighbors_k)
            neighbors_logits = neighbors_logits / math.sqrt(Dh)

            # 7. Apply Causal Mask
            if mask is not None:
                neighbors_logits = neighbors_logits.masked_fill(neighbor_causal_mask, float('-inf'))

            # 8. Concatenate (Self + Neighbors)
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
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0. , max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0. , d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1), :], requires_grad=False)
        return self.dropout(x)

# --- IMPROVED DECODER LAYER WITH CHECKPOINTING ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = HierarchicalSparseAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, trg_mask=None, return_attention=False):
        norm_x = self.norm1(x)
        
        # ======================================================
        # FIX 2: GRADIENT CHECKPOINTING (Essential for Custom Attention)
        # ======================================================
        if self.training and x.requires_grad and not return_attention:
            # Checkpointing recomputes the forward pass during backward pass
            # This saves massive amounts of memory for tree structures
            attn_output = checkpoint(self._custom_attn_wrapper, norm_x, trg_mask, use_reentrant=False)
        else:
            attn_output = self.self_attn(norm_x, norm_x, norm_x, mask=trg_mask)
            if return_attention:
                if isinstance(attn_output, tuple):
                     attn_output, self_attn_weights = attn_output
            
        x = x + self.dropout(attn_output) # Residual
        
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output) # Residual
        
        if return_attention:
            return x, self_attn_weights
        return x

    def _custom_attn_wrapper(self, x, mask):
        # Helper for checkpoint to unpack arguments
        return self.self_attn(x, x, x, mask=mask)

# --- IMPROVED MODEL CLASS (Added Initialization) ---
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
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
        trg_mask = self.make_causal_mask(trg)
        
        x = self.embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        attentions = [] if return_attention else None
        
        for layer in self.layers:
            if return_attention:
                x, attention = layer(x, trg_mask=trg_mask, return_attention=True)
                attentions.append(attention)
            else:
                x = layer(x, trg_mask=trg_mask)
        
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

# ==========================================
# 3. TRAINING LOOP (FIXED OOM with Mixed Precision)
# ==========================================

def train_transformer_model(model, train_loader, valid_loader, criterion=None, num_epochs=100, learning_rate=5e-4, patience=6):
    if criterion is None:
        # NOTE: Using Label Smoothing for better generalization
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    device = next(model.parameters()).device
    
    print(f"\n{'='*50}")
    print(f"<> Training Transformer Model")
    print(f"{'='*50}")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # FIX 2: Initialize GradScaler for AMP
    scaler = GradScaler()

    # CHANGED: Use OneCycleLR for better convergence
    # Note: total_steps is usually len(train_loader) * num_epochs
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    
    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    epoch_times = []
    
    # --- EARLY STOPPING VARIABLES ---
    patience_counter = 0
    
    # Gradient Accumulation Steps (Simulate larger batch size)
    # FIX 3: Increased accumulation to compensate for smaller batch size
    # 16 batch_size * 8 accumulation = 128 effective batch size
    accumulation_steps = 8

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), 
                          total=len(train_loader),
                          desc=f'Epoch {epoch+1}/{num_epochs}',
                          leave=True)
        
        optimizer.zero_grad()
        
        for batch_idx, batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # --- FIX 4: Autocast context manager for AMP ---
            with autocast():
                # --- IMPROVEMENT: Only pass input_ids (Decoder Only) ---
                outputs = model(input_ids)
                
                output_dim = outputs.shape[-1]
                outputs = outputs.contiguous().view(-1, output_dim)
                labels = labels.contiguous().view(-1)
                
                loss = criterion(outputs, labels)
                
                # Divide loss by accumulation steps
                loss = loss / accumulation_steps
            
            # --- FIX 5: Scale loss and backward ---
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step with scaler
                scaler.step(optimizer)
                scaler.update()
                
                # OneCycleLR steps per batch
                scheduler.step()
                optimizer.zero_grad()
            
            # Multiply back for reporting
            total_train_loss += loss.item() * accumulation_steps
            
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'ppl': f'{math.exp(min(loss.item() * accumulation_steps, 10)):.2f}',
                'lr': f'{current_lr:.6f}'
            })
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_valid_loss = 0
        
        print("\n<> Validating...")
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                
                # Cast validation to fp16 as well to save memory
                with autocast():
                    outputs = model(input_ids)
                    output_dim = outputs.shape[-1]
                    outputs = outputs.contiguous().view(-1, output_dim)
                    labels = labels.contiguous().view(-1)
                    loss = criterion(outputs, labels)
                
                total_valid_loss += loss.item()
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f'\n<> Epoch {epoch+1} Results:')
        print(f'<>  Time: {timedelta(seconds=int(epoch_time))}')
        print(f'Train Loss: {avg_train_loss:.4f} | Train PPL: {math.exp(min(avg_train_loss, 20)):.2f}')
        print(f'Valid Loss: {avg_valid_loss:.4f} | Valid PPL: {math.exp(min(avg_valid_loss, 20)):.2f}')
        
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        
        # --- AUTO-EXIT (EARLY STOPPING) LOGIC ---
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0 # Reset counter
            
            # Save the underlying model
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_valid_loss,
            }, 'best_transformer_wikitext.pt')
            print(f'<> Saved new best model with validation loss: {best_valid_loss:.4f}')
        else:
            patience_counter += 1
            print(f"<> No improvement for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print(f"\n<> Auto-Exit Triggered: Validation loss has not improved for {patience} epochs.")
                break
        
        # Clean memory manually
        gc.collect()
        torch.cuda.empty_cache()
    
    total_time = sum(epoch_times)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epoch_times)
    plt.title('Epoch Times')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('transformer_wikitext_history.png')
    plt.close()
    
    print(f"\n{'='*50}")
    print("<> Training Complete!")
    print(f"{'='*50}")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    
    results = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'best_valid_loss': best_valid_loss,
        'epoch_times': epoch_times,
        'total_time': total_time
    }
    
    return results

def evaluate_test_set(model, test_loader):
    print(f"\n{'='*50}")
    print(f"<> FINAL TEST EVALUATION")
    print(f"{'='*50}")
    
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    
    # Use standard CrossEntropy (no smoothing) for fair score comparison
    eval_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Use autocast for eval too (faster)
            with autocast():
                # --- IMPROVEMENT: Only pass input_ids ---
                outputs = model(input_ids)
                
                output_dim = outputs.shape[-1]
                outputs = outputs.contiguous().view(-1, output_dim)
                labels = labels.contiguous().view(-1)
                
                loss = eval_criterion(outputs, labels)
            
            total_loss += loss.item()
            
    avg_loss = total_loss / len(test_loader)
    perplexity = math.exp(avg_loss)
    
    print(f"\n>> Final Test Loss: {avg_loss:.4f}")
    print(f">> Final Test Perplexity: {perplexity:.2f}")
    print(f"{'='*50}")

# ==========================================
# 4. INITIALIZATION AND RUN
# ==========================================

print("\n<> Initializing Transformer Model...")
# Ensure vocab size matches builder
vocab_size = len(vocab_builder.word2idx)
print(f"Actual Vocab Size: {vocab_size}")

# RESTORED HYPERPARAMETERS AS REQUESTED
model = TransformerLM(
    vocab_size=vocab_size,
    d_model=512,        # RESTORED: 512
    num_heads=8,        # RESTORED: 8
    d_ff=2048,          # RESTORED: 2048
    num_layers=12,      # RESTORED: 12
    dropout=0.1         
)

# ENABLE MULTI-GPU SUPPORT (DataParallel)
if torch.cuda.device_count() > 1:
    print(f"<> Utilizing {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

print(f"Using device: {device}")

# Training parameters
num_epochs = 100     # SET HIGH as requested (auto-exit will stop it)
learning_rate = 5e-4 # INCREASED slightly for OneCycleLR

print("\n<> Starting Training...")
results = train_transformer_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=None, 
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    patience=6 # Auto-exit if no improvement for 6 epochs
)

# Run Final Evaluation on Test Set
print("\n<> Loading Best Model for Evaluation...")
checkpoint = torch.load('best_transformer_wikitext.pt')
# We need to load state dict carefully depending on if DataParallel was used during save
# My training loop logic saves model.module if available, so we load into a fresh instance
best_model = TransformerLM(
    vocab_size=vocab_size,
    d_model=512,        # RESTORED: 512
    num_heads=8,        # RESTORED: 8
    d_ff=2048,          # RESTORED: 2048
    num_layers=12,      # RESTORED: 12
    dropout=0.1         
)
best_model.load_state_dict(checkpoint['model_state_dict'])
best_model = best_model.to(device)

if torch.cuda.device_count() > 1:
     best_model = nn.DataParallel(best_model)

evaluate_test_set(best_model, test_loader)