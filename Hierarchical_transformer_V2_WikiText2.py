# -*- coding: utf-8 -*-

import os
# ==========================================
# CRITICAL FIX: GPU SELECTION MUST BE FIRST
# ==========================================
# Set this before importing torch or calling torch.cuda to avoid OOM
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import datasets
# Essential PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
MAX_LEN = 4096
train_dataset = WikiTextDataset(train_texts, vocab_builder, max_len=MAX_LEN)
valid_dataset = WikiTextDataset(valid_texts, vocab_builder, max_len=MAX_LEN)
test_dataset = WikiTextDataset(test_texts, vocab_builder, max_len=MAX_LEN)

# Create dataloaders
# HYPERPARAMETER: batch_size
batch_size = 4
# CHANGED: shuffle=False (Point 3: Maintain continuity)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

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

    @staticmethod
    def build_parent_child_mask(parent_count, child_count, device):
        mask = torch.full((parent_count, child_count), float('-inf'), device=device)
        for i in range(parent_count):
            if 2*i < child_count: mask[i, 2*i] = 0.0
            if 2*i+1 < child_count: mask[i, 2*i+1] = 0.0
        return mask

    def cross_update_Y(self, x, y_in):
        """
        Updates y using x bottom-up.
        Strictly enforces that y_in exists.
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        
        assert y_in is not None, "y_in cannot be None in cross_update_Y"
        assert y_in.size(1) > 0, "y_in must have valid tokens"
        
        y_source = y_in

        if self.sizes is None: self.sizes, self.offsets = self.build_level_info(N)
        
        new_Y_levels = []
        prev_sources = x
        
        for level, parent_count in enumerate(self.sizes):
            offset = self.offsets[level]
            
            # Read from provided y_source
            Y_slice = y_source[:, offset:offset + parent_count, :]

            # --- Inline Split Heads ---
            # Projects (B, N, D) -> (B, N, H, Dh) -> (B, H, N, Dh)
            Q = self.Wq_y(Y_slice).view(B, -1, H, Dh).transpose(1, 2)
            K = self.Wk_y(prev_sources).view(B, -1, H, Dh).transpose(1, 2)
            V = self.Wv_y(prev_sources).view(B, -1, H, Dh).transpose(1, 2)

            V = self.dropout(V) # Apply dropout to the projected values

            mask = self.build_parent_child_mask(parent_count, prev_sources.size(1), x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)

            # Optimization: Removed explicit .contiguous() calls here.
            # matmul handles strided tensors fine.
            attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Dh)
            attn_logits += mask
  
            attn_weights = F.softmax(attn_logits, dim=-1)
            #attn_weights = self.dropout(attn_weights)
            updated = torch.matmul(attn_weights, V)
            
            # --- Inline Merge Heads ---
            # (B, H, N, Dh) -> (B, N, H, Dh) -> (B, N, D)
            updated_merged = updated.transpose(1, 2).reshape(B, -1, D)
            
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

# --- IMPROVED DECODER LAYER (Pre-LN Architecture) ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Note: dim=d_model to match your previous code hyperparameters
        self.self_attn = HierarchicalSparseAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # --- NEW: Norm for Y ---
        self.norm_y = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y, mask=None, return_attention=False):
        ## Update Y (Hierarchy)
        ## Using the specific cross_update_Y method from your Attention class
        #y_next = self.self_attn.cross_update_Y(x, y_in=y)
        
        # 1. Update Y (Hierarchy) with Residual + Norm
        # We use 'norm_y(y)' as input to be safe, similar to Pre-LN for x
        y_norm = self.norm_y(y)
        
        # Calculate the update (delta)
        y_delta = self.self_attn.cross_update_Y(x, y_in=y_norm)
        
        # Apply Residual Connection to Y
        y_next = y + self.dropout(y_delta)

    
        # PRE-LAYER NORMALIZATION (Apply Norm BEFORE Attention)
        # This significantly improves stability and convergence speed
        
        # Norm -> Attention -> Add
        norm_x = self.norm1(x)

        if return_attention:
            attn_output, self_attn_weights = self.self_attn(norm_x, norm_x, norm_x, y=y_next, mask=mask, return_attention=True)
        else:
            attn_output = self.self_attn(norm_x, norm_x, norm_x, y=y_next, mask=mask)
            
        x = x + self.dropout(attn_output) # Residual
        
        # FF
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        if return_attention:
            return x, y_next, self_attn_weights
        return x, y_next

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
        x = self.embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Initialize Y from the embeddings using the static method
        y = HierarchicalSparseAttention.generate_span_input_Y(x)
        
        trg_mask = self.make_causal_mask(trg)

        attentions = [] if return_attention else None
        
        for layer in self.layers:
            if return_attention:
                x, y, attention = layer(x, y, mask=trg_mask, return_attention=True)
                attentions.append(attention)
            else:
                x, y = layer(x, y, mask=trg_mask)
        
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
# 3. TRAINING LOOP (Added Auto-Exit / Early Stopping)
# ==========================================

def train_transformer_model(model, train_loader, valid_loader, criterion=None, num_epochs=100, learning_rate=3e-4, patience=10):
    if criterion is None:
        # NOTE: Using Label Smoothing for better generalization
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    device = next(model.parameters()).device
    
    print(f"\n{'='*50}")
    print(f"<> Training Transformer Model (AMP Enabled)")
    print(f"{'='*50}")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # CHANGED: Use Cosine Annealing (Point 7: Better scheduler)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=False)
    
    # --- AMP CHANGE 1: Initialize GradScaler ---
    # This manages the dynamic loss scaling (critical for FP16 stability)
    scaler = torch.cuda.amp.GradScaler() 

    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    epoch_times = []
    
    # --- EARLY STOPPING VARIABLES ---
    patience_counter = 0
    
    # Gradient Accumulation Steps (Simulate larger batch size)
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
            
            # --- AMP CHANGE 2: Run forward pass in autocast context ---
            # Operations here will automatically choose FP16 or FP32
            with torch.cuda.amp.autocast():
                outputs = model(input_ids)
                
                output_dim = outputs.shape[-1]
                outputs = outputs.contiguous().view(-1, output_dim)
                labels = labels.contiguous().view(-1)
                
                loss = criterion(outputs, labels)
                
                # Divide loss by accumulation steps
                loss = loss / accumulation_steps
            
            # --- AMP CHANGE 3: Scale loss and backward ---
            # Instead of loss.backward(), we scale it first to prevent underflow
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Unscale gradients before clipping (required for correct clipping)
                scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step with scaler (it will skip update if NaNs found)
                scaler.step(optimizer)
                
                # Update scaler factor for next iteration
                scaler.update()
                
                optimizer.zero_grad()
            
            # Multiply back for reporting (use .item() to detach from graph)
            total_train_loss += loss.item() * accumulation_steps
            
            progress_bar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'ppl': f'{math.exp(min(loss.item() * accumulation_steps, 10)):.2f}'
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
                
                # --- AMP CHANGE 4: Validation also benefits from AMP speedup ---
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids)
                    
                    output_dim = outputs.shape[-1]
                    outputs = outputs.contiguous().view(-1, output_dim)
                    labels = labels.contiguous().view(-1)
                    
                    loss = criterion(outputs, labels)
                
                total_valid_loss += loss.item()
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        
        # CHANGED: Scheduler step now per epoch without metric
        scheduler.step()
        
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
                # Good practice: Save scaler state too
                'scaler_state_dict': scaler.state_dict()
            }, 'best_transformer_wikitext.pt')
            print(f'<> Saved new best model with validation loss: {best_valid_loss:.4f}')
        else:
            patience_counter += 1
            print(f"<> No improvement for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print(f"\n<> Auto-Exit Triggered: Validation loss has not improved for {patience} epochs.")
                break
    
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

# OPTIMIZED HYPERPARAMETERS
model = TransformerLM(
    vocab_size=vocab_size,
    d_model=512,        # INCREASED from 256
    num_heads=8,        
    d_ff=2048,          # INCREASED from 1024
    num_layers=12,       
    dropout=0.2         # CHANGED: Increased from 0.1 to 0.2 (Point 7)
)

# ENABLE MULTI-GPU SUPPORT (DataParallel)
if torch.cuda.device_count() > 1:
    print(f"<> Utilizing {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

print(f"Using device: {device}")

# Training parameters
num_epochs = 100     # SET HIGH as requested (auto-exit will stop it)
learning_rate = 1e-4 # LOWERED for stability with larger model

print("\n<> Starting Training...")
results = train_transformer_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=None, 
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    patience=10 # Auto-exit if no improvement for 6 epochs
)

# Run Final Evaluation on Test Set
print("\n<> Loading Best Model for Evaluation...")
checkpoint = torch.load('best_transformer_wikitext.pt')
# We need to load state dict carefully depending on if DataParallel was used during save
# My training loop logic saves model.module if available, so we load into a fresh instance
best_model = TransformerLM(
    vocab_size=vocab_size,
    d_model=512,
    num_heads=8,        
    d_ff=2048,
    num_layers=12,       
    dropout=0.2
)
best_model.load_state_dict(checkpoint['model_state_dict'])
best_model = best_model.to(device)

if torch.cuda.device_count() > 1:
     best_model = nn.DataParallel(best_model)

evaluate_test_set(best_model, test_loader)