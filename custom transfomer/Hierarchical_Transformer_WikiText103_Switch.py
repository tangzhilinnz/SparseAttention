# -*- coding: utf-8 -*-

import os
# ==========================================
# CRITICAL FIX: GPU SELECTION MUST BE FIRST
# ==========================================
# Set this before importing torch or calling torch.cuda to avoid OOM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
# 1. DATA PROCESSING (WikiText-103 Optimized)
# ==========================================
import collections

class EfficientVocabBuilder:
    def __init__(self, dataset_split, max_vocab_size=50000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.build_vocab(dataset_split)
        
    def build_vocab(self, dataset_split):
        print("Counting words in WikiText-103 (Streamed)...")
        word_counts = collections.Counter()
        
        # Iterate without loading everything to RAM
        for item in tqdm(dataset_split, desc="Building Vocab"):
            text = item['text']
            if len(text.strip()) > 0:
                # WikiText-103 is already space-tokenized, but .split() is safe
                word_counts.update(text.lower().split())
        
        # Take most common words
        most_common = word_counts.most_common(self.max_vocab_size - 4)
        for word, count in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        print(f"\nVocabulary Statistics:")
        print(f"Vocabulary limit: {self.max_vocab_size}")
        print(f"Total unique words seen: {len(word_counts)}")

class LargeScaleWikiTextDataset(Dataset):
    def __init__(self, dataset_split, vocab, max_len=4096, block_size=1000000):
        """
        Args:
            block_size: Number of tokens to process before converting to Tensor
                        (prevents Python list RAM explosion)
        """
        self.vocab = vocab
        self.max_len = max_len
        self.data = self._tokenize_and_flatten(dataset_split, block_size)
        
        # Calculate number of full chunks
        self.num_samples = len(self.data) // (self.max_len + 1)
        print(f"Total Tokens: {len(self.data)}")
        print(f"Total Sequences: {self.num_samples}")

    def _tokenize_and_flatten(self, dataset_split, block_size):
        print(f"Tokenizing and flattening data...")
        
        # We use a list of tensors to avoid one massive Python list resizing
        tensor_chunks = []
        current_chunk = []
        
        unk_idx = self.vocab.word2idx['<UNK>']
        eos_idx = self.vocab.word2idx['<EOS>']
        
        for item in tqdm(dataset_split, desc="Tokenizing"):
            text = item['text']
            if len(text.strip()) > 0:
                words = text.lower().split()
                # Optimized list comprehension lookups
                ids = [self.vocab.word2idx.get(w, unk_idx) for w in words]
                ids.append(eos_idx)
                current_chunk.extend(ids)
                
                # If chunk gets too big, convert to tensor and clear list
                if len(current_chunk) > block_size:
                    tensor_chunks.append(torch.tensor(current_chunk, dtype=torch.long))
                    current_chunk = []
        
        # Process remaining
        if current_chunk:
            tensor_chunks.append(torch.tensor(current_chunk, dtype=torch.long))
            
        # Concatenate all tensors into one massive 1D tensor
        # This fits in RAM (100M tokens * 8 bytes = ~800MB)
        return torch.cat(tensor_chunks)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.max_len
        end_idx = start_idx + self.max_len + 1
        
        # Slicing a tensor is zero-copy (efficient)
        chunk = self.data[start_idx:end_idx]
        
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        
        return {
            'input_ids': input_ids,
            'label': target_ids
        }

# --- LOAD DATA ---
print("\n<> Loading WikiText-103 Dataset...")
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Standard WT103 is ~267k. Setting to 50k for efficiency with custom attention.
VOCAB_SIZE = 50000 
vocab_builder = EfficientVocabBuilder(dataset['train'], max_vocab_size=VOCAB_SIZE)

# Create Datasets
print("\n<> Processing Training Data (This may take 1-2 mins)...")
MAX_LEN = 2048 
train_dataset = LargeScaleWikiTextDataset(dataset['train'], vocab_builder, max_len=MAX_LEN)

print("\n<> Processing Validation Data...")
valid_dataset = LargeScaleWikiTextDataset(dataset['validation'], vocab_builder, max_len=MAX_LEN)

print("\n<> Processing Test Data...")
test_dataset = LargeScaleWikiTextDataset(dataset['test'], vocab_builder, max_len=MAX_LEN)

# Dataloaders
batch_size = 8 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=4)


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
    causal_mask = torch.full((seq_len, level_num), False, dtype=torch.bool, device=device)
    idx_map = torch.full((seq_len, level_num), -1, dtype=dtype, device=device)

    for n in range(seq_len):
        n_cur = n # Starts as the leaf index
        
        for lvl in range(level_num):
            if lvl == 0:
                n_next = n_cur ^ 1  # Sibling leaf
                pair = n_cur        # The leaf itself
            else:
                n_next = (n_cur // 2 + seq_len) ^ 1 # Uncle
                pair = (n_cur // 2 + seq_len)       # Parent

            if n_next > max_valid:
                break

            if pair < n_next:
                causal_mask[n, lvl] = True

            idx_map[n, lvl] = n_next
            n_cur = n_next 

    return idx_map, causal_mask


class HierarchicalSparseAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, window_size=22):
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

        self.merge_layer = nn.Linear(2 * dim, dim)

        # --- X Updates (Top-Down) ---
        self.Wq_x = nn.Linear(dim, dim, bias=False)
        self.Wk_x = nn.Linear(dim, dim, bias=False)
        self.Wv_x = nn.Linear(dim, dim, bias=False)
        self.out_proj_x = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        
        # --- Caching ---
        self.cached_idx_table = None
        self.cached_causal_mask = None
        self.cached_seq_len = -1
        
        self.cached_swa_indices = None
        self.cached_swa_offsets_len = -1

        self.sizes = None
        self.offsets = None

    def _get_lookup_table(self, L, device):
        if (self.cached_idx_table is not None and 
            self.cached_seq_len == L and 
            self.cached_idx_table.device == device):
            return self.cached_idx_table, self.cached_causal_mask

        # DIRECT CALL: Removed try/except to prevent silent failure
        idx_table, mask = build_hierarchical_index_lookup_table(L, device=device, dtype=torch.int64)
        
        self.cached_idx_table = idx_table
        self.cached_causal_mask = mask
        self.cached_seq_len = L
        return idx_table, mask

    def _get_swa_indices(self, L, device):
        offsets = torch.arange(-self.window_size + 1, self.window_size, device=device)
        
        if (self.cached_swa_indices is not None and 
            self.cached_swa_indices.shape[0] == L and
            self.cached_swa_offsets_len == len(offsets) and
            self.cached_swa_indices.device == device):
            return self.cached_swa_indices

        offsets_unsqueezed = offsets.unsqueeze(0) 
        base_indices = torch.arange(L, device=device).unsqueeze(1)
        swa_indices = base_indices + offsets_unsqueezed
        
        self.cached_swa_indices = swa_indices
        self.cached_swa_offsets_len = len(offsets)
        return swa_indices

    @staticmethod
    def generate_span_input_Y(x):
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
        # Bottom-Up Logic
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        assert y_in is not None

        if self.sizes is None or (self.sizes[0] != N // 2): 
            self.sizes, self.offsets = self.build_level_info(N)

        Q_p_all = self.Wq_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        K_p_all = self.Wk_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        V_p_all = self.Wv_y(y_in).view(B, -1, H, Dh).transpose(1, 2)
        
        new_Y_levels = []
        prev_sources = x 

        for level, parent_count in enumerate(self.sizes):
            useful_len = parent_count * 2
            children = prev_sources[:, :useful_len, :] 
            
            K_c = self.Wk_y(children).view(B, -1, H, Dh).transpose(1, 2)
            V_c = self.Wv_y(children).view(B, -1, H, Dh).transpose(1, 2)
            V_c = self.dropout(V_c)

            K_c_pairs = K_c.view(B, H, parent_count, 2, Dh)
            V_c_pairs = V_c.view(B, H, parent_count, 2, Dh)

            offset = self.offsets[level]
            Q_p = Q_p_all[:, :, offset : offset + parent_count, :]
            K_p = K_p_all[:, :, offset : offset + parent_count, :]
            V_p = V_p_all[:, :, offset : offset + parent_count, :]

            K_pool = torch.cat([K_p.unsqueeze(3), K_c_pairs], dim=3)
            V_pool = torch.cat([V_p.unsqueeze(3), V_c_pairs], dim=3)

            logits = torch.matmul(Q_p.unsqueeze(3), K_pool.transpose(-1, -2))
            logits = logits / math.sqrt(Dh)
            
            weights = F.softmax(logits, dim=-1)
            attn_out = torch.matmul(weights, V_pool)
            
            attn_out = attn_out.squeeze(3).transpose(1, 2).contiguous().reshape(B, parent_count, D)
            
            new_Y_levels.append(attn_out)
            prev_sources = attn_out

        return self.out_proj_y(torch.cat(new_Y_levels, dim=1))

    def update_X_from_Y(self, x, y, mask=None, disable_hierarchy=False):
        """
        Fused Attention: SWA + (Optional) Hierarchical Connections
        Top-Down Logic
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # 1. Project Q (always from X)
        Q = self.Wq_x(x).view(B, N, H, Dh).transpose(1, 2)
        
        # 2. Project K, V based on ablation mode
        if disable_hierarchy or y is None:
            # Pure SWA: Only look at leaf nodes (x)
            K_full = self.Wk_x(x).view(B, N, H, Dh).transpose(1, 2)
            V_full = self.Wv_x(x).view(B, N, H, Dh).transpose(1, 2)
        else:
            # Full Model: Look at leaves (x) + hierarchy (y)
            XY = torch.cat([x, y], dim=1)
            K_full = self.Wk_x(XY).view(B, -1, H, Dh).transpose(1, 2)
            V_full = self.Wv_x(XY).view(B, -1, H, Dh).transpose(1, 2)

        V_full = self.dropout(V_full)

        # --- SLIDING WINDOW LOGITS ---
        swa_indices = self._get_swa_indices(N, device=x.device)
        swa_valid_mask = (swa_indices >= 0) & (swa_indices < N)
        safe_swa_indices = swa_indices.clamp(0, N - 1)
        
        swa_k = K_full[:, :, safe_swa_indices, :]
        swa_v = V_full[:, :, safe_swa_indices, :]
        
        swa_logits = torch.einsum('b h l d, b h l n d -> b h l n', Q, swa_k)
        swa_logits = swa_logits / math.sqrt(Dh)
        swa_logits = swa_logits.masked_fill(~swa_valid_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            row_indices = torch.arange(N, device=x.device).unsqueeze(1)
            causal_mask_bool = safe_swa_indices > row_indices
            swa_logits = swa_logits.masked_fill(causal_mask_bool.unsqueeze(0).unsqueeze(0), float('-inf'))

        # --- HIERARCHICAL LOGITS (Conditional) ---
        if not disable_hierarchy and y is not None:
            idx_table, neighbor_causal_mask = self._get_lookup_table(N, device=x.device)
            hier_k = K_full[:, :, idx_table, :] 
            hier_v = V_full[:, :, idx_table, :]

            hier_logits = torch.einsum('b h l d, b h l n d -> b h l n', Q, hier_k)
            hier_logits = hier_logits / math.sqrt(Dh)

            if mask is not None:
                hier_logits = hier_logits.masked_fill(neighbor_causal_mask, float('-inf'))

            # Combine SWA and Hierarchy
            all_logits = torch.cat([swa_logits, hier_logits], dim=3)
            all_v = torch.cat([swa_v, hier_v], dim=3)
        else:
            # Pure SWA mode
            all_logits = swa_logits
            all_v = swa_v

        # --- OUTPUT ---
        weights = F.softmax(all_logits, dim=-1)
        output_leaf = torch.einsum('b h l n, b h l n d -> b h l d', weights, all_v)
        output = output_leaf.transpose(1, 2).reshape(B, N, D)

        return self.out_proj_x(output)

    def _standard_attention(self, Q, K, V, mask):
        D_head = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(D_head)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(attn, V), attn

    def forward(self, query, key, value, y=None, mask=None, return_attention=False, disable_hierarchy=False):
        x = query 
        B, L_Q, D = x.size()
        H, Dh = self.num_heads, self.head_dim
        
        # Use Hierarchical/SWA path if inputs match and we are in self-attention mode
        if L_Q == key.size(1) == value.size(1):
            # We assume Bottom-Up (cross_update_Y) is done by DecoderLayer before calling this
            output = self.update_X_from_Y(x, y, mask, disable_hierarchy=disable_hierarchy)
            return (output, None) if return_attention else output
        
        # Fallback for Cross-Attention or non-matching lengths
        else:
            Q = self.Wq_x(query).view(B, L_Q, H, Dh).transpose(1, 2)
            K = self.Wk_x(key).view(B, -1, H, Dh).transpose(1, 2)
            V = self.Wv_x(value).view(B, -1, H, Dh).transpose(1, 2)
            
            output_leaf, attn_weights = self._standard_attention(Q, K, V, mask)
            output = self.out_proj_x(output_leaf.transpose(1, 2).reshape(B, L_Q, D))
            
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
        self.self_attn = HierarchicalSparseAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_y = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y, mask=None, return_attention=False, disable_hierarchy=False):
        
        # --- 1. Update Y (Hierarchy) with Residual + Norm ---
        if not disable_hierarchy and y is not None:
            # Bottom-Up Logic
            y_norm = self.norm_y(y)
            y_delta = self.self_attn.cross_update_Y(x, y_in=y_norm)
            y_next = y + self.dropout(y_delta)
        else:
            # Disable hierarchy: y_next is None or bypassed
            y_next = y if y is not None else None 
        
        # --- 2. Update X (Pre-LN) ---
        norm_x = self.norm1(x)

        if return_attention:
            attn_output, self_attn_weights = self.self_attn(
                norm_x, norm_x, norm_x, 
                y=y_next, 
                mask=mask, 
                return_attention=True,
                disable_hierarchy=disable_hierarchy
            )
        else:
            attn_output = self.self_attn(
                norm_x, norm_x, norm_x, 
                y=y_next, 
                mask=mask,
                disable_hierarchy=disable_hierarchy
            )
            
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

    def forward(self, trg, return_attention=False, disable_hierarchy=False):
        x = self.embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Initialize Y from the embeddings using the static method
        y = HierarchicalSparseAttention.generate_span_input_Y(x)
        
        trg_mask = self.make_causal_mask(trg)

        attentions = [] if return_attention else None
        
        for layer in self.layers:
            if return_attention:
                x, y, attention = layer(x, y, mask=trg_mask, return_attention=True, disable_hierarchy=disable_hierarchy)
                attentions.append(attention)
            else:
                x, y = layer(x, y, mask=trg_mask, disable_hierarchy=disable_hierarchy)
        
        output = self.fc_out(x)
        return output
    
    def generate(self, src, start_token=2, max_len=50, temperature=1.0, disable_hierarchy=False):
        self.eval()
        device = next(self.parameters()).device
        
        # Src is treated as prompt in Decoder-Only
        current_tokens = src.to(device)
        if current_tokens.dim() == 1:
            current_tokens = current_tokens.unsqueeze(0)
            
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(current_tokens, disable_hierarchy=disable_hierarchy)
                last_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(last_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                
                if next_token.item() == 3: # EOS token ID
                    break
        
        return current_tokens

# ==========================================
# 3. TRAINING LOOP
# ==========================================

def train_transformer_model(model, train_loader, valid_loader, criterion=None, num_epochs=100, learning_rate=3e-4, patience=10, disable_hierarchy=False):
    if criterion is None:
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    device = next(model.parameters()).device
    
    print(f"\n{'='*50}")
    print(f"<> Training Transformer Model (AMP Enabled)")
    print(f"<> Mode: {'SWA Only (Ablation)' if disable_hierarchy else 'Full Hierarchical Attention'}")
    print(f"{'='*50}")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=False)
    scaler = torch.cuda.amp.GradScaler() 

    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    epoch_times = []
    
    patience_counter = 0
    accumulation_steps = 16 

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
            
            with torch.cuda.amp.autocast():
                # PASS ABLATION FLAG HERE
                outputs = model(input_ids, disable_hierarchy=disable_hierarchy)
                
                output_dim = outputs.shape[-1]
                outputs = outputs.contiguous().view(-1, output_dim)
                labels = labels.contiguous().view(-1)
                
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
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
                
                with torch.cuda.amp.autocast():
                    # PASS ABLATION FLAG HERE
                    outputs = model(input_ids, disable_hierarchy=disable_hierarchy)
                    
                    output_dim = outputs.shape[-1]
                    outputs = outputs.contiguous().view(-1, output_dim)
                    labels = labels.contiguous().view(-1)
                    
                    loss = criterion(outputs, labels)
                
                total_valid_loss += loss.item()
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        
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
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0 
            
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_valid_loss,
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

def evaluate_test_set(model, test_loader, disable_hierarchy=False):
    print(f"\n{'='*50}")
    print(f"<> FINAL TEST EVALUATION")
    print(f"{'='*50}")
    
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    
    eval_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, disable_hierarchy=disable_hierarchy)
            
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
# 4. INITIALIZATION AND RUN (A100 Configuration)
# ==========================================

print("\n<> Initializing Transformer Model...")
vocab_size = len(vocab_builder.word2idx)
print(f"Actual Vocab Size: {vocab_size}")

# --- ABLATION TOGGLE ---
# Set this to True to train only SWA (No Hierarchy)
DISABLE_HIERARCHY = True 

model = TransformerLM(
    vocab_size=vocab_size,
    d_model=768,          
    num_heads=12,         
    d_ff=3072,            
    num_layers=12,        
    dropout=0.15          
)

if torch.cuda.device_count() > 1:
    print(f"<> Utilizing {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

num_epochs = 30       
learning_rate = 2e-4  

print(f"\n<> Starting Training on WikiText-103 (Hierarchy Disabled: {DISABLE_HIERARCHY})...")
results = train_transformer_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=None, 
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    patience=100,
    disable_hierarchy=DISABLE_HIERARCHY
)


# ==========================================
# 5. ADVANCED EVALUATION (Sliding Window)
# ==========================================

def evaluate_wikitext_103(model, test_loader, device, sliding_window=False, stride=512, disable_hierarchy=False):
    """
    Evaluates the model on WikiText-103 using either:
    1. Standard Chunked evaluation (Fast, slightly higher PPL)
    2. Sliding Window evaluation (Slower, accurate SOTA PPL)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
    
    print(f"\n{'='*50}")
    if sliding_window:
        print(f"<> Starting SLIDING WINDOW Evaluation (Stride={stride})...")
        print(f"<> Note: This provides the most accurate Perplexity.")
    else:
        print(f"<> Starting STANDARD CHUNKED Evaluation...")
    print(f"{'='*50}")

    if not sliding_window:
        # --- METHOD 1: Standard Chunked Evaluation ---
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, disable_hierarchy=disable_hierarchy)
                
                # Flatten
                shift_logits = outputs.view(-1, outputs.size(-1))
                shift_labels = labels.view(-1)
                
                loss = criterion(shift_logits, shift_labels)
                total_loss += loss.item()
                total_tokens += (shift_labels != 0).sum().item()

    else:
        # --- METHOD 2: Sliding Window Evaluation (SOTA Standard) ---
        raw_data = []
        print(">> Flattening test data for sliding window...")
        for batch in test_loader:
            raw_data.append(batch['input_ids'].cpu())
        
        full_seq = torch.cat(raw_data).view(-1).to(device)
        max_len = 2048

        with torch.no_grad():
            for i in tqdm(range(0, len(full_seq) - max_len, stride), desc="Sliding Window"):
                input_window = full_seq[i : i + max_len].unsqueeze(0) # [1, Seq_Len]
                target_window = full_seq[i+1 : i + max_len + 1].unsqueeze(0)

                outputs = model(input_window, disable_hierarchy=disable_hierarchy) # [1, Seq_Len, Vocab]

                logits_stride = outputs[:, -stride:, :]
                labels_stride = target_window[:, -stride:]

                loss = criterion(logits_stride.reshape(-1, logits_stride.size(-1)), 
                                 labels_stride.reshape(-1))
                
                total_loss += loss.item()
                total_tokens += labels_stride.numel()

    # --- FINAL CALCULATION ---
    if total_tokens == 0:
        print("Error: No tokens evaluated.")
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    print(f"\n{'='*50}")
    print(f"RESULTS: {'Sliding Window' if sliding_window else 'Standard Chunked'}")
    print(f"{'='*50}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"PERPLEXITY  : {perplexity:.2f}")
    print(f"{'='*50}\n")
    
    return perplexity

# ==========================================
# 6. FINAL EXECUTION
# ==========================================

print("\n<> Training Finished. Loading Best Model for Evaluation...")

# 1. Initialize a fresh model instance to ensure clean state
best_model = TransformerLM(
    vocab_size=vocab_size,
    d_model=768, 
    num_heads=12,         
    d_ff=3072,
    num_layers=12,        
    dropout=0.15          
)

# 2. Load Checkpoint
checkpoint_path = 'best_transformer_wikitext.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') # remove 'module.'
        new_state_dict[name] = v
        
    best_model.load_state_dict(new_state_dict)
    print(f"<> Loaded checkpoint from epoch {checkpoint['epoch']} (Loss: {checkpoint['loss']:.4f})")
else:
    print(f"<> Warning: Checkpoint {checkpoint_path} not found. Using current model state.")
    best_model = model

best_model = best_model.to(device)
if torch.cuda.device_count() > 1:
     best_model = nn.DataParallel(best_model)

# 3. Run Standard Evaluation (Fast check)
print("\nRunning Standard Evaluation...")
evaluate_wikitext_103(best_model, test_loader, device, sliding_window=False, disable_hierarchy=DISABLE_HIERARCHY)

# 4. Run Sliding Window Evaluation (Accurate / Publication Ready)
print("\nRunning Sliding Window Evaluation (This will take longer)...")
evaluate_wikitext_103(best_model, test_loader, device, sliding_window=True, stride=512, disable_hierarchy=DISABLE_HIERARCHY)