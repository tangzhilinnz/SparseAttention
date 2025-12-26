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
# CHANGED: 'wikitext-103-v1'
dataset = load_dataset("wikitext", "wikitext-103-v1")

# CHANGED: Increased Vocab Size for WT103
# Standard WT103 is ~267k. Setting to 50k for efficiency with custom attention.
# If you want the full benchmark, set max_vocab_size=267735
VOCAB_SIZE = 50000 
vocab_builder = EfficientVocabBuilder(dataset['train'], max_vocab_size=VOCAB_SIZE)

# Create Datasets
print("\n<> Processing Training Data (This may take 1-2 mins)...")
MAX_LEN = 2048 # Reduced slightly from 4096 to ensure safety with batching on A100
train_dataset = LargeScaleWikiTextDataset(dataset['train'], vocab_builder, max_len=MAX_LEN)

print("\n<> Processing Validation Data...")
valid_dataset = LargeScaleWikiTextDataset(dataset['validation'], vocab_builder, max_len=MAX_LEN)

print("\n<> Processing Test Data...")
test_dataset = LargeScaleWikiTextDataset(dataset['test'], vocab_builder, max_len=MAX_LEN)

# Dataloaders
# Increased Batch Size for A100 80G
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
        # NOTE: Wq_y, Wk_y, Wv_y are kept in case you want to switch back to attention,
        # but for MLP mode, we primarily use merge_layer.
        self.Wq_y = nn.Linear(dim, dim, bias=False)
        self.Wk_y = nn.Linear(dim, dim, bias=False)
        self.Wv_y = nn.Linear(dim, dim, bias=False)
        self.out_proj_y = nn.Linear(dim, dim)

        # --- NEW: MLP Merge Layer for cross_update_Y_MLP ---
        # Takes concatenated Left+Right (2*dim) and projects to Parent (dim)
        self.merge_layer = nn.Linear(2 * dim, dim)

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
        """Smart retrieval: Returns cached table if L matches, otherwise recomputes."""
        if (self.cached_idx_table is not None and 
            self.cached_seq_len == L and 
            self.cached_idx_table.device == device):
            return self.cached_idx_table, self.cached_causal_mask

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

    def cross_update_Y(self, x, y_in):
        """
        Architecture: Recursive Parent-Self Attention.
        
        Logic:
           Pool = [Parent_Old, Child_Left, Child_Right]
           Query = Parent_Old
           Output = Attention(Q, K=Pool, V=Pool)
           
        This allows the parent to selectively 'read' from itself or its children.
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        assert y_in is not None

        if self.sizes is None: self.sizes, self.offsets = self.build_level_info(N)

        # -------------------------------------------------------
        # OPTIMIZATION: Global Parent Projection
        # -------------------------------------------------------
        # Since y_in is static, we project ALL parents at once outside the loop.
        # This saves massive compute compared to projecting inside the loop.
        
        # Q_parents: The parent acts as the Query
        Q_p_all = self.Wq_y(y_in).view(B, -1, H, Dh)
        
        # K_p_all, V_p_all: The parent also acts as a Key/Value (Target for attending to self)
        K_p_all = self.Wk_y(y_in).view(B, -1, H, Dh)
        V_p_all = self.Wv_y(y_in).view(B, -1, H, Dh)
        
        new_Y_levels = []
        prev_sources = x 

        for level, parent_count in enumerate(self.sizes):
            # 1. Prepare Children
            useful_len = parent_count * 2
            children = prev_sources[:, :useful_len, :] 
            
            # Project Children (Must be done in loop as prev_sources changes)
            # [B, 2*P, H, Dh]
            K_c = self.Wk_y(children).view(B, -1, H, Dh)
            V_c = self.Wv_y(children).view(B, -1, H, Dh)
            V_c = self.dropout(V_c)

            # 2. Reshape Children to Pairs [B, P, 2, H, Dh]
            K_c_pairs = K_c.view(B, parent_count, 2, H, Dh)
            V_c_pairs = V_c.view(B, parent_count, 2, H, Dh)

            # 3. Slice Parents for this Level [B, P, H, Dh]
            offset = self.offsets[level]
            Q_p = Q_p_all[:, offset : offset + parent_count, :, :]
            K_p = K_p_all[:, offset : offset + parent_count, :, :]
            V_p = V_p_all[:, offset : offset + parent_count, :, :]

            # 4. Form the Attention Pool: [Parent, Child_L, Child_R]
            # K shape: [B, P, 3, H, Dh]
            # We unsqueeze Parent to [B, P, 1, H, Dh] to stack
            K_pool = torch.cat([K_p.unsqueeze(2), K_c_pairs], dim=2)
            V_pool = torch.cat([V_p.unsqueeze(2), V_c_pairs], dim=2)

            # 5. Compute Attention Scores
            # Q: [B, P, H, Dh] -> [B, P, H, 1, Dh]
            # K: [B, P, 3, H, Dh] -> [B, P, H, Dh, 3] (Permute for dot product)
            
            q_vec = Q_p.unsqueeze(2).transpose(2, 3) # [B, P, H, 1, Dh]
            k_vec = K_pool.permute(0, 1, 3, 4, 2)    # [B, P, H, Dh, 3]
            
            # Dot Product: [B, P, H, 1, 3]
            # Result is 3 scores per head: [Score_Self, Score_Left, Score_Right]
            logits = torch.matmul(q_vec, k_vec) / math.sqrt(Dh)
            
            weights = F.softmax(logits, dim=-1) # [B, P, H, 1, 3]
            # weights = self.dropout(weights)

            # 6. Weighted Sum
            # V_pool: [B, P, 3, H, Dh] -> [B, P, H, 3, Dh] (Permute to match weights)
            v_vec = V_pool.permute(0, 1, 3, 2, 4)
            
            # [B, P, H, 1, 3] * [B, P, H, 3, Dh] -> [B, P, H, 1, Dh]
            attn_out = torch.matmul(weights, v_vec)
            
            # 7. Merge Heads
            # Remove the '1' dim, transpose back
            # [B, P, H, 1, Dh] -> [B, P, H, Dh] -> [B, P, D]
            attn_out = attn_out.squeeze(3).transpose(2, 3).reshape(B, parent_count, D)
            
            new_Y_levels.append(attn_out)
            prev_sources = attn_out

        return self.out_proj_y(torch.cat(new_Y_levels, dim=1))

    def update_X_from_Y(self, x, y, mask=None):
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        if y is None: return x

        # Concatenate inputs once to allow unified K/V calculation
        XY = torch.cat([x, y], dim=1)

        # --- Inline Split Heads ---
        Q = self.Wq_x(x).view(B, N, H, Dh).transpose(1, 2)
        
        # Calculate K, V on combined input
        kv_input = self.Wk_x(XY).view(B, -1, H, Dh).transpose(1, 2)
        v_input = self.Wv_x(XY).view(B, -1, H, Dh).transpose(1, 2)

        K_full = kv_input
        V_full = self.dropout(v_input)

        idx_table, neighbor_causal_mask = self._get_lookup_table(N, device=x.device)
            
        # Self: The leaves attending to themselves
        K_self = K_full[:, :, :N, :]                 
        V_self = V_full[:, :, :N, :]                 

        # Gather neighbors using index table
        gather_indices = idx_table
        
        neighbors_k = K_full[:, :, gather_indices, :] 
        neighbors_v = V_full[:, :, gather_indices, :]

        # Compute Self Logits
        self_logits = torch.einsum('b h n d, b h n d -> b h n', Q, K_self) / math.sqrt(Dh)

        # Compute Neighbor Logits
        neighbors_logits = torch.einsum('b h l x d, b h l n d -> b h l n', Q.unsqueeze(3), neighbors_k)
        neighbors_logits = neighbors_logits / math.sqrt(Dh)

        # Apply Causal Mask
        if mask is not None:
            neighbors_logits = neighbors_logits.masked_fill(neighbor_causal_mask, float('-inf'))

        # Concatenate (Self + Neighbors)
        all_v = torch.cat([V_self.unsqueeze(3), neighbors_v], dim=3)             
        all_logits = torch.cat([self_logits.unsqueeze(3), neighbors_logits], dim=3) 

        # Attention Softmax & Weighted Sum
        max_logits = all_logits.max(dim=-1, keepdim=True)[0]
        weights = F.softmax(all_logits - max_logits, dim=-1)             
            
        output_leaf = torch.einsum('b h l n, b h l n d -> b h l d', weights, all_v)

        # --- Inline Merge Heads ---
        return output_leaf.transpose(1, 2).reshape(B, N, D)

    def _standard_attention(self, Q, K, V, mask):
        D_head = Q.size(-1)
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
            Q = self.Wq_x(query).view(B, L_Q, H, Dh).transpose(1, 2)
            K = self.Wk_x(key).view(B, L_K, H, Dh).transpose(1, 2)
            V = self.Wv_x(value).view(B, L_V, H, Dh).transpose(1, 2)
            
            output_leaf, attn_weights = self._standard_attention(Q, K, V, mask)
            
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
# 4. INITIALIZATION AND RUN (A100 Configuration)
# ==========================================

print("\n<> Initializing Transformer Model...")
vocab_size = len(vocab_builder.word2idx)
print(f"Actual Vocab Size: {vocab_size}")

# OPTIMIZED HYPERPARAMETERS FOR A100 & WT-103
model = TransformerLM(
    vocab_size=vocab_size,
    d_model=768,          # Standard Base size (up from 512)
    num_heads=12,         # Standard Base heads (up from 8)
    d_ff=3072,            # Standard FFN size (4x d_model)
    num_layers=12,        # Deep enough for WT103
    dropout=0.15          # Slightly lower dropout as we have more data
)

# ENABLE MULTI-GPU SUPPORT (DataParallel)
if torch.cuda.device_count() > 1:
    print(f"<> Utilizing {torch.cuda.device_count()} GPUs!")
    # Using DataParallel is okay, but DistributedDataParallel (DDP) is better for A100s.
    # For a single script, DataParallel is easier to implement.
    model = nn.DataParallel(model)

model = model.to(device)

# Training parameters
num_epochs = 100       # WT103 converges slower, but 40 epochs is usually plenty
learning_rate = 2e-4  # Standard LR for this model size

print("\n<> Starting Training on WikiText-103...")
results = train_transformer_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=None, 
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    patience=100 
)


# ==========================================
# 5. ADVANCED EVALUATION (Sliding Window)
# ==========================================

def evaluate_wikitext_103(model, test_loader, device, sliding_window=False, stride=512):
    """
    Evaluates the model on WikiText-103 using either:
    1. Standard Chunked evaluation (Fast, slightly higher PPL)
    2. Sliding Window evaluation (Slower, accurate SOTA PPL)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Use standard CrossEntropy with sum reduction to aggregate manually
    # ignore_index=0 handles padding
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

                outputs = model(input_ids)
                
                # Flatten
                shift_logits = outputs.view(-1, outputs.size(-1))
                shift_labels = labels.view(-1)
                
                loss = criterion(shift_logits, shift_labels)
                total_loss += loss.item()
                total_tokens += (shift_labels != 0).sum().item()

    else:
        # --- METHOD 2: Sliding Window Evaluation (SOTA Standard) ---
        # 1. Reconstruct the full token stream from the loader
        raw_data = []
        print(">> Flattening test data for sliding window...")
        for batch in test_loader:
            # batch['input_ids'] is [B, Seq_Len]
            raw_data.append(batch['input_ids'].cpu())
        
        # Concatenate into one massive 1D tensor: [Total_Tokens]
        full_seq = torch.cat(raw_data).view(-1).to(device)
        
        # Determine context length from model config
        # Handle DataParallel wrapper if present
        if isinstance(model, nn.DataParallel):
            max_len = model.module.d_model if hasattr(model.module, 'd_model') else 2048
        else:
            max_len = model.d_model if hasattr(model, 'd_model') else 2048

        # 2. Iterate with stride
        with torch.no_grad():
            # Loop stops when we can't form a full window
            for i in tqdm(range(0, len(full_seq) - max_len, stride), desc="Sliding Window"):
                # Input: [i : i+max_len]
                input_window = full_seq[i : i + max_len].unsqueeze(0) # [1, Seq_Len]
                
                # Target: [i+1 : i+max_len+1]
                # We only care about the targets corresponding to the STRIDE (the new tokens)
                target_window = full_seq[i+1 : i + max_len + 1].unsqueeze(0)

                outputs = model(input_window) # [1, Seq_Len, Vocab]

                # Focus on the last 'stride' tokens (where context is fullest)
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
# Ensure these params match your training config!
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
    
    # Handle DataParallel prefix ('module.') if it exists in saved dict but not in new model
    # or vice versa.
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
evaluate_wikitext_103(best_model, test_loader, device, sliding_window=False)

# 4. Run Sliding Window Evaluation (Accurate / Publication Ready)
# Stride 512 is a good balance between speed and accuracy
print("\nRunning Sliding Window Evaluation (This will take longer)...")
evaluate_wikitext_103(best_model, test_loader, device, sliding_window=True, stride=512)