# -*- coding: utf-8 -*-

import os
import re
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Hugging Face Datasets
import datasets
from datasets import load_dataset

# ==========================================
# 1. SETUP & UTILS
# ==========================================

# Set this if you need to target specific GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
# 2. DATA PROCESSING (Twitter Specific)
# ==========================================

def normalize_tweet(text):
    """
    Cleans tweet text by normalizing handles and URLs.
    This significantly improves vocabulary efficiency.
    """
    # Replace URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '<URL>', text, flags=re.MULTILINE)
    # Replace @mentions
    text = re.sub(r'@\w+', '<USER>', text)
    return text.strip()

class VocabBuilder:
    def __init__(self, texts, max_vocab_size=15000):
        self.max_vocab_size = max_vocab_size
        # Pre-define special tokens
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<URL>': 2, '<USER>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<URL>', 3: '<USER>'}
        self.build_vocab(texts)
        
    def build_vocab(self, texts):
        print("Counting words...")
        word_counts = Counter()
        for text in texts:
            text = normalize_tweet(text)
            if len(text) > 0:
                words = text.lower().split()
                word_counts.update(words)
        
        # Take most common words (-4 for our special tokens)
        most_common = word_counts.most_common(self.max_vocab_size - 4)
        for word, count in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
            
        print(f"Vocabulary Statistics:")
        print(f"Total words found: {len(word_counts)}")
        print(f"Final Vocab Size: {len(self.word2idx)}")

class TwitterDataset(Dataset):
    def __init__(self, texts, labels, vocab_builder, max_len=128):
        self.vocab = vocab_builder
        self.max_len = max_len
        self.texts = texts
        self.labels = labels
        
        # CRITICAL: Check power of 2 for Hierarchical Attention
        assert (max_len & (max_len - 1)) == 0, "MAX_LEN must be a power of 2"

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 1. Normalize
        text = normalize_tweet(text)
        words = text.lower().split()
        
        # 2. Truncate
        words = words[:self.max_len]
        
        # 3. Encode
        ids = [self.vocab.word2idx.get(w, self.vocab.word2idx['<UNK>']) for w in words]
        
        # 4. Pad (Right Padding)
        if len(ids) < self.max_len:
            padding = [self.vocab.word2idx['<PAD>']] * (self.max_len - len(ids))
            ids = ids + padding
            
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 3. FIXED HIERARCHICAL ATTENTION (Unchanged)
# ==========================================

def build_hierarchical_index_lookup_table(seq_len, device="cuda", dtype=torch.int32):
    assert (seq_len & (seq_len - 1)) == 0, "seq_len must be a power of 2"
    total_nodes = 2 * seq_len - 1
    max_valid = total_nodes - 2
    level_num = int(math.log2(seq_len))
    
    causal_mask = torch.full((seq_len, level_num), False, dtype=torch.bool, device=device)
    idx_map = torch.full((seq_len, level_num), -1, dtype=dtype, device=device)

    for n in range(seq_len):
        n_cur = n 
        for lvl in range(level_num):
            if lvl == 0:
                n_next = n_cur ^ 1  
                pair = n_cur        
            else:
                n_next = (n_cur // 2 + seq_len) ^ 1 
                pair = (n_cur // 2 + seq_len)       

            if n_next > max_valid:
                break
            if pair < n_next:
                causal_mask[n, lvl] = True
            idx_map[n, lvl] = n_next
            n_cur = n_next 

    return idx_map, causal_mask

class HierarchicalSparseAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.Wq_y = nn.Linear(dim, dim, bias=False)
        self.Wk_y = nn.Linear(dim, dim, bias=False)
        self.Wv_y = nn.Linear(dim, dim, bias=False)
        self.out_proj_y = nn.Linear(dim, dim)

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
            Y_slice = y_source[:, offset:offset + parent_count, :]

            Q = self.Wq_y(Y_slice).view(B, -1, H, Dh).transpose(1, 2)
            K = self.Wk_y(prev_sources).view(B, -1, H, Dh).transpose(1, 2)
            V = self.Wv_y(prev_sources).view(B, -1, H, Dh).transpose(1, 2)
            V = self.dropout(V)

            mask = self.build_parent_child_mask(parent_count, prev_sources.size(1), x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)

            attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Dh)
            attn_logits += mask
            attn_weights = F.softmax(attn_logits, dim=-1)
            updated = torch.matmul(attn_weights, V)
            updated_merged = updated.transpose(1, 2).reshape(B, -1, D)
            new_Y_levels.append(updated_merged)
            prev_sources = updated_merged

        Y_new = torch.cat(new_Y_levels, dim=1)
        Y_new = self.out_proj_y(Y_new)
        return Y_new

    def update_X_from_Y(self, x, y, mask=None):
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        if y is None: return x
        XY = torch.cat([x, y], dim=1)
        Q = self.Wq_x(x).view(B, N, H, Dh).transpose(1, 2)
        kv_input = self.Wk_x(XY).view(B, -1, H, Dh).transpose(1, 2)
        v_input = self.Wv_x(XY).view(B, -1, H, Dh).transpose(1, 2)

        K_full = kv_input
        V_full = self.dropout(v_input)

        idx_table, neighbor_causal_mask = self._get_lookup_table(N, device=x.device)
        K_self = K_full[:, :, :N, :]                
        V_self = V_full[:, :, :N, :]                 

        gather_indices = idx_table
        neighbors_k = K_full[:, :, gather_indices, :] 
        neighbors_v = V_full[:, :, gather_indices, :]

        self_logits = torch.einsum('b h n d, b h n d -> b h n', Q, K_self) / math.sqrt(Dh)
        neighbors_logits = torch.einsum('b h l x d, b h l n d -> b h l n', Q.unsqueeze(3), neighbors_k)
        neighbors_logits = neighbors_logits / math.sqrt(Dh)

        # if mask is not None:
        #     neighbors_logits = neighbors_logits.masked_fill(neighbor_causal_mask, float('-inf'))

        all_v = torch.cat([V_self.unsqueeze(3), neighbors_v], dim=3)              
        all_logits = torch.cat([self_logits.unsqueeze(3), neighbors_logits], dim=3) 

        max_logits = all_logits.max(dim=-1, keepdim=True)[0]
        weights = F.softmax(all_logits - max_logits, dim=-1)              
        output_leaf = torch.einsum('b h l n, b h l n d -> b h l d', weights, all_v)
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
            output_leaf = self.update_X_from_Y(x, y, mask)
            output = self.out_proj_x(output_leaf)
            return (output, None) if return_attention else output
        else:
            Q = self.Wq_x(query).view(B, L_Q, H, Dh).transpose(1, 2)
            K = self.Wk_x(key).view(B, L_K, H, Dh).transpose(1, 2)
            V = self.Wv_x(value).view(B, L_V, H, Dh).transpose(1, 2)
            output_leaf, attn_weights = self._standard_attention(Q, K, V, mask)
            output = output_leaf.transpose(1, 2).reshape(B, L_Q, D)
            output = self.out_proj_x(output)
            return (output, attn_weights) if return_attention else output

# ==========================================
# 4. OPTIMIZED TRANSFORMER CLASSIFIER
# ==========================================

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        # OPTIMIZATION: GELU is standard for modern transformers
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = HierarchicalSparseAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_y = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y, mask=None):
        # 1. Update Y (Hierarchical Stream)
        y_norm = self.norm_y(y)
        y_delta = self.self_attn.cross_update_Y(x, y_in=y_norm)
        y_next = y + self.dropout(y_delta)

        # 2. Update X (Token Stream) with Pre-LN
        norm_x = self.norm1(x)
        # Self Attention (using y_next)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, y=y_next, mask=mask)
        x = x + self.dropout(attn_output) 
        
        # FF
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        return x, y_next

class TwitterSentimentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes=3, max_len=128, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # OPTIMIZATION: Learnable Positional Embeddings often work better for short fixed sequences
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification Head (Pooling + Norm + Linear)
        self.norm_final = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self.d_model = d_model
        self.max_len = max_len
        
        self.apply(self._init_weights)

        # OPTIMIZATION: GPT-2 style initialization for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('out_proj_x.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

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

    def forward(self, trg):
        B, L = trg.shape
        
        # Embeddings
        positions = torch.arange(0, L, device=trg.device).unsqueeze(0)
        x = self.embedding(trg) * math.sqrt(self.d_model) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Initialize Y from X
        y = HierarchicalSparseAttention.generate_span_input_Y(x)
        trg_mask = self.make_causal_mask(trg)

        # Layers
        for layer in self.layers:
            x, y = layer(x, y, mask=trg_mask)
        
        # POOLING:
        # In a Causal model, the last token (index -1) aggregates the full context.
        # Since we use fixed padding, index -1 is always the end of the sequence (or padding).
        # (Note: For strictly variable lengths, you'd gather by index, but with fixed padding this works).
        last_token_rep = x[:, -1, :] 
        
        normalized_rep = self.norm_final(last_token_rep)
        logits = self.classifier(normalized_rep)
        
        return logits

# ==========================================
# 5. TRAINING LOOP (With Plotting)
# ==========================================

def evaluate(model, loader, criterion):
    """
    Helper to get both Accuracy and Loss
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            lbl = batch['label'].to(device)
            
            logits = model(ids)
            loss = criterion(logits, lbl)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)
            
    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return acc, avg_loss

def train_sentiment_model(model, train_loader, valid_loader, num_epochs=10):
    # Using Label Smoothing to prevent overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Weight Decay: Apply only to weights, not biases/LN
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    # --- TRACKING VARIABLES ---
    train_losses = []
    valid_losses = []
    epoch_times = []
    best_valid_loss = float('inf')
    best_acc = 0.0
    
    print(f"\n{'='*50}")
    print(f"<> Starting Training...")
    print(f"{'='*50}")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            ids = batch['input_ids'].to(device)
            lbl = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(ids)
                loss = criterion(logits, lbl)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)
            
            pbar.set_postfix({'acc': f"{100*correct/total:.2f}%", 'loss': f"{loss.item():.4f}"})
        
        # Epoch metrics
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        val_acc, avg_valid_loss = evaluate(model, valid_loader, criterion)
        
        # Update trackers
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch+1} | Valid Acc: {val_acc:.2f}% | Valid Loss: {avg_valid_loss:.4f} | Time: {int(epoch_time)}s")
        
        # Save Best
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_twitter_model.pt")
            print(f"<> Saved New Best Model (Acc: {val_acc:.2f}%)")
            
        scheduler.step()

    # --- PLOTTING CODE BLOCK ---
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
    plt.savefig('twitter_sentiment_history.png')
    plt.close()
    
    print(f"\n{'='*50}")
    print("<> Training Complete!")
    print(f"{'='*50}")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    
    results = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'best_valid_loss': best_valid_loss,
        'epoch_times': epoch_times,
        'total_time': total_time
    }
    
    return results

# ==========================================
# 6. EXECUTION
# ==========================================

def main():
    print("\n<> Loading TweetEval (Sentiment)...")
    try:
        dataset = load_dataset("tweet_eval", "sentiment")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure 'tweet_eval' is available via Hugging Face Datasets.")
        return

    # Extract splits
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    valid_texts = dataset['validation']['text']
    valid_labels = dataset['validation']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    print("\n<> Building Vocabulary...")
    vocab_builder = VocabBuilder(train_texts, max_vocab_size=15000)

    # Dataset Config
    # MUST BE POWER OF 2 for your Hierarchical Attention (e.g., 64, 128, 256)
    MAX_LEN = 128 
    BATCH_SIZE = 64

    print(f"\n<> Creating Datasets (Max Len: {MAX_LEN})...")
    train_dataset = TwitterDataset(train_texts, train_labels, vocab_builder, MAX_LEN)
    valid_dataset = TwitterDataset(valid_texts, valid_labels, vocab_builder, MAX_LEN)
    test_dataset = TwitterDataset(test_texts, test_labels, vocab_builder, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)

    # Initialize Model
    print("\n<> Initializing Transformer Sentiment Classifier...")
    model = TwitterSentimentTransformer(
        vocab_size=len(vocab_builder.word2idx),
        d_model=256,       # Tuned for Twitter (Short text, high noise)
        num_heads=4,
        d_ff=1024,
        num_layers=6,      # 6 Layers is standard for this complexity
        num_classes=3,     # Neg, Neu, Pos
        max_len=MAX_LEN,
        dropout=0.1
    )

    if torch.cuda.device_count() > 1:
        print(f"<> Utilizing {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)

    # Train
    train_sentiment_model(model, train_loader, valid_loader, num_epochs=20)

    # Final Test
    print("\n<> Running Final Test Evaluation...")
    # Reload best
    if torch.cuda.device_count() > 1:
        # Load state dict handling for DataParallel
        state_dict = torch.load("best_twitter_model.pt")
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load("best_twitter_model.pt"))
        
    criterion = nn.CrossEntropyLoss()
    test_acc, test_loss = evaluate(model, test_loader, criterion)
    print(f">> Final Test Accuracy: {test_acc:.2f}%")
    print(f">> Final Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()