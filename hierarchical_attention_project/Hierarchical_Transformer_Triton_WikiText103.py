# -*- coding: utf-8 -*-

import os
# ==========================================
# CRITICAL FIX: GPU SELECTION MUST BE FIRST
# ==========================================
# Set this before importing torch or calling torch.cuda to avoid OOM
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ==========================================
# NEW LIBRARY IMPORT
# ==========================================
from hierarchical_attn import HierarchicalAttention

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
        # [MODIFIED] Use the class from the new library
        self.self_attn = HierarchicalAttention(d_model, num_heads, dropout)
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
        # [MODIFIED] Calls the library method
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
        # [MODIFIED] Use HierarchicalAttention static method
        y = HierarchicalAttention.generate_span_input_Y(x)
        
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