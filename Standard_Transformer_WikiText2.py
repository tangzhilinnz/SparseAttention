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
import os
import gc
import re

# FORCE GPU 3 SELECTION
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

# OPTIMIZATION: Basic Tokenizer to separate punctuation
def basic_tokenizer(text):
    # Splits words and punctuation (e.g. "hello," -> "hello", ",")
    return re.findall(r"[\w']+|[.,!?;]", text.lower())

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
                # OPTIMIZED: Use regex tokenizer instead of .split()
                words = basic_tokenizer(text)
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
                # OPTIMIZED: Use regex tokenizer
                words = basic_tokenizer(text)
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
# Keeping at 128 (Safe Floor)
MAX_LEN = 512
train_dataset = WikiTextDataset(train_texts, vocab_builder, max_len=MAX_LEN)
valid_dataset = WikiTextDataset(valid_texts, vocab_builder, max_len=MAX_LEN)
test_dataset = WikiTextDataset(test_texts, vocab_builder, max_len=MAX_LEN)

# Create dataloaders
# HYPERPARAMETER: batch_size
# Keeping at 16 (High updates)
batch_size = 16
# CHANGED: shuffle=True 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None, return_attention=False):
        batch_size = query.shape[0]
        
        # 1. Project and Reshape [Batch, Seq, Heads, Dim] -> [Batch, Heads, Seq, Dim]
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Flash Attention (Optimized Kernel for A100)
        output = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=mask, 
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # 3. CRITICAL FIX: Reshape back to [Batch, Seq_Len, d_model]
        output = output.transpose(1, 2).contiguous() # [Batch, Seq, Heads, Dim]
        output = output.view(batch_size, -1, self.d_model) # Flatten heads
        
        # 4. Final Projection
        output = self.out_proj(output)
        
        if return_attention:
            # FlashAttention doesn't return weights easily, returning None to avoid breaking loop.
            return output, None 
        
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # OPTIMIZATION: Switched to GELU for better Transformer convergence
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

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
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, trg_mask=None, return_attention=False):
        # PRE-LAYER NORMALIZATION (Apply Norm BEFORE Attention)
        
        # 1. Norm -> Attention -> Add
        norm_x = self.norm1(x)
        if return_attention:
            attn_output, self_attn_weights = self.self_attn(norm_x, norm_x, norm_x, mask=trg_mask, return_attention=True)
        else:
            attn_output = self.self_attn(norm_x, norm_x, norm_x, mask=trg_mask)
            
        x = x + self.dropout(attn_output) # Residual
        
        # 2. Norm -> FeedForward -> Add
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output) # Residual
        
        if return_attention:
            return x, self_attn_weights
        return x

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
# 3. TRAINING LOOP (Added Auto-Exit / Early Stopping)
# ==========================================

def train_transformer_model(model, train_loader, valid_loader, criterion=None, num_epochs=100, learning_rate=3e-4, patience=10):
    if criterion is None:
        # OPTIMIZATION: NO Label Smoothing for PPL tasks
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    device = next(model.parameters()).device
    
    print(f"\n{'='*50}")
    print(f"<> Training Transformer Model")
    print(f"{'='*50}")
    
    # OPTIMIZATION: High Weight Decay to punish complexity
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    # CHANGED: Use OneCycleLR for better convergence
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        total_steps=total_steps,
        pct_start=0.2, # Warmup slightly faster
        div_factor=10,
        final_div_factor=1000
    )
    
    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    epoch_times = []
    
    # --- EARLY STOPPING VARIABLES ---
    patience_counter = 0
    
    # Gradient Accumulation Steps 
    # KEPT AT 1 to ensure effective batch size remains small (16) for regularization
    accumulation_steps = 1 

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
            
            # --- IMPROVEMENT: Only pass input_ids (Decoder Only) ---
            outputs = model(input_ids)
            
            output_dim = outputs.shape[-1]
            outputs = outputs.contiguous().view(-1, output_dim)
            labels = labels.contiguous().view(-1)
            
            loss = criterion(outputs, labels)
            
            # Divide loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step() # Step per batch for OneCycle
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
                
                # --- IMPROVEMENT: Only pass input_ids ---
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

# OPTIMIZED HYPERPARAMETERS ("TINY" MODEL FOR WIKITEXT-2)
# Drastically reduced d_model to 256 to close the overfitting gap
model = TransformerLM(
    vocab_size=vocab_size,
    d_model=256,        # DOWN from 512. Essential for small data.
    num_heads=4,        # 256 / 4 = 64
    d_ff=1024,          # 4 * 256
    num_layers=4,       # DOWN from 6. Reduces depth.
    dropout=0.4         # UP from 0.3. High regularization.
)

# ENABLE MULTI-GPU SUPPORT (DataParallel)
if torch.cuda.device_count() > 1:
    print(f"<> Utilizing {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

print(f"Using device: {device}")

# Training parameters
num_epochs = 100     
learning_rate = 3e-4 # Standard

print("\n<> Starting Training...")
results = train_transformer_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=None, 
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    patience=10 # Higher patience for smaller batches
)

# Run Final Evaluation on Test Set
print("\n<> Loading Best Model for Evaluation...")
checkpoint = torch.load('best_transformer_wikitext.pt')
# We need to load state dict carefully depending on if DataParallel was used during save
# My training loop logic saves model.module if available, so we load into a fresh instance
best_model = TransformerLM(
    vocab_size=vocab_size,
    d_model=256,        # Match "Tiny" model
    num_heads=4,        
    d_ff=1024,          
    num_layers=4,         
    dropout=0.4
)
best_model.load_state_dict(checkpoint['model_state_dict'])
best_model = best_model.to(device)

if torch.cuda.device_count() > 1:
     best_model = nn.DataParallel(best_model)

evaluate_test_set(best_model, test_loader)