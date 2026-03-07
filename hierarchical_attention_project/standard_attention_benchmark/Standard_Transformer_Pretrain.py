# -*- coding: utf-8 -*-

import sys
import os
# ==========================================
# CRITICAL FIX: GPU SELECTION MUST BE FIRST
# ==========================================
# Set this before importing torch or calling torch.cuda to avoid OOM
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ==========================================
# NEW LIBRARY IMPORT
# ==========================================
# 1. Get the folder where this script lives
# (.../hierarchical_attention_project)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the PARENT folder
# (.../workspace/SparseAttention)
parent_dir = os.path.dirname(script_dir)

# 3. Add the PARENT folder to Python's search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Debug: Prove we can see it
print(f"Script location: {script_dir}")
print(f"Parent location: {parent_dir}")
print(f"Checking for library: {os.path.join(parent_dir, 'hierarchical_attention')}")

# REMOVED: hierarchical_attention and generate_span_input_Y imports
from torch.nn.utils.rnn import pad_sequence # <--- INJECTED FOR DYNAMIC PADDING

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

# --- INJECTED: HUGGINGFACE TOKENIZER ---
from transformers import AutoTokenizer

# Data manipulation and utilities
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from datetime import timedelta
from torch.autograd import Variable

import math
import gc

# ==========================================
# PIPELINE TOGGLE
# ==========================================
# Change to "FINETUNE" to run the QMSum Downstream task
RUN_MODE = "PRETRAIN" 

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
# --- GLOBAL TOKENIZER INITIALIZATION ---
# Initialize GPT-2 BPE Tokenizer for translation/downstream compatibility
# ==========================================
print("\n<> Initializing GPT-2 Tokenizer...")
global_tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=1000000)
if global_tokenizer.pad_token is None:
    global_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
GLOBAL_PAD_TOKEN_ID = global_tokenizer.pad_token_id

# ==========================================
# 1. DATA PROCESSING (WikiText-103 Optimized)
# ==========================================
import collections

# NOTE: Left your original builder here so you don't lose the code, 
# but the pipeline now uses the BPE tokenizer below for better generalization.
class EfficientVocabBuilder:
    def __init__(self, dataset_split, max_vocab_size=50000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.build_vocab(dataset_split)
        
    def build_vocab(self, dataset_split):
        print("Counting words in dataset (Streamed)...")
        word_counts = collections.Counter()
        
        # Iterate without loading everything to RAM
        for item in tqdm(dataset_split, desc="Building Vocab"):
            text = item.get('text', item.get('document', ''))
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
    # UPDATED: Swapped 'vocab' for 'tokenizer'
    def __init__(self, dataset_split, tokenizer, max_len=4096, block_size=1000000):
        """
        Args:
            block_size: Number of tokens to process before converting to Tensor
                        (prevents Python list RAM explosion)
        """
        self.tokenizer = tokenizer
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
        
        for item in tqdm(dataset_split, desc="Tokenizing"):
            text = item['text']
            if len(text.strip()) > 0:
                # UPDATED: Use BPE Tokenizer encoding
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                ids.append(self.tokenizer.eos_token_id)
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

# ==========================================
# --- MODULE 1: PROMPT ABSTRACTION & COLLATOR ---
# ==========================================
class InstructionDataset(Dataset):
    # UPDATED: Swapped 'vocab' for 'tokenizer' and default max_len to 4096
    def __init__(self, dataset_split, tokenizer, max_len=4096):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        print(f"Formatting data for Downstream Task...")
        for item in tqdm(dataset_split, desc="Tokenizing Prompts"):
            # Handles QMSum mapping
            source_text = item.get('document', item.get('meeting_transcript', '')) 
            query = item.get('query', 'Summarize this meeting.')
            target_text = item.get('summary', '')
            
            prompt = f"Query: {query}\n\nTranscript:\n{source_text}\n\nSummary:\n{target_text}"
            
            # UPDATED: Use BPE Tokenizer with truncation
            ids = self.tokenizer.encode(prompt, truncation=True, max_length=max_len-1)
            ids.append(self.tokenizer.eos_token_id)
            
            self.data.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {
            'input_ids': chunk[:-1],
            'label': chunk[1:]
        }

def dynamic_padding_collate_fn(batch):
    """ Pads sequences to the max length of the current batch only """
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]
    PAD_TOKEN_ID = GLOBAL_PAD_TOKEN_ID # UPDATED: Uses the tokenizer's specific pad ID
    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=PAD_TOKEN_ID)
    return {'input_ids': padded_inputs, 'label': padded_labels}
# ==========================================


# --- LOAD DATA (TOGGLED BY MODE) ---
if RUN_MODE == "PRETRAIN":
    print("\n<> Loading WikiText-103 Dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # UPDATED: Using GPT2 Vocab length
    vocab_size = len(global_tokenizer)

    print("\n<> Processing Training Data (This may take 1-2 mins)...")
    MAX_LEN = 4096 # UPDATED: Set to 4096 for long-context prep
    train_dataset = LargeScaleWikiTextDataset(dataset['train'], global_tokenizer, max_len=MAX_LEN)
    print("\n<> Processing Validation Data...")
    valid_dataset = LargeScaleWikiTextDataset(dataset['validation'], global_tokenizer, max_len=MAX_LEN)
    print("\n<> Processing Test Data...")
    test_dataset = LargeScaleWikiTextDataset(dataset['test'], global_tokenizer, max_len=MAX_LEN)

    batch_size = 8 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=4)

elif RUN_MODE == "FINETUNE":
    print("\n<> Loading QMSum Dataset...")
    dataset = load_dataset("tau/qmsum")
    
    # UPDATED: Using GPT2 Vocab length
    vocab_size = len(global_tokenizer)
    
    MAX_LEN = 4096 # UPDATED: Set to 4096 to capture long meeting transcripts
    train_dataset = InstructionDataset(dataset['train'], global_tokenizer, max_len=MAX_LEN)
    valid_dataset = InstructionDataset(dataset['validation'], global_tokenizer, max_len=MAX_LEN)
    test_dataset = InstructionDataset(dataset['test'], global_tokenizer, max_len=MAX_LEN)

    # Note: Using dynamic_padding_collate_fn and shuffling for finetuning
    batch_size = 16 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=dynamic_padding_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, num_workers=4, collate_fn=dynamic_padding_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=4, collate_fn=dynamic_padding_collate_fn)


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================

# ==========================================
# --- MODULE 2: LORA ADAPTATION ---
# ==========================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.pretrained = nn.Linear(in_features, out_features)
        self.pretrained.weight.requires_grad = False
        if self.pretrained.bias is not None:
            self.pretrained.bias.requires_grad = False
            
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        frozen_out = self.pretrained(x)
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return frozen_out + lora_out

def inject_lora_and_freeze(model):
    print("\n<> Injecting LoRA Matrices and Freezing Base Weights...")
    for param in model.parameters():
        param.requires_grad = False
        
    for layer in model.layers:
        in_feat1 = layer.feed_forward.fc1.in_features
        out_feat1 = layer.feed_forward.fc1.out_features
        layer.feed_forward.fc1 = LoRALinear(in_feat1, out_feat1, r=8)
        
        in_feat2 = layer.feed_forward.fc2.in_features
        out_feat2 = layer.feed_forward.fc2.out_features
        layer.feed_forward.fc2 = LoRALinear(in_feat2, out_feat2, r=8)
    
    # REMOVED: Gated delta projection logic. Only standard LoRA weights are trainable.
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    return model
# ==========================================

# ==========================================
# MultiHeadAttention using standard FlashAttention
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
        
        # 1. Project and Reshape
        # Resulting shape: [Batch, Heads, SeqLen, HeadDim]
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Standard PyTorch FlashAttention Implementation
        if not return_attention:
            output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None, 
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True # We rely on internal causal masking
            )
            attn_weights = None 
            
        else:
            # 3. Fallback for Visualization
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
            
            if mask is not None:
                min_value = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(mask == 0, min_value)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, V)

        # 4. Reassemble Heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_weights
        
        return output

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
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, trg_mask=None, return_attention=False):
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

# --- STANDARD TRANSFORMER LM ---
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

    def forward(self, trg, return_attention=False):
        x = self.embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        trg_mask = self.make_causal_mask(trg)

        attentions = [] if return_attention else None
        
        # CLEANUP: Removed y-stream injection logic. Iterating purely over x now.
        for layer in self.layers:
            if return_attention:
                x, attention = layer(x, trg_mask=trg_mask, return_attention=True)
                attentions.append(attention)
            else:
                x = layer(x, trg_mask=trg_mask)
        
        output = self.fc_out(x)
        if return_attention:
            return output, attentions
        return output
    
    def generate(self, src, start_token=2, max_len=50, temperature=1.0):
        self.eval()
        device = next(self.parameters()).device
        
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
        criterion = nn.CrossEntropyLoss(ignore_index=GLOBAL_PAD_TOKEN_ID, label_smoothing=0.1)
    
    device = next(model.parameters()).device
    
    print(f"\n{'='*50}")
    print(f"<> Training Transformer Model (AMP Enabled)")
    print(f"{'='*50}")
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=False)
    scaler = torch.cuda.amp.GradScaler() 

    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    epoch_times = []
    
    patience_counter = 0
    accumulation_steps = 15 

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
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
                outputs = model(input_ids)
                
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
                    outputs = model(input_ids)
                    
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

def evaluate_test_set(model, test_loader):
    print(f"\n{'='*50}")
    print(f"<> FINAL TEST EVALUATION")
    print(f"{'='*50}")
    
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    
    eval_criterion = nn.CrossEntropyLoss(ignore_index=GLOBAL_PAD_TOKEN_ID)
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
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
if RUN_MODE == "PRETRAIN":
    print("\n<> Initializing Transformer Model...")
    print(f"Actual Vocab Size: {vocab_size}")

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=768,          
        num_heads=12,         
        d_ff=3072,            
        num_layers=12,        
        dropout=0.1           
    )

    model = model.to(device)

    num_epochs = 50        
    learning_rate = 4e-4   

    print("\n<> Starting Training on WikiText-103...")
    results = train_transformer_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=None, 
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        patience=50 
    )


# ==========================================
# 5. ADVANCED EVALUATION (Sliding Window)
# ==========================================

def evaluate_wikitext_103(model, test_loader, device, sliding_window=False, stride=512):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=GLOBAL_PAD_TOKEN_ID)
    
    print(f"\n{'='*50}")
    if sliding_window:
        print(f"<> Starting SLIDING WINDOW Evaluation (Stride={stride})...")
        print(f"<> Note: This provides the most accurate Perplexity.")
    else:
        print(f"<> Starting STANDARD CHUNKED Evaluation...")
    print(f"{'='*50}")

    if not sliding_window:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids)
                
                shift_logits = outputs.view(-1, outputs.size(-1))
                shift_labels = labels.view(-1)
                
                loss = criterion(shift_logits, shift_labels)
                total_loss += loss.item()
                total_tokens += (shift_labels != GLOBAL_PAD_TOKEN_ID).sum().item()

    else:
        raw_data = []
        print(">> Flattening test data for sliding window...")
        for batch in test_loader:
            raw_data.append(batch['input_ids'].cpu())
        
        full_seq = torch.cat(raw_data).view(-1).to(device)
        
        max_len = 4096

        with torch.no_grad():
            for i in tqdm(range(0, len(full_seq) - max_len, stride), desc="Sliding Window"):
                input_window = full_seq[i : i + max_len].unsqueeze(0) 
                
                target_window = full_seq[i+1 : i + max_len + 1].unsqueeze(0)

                outputs = model(input_window) 

                logits_stride = outputs[:, -stride:, :]
                labels_stride = target_window[:, -stride:]

                loss = criterion(logits_stride.reshape(-1, logits_stride.size(-1)), 
                                 labels_stride.reshape(-1))
                
                total_loss += loss.item()
                total_tokens += labels_stride.numel()

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
if RUN_MODE == "PRETRAIN":
    print("\n<> Training Finished. Loading Best Model for Evaluation...")

    best_model = TransformerLM(
        vocab_size=vocab_size,
        d_model=768,
        num_heads=12,
        d_ff=3072,
        num_layers=12,
        dropout=0.1 
    )

    checkpoint_path = 'best_transformer_wikitext.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') 
            new_state_dict[name] = v
            
        best_model.load_state_dict(new_state_dict)
        print(f"<> Loaded checkpoint from epoch {checkpoint['epoch']} (Loss: {checkpoint['loss']:.4f})")
    else:
        print(f"<> Warning: Checkpoint {checkpoint_path} not found. Using current model state.")
        best_model = model

    best_model = best_model.to(device)

    print("\nRunning Standard Evaluation...")
    evaluate_wikitext_103(best_model, test_loader, device, sliding_window=False)

    print("\nRunning Sliding Window Evaluation (This will take longer)...")
    evaluate_wikitext_103(best_model, test_loader, device, sliding_window=True, stride=512)

# ==========================================
# 7. DOWNSTREAM FINETUNING EXECUTION
# ==========================================
elif RUN_MODE == "FINETUNE":
    print("\n<> Starting FINETUNING Phase on Downstream Task...")
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=768, 
        num_heads=12,          
        d_ff=3072,
        num_layers=12,        
        dropout=0.1         
    )
    
    checkpoint_path = 'best_transformer_wikitext.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k.replace('module.', '') 
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print(f"<> Loaded pretrained weights from {checkpoint_path}")
    else:
        print(f"<> WARNING: Pretrained weights not found. Fine-tuning from random initialization.")

    model = inject_lora_and_freeze(model)
        
    model = model.to(device)

    num_epochs = 5        
    learning_rate = 5e-5  

    print("\n<> Starting Fine-Tuning Training Loop...")
    results = train_transformer_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=None, 
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        patience=3 
    )
    
    evaluate_test_set(model, test_loader)