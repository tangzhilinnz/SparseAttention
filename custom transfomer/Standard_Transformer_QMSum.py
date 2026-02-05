# -*- coding: utf-8 -*-
import os
# ==========================================
# 1. SETUP & IMPORTS
# ==========================================
# CRITICAL: Set GPU before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import collections
from datetime import timedelta
import time
import urllib.request

# Set device
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
# 2. DATA PROCESSING (QMSum Fixed)
# ==========================================

def download_qmsum_data():
    """
    Downloads QMSum JSONL files locally to avoid HTTP errors during load_dataset.
    """
    base_url = "https://raw.githubusercontent.com/Yale-LILY/QMSum/master/data/ALL/jsonl/"
    files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    save_dir = "qmsum_data"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    local_paths = {}
    print(f"\n<> Checking local QMSum data in './{save_dir}'...")
    
    for file in files:
        url = base_url + file
        path = os.path.join(save_dir, file)
        local_paths[file.split('.')[0]] = path
        
        if not os.path.exists(path):
            print(f"   Downloading {file} from {url}...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"   Error downloading {file}: {e}")
                print("   Please check your internet connection or URL.")
                raise e
        else:
            print(f"   Found {file}, skipping download.")
            
    # Map 'val' from filename to 'validation' for datasets standard
    return {
        "train": local_paths["train"],
        "validation": local_paths["val"], 
        "test": local_paths["test"]
    }

class EfficientVocabBuilder:
    def __init__(self, dataset_split, max_vocab_size=30000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3, '<SEP>': 4}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>', 4: '<SEP>'}
        self.build_vocab(dataset_split)
        
    def build_vocab(self, dataset_split):
        print("Counting words in QMSum (Streamed)...")
        word_counts = collections.Counter()
        
        for item in tqdm(dataset_split, desc="Building Vocab"):
            # 1. Add Transcripts
            if 'meeting_transcripts' in item:
                for turn in item['meeting_transcripts']:
                    word_counts.update(turn['speaker'].lower().split())
                    word_counts.update(turn['content'].lower().split())
            
            # 2. Add Queries & Answers
            if 'specific_query_list' in item:
                for qa in item['specific_query_list']:
                    word_counts.update(qa['query'].lower().split())
                    word_counts.update(qa['answer'].lower().split())
        
        most_common = word_counts.most_common(self.max_vocab_size - 5)
        for word, count in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        print(f"\nVocabulary Statistics: {len(self.word2idx)} words")

class QMSumDataset(Dataset):
    def __init__(self, dataset_split, vocab, max_src_len=4096, max_trg_len=512):
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len
        self.data = self._process_data(dataset_split)
        
    def _text_to_indices(self, text, max_len, add_eos=True):
        words = text.lower().split()
        unk_idx = self.vocab.word2idx['<UNK>']
        eos_idx = self.vocab.word2idx['<EOS>']
        
        ids = [self.vocab.word2idx.get(w, unk_idx) for w in words]
        if add_eos: ids.append(eos_idx)
        
        # Truncate
        if len(ids) > max_len:
            ids = ids[:max_len-1] + [eos_idx] if add_eos else ids[:max_len]
        # Pad
        if len(ids) < max_len:
            ids = ids + [self.vocab.word2idx['<PAD>']] * (max_len - len(ids))
            
        return torch.tensor(ids, dtype=torch.long)

    def _process_data(self, dataset_split):
        processed = []
        sep_token = " <sep> " 
        
        print("Formatting QMSum samples...")
        for item in tqdm(dataset_split):
            # 1. Flatten the Transcript once per meeting
            transcript = " context: "
            if 'meeting_transcripts' in item:
                for turn in item['meeting_transcripts']:
                    transcript += f"{turn['speaker']}: {turn['content']} "
            
            # 2. Create one sample for EACH Query in the meeting
            if 'specific_query_list' in item:
                for qa in item['specific_query_list']:
                    query = "query: " + qa['query']
                    full_src_text = query + sep_token + transcript
                    summary_text = qa['answer']
                    
                    src_tensor = self._text_to_indices(full_src_text, self.max_src_len)
                    trg_tensor = self._text_to_indices(summary_text, self.max_trg_len)
                    
                    processed.append({'src': src_tensor, 'trg': trg_tensor})
            
        return processed

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# 3. MODEL ARCHITECTURE (Standard Transformer)
# ==========================================

class StandardMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.shape[0]
        Q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot Product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # Mask shape (B, 1, 1, Seq) or (B, 1, Seq, Seq)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_proj(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = StandardMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, norm_x, norm_x, mask=mask)
        x = x + self.dropout(attn_out)
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = StandardMultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = StandardMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        norm_x = self.norm1(x)
        self_attn_out = self.self_attn(norm_x, norm_x, norm_x, mask=trg_mask)
        x = x + self.dropout(self_attn_out)
        norm_x = self.norm2(x)
        cross_attn_out = self.cross_attn(norm_x, enc_output, enc_output, mask=src_mask)
        x = x + self.dropout(cross_attn_out)
        norm_x = self.norm3(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        return x

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = self._create_pos_encoding(d_model, 5000)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
        
    def _create_pos_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def make_src_mask(self, src):
        # (B, 1, 1, Seq)
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def make_trg_mask(self, trg):
        # (B, 1, Seq, Seq)
        N, trg_len = trg.shape
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        x = self.embedding(src) * self.scale
        x = x + self.pos_encoding[:, :src.shape[1], :]
        x = self.dropout(x)
        for layer in self.encoder_layers: x = layer(x, mask=src_mask)
        enc_output = x
        x = self.embedding(trg) * self.scale
        x = x + self.pos_encoding[:, :trg.shape[1], :]
        x = self.dropout(x)
        for layer in self.decoder_layers: x = layer(x, enc_output, src_mask=src_mask, trg_mask=trg_mask)
        return self.fc_out(x)

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def validate_seq2seq(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            sos_tokens = torch.full((trg.shape[0], 1), 2, device=device, dtype=torch.long)
            trg_input = torch.cat([sos_tokens, trg[:, :-1]], dim=1)
            with torch.cuda.amp.autocast():
                output = model(src, trg_input)
                loss = criterion(output.reshape(-1, output.shape[-1]), trg.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def train_seq2seq(model, train_loader, valid_loader, num_epochs=20, patience=20):
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    train_losses, valid_losses, epoch_times = [], [], []
    best_valid_loss = float('inf')
    total_start_time = time.time()
    patience_counter = 0
    
    print(f"\n{'='*50}")
    print(f"<> Training QMSum (Standard Encoder-Decoder)")
    print(f"<> Patience: {patience}")
    print(f"{'='*50}")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress:
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            sos_tokens = torch.full((trg.shape[0], 1), 2, device=device, dtype=torch.long)
            trg_input = torch.cat([sos_tokens, trg[:, :-1]], dim=1)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(src, trg_input)
                loss = criterion(output.reshape(-1, output.shape[-1]), trg.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            progress.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        avg_valid_loss = validate_seq2seq(model, valid_loader, criterion)
        epoch_duration = time.time() - epoch_start_time
        
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        epoch_times.append(epoch_duration)
        
        print(f"Summary: Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_qmsum_model.pt")
            print(f"  > New Best Model Saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n<> Patience limit reached ({patience_counter}). Stopping.")
                break

    # --- PLOTTING ---
    total_time = time.time() - total_start_time
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('QMSum Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epoch_times, color='orange', marker='o')
    plt.title('Epoch Duration')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('qmsum_training_history.png')
    plt.close()

    print(f"\n{'='*50}")
    print("<> Training Complete!")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")

# ==========================================
# 5. EXECUTION
# ==========================================
# 1. Download data locally
data_files = download_qmsum_data()

# 2. Load from local files
print("\n<> Loading QMSum Dataset from local files...")
dataset = load_dataset("json", data_files=data_files)

vocab_builder = EfficientVocabBuilder(dataset['train'], max_vocab_size=30000)
train_dataset = QMSumDataset(dataset['train'], vocab_builder)
valid_dataset = QMSumDataset(dataset['validation'], vocab_builder)

BATCH_SIZE = 4 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, drop_last=True)

print("\n<> Initializing Model...")
model = TransformerSeq2Seq(
    vocab_size=len(vocab_builder.word2idx),
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    dropout=0.2,
    pad_idx=0
)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
model = model.to(device)

train_seq2seq(model, train_loader, valid_loader, num_epochs=20, patience=20)