import numpy as np
import torch
import torch.nn as nn

class LinformerSelfAttention(nn.Module):
    def __init__(self, input_dim, seq_len, num_heads, k=32, dropout=0.1):
        super(LinformerSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.k = k  # Low-rank projection dimension
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        # Linear projections for queries, keys, and values
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # Low-rank projection matrices for keys and values
        self.proj_key = nn.Linear(seq_len, seq_len)  # 수정: 투영 차원을 원래 시퀀스 길이와 일치시킴
        self.proj_value = nn.Linear(seq_len, seq_len)  # 수정: 투영 차원을 맞춤

        # Output projection
        self.out = nn.Linear(input_dim, input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, seq_len, _ = x.size()

        # Project to queries, keys, and values
        queries = self.query(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply low-rank projection to keys and values
        keys = self.proj_key(keys).transpose(1, 2)  # 투영 후 크기 맞춤
        values = self.proj_value(values).transpose(1, 2)  # 투영 후 크기 맞춤

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Attention output
        output = torch.matmul(attn, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)

        return self.out(output)

class PerformerSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, kernel_size=32, dropout=0.1):
        super(PerformerSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.kernel_size = kernel_size
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        # Linear projections for queries, keys, and values
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # Output projection
        self.out = nn.Linear(input_dim, input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def feature_map(self, x):
        # Random Fourier feature mapping (or kernel approximation)
        return torch.exp(-x ** 2 / 2)

    def forward(self, x):
        bsz, seq_len, _ = x.size()

        # Project to queries, keys, and values
        queries = self.feature_map(self.query(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2))
        keys = self.feature_map(self.key(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2))
        values = self.value(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform efficient attention
        kv = torch.einsum('bhse,bhsc->bhsec', keys, values)
        qkv = torch.einsum('bhse,bhsec->bhsc', queries, kv)

        # Attention output
        output = qkv.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.out(output)

class Time2Vec(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        periodic_dim = (input_dim-1) // 2
        self.linear = nn.Linear(input_dim, input_dim - periodic_dim*2)
        self.periodic = nn.Linear(input_dim, periodic_dim)
    
    def forward(self, x):
        linear_out = self.linear(x)
        periodic_sin = torch.sin(self.periodic(x))
        periodic_cos = torch.cos(self.periodic(x))  # cosine 추가
        periodic_out = torch.cat([periodic_sin, periodic_cos], dim=-1)  # sin과 cos 결합
        return torch.cat([linear_out, periodic_out], dim=-1)

# Define Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout, method='multihead', seq_len=None):
        super().__init__()
        self.method = method
        
        if method == 'multihead':
            self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        elif method == 'linformer':
            self.attention = LinformerSelfAttention(input_dim, seq_len, num_heads, dropout=dropout)
        elif method == 'performer':
            self.attention = PerformerSelfAttention(input_dim, num_heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.method=='multihead':
            attended, _ = self.attention(x, x, x) # 1. Attention
        else:
            attended = self.attention(x)
        x = self.norm1(attended + x)              # 2. 잔차 연결 + Layer Normalization
        feedforward = self.ff(x)                  # 3. Feedforward + BatchNorm 적용
        x = self.norm2(feedforward + x)           # 4. 잔차 연결 + Layer Normalization
        return x

# Define main model architecture
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_len = 10
        self.d_model = config.d_model
        self.hidden_size = config.hidden_size
        self.dropout_rate = config.dropout
        self.num_heads = config.num_heads
        self.method = config.method
        self.seq_len = config.seq_len
        self.num_layers = config.num_layers
        self.pred_len = config.pred_len

        self.time2vec = Time2Vec(self.d_model)
        self.embedding = nn.Linear(self.d_model, self.hidden_size)
        self.position_encoding = self.generate_position_encoding(self.hidden_size, self.max_len)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.hidden_size, self.num_heads, self.dropout_rate, self.method, self.seq_len) 
            for _ in range(self.num_layers)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.pred_len)
        )

    def generate_position_encoding(self, hidden_size, max_len):
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        b, s, f = x.shape
        x = self.time2vec(x)
        x = self.embedding(x)
        x = x + self.position_encoding[:, :s, :].to(x.device)
        x = self.dropout(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.mean(dim=1)
        x = self.output_layer(x)
        return x