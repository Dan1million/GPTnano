import torch
import torch.nn as nn
from torch.nn import functional as F

# Multiple Heads of self attention
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, block_size, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, head_size, n_embd) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim = -1)

# Single head of self attention
class Head(nn.Module):
    
    def __init__(self, block_size, head_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores using Attention(Q, K, V) = softmax(QK/sqrt(head_size)) * V
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # masking weights --> set top right to -inf. Keeps from channging relative weights --> prevents using forward nodes to provide context "encoder block"
        wei = F.softmax(wei, dim=-1)

        # perform weighted aggregation --> the "V" at the end
        v = self.value(x)
        out = wei @ v # (B, T, C)
        return out
        