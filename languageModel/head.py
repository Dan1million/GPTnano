import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    """
        Multi-headed self attention block that supports configuring the number of heads,
        size of the heads, number of embeddings (dimensions in the embedding), and a 
        dropout value (helps negate overfitting).

        Implementation of "Attention Is All You Need" pg. 3 with only the Decoder block

        Note: This is a decoder only implmeentation since we are not working with a data
        set that requires any kind of additional encoding.
    """

    def __init__(self, n_heads, block_size, n_embd, dropout):
        """
            Initializes a multi-headed attention block
        
            Args:
                n_heads int: Nubmer of heads in the attention block
                block_size int: Maximum nubmer of tokens processed at once
                n_embd int: number of dimensions in the embedding
                dropout float32: percentage of results to "dropout" to maintain evolution --> See: https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b4
        """
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(block_size, head_size, n_embd, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        """
            Concatenates the multi headed outputs into a single tensor. Projects the output
            To allow the heads to forward communicate with eachother.

            Args:
                x tensor: the input tensor for the multi-headed self-attention block
        """
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out

class Head(nn.Module):
    """
        Single head of self-attention as outlined in "Attention Is All You Need" pg. 4
        Performs the scaled dot-product attention operation as outlined in the 
        Attention(Q, K, V) equation
    """

    def __init__(self, block_size, head_size, n_embd, dropout):
        """
            Initializes a single head of a self attention

            Args:
                block_size int: Maximum nubmer of tokens processed at once
                head_size int: the size of this individual attention head
                n_embd int: number of dimensions in the embedding
                dropout float32: percentage of results to "dropout" to maintain evolution
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # initializes the 1's triangle that enforces only using context previous to the current token
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
            The real bread and butter of the self attention block. Performs the now
            famous Attention(Q, K, V) calculation

            Args:
                x tensor: the input tensor for the self-attention block
        """
        B,T,C = x.shape # (B,T,C) --> (Batch Size, Sequence Length, Embedding Size)
        k = self.key(x) # (B, T, C) --> Get the K value
        q = self.query(x) # (B, T, C) --> Get the Q value

        # Compute attention scores using Attention(Q, K, V) = softmax(QK/sqrt(head_size)) * V
        wei = q @ k.transpose(-2, -1) * C**-0.5 # Matrix Multiplication Step
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Masking weights --> set top right triangle to -inf to avoid altering relative weights
        wei = F.softmax(wei, dim=-1) # Perform the softmax
        wei = self.dropout(wei) # Perform dropout to avoid overfitting

        # Perform weighted aggregation --> the "V" at the end
        v = self.value(x)
        out = wei @ v
        return out
