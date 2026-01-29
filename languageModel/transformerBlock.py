import torch.nn as nn
from languageModel.head import MultiHeadAttention

class TransformerBlock(nn.Module):
    """
        Implementation of a decode only transformer block as outlined in "Attention Is All You Need"
    """

    def __init__(self, n_heads, block_size, n_embd, dropout):
        """
            Initializes a decode only transformer block with feed-forward and layer-normalization
            capabilities.

            Args:
                n_heads int: Nubmer of heads in the attention block
                block_size int: Maximum nubmer of tokens processed at once
                n_embd int: number of dimensions in the embedding
                dropout float32: percentage of results to "dropout" to maintain evolution --> See: https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b4
        """
        super().__init__()
        self.sa = MultiHeadAttention(n_heads, block_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        """
            Performs the Multi-Head Attantion step and the preceding and proceeding layer normalization

            Args:
                x tensor: the input tensor for the transformer
        """
        x = x + self.sa(self.ln1(x)) # Performs the layer normalization step before the Multi-Head Attention step
        x = x + self.ffwd(self.ln2(x)) # Performs the layer normalization step after the Multi-Head Attention step
        return x

class FeedForward(nn.Module):
    """
        Simple feed forward implementation. Ensures that the context from previous tokens in
        the batch are forwarded to the tokens ahead.
    """

    def __init__(self, n_embd, dropout):
        """
            Initializes a simple feed forward with ReLU and dropout. Explained on pg 5. section 3.3
            in "Attention Is All You Need"

            Args:
                n_embd int: number of dimensions in the embedding
                dropout float32: percentage of results to "dropout" to maintain evolution
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
            Performs the feed forward operation

            Args:
                x tensor: the input tensor for the feed forward operation
        """
        return self.net(x)
