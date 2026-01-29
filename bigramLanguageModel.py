import torch
import torch.nn as nn
from torch.nn import functional as F
from head import MultiHeadAttention

class BigramLanguageModel(nn.Module):

    def __init__(self, block_size, device, vocab_size, n_embd, head_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, 4, block_size),
            Block(n_embd, 4, block_size),
            Block(n_embd, 4, block_size)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.device = device
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Create Batch * time * channels tensor of logits
        token_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = token_emb + pos_emb # (B, T, C) --> holds token identites and positions that they occur
        x = self.blocks(x) # apply multi headed self attention
        logits = self.lm_head(x) # (B, T, vocab_size) (4, 8, 65)

        if targets is None:
            loss = None
        else:
            # pytorch wants channels to be the second dimension for this operation --> (B, C, T)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # Calculate how well we are calculating targets based on the logits 
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Take (B, T) and generate new tokens up to max_new_tokens length
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # apply softmax --> get probabilities for each token
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
# Transformer block
class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, block_size, head_size, n_embd)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x

# Simple feed forward layer --> Tokens "think" on data individually
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)