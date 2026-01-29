import torch;
import torch.nn as nn
from torch.nn import functional as F
import bigramLanguageModel

# Parameters
block_size = 8 # Maximum context length
batch_size = 32 # Maximum number of parallel executions
max_iters = 5000 # Number of learning iterations
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
head_size = 32

# set torch seed
torch.manual_seed(1337)

# Read data set
with open('input.txt', 'r', encoding='utf-8') as f :
    text = f.read()

# Get unique characters in the text
chars = sorted(list(set(text))) # create the vocabulary by parsing the unique characters
vocab_size = len(chars)
# Create mappint to convert characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda string: [stoi[c] for c in string] # encoder to convert string to a list of integers representing the characters
decode = lambda list: ''.join([itos[i] for i in list]) # decoder to convert a list of integers to it's string equivalent

# Create a training dataset and validation dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n] # 90% to train
val_data = data[n:] # 10% to test

def get_batch(split):
    # generate a small batch of inputs x and outputs y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random offset in training set
    x = torch.stack([data[i:i+block_size] for i in ix]) # block of characters chunk 'training data for tensor'
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # off by one block of characters chunk 'correct predictions'
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # we will not call backward for back propogation in this function --> removes memory
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create language model and offload to GPU if possible
model = bigramLanguageModel.BigramLanguageModel(block_size, device, vocab_size, n_embd, head_size)
m = model.to(device)

# Create a pytorch optimizer --> gradient descent!
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample the training data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx = context, max_new_tokens=500)[0].tolist()))
