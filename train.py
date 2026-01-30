import json
import os
import torch
from datetime import datetime
from languageModel.bigramLanguageModel import BigramLanguageModel
from tokenizer.tokenizer import Tokenizer

# Parameters From Config File
with open('config/config.json', 'r') as configuration:
    config_data = json.load(configuration)

block_size = config_data['block_size'] # Maximum context length
n_embd = config_data['n_embd'] # Number of embedding dimensions to use for embeddings
n_layer = config_data['n_layer'] # Number of transformers used in the language model
n_heads = config_data['n_heads'] # Number of heads in each multi-headed attention block
dropout = config_data['dropout'] # Dropout percentage to maintain evolution

# Training Specific Parameters
batch_size = 256 # Maximum number of parallel executions
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Device to run the language model on
eval_interval = 300 # Number of iterations break before outputing evaluation
eval_iters = 200 # Number of batches to evaluate in the evaluation step
learning_rate = 3e-4 # Learning rate
max_iters = 5000 # Number of learning iterations


# Read in the data set
with open(f'datasets/{config_data['dataset']}', 'r', encoding='utf-8') as f :
    text = f.read()


# Split the dataset into a training dataset and validation dataset
tokenizer = Tokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.8*len(data))
train_data = data[:n] # 90% to train
val_data = data[n:] # 10% to test


def get_batch(dataset):
    """
        Randomly samples the training or validation data set

        Args:
            dataset string: 'train' if usign the training data set, 'val' if using the validation training set
    """
    data = train_data if dataset == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Create a random offset for a batch
    x = torch.stack([data[i:i+block_size] for i in ix]) # Create block of characters chunk 'training data for tensor'
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Create off by one block of characters chunk 'correct predictions'
    x, y = x.to(device), y.to(device) # Move the data set to the device
    return x, y

@torch.no_grad() # we will not call backward for back propogation in this function --> reduces memory footprint
def estimate_loss():
    """
        Esitmates the current training data loss and validation data loss
    """
    out = {}
    model.eval()
    for dataset in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(dataset)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[dataset] = losses.mean()
    model.train()
    return out

# Create the bigram language model and offload to GPU if possible
model = BigramLanguageModel(block_size, device, dropout, n_embd, n_heads, n_layer, tokenizer.vocab_size())
m = model.to(device)

# Output number of parameters to command line
print(sum(p.numel() for p in m.parameters()), 'Parameters')

# Using the AdamW pytorch optimizer --> performs the gradient descent calculation gradient descent
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0: # Occasionaly output current state of training
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample the training data
    xb, yb = get_batch('train')

    # Evaluate the loss and perform gradient descent
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Create a directory for the current time and save the configuration and trained result
date_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(f'savedResults/{date_string}', exist_ok=True)
torch.save(m.state_dict(), f'savedResults/{date_string}/result.pt')

with open(f'savedResults/{date_string}/config.json', 'w') as json_file:
    json.dump(config_data, json_file, indent=4)
