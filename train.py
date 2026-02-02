import argparse
import json
import os
import torch
from datetime import datetime
from languageModel.bigramLanguageModel import BigramLanguageModel
from tokenizer.tokenizer import Tokenizer

# CLI Parameters
parser = argparse.ArgumentParser(description="Train a Bigram Language Model")
parser.add_argument('--config', type=str, default='config/config.json', help='Path to the configuration file')
parser.add_argument('--dataset', type=str, default='datasets/tinyShakespeare.txt', help='Path to the dataset file')
parser.add_argument('--output', type=str, default=f'savedResults/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', help='Directory to save the trained model, cofiguration, and checkpoints')
args = parser.parse_args()

# Parameters From Config File
with open(args.config, 'r') as configuration:
    config_data = json.load(configuration)

batch_size = config_data['batch_size'] # Maximum number of parallel executions
block_size = config_data['block_size'] # Maximum context length
checkpoint_interval = config_data['checkpoint_interval'] # Number of iterations between saving model checkpoints
eval_interval = config_data['eval_interval'] # Number of iterations break before outputing evaluation
eval_iters = config_data['eval_iters'] # Number of batches to evaluate in the evaluation step
learning_rate = config_data['learning_rate'] # Learning rate
max_iters = config_data['max_iters'] # Number of learning iterations
n_embd = config_data['n_embd'] # Number of embedding dimensions to use for embeddings
n_layer = config_data['n_layer'] # Number of transformers used in the language model
n_heads = config_data['n_heads'] # Number of heads in each multi-headed attention block
dropout = config_data['dropout'] # Dropout percentage to maintain evolution
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Device to run the language model on


with open(args.dataset, 'r', encoding='utf-8') as f : # Read in the data set
    text = f.read()

os.makedirs(args.output, exist_ok=True) # Setup output directory
os.makedirs(f'{args.output}/checkpoints', exist_ok=True) # Setup checkpoints directory

# Split the dataset into a training dataset and validation dataset
tokenizer = Tokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.8*len(data))
train_data = data[:n] # 80% to train
val_data = data[n:] # 20% to test


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

# Learning rate warmup with cosine decay
warmup_iters = int(0.1*max_iters)
cosine_iters = max_iters - warmup_iters

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_iters)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0: # Occasionaly output current state of training
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {scheduler.get_last_lr()[0]:.6f}")

    if iter % checkpoint_interval == 0: # Save model checkpoints
        os.makedirs(f'{args.output}/checkpoints/iteration_{iter}', exist_ok=True)
        torch.save(m.state_dict(), f'{args.output}/checkpoints/iteration_{iter}/result.pt')
        torch.save(optimizer.state_dict(), f'{args.output}/checkpoints/iteration_{iter}/optimizer.pt')
        with open(f'{args.output}/checkpoints/iteration_{iter}/config.json', 'w') as json_file:
            json.dump(config_data, json_file, indent=4)

    # Sample the training data
    xb, yb = get_batch('train')

    # Evaluate the loss and perform gradient descent
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0) # Gradient clipping to avoid exploding gradients
    optimizer.step()
    scheduler.step()

# Save final results
os.makedirs(f'{args.output}/checkpoints/final', exist_ok=True)
torch.save(m.state_dict(), f'{args.output}/checkpoints/final/result.pt')
torch.save(optimizer.state_dict(), f'{args.output}/checkpoints/final/optimizer.pt')
with open(f'{args.output}/checkpoints/final/config.json', 'w') as json_file:
    json.dump(config_data, json_file, indent=4)
