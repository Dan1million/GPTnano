import json
import os
import sys
import torch
from languageModel.bigramLanguageModel import BigramLanguageModel

if len(sys.argv) > 1:
    trained_data_path = sys.argv[1]
else:
    print("ERROR: No arguments passed")
    print("USAGE: py generate.py <date_time_string>")
    sys.exit()

# Parameters From Config File
if os.path.isdir(f'savedResults/{trained_data_path}'):
    with open(f'savedResults/{trained_data_path}/config.json', 'r') as configuration:
        config_data = json.load(configuration)
else:
    print(f'ERROR: Folder at folder path {trained_data_path} does not exist')
    sys.exit()

block_size = config_data['block_size'] # Maximum context length
n_embd = config_data['n_embd'] # Number of embedding dimensions to use for embeddings
n_layer = config_data['n_layer'] # Number of transformers used in the language model
n_heads = config_data['n_heads'] # Number of heads in each multi-headed attention block
dropout = config_data['dropout'] # Dropout percentage to maintain evolution
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Device to run the language model on

with open(f'datasets/{config_data['dataset']}', 'r', encoding='utf-8') as f :
    text = f.read()

# Tokenizaiton --> Each unique character is a token
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create mapping to convert tokens to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda string: [stoi[c] for c in string] # Encoder to convert string to a list of integers representing the characters
decode = lambda list: ''.join([itos[i] for i in list]) # Decoder to convert a list of integers to it's string equivalent

model = BigramLanguageModel(block_size, device, dropout, n_embd, n_heads, n_layer, vocab_size)
model.load_state_dict(torch.load(f'savedResults/{trained_data_path}/result.pt', weights_only=True))
model.eval()
m = model.to(device)

# Create input vector representing input token index
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate new tokens based on our trained ML model!
print(decode(m.generate(idx = context, max_new_tokens=500)[0].tolist()))
