import json
import os
import sys
import torch
from languageModel.bigramLanguageModel import BigramLanguageModel
from tokenizer.tokenizer import Tokenizer

# Parse argument for directory holding the GPT model
if len(sys.argv) > 1:
    trained_data_path = sys.argv[1]
else:
    print("ERROR: No arguments passed")
    print("USAGE: py generate.py <date_time_string>")
    sys.exit()

# Check that GPT directory exists
if os.path.isdir(f'savedResults/{trained_data_path}'):
    with open(f'savedResults/{trained_data_path}/config.json', 'r') as configuration:
        config_data = json.load(configuration)
else:
    print(f'ERROR: Folder at folder path {trained_data_path} does not exist')
    sys.exit()

# Parameters From Config File
block_size = config_data['block_size'] # Maximum context length
n_embd = config_data['n_embd'] # Number of embedding dimensions to use for embeddings
n_layer = config_data['n_layer'] # Number of transformers used in the language model
n_heads = config_data['n_heads'] # Number of heads in each multi-headed attention block
dropout = config_data['dropout'] # Dropout percentage to maintain evolution
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Device to run the language model on

with open(f'datasets/{config_data['dataset']}', 'r', encoding='utf-8') as f :
    text = f.read()

tokenizer = Tokenizer(text)
model = BigramLanguageModel(block_size, device, dropout, n_embd, n_heads, n_layer, tokenizer.vocab_size())
model.load_state_dict(torch.load(f'savedResults/{trained_data_path}/result.pt', weights_only=True))
model.eval()
m = model.to(device)

# Create input vector representing input token index
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate new tokens using the GPT model!
print(tokenizer.decode(m.generate(idx = context, max_new_tokens=500)[0].tolist()))
