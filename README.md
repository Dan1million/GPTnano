# CornellNano-172

A decoder-only GPT model built from scratch in PyTorch for learning and experimentation.

## Model Overview

**Architecture**: Decoder-only Transformer<br>
**Parameters**: ~172 million<br>
**Context Window**: 256 tokens<br>
**Training Dataset**: 500 million tokens from OpenWebText<br>
**Tokenization**: Character-level tokenization<br>
**Final Loss**: ~1.04 for train ~1.07 for val<br>

### Current Configuration
- **Embedding Dimensions**: 1,280
- **Transformer Layers**: 8
- **Attention Heads**: 8 per layer
- **Dropout Rate**: 0.2
- **Batch Size**: 48
- **Learning Rate**: 0.0003 with warmup and cosine decay

### Key Components
1. **Token & Position Embeddings**: Converts characters into numerical representations
2. **Multi-Head Self-Attention**: Allows the model to focus on different parts of the input simultaneously
3. **Feed-Forward Networks**: Processes the attended information
4. **Layer Normalization**: Stabilizes training
5. **Residual Connections**: Helps gradients flow during training

## Quick Start

### Installation
```bash
# 1. Clone the repository
git clone <repository-url>
cd GPTnano

# 2. Create a virtual environment
python -m venv .env

# 3. Activate the environment
.\.env\Scripts\activate  # Windows PowerShell
# source .env/bin/activate  # Linux/Mac

# 4. Install PyTorch (CUDA 13.0)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 5. Install Hugging Face datasets
pip install datasets
```

### Training the Model
```bash
# Train with default configuration
python train.py

# Train with custom config
python train.py --config config/config.json

# Resume from checkpoint
python train.py --resume savedResults/2026-02-04_12-10-25/checkpoints/iteration_10000
```

### Generating Text
```bash
# Generate from the latest trained model
python generate.py

# Generate from a specific model
python generate.py --gpt_path savedResults/40000_iterations

# Generate more tokens
python generate.py --tokens 1000

# Generate with a prompt (seed text)
python generate.py --prompt "I am a robot that likes"

# Generate with prompt and custom token count
python generate.py --prompt "Once upon a time" --tokens 500
```

> **Note**: The 40,000 iteration trained model is not included in this repository due to its size. If you would like access to this pre-trained model, please contact me at dcornell314@gmail.com and I will send it to you.

## Configuration

Edit `config/config.json` to customize training:

```json
{
    "batch_size": 48,             // Parallel sequences processed
    "block_size": 256,            // Context window size
    "checkpoint_interval": 10000, // Save checkpoints every N iterations
    "eval_interval": 2000,        // Evaluate loss every N iterations
    "eval_iters": 10,             // Number of batches for evaluation
    "learning_rate": 0.0003,      // Initial learning rate
    "max_iters": 100000,          // Total training iterations
    "n_embd": 1280,               // Embedding dimensions
    "n_layer": 8,                 // Number of transformer blocks
    "n_heads": 8,                 // Attention heads per layer
    "dropout": 0.2,               // Dropout rate for regularization
    "use_openwebtext": true,      // Use OpenWebText dataset (or custom)
    "dataset": "data.txt"         // Custom dataset file
}
```

## Training Features

### Modern Training Techniques
- **Mixed Precision Training**: Faster training on GPUs with automatic mixed precision (AMP)
- **Learning Rate Scheduling**: 10% warmup followed by cosine annealing decay
- **Gradient Clipping**: Prevents exploding gradients (clipped at 1.0)
- **Checkpointing**: Automatically saves model, optimizer, and scheduler state
- **Resume Training**: Continue from any saved checkpoint
- **Vocabulary Caching**: Save and reuse tokenizer vocabularies

### Monitoring
During training the following is logged:
- Current iteration and progress
- Training and validation loss
- Current learning rate
- Time per iteration
- Estimated time remaining

## Hardware Training Requirements

### Tested Configuration
- **GPU**: NVIDIA RTX 3080 (16GB VRAM)
- **RAM**: 64GB system memory
- **CPU**: Intel Core Ultra 9

### Performance
- ~50 minutes per 10,000 iterations with current configuration
- Overfitting begins occuring at 

## Advanced Usage

### Command-Line Arguments

#### train.py
```bash
python train.py [-h] [--config CONFIG] [--vocab VOCAB] [--output OUTPUT] [--resume RESUME]

Options:
  -h, --help         Show help message
  --config CONFIG    Path to config file (default: config/config.json)
  --vocab VOCAB      Path to pre-built vocabulary (speeds up training start)
  --output OUTPUT    Output directory (default: savedResults/<timestamp>)
  --resume RESUME    Resume from checkpoint directory
```

#### generate.py
```bash
python generate.py [-h] [--gpt_path GPT_PATH] [--tokens TOKENS] [--prompt PROMPT]

Options:
  -h, --help              Show help message
  --gpt_path GPT_PATH     Path to trained model directory
  --tokens TOKENS         Number of tokens to generate (default: 500)
  --prompt PROMPT         Seed text to start generation (optional)
```

### Using Pre-built Vocabularies
If training with OpenWebText, save time by reusing vocabularies:
```bash
# First training run creates vocab_openwebtext.json
python train.py --config config/config.json

# Subsequent runs can reuse it
python train.py --config config/config.json --vocab vocab_openwebtext.json
```

### Working with Checkpoints
Checkpoints are saved at regular intervals and include:
- Model weights (`result.pt`)
- Optimizer state (`optimizer.pt`)
- Scheduler state (`scheduler.pt`)
- Configuration (`config.json`)

Resume from any checkpoint:
```bash
python train.py --resume savedResults/2026-02-04_12-10-25/checkpoints/iteration_20000
```

## Project Structure
```
GPTnano/
├── config/
│   └── config.json              # Training configuration
├── datasets/
│   └── tinyShakespeare.txt      # Example small dataset
├── languageModel/
│   ├── bigramLanguageModel.py   # Main GPT model
│   ├── transformerBlock.py      # Transformer block implementation
│   └── head.py                  # Attention head implementation
├── tokenizer/
│   └── tokenizer.py             # Character-level tokenizer
├── savedResults/                # Trained models and checkpoints
├── train.py                     # Training script
└── generate.py                  # Text generation script
```

## Learning Resources

This implementation is inspired by:
- **[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)** - The original Transformer paper
- **[Dropout Paper](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)** - Regularization technique
- **[Andrej Karpathy's GPT Tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)** - Fundamentals of implementing a GPT model from scratch
- **[3Blue1Brown Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)** - Visual explanations and mathematical background

## What I Learned

### Core Concepts
- **Tokenization**: Taking the dataset and splitting it into tokens for training
- **Self-Attention Mechanism**: How models "learn" the connections between tokens
- **Transformer Architecture**: Layer normalization, residual connections, feed-forward networks
- **Training Dynamics**: Loss curves, overfitting detection, learning rate schedules
- **Optimization**: AdamW optimizer, gradient clipping, mixed precision training

### PyTorch Skills
- Building custom neural network modules
- Dataset batching and efficient sampling
- GPU memory management and mixed precision
- Checkpointing and state management
- Cross-entropy loss for language modeling

### Enhancements Beyond Andrej Karpathy's Tutorial
- Learning rate warmup and cosine decay scheduling
- Gradient clipping to improve stability
- Checkpointing system with resume capability
- Use of an external configuration file
- Itnegrating OpenWebText using Hugging Face
- Vocabulary caching to reduce the amount of time to restart training
- Automatic Mixed Precision training to speed up training

## Example Output

After training, the model can generate coherent text in the style of its training data:

```
----- CornellNano-172 Text Generation -----

Post: 527/2010

Simpson:

DT: Doesn’t do that in your life? Can you expect anything to do or follow what you’re doing for?

JS: Right now I’m going to announce all kinds of resolutions that apply to just file the links. Standard applications by calling in compared services most likely to increase the production of information that allows new connection between just above, per Attorney General Eric DeCoy conducted in June 2004. Photo courtesy of Standard Allowance Exhibitions-y JustinATTORNEY(D-

----- Generation Complete -----
```

## How I got better results

1. **Increased Training Time**: Ran for 40,000 iterations over ~4 hours --> any longer resulted in overfitting
2. **Monitored Validation Loss**: Stopped training when validation started rising --> overfitting
3. **Experimented with Parameters**: Try different values for `n_embd`, `n_layer`, `n_heads`, etc. Helped extract significantly more power from my RTX 5080
4. **Added Automatic Mixed Precision**: Increased training speed for my model by over 2x vs the simpler self attantion implemented in Andrej Karpathy's tutorial
5. **Increased Dataset Size**: Initially I used the tiny Shakespeare data set which was ~1MB. I quickly ran into the issue of overfitting. In the current configuration the OpenWebText dataset is used which is about 14GB

## Acknowledgments

Special thanks to Andrej Karpathy and 3Blue1Brown for their excellent educational content. Also thank you to the PyTorch team for providing an incredibly powerful tool.
