# Super Nano GPT Model

## Overview
This is a basic implementation of a **decoder-only GPT model** built from scratch in PyTorch.

The architecture is inspired by:
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- Andrej Karpathy’s walkthrough: [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)

This project focuses on understanding GPT internals.

## Setup

1. Install PyTorch  
2. Clone this repository

### Training
Run ```py train.py``` which does the following
1. Read the current configuration under config\config.json
2. Train the model based on the configuration
3. Save the configuration used and training result to **savedResults/\<dateTimeString\>/**

### Run The GPT Model
Run ```py generate.py <dateTimeString>```  which does the following
1. Read the configuration used and trained data from **savedResults/\<dateTimeString\>/**
2. Run the GPT model

Note: an example trained model is under **savedResults/2026-01-29_17-30-23**

## Learnings
This was incredibly fun and difficult to implement. Before following along with Andrej Karpathy's video I had watched the series by 3Blue1Brown that explains at a high level the math and computations that go into creating a basic neural network and GPT model [link to the playlist here](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). From there Andrej Karpathy's video provided a step by step guide on the actual implementation of the theory that I had learned in a way that made the concepts approachable and simple to implement.

From this project, I gained hands-on experience with:
* Tokenization and vocabulary building
* Dataset batching and sampling
* Cross-entropy loss calculation
* Multi-head self-attention
* Feed-forward networks and layer normalization
* Dropout for regularization

## Specs
**Parameters**: 10,715,201<br>
**Dataset**: Mini Shakespeare<br>
**Architecture**: Decoder-only Transformer<br>
**Context Window**: 64 characters

## Training Details
Testing was performed on a local machine with:
- NVIDIA RTX 3080 (16GB VRAM)
- 64GB RAM
- Intel Core Ultra 9 CPU

## Example Output From 5000 Iterations
To appare my hunts, Imagine unform,
As 'twere not by the gentleman you ope,
Things, proceeded by my house.
Shall you save the man present thou hast, and not mine
To-morrow more her, you can so strong
and spiders of gock.

KING RICHARD III:
Let'st meet; and I shall be spiced with
To faint my friends and consent way to thee.

FLORIZEL:
What said Warwick,
As caugh I do, you were hused.

LEONTES:
Ay, so but a horse, good sir, a word;
And to your mother hath young.

HARTIUS:
He will a woman of brain,
