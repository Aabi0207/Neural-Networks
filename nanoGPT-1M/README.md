# nanoGPT-1M

This repository implements a lightweight GPT-style language model based on Andrej Karpathy's nanoGPT framework. The project demonstrates training a 1M-parameter GPT model for text generation tasks.

## Features
- **Transformer Architecture:** Implements a minimal GPT model with self-attention and feed-forward layers.
- **Custom Dataset Support:** Allows training on any text dataset with preprocessing utilities.
- **Efficient Training:** Optimized for quick experimentation using PyTorch.
- **Scalable Design:** Easy to modify model size and hyperparameters.

## Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy
- tqdm
- Transformers (optional, for tokenizer support)

Install dependencies:
```bash
pip install torch numpy tqdm transformers
```

## Usage
### 1. Preprocess Dataset
Place your dataset in the `data/` directory and preprocess it:
```bash
python preprocess.py --input data/input.txt --output data/output.bin
```

### 2. Train the Model
```bash
python train.py --dataset data/output.bin --epochs 10 --batch_size 64
```

### 3. Generate Text
```bash
python generate.py --checkpoint model_checkpoint.pth --prompt "Once upon a time"
```

## Model Configuration
Modify `config.py` to customize model parameters such as:
- Embedding size
- Number of layers
- Attention heads
- Dropout rate

## File Structure
```
├── data/
│   └── input.txt            # Input dataset
├── model/
│   ├── gpt.py               # Model implementation
│   ├── train.py             # Training script
│   ├── generate.py          # Text generation script
├── utils/
│   ├── preprocess.py        # Preprocessing utilities
│   ├── config.py            # Model configuration
├── checkpoints/
│   └── model_checkpoint.pth # Saved model checkpoints
└── README.md
```

## Acknowledgements
Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).


