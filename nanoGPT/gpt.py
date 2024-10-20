# import all the dependancie's

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters

batch_size = 64
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 5000
block_size = 256
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# Load the dataset
with open('input.txt', 'r') as d:
    text = d.read()

# Create the mappings 
chars = sorted(list(set(text)))
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)
print(f"The vocab size is {vocab_size}")

# Create the functions to encode and decode the context
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join(itos[i] for i in l)

# Make for train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

# Data loading 
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(0, len(train_data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        for i in range(eval_iters):
            