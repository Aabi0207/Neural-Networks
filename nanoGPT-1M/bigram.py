import torch
import torch.nn as nn
import torch.nn.functional as F

#--------------------Constants-----------------------#

max_iter = 3000
lr = 1e-2
block_size = 8
batch_size = 32
eval_interval = 300
eval_iter = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------#

# Seed the torch for consistent result
torch.manual_seed(1337)

# Open the input file which is consisting the data complied from all the Shakespere dataset
with open('input.txt', 'r') as file:
    text = file.read()

# Tokenize the data

chars = list(sorted(set(text)))
vocab_size = len(chars)

stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in stoi.items()}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[val] for val in l])

# Split the dataset into train and test dataset

encoded_data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(encoded_data))
train_data = encoded_data[:n]
test_data = encoded_data[n:]

#-----------------Helper Functions-------------------#

# Data loader
def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    X = torch.stack([data[i: i + block_size] for i in ix])
    Y = torch.stack([data[i+1: i+1+block_size] for i in ix])
    X, Y = X.to(device), Y.to(device)
    return X, Y

