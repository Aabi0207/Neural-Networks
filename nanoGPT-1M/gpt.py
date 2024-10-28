import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings 
warnings.filterwarnings('ignore')
torch.manual_seed(42)

#---------------------------------Constants----------------------------------#
max_iter = 5000
lr = 3e-4
block_size = 256
batch_size = 32
eval_interval = 500
eval_iter = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
#-----------------------------------------------------------------------------#

#--------------------------Data Loading---------------------------------------#
with open('input.txt', 'r') as f:
    text = f.read()

chars = list(sorted(set(text)))
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[val] for val in l])

encoded_data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(encoded_data))
train_data = encoded_data[:n]
test_data = encoded_data[n:]

#------------------------Helper Functions ------------------------------------#
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))    # B, 1
    xb = torch.stack([data[i:i+block_size] for i in ix])     # B, block_size/ B, T
    yb = torch.stack([data[i+1:i+1+block_size] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iter)
        for i in range(eval_iter):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#---------------------Transformer Architecture--------------------------------#

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value =nn.Linear(n_embd, head_size=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):   # B, T, C
        B,T,C = x.shape
        k = self.key(x)  # B, T, hs
        q = self.query(x)# B, T, hs
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # B, T, HS @ B, HS, T == B, T, T
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('inf'))
        wei = nn.Softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) #B, T, HS
        out = wei @ v # B,T, HS
        return out