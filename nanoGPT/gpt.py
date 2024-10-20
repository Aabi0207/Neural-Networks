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

# Function to estimate the loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()   # set the modal in evaluation mode for layers such as batch norm
    for split in ['train', 'test']:
        X, Y = get_batch(split)
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Write the function to determine the head of the transformer
class Head(nn.Module):
    """One head of the self Attention"""
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # B, T, head_size
        q = self.query(x)

        # Compute attention score (affinities)
        wei = q @ k.T(-2, -1) * k.shape(-1) **-0.5  ## Kaning initilization
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('inf'))
        wei = nn.Softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Perform weighted aggeragation of the weights
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Mltiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = torch.cat(h(x) for h in self.heads)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.layers = nn.Sequential([
            nn.Linear(n_embd, n_embd * 4), 
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        ])
        
    def forward(self, x):
        out = self.layers(x)
        return out

    
class Block(nn.Module):
    """Transformer block which holds the process together"""

    def __init__(self, n_head, n_embd):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head, n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        tok_embedding = self.token_embedding_table(idx)
        pos_embedding = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_embedding + pos_embedding
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, target=targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
        

model = GPTLanguageModel()
m = model.to(device)
print(f"Total No. of parameters are: {sum(p.numel() for p in model.parameters())}")
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    # get batch
    xb, yb = get_batch('train')
    # forward pass
    logits, loss = model(xb, yb)

    # Backward pass
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    if i % eval_interval == 0 or i == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

