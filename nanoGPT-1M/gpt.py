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


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head*head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # B,T,HS*N_HEAD
        out = self.proj(out) # B,T,C
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super.__init__()
        self.net = nn.Sequential([
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        ])

    def forward(self, x):
        return self.net(x)  #B,T,C


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(head_size)
        self.ffw = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x   # B,T,C


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.lif = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, model):
        if isinstance(model, nn.Linear):
            torch.nn.init.normal_(model.weight, mean=0.0, std=0.02)
            if model.bias is not None:
                torch.nn.init.zeros_(model.bias)
        if isinstance(model, nn.Embedding):
            torch.nn.init.normal_(model.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.lif(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_token):
        for _ in range(max_token):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = nn.Softmax(logits)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


