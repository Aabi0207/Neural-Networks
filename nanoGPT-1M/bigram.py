import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

#--------------------Constants-----------------------#

max_iter = 10000
lr = 1e-2
block_size = 16
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
itos = {i: s for s, i in stoi.items()}
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

# Calculate the loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iter)
        for i in range(eval_iter):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------Bigram Model---------------------#

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx, target=None):
        logits = self.token_embedding_table(idx) # B, T, C

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_token):
        for i in range(max_token):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits)
            next_ids = torch.multinomial(probs, num_samples=1)  # B, 1
            idx = torch.cat((idx, next_ids), dim=1)
        return idx


model = BigramLanguageModel(vocab_size, 356)
model = model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(max_iter):
    # forward pass
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    # backward pass
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_token=500)[0].tolist()))


