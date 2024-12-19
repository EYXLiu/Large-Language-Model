import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 16
max_iters = 5001
eval_interval = 500
learning_rate = 1e-2
device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu' no gpu pensive
eval_iters = 200
n_embd = 64
head_size = 32
n_head = 4
n_layer = 4
# dropout drops a percent of the neurons in the previous layer, which is helpful for both speed and for testing
dropout = 0.2
temperature = 1

with open('nanoGPT/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# encode and decode to numbers to be able to access/edit
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#self attention -> masks later tokens compared to cross attention, in which theres no masking because each value depends on all the others (not previous/next)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # attention(Query, Key, Value)
        self.key = nn.Linear(n_embd, head_size, bias=False) #answers queries, dot product -> if yes then large, otherwise small/negative
        self.query = nn.Linear(n_embd, head_size, bias=False) #smaller vector that asks question 
        self.value = nn.Linear(n_embd, head_size, bias=False) # current value -> normally the size is key + query parameters 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape 
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5 #divide by sqrt for numerical stability
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # only apply to before, due to predictive nature - remove this line if you want all words to affect this value
        # setting values to -inf allow softmax to normalize the collumns still 
        wei = F.softmax(wei, dim=-1) #normalize values, weights to how relevant the word is to the corresponding value -> attention pattern
        # matrix is context size squared, larger context means much slower and more gpu/cpu tasking
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v # multiply by value to update embedding, vector of how much each value before affects the current one (the head of the attention)
        
        return out
# runs many heads of attention in parallel, each head produces a proposed change and adds to original embedding
# each head has distinct key, query, and value matrices, in such that it produces a different change
# learn many ways that context changes meaning
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # add each head to the original embedding
        # when attention is passed, each vector encodes everything previous, EG. the Name vector in "First Name" would now be pointing at the "First Name" location in the space
        # the resulting vector would have a dot product of 1 with anything its related too, eg. "First Name" dot product with both "Name" and "First" would be 1
        out = self.dropout(self.proj(out))
        return out
    
# MLP/MultiLayer Perceptron/FeedForward
# Each vector goes through a series of transformations (linear, relu, linear)
# Each nn is the same, equally applied, and equally modified if needed
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # first layer probs for features -> resulting value in vector is how much of a yes that answer is
            # often (not now) add bias after which decreases or increases depending on parameter
            # eg. if you want it to be negative if theres multiple questions -> F(E) + N(E) = 2 if First and Name, then -1 so that its 1 if true and <= 0 if false (importance below)
            # each row is "number of questions asked"
            nn.Linear(n_embd, 4 * n_embd), # growing residual block on the side 
            # maps all negative values to 0 and positives stay same -> And Gate
            # 0 if No, positive if Yes 
            nn.ReLU(),
            # output is "Neurons" -> active if positive, inactive if negative
            # this matrix is for "Facts" -> often a certain set of values in the vector due to superposition
            # applied to the ReLU'd vector, each value being true and how true it is 
            # bias can also be added here, just for bookkeeping
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# The circled "Block" in the 'Attention is all you need' paper diagram
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # added vector to itself so facts are added on to vector
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x)) # fork off, do computation, come back 
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        x = tok_emb + pos_emb # B, T, C
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #B, T, vocab_size

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # last vector is for the next character/token
            logits = logits / temperature # scale by temperature to make it more or less random
            probs = F.softmax(logits, dim=-1) # e^x/t / ∑e^xn/t to generate probability -> t for temperature, how fucked the output is (lower means more likely to just pick biggest number, higher means distributes similar)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter%eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    xb, xy = get_batch('train')

    logits, loss = model(xb, xy)
    optimizer.zero_grad(set_to_none=True)
    # backpropagation
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
open('nanoGPT/more.txt', 'w', encoding='utf-8').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))