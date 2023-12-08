import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.stack(labels)

def train(net, loader, criterion, optimizer):
    net.train()
    total_loss = 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(net, loader, criterion):
    net.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            total_correct += (out.argmax(1) == y).float().sum().item()
    return total_loss / len(loader), total_correct / len(loader.dataset)


class Network(nn.Module):
    def __init__(self, vocab_size, emb, hidden, num_classes):
        super(Network, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb)
        self.linear1 = nn.Linear(emb, hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.relu(x)
        x, _ = torch.max(x, dim=1) #global max pool
        x = self.linear2(x)
        return x
    
    
class IMDBDataset(Dataset):
    def __init__(self, x, y):
        self.x = [torch.LongTensor(seq) for seq in x]
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ARDataset:
    def __init__(self, x, w2i, maxsize=10000, bs=32):
        self.x = [[w2i['.start']] + i + [w2i['.end']] for i in x]
        self.c = 0
        self.maxsize = max(maxsize, len(max(self.x, key=lambda x: len(x))) + 1)
        self.bs = bs
        self.w2i = w2i
    
    def shuffle(self):
        self.c = 0
        self.x = [self.x[i] for i in np.random.permutation(range(len(self.x)))]
    
    def get(self):
        temp = []
        while len(temp) < self.maxsize:
            if self.c >= len(self.x) or len(temp) + len(self.x[self.c]) + 1 > self.maxsize:
                temp.extend([self.w2i['.pad']] * (self.maxsize - len(temp)))
            else:
                temp.extend(self.x[self.c] + [self.w2i['.pad']])
                self.c += 1
        return torch.tensor(temp, dtype=torch.long)
    
    def dataloader(self):
        while self.c < len(self.x):
            x = torch.concat([self.get().view(1, -1) for _ in range(self.bs)]).to(device)
            x = x[x.count_nonzero(dim=1) > 1]
            y = torch.zeros_like(x)
            y[:, :-1] = x[:, 1:]
            yield x, y.view(-1)


class Elman(nn.Module):
    def __init__(self, insize=300, outsize=300, hsize=300):
        super().__init__()
        self.lin1 = nn.Linear(insize + hsize, hsize)
        self.lin2 = nn.Linear(hsize, outsize)
        self.sig = nn.Sigmoid()
    
    def forward(self, x, hidden=None):
        b, t, e = x.size()
        if hidden is None:
            hidden = torch.zeros(b, e, dtype=torch.float, device=x.device)
            
        outs = []
        for i in range(t):
            inp = torch.cat([x[:, i, :], hidden], dim=1)
            hidden = self.sig(self.lin1(inp))
            out = self.lin2(hidden)
            outs.append(out[:, None, :])
            
        return torch.cat(outs, dim=1), hidden


class ElmanNetwork(nn.Module):
    def __init__(self, vocab_size, emb, hidden, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb)
        self.rnn = Elman(emb, hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.relu(x)
        x, _ = torch.max(x, dim=1) #global max pool
        x = self.linear2(x)
        return x


class RNNNetwork(nn.Module):
    def __init__(self, vocab_size, emb, hidden, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb)
        self.rnn = nn.RNN(emb, hidden, batch_first=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.relu(x)
        x, _ = torch.max(x, dim=1) #global max pool
        x = self.linear2(x)
        return x


class LSTMNetwork(nn.Module):
    def __init__(self, vocab_size, emb, hidden, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb)
        self.rnn = nn.LSTM(emb, hidden, batch_first=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.relu(x)
        x, _ = torch.max(x, dim=1) #global max pool
        x = self.linear2(x)
        return x


class AutoRegressiveNetwork(nn.Module):
    def __init__(self, w2i, emb=32, h=16):
        super().__init__()
        self.emb = nn.Embedding(len(w2i), emb)
        self.lstm = nn.LSTM(emb, h, batch_first=True)
        self.linear = nn.Linear(h, len(w2i))
        self.w2i = w2i
    
    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x.view(-1, len(self.w2i))