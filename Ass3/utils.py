import torch
import torch.nn as nn
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