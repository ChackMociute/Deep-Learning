import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_rnn import load_imdb

# Network
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

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

emb = 300
hidden = 300
vocab_size = len(w2i)
num_classes = numcls

net = Network(vocab_size, emb, hidden, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

# Dataloader
class IMDBDataset(Dataset):
    def __init__(self, x, y):
        self.x = [torch.LongTensor(seq) for seq in x]
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.stack(labels)

train_dataset = IMDBDataset(x_train, y_train)
val_dataset = IMDBDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

# GPU check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net.to(device)

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

train_losses = []
val_accu = []

# Training
for epoch in range(10):
    train_loss = train(net, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(net, val_loader, criterion)
    train_losses.append(train_loss)
    val_accu.append(val_acc)
    print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, Val Acc: {val_acc:.2f}')


# Testing
(x_train, y_train), (x_test, y_test), (i2w, w2i), numcls = load_imdb(final=True)
test_dataset = IMDBDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
test_loss, test_acc = evaluate(net, test_loader, criterion)
print(f'Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}')

test_loss, test_acc = evaluate(net, test_loader, criterion)
print(f'Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}')



# Plotting
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
axs[0].set_title('Training Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(range(1, len(val_accu) + 1), val_accu, label='Val Acc', color='green')
axs[1].set_title('Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()