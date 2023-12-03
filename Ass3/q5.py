import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_rnn import load_imdb
from utils import ElmanNetwork, IMDBDataset, collate_fn, train, evaluate, device


(x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = load_imdb(final=False)

emb = 300
hidden = 300
vocab_size = len(w2i)
bs = 64

net = ElmanNetwork(vocab_size, emb, hidden, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

train_loader = DataLoader(IMDBDataset(x_train, y_train), batch_size=bs, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(IMDBDataset(x_val, y_val), batch_size=bs, shuffle=False, collate_fn=collate_fn)

net.to(device)

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
test_loader = DataLoader(IMDBDataset(x_test, y_test), batch_size=bs, shuffle=False, collate_fn=collate_fn)
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