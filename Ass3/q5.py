import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_rnn import load_imdb
from utils import IMDBDataset, collate_fn, train, evaluate, device


(x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = load_imdb(final=False)

emb = 300
vocab_size = len(w2i)
hiddens = [100, 300, 500]
bss = [32, 16, 8]
lrs = [1e-2, 1e-3, 1e-4]

criterion = nn.CrossEntropyLoss()
networks = ['Network', 'RNNNetwork', 'LSTMNetwork']

for network in networks:
    for lr in lrs:
        for bs in bss:
            train_loader = DataLoader(IMDBDataset(x_train, y_train), batch_size=bs, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(IMDBDataset(x_val, y_val), batch_size=bs, shuffle=False, collate_fn=collate_fn)
            for hidden in hiddens:
                print()
                net = globals()[network](vocab_size, emb, hidden, num_classes)
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)

                net.to(device)

                train_losses, val_losses, val_accu = [], [], []

                for epoch in range(10):
                    train_loss = train(net, train_loader, criterion, optimizer)
                    val_loss, val_acc = evaluate(net, val_loader, criterion)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    val_accu.append(val_acc)
                    
                path = f"results\\{network}\\lr={lr},bs={bs},h={hidden}.json"
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                with open(path, 'w') as f:
                    f.write(json.dumps({'train_loss': train_losses, 'val_loss': val_losses, 'acc': val_accu}))