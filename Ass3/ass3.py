import torch
import torch.nn as nn


class Elman(nn.Module):
    def __init__(self, insize=300, outsize=300, hsize=300):
        super().__init__()
        self.lin1 = nn.Linear(insize + hsize, hsize)
        self.lin2 = nn.Linear(hsize, outsize)
        self.sig = nn.Sigmoid()
    
    def forward(self, x, hidden=None):
        b, t, e = x.size()
        if hidden is None:
            hidden = torch.zeros(b, e, dtype=torch.float)
            
        outs = []
        for i in range(t):
            inp = torch.cat([x[:, i, :], hidden], dim=1)
            hidden = self.sig(self.lin1(inp))
            out = self.lin2(hidden)
            outs.append(out[:, None, :])
            
        return torch.cat(outs, dim=1), hidden