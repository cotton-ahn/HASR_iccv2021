import torch
import torch.nn as nn

class mGRU(nn.Module):
    def __init__(self, num_layers, feat_dim, inp_dim, out_dim):
        super(mGRU, self).__init__()
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        
        self.in_linear = nn.Linear(inp_dim, feat_dim)
        self.gru = nn.GRU(feat_dim, feat_dim, num_layers = num_layers, batch_first=True, bidirectional=True)
        self.out_linear = nn.Linear(feat_dim*2, out_dim)
        
    def forward(self, inp):
        # inp : B D L
        inp = inp.permute(0, 2, 1) # B L D
        inp = self.in_linear(inp)
        out, _ = self.gru(inp)
        out = self.out_linear(out) # B L D
        out = out.permute(0, 2, 1) # B D L
        
        return out