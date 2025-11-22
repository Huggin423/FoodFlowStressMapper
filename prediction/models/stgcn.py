"""Minimal STGCN-like model implementation.
If PyTorch is unavailable, importing this module will raise a RuntimeError when attempting model instantiation.
"""
from __future__ import annotations
try:
    import torch
    import torch.nn as nn
except ImportError as e:  # graceful fallback
    torch = None
    nn = object

class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
    def forward(self, x, adj):
        # x: (batch, nodes, in_features); adj: (batch, nodes, nodes)
        support = self.lin(x)
        out = torch.bmm(adj, support)  # simple aggregation
        return out

class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, k: int = 3):
        super().__init__()
        self.temporal = nn.GRU(input_size=in_channels, hidden_size=hidden_channels, batch_first=True)
        self.graph = GraphConv(hidden_channels, hidden_channels)
        self.act = nn.ReLU()
    def forward(self, x, adj):
        # x: (batch, time, nodes, features)
        b,t,n,f = x.shape
        x_reshaped = x.permute(0,2,1,3).reshape(b*n, t, f)  # (b*n, t, f)
        out_seq,_ = self.temporal(x_reshaped)  # (b*n, t, hidden)
        last = out_seq[:,-1,:]  # (b*n, hidden)
        node_emb = last.view(b,n,-1)
        # adjacency (batch,n,n)
        gc_out = self.graph(node_emb, adj)
        return self.act(gc_out)

class SimpleSTGCN(nn.Module):
    def __init__(self, in_features: int, hidden: int = 32, out_features: int = 1, blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([STGCNBlock(in_features if i==0 else hidden, hidden) for i in range(blocks)])
        self.out = nn.Linear(hidden, out_features)
    def forward(self, x, adj):
        # x: (batch, time, nodes, features); adj: (batch, nodes, nodes)
        h = x
        for blk in self.blocks:
            h = blk(h, adj)
            # expand back to time dimension for next block: replicate along time axis
            h = h.unsqueeze(1).repeat(1, x.size(1), 1, 1)
        # final prediction uses last node embeddings (from last block) before replication
        # Extract last block output (after replication h is (batch,time,n,hidden)) -> take last time
        last_time = h[:,-1,:,:]  # (batch,n,hidden)
        pred = self.out(last_time)  # (batch,n,out)
        return pred.squeeze(-1)  # (batch,n)

def build_stgcn(in_features: int, hidden: int = 32, out_features: int = 1, blocks: int = 2):
    if torch is None:
        raise RuntimeError("PyTorch not installed. Install torch to use STGCN.")
    return SimpleSTGCN(in_features, hidden, out_features, blocks)
