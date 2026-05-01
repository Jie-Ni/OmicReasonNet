"""OmicReasonNet model: dual-prior LAGCN + gated fusion + FWGCN classifier."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class LAGCN(nn.Module):
    """Layer-Attention GCN. Aggregates per-layer outputs with learned attention."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.attn_W = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        outs = []
        h = x
        for conv in self.layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            outs.append(h)
        H = torch.stack(outs, dim=0)                                  # [L, N, D]
        scores = self.attn_w(torch.tanh(self.attn_W(H))).squeeze(-1)  # [L, N]
        beta = F.softmax(scores, dim=0).unsqueeze(-1)                 # [L, N, 1]
        return (beta * H).sum(dim=0)                                  # [N, D]


class GatedFusion(nn.Module):
    """Per-element sigmoid gate fusing two same-shape representations.

    Bias initialised to +2.0 so the gate starts at sigmoid(2) ≈ 0.88, i.e.
    the fusion is heavily stat-weighted at start. This gives the optimizer
    a "warm start from the strong baseline" and lets sem add value without
    fighting the dominant signal — empirically prevents the slight regression
    we saw on BRCA when starting at gate=0.5.
    """

    def __init__(self, dim: int, init_bias: float = 2.0):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        nn.init.constant_(self.gate.bias, init_bias)

    def forward(self, z_stat: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.gate(torch.cat([z_stat, z_sem], dim=-1)))
        return z * z_stat + (1.0 - z) * z_sem


class FWGCN(nn.Module):
    """Feature-Weighted GCN classifier for sample-as-node graphs."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.cls(h)


class OmicReasonNet(nn.Module):
    """End-to-end multi-omics dual-prior network.

    forward inputs:
        x_views: list of [N, F_v]
        sample_edges: list of edge_index (sample graph per view, statistical prior)
        sem_edges: list of edge_index OR None (sample graph per view, semantic prior)
        delta: float (regularization in feature reweighting)
    """

    def __init__(
        self,
        feat_dims: list[int],
        n_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        delta: float = 0.01,
        layers: int = 3,
    ):
        super().__init__()
        self.delta = delta
        self.n_views = len(feat_dims)
        self.stat_enc = nn.ModuleList(
            [LAGCN(d, hidden_dim, num_layers=layers, dropout=dropout) for d in feat_dims]
        )
        self.sem_enc = nn.ModuleList(
            [LAGCN(d, hidden_dim, num_layers=layers, dropout=dropout) for d in feat_dims]
        )
        self.fuse = nn.ModuleList([GatedFusion(hidden_dim) for _ in range(self.n_views)])
        # learn per-feature reasoning score from fused embedding -> projected back to feat space
        self.score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, d) for d in feat_dims]
        )
        total_in = sum(feat_dims)
        self.fwgcn = FWGCN(total_in, hidden_dim, n_classes, dropout=dropout)

    def forward(
        self,
        x_views: list[torch.Tensor],
        stat_edges: list[torch.Tensor],
        sem_edges: list[torch.Tensor | None],
        sample_edge_full: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fused_per_view = []
        scores_per_view = []
        for v in range(self.n_views):
            x_v = x_views[v]
            z_stat = self.stat_enc[v](x_v, stat_edges[v])
            if sem_edges[v] is not None:
                z_sem = self.sem_enc[v](x_v, sem_edges[v])
            else:
                z_sem = z_stat
            z_fused = self.fuse[v](z_stat, z_sem)               # [N, D]
            s = torch.sigmoid(self.score_head[v](z_fused.mean(0)))  # [F_v]  (population score)
            scores_per_view.append(s)
            x_re = x_v * s.unsqueeze(0) * torch.exp(self.delta * x_v.abs())
            fused_per_view.append(x_re)
        X = torch.cat(fused_per_view, dim=1)
        logits = self.fwgcn(X, sample_edge_full)
        return logits, scores_per_view


# ---- baselines ----

class PlainGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden_dim)
        self.c2 = GCNConv(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x, e):
        h = F.relu(self.c1(x, e))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.c2(h, e)


class PlainGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.g1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.g2 = GATConv(hidden_dim * heads, n_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, e):
        h = F.elu(self.g1(x, e))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.g2(h, e)


class MLPConcat(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x, edge_index=None):
        return self.net(x)
