"""Architectural ablations for OmicReasonNet.

Variants:
  - full           : OmicReasonNet (default)
  - no_stat        : drop statistical-prior branch (use only LLM semantic)
  - no_sem         : drop semantic-prior branch (use only statistical)
  - no_gate        : replace gated fusion with simple sum
  - no_fwgcn       : skip FWGCN, use MLP head on concatenated reweighted features
  - no_reasoning   : skip per-feature reasoning score (no exp(delta*|x|) reweight)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import LAGCN, GatedFusion, FWGCN


class AblatedOmicReasonNet(nn.Module):
    def __init__(self, feat_dims, n_classes, variant="full",
                 hidden_dim=64, dropout=0.5, delta=1e-2, layers=3):
        super().__init__()
        assert variant in {"full", "no_stat", "no_sem", "no_gate", "no_fwgcn", "no_reasoning"}
        self.variant = variant
        self.delta = delta
        self.n_views = len(feat_dims)
        self.stat_enc = nn.ModuleList([LAGCN(d, hidden_dim, num_layers=layers, dropout=dropout) for d in feat_dims])
        self.sem_enc = nn.ModuleList([LAGCN(d, hidden_dim, num_layers=layers, dropout=dropout) for d in feat_dims])
        if variant != "no_gate":
            self.fuse = nn.ModuleList([GatedFusion(hidden_dim) for _ in range(self.n_views)])
        self.score_head = nn.ModuleList([nn.Linear(hidden_dim, d) for d in feat_dims])
        total_in = sum(feat_dims)
        if variant == "no_fwgcn":
            self.head = nn.Sequential(
                nn.Linear(total_in, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            self.head = FWGCN(total_in, hidden_dim, n_classes, dropout=dropout)

    def forward(self, x_views, stat_edges, sem_edges, sample_edge_full):
        fused = []
        scores = []
        for v in range(self.n_views):
            x_v = x_views[v]
            if self.variant == "no_stat":
                z = self.sem_enc[v](x_v, sem_edges[v] if sem_edges[v] is not None else stat_edges[v])
            elif self.variant == "no_sem":
                z = self.stat_enc[v](x_v, stat_edges[v])
            else:
                z_stat = self.stat_enc[v](x_v, stat_edges[v])
                z_sem = self.sem_enc[v](x_v, sem_edges[v] if sem_edges[v] is not None else stat_edges[v])
                if self.variant == "no_gate":
                    z = 0.5 * (z_stat + z_sem)
                else:
                    z = self.fuse[v](z_stat, z_sem)
            s = torch.sigmoid(self.score_head[v](z.mean(0)))
            scores.append(s)
            if self.variant == "no_reasoning":
                fused.append(x_v)
            else:
                fused.append(x_v * s.unsqueeze(0) * torch.exp(self.delta * x_v.abs()))
        X = torch.cat(fused, dim=1)
        if self.variant == "no_fwgcn":
            return self.head(X), scores
        return self.head(X, sample_edge_full), scores
