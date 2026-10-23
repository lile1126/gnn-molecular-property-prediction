# -*- coding: utf-8 -*-
"""探查 Clintox 标签结构与 MoleculeNet 特征（临时）"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch_geometric.datasets import MoleculeNet

for name in ["Clintox", "ESOL"]:
    ds = MoleculeNet(root=r"D:\桌面\工程实践\GNN分子性质预测\data", name=name)
    d = ds[0]
    print(f"[{name}] n={len(ds)} num_features={ds.num_features} num_edge_features={ds.num_edge_features}")
    print("  y shape:", tuple(d.y.shape), "y[0]:", d.y[0].tolist()[:12])
    print("  x shape:", tuple(d.x.shape), "x[0] sum:", float(d.x[0].sum()))
    print("  edge_index shape:", tuple(d.edge_index.shape), "edge_attr:", getattr(d, "edge_attr", None))
    print("  keys:", d.keys)
