# -*- coding: utf-8 -*-
"""检查 MoleculeNet 各数据集特征维度与规模，确认预训练池可行性"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from torch_geometric.datasets import MoleculeNet

ROOT = r"D:\桌面\工程实践\GNN分子性质预测"
DATA_ROOT = os.path.join(ROOT, "data")

names = ["ESOL", "FreeSolv", "BBBP", "ClinTox", "BACE", "SIDER", "Tox21"]
for name in names:
    try:
        ds = MoleculeNet(root=DATA_ROOT, name=name)
        info = {
            "samples": len(ds),
            "num_features": ds.num_features,
            "num_edge_features": ds.num_edge_features,
            "x_dtype": str(ds[0].x.dtype),
            "x_shape": list(ds[0].x.shape),
        }
        print(f"[{name}] {info}")
    except Exception as e:
        print(f"[{name}] ERROR {type(e).__name__}: {e}")
