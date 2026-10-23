# -*- coding: utf-8 -*-
"""验证 Clintox 基线模型训练时用的到底是哪一列标签（col0 FDA vs col1 CT_TOX）。
用已训练模型对测试集分别按 col0/col1 计算 AUC，与 clintox_baseline.json 对比定位。"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from data_prep import load
from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
HEADS = 4


def main():
    dataset, split = load("Clintox", verbose=False)
    test_loader = DataLoader(dataset[[int(i) for i in split["test"]]], batch_size=BATCH_SIZE, shuffle=False)
    y0, y1, probs = [], [], {}
    with torch.no_grad():
        for conv in ["GCN", "GAT", "GraphSAGE"]:
            path = os.path.join(ROOT, "models", f"{conv}_Clintox.pt")
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            m = build_model(conv, ckpt["in_dim"], ckpt["hidden"], 2, ckpt["layers"],
                            heads=ckpt.get("heads", 4)).to(DEVICE)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            ps = []
            for data in test_loader:
                data = data.to(DEVICE)
                out = m(data.x, data.edge_index, data.batch)
                ps.append(F.softmax(out, dim=1)[:, 1].cpu())
            probs[conv] = torch.cat(ps).numpy()
            y0 = torch.cat([d.y[:, 0] for d in dataset[[int(i) for i in split["test"]]]]).numpy()
            y1 = torch.cat([d.y[:, 1] for d in dataset[[int(i) for i in split["test"]]]]).numpy()
    baseline = json.load(open(os.path.join(ROOT, "reports", "clintox_baseline.json"), encoding="utf-8"))
    print("clintox_baseline.json 记录值:", {k: v.get("AUC") for k, v in baseline["Clintox"].items()})
    print(f"col0 FDA_APPROVED: 正类 {int(y0.sum())}/{len(y0)} ({y0.mean():.1%})")
    print(f"col1 CT_TOX:       正类 {int(y1.sum())}/{len(y1)} ({y1.mean():.1%})")
    for conv in ["GCN", "GAT", "GraphSAGE"]:
        p = probs[conv]
        print(f"{conv}: AUC(col0)={roc_auc_score(y0, p):.4f}  AUC(col1)={roc_auc_score(y1, p):.4f}")


if __name__ == "__main__":
    main()
