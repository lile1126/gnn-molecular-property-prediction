# -*- coding: utf-8 -*-
"""Clintox 深度集成评估（整体检查收尾）：3 架构 × 4 成员（seed 42/7/13/21）。
成员测试集 AUC + 集成均值 AUC + ECE，统一概率口径，结果存 reports/clintox_ensemble.json。"""
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
ENSEMBLE_DIR = os.path.join(ROOT, "models", "ensemble")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
HEADS = 4


def ece(probs, targets, n_bins=10):
    probs = np.asarray(probs)
    targets = np.asarray(targets)
    conf = np.clip(probs, 0, 1)
    acc = (conf >= 0.5).astype(int) == targets
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / len(conf) * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def main():
    dataset, split = load("Clintox", verbose=False)
    test_loader = DataLoader(dataset[[int(i) for i in split["test"]]], batch_size=BATCH_SIZE, shuffle=False)
    Y_COL = 1  # CT_TOX

    results = {"task": "Clintox CT_TOX (classification, pos 7.6%)",
               "config": "hidden=64 layers=3 heads=4 成员=seed42基线+s7+s13+s21（类别加权[1,12]）"}
    for conv in ["GCN", "GAT", "GraphSAGE"]:
        base = os.path.join(ROOT, "models", f"{conv}_Clintox.pt")
        paths = [base] if os.path.exists(base) else []
        paths += [os.path.join(ENSEMBLE_DIR, p) for p in sorted(os.listdir(ENSEMBLE_DIR))
                  if p.startswith(f"{conv}_Clintox_s") and p.endswith(".pt")]
        probs_all = []
        members = []
        with torch.no_grad():
            for path in paths:
                ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
                m = build_model(conv, ckpt["in_dim"], ckpt["hidden"], 2, ckpt["layers"],
                                heads=ckpt.get("heads", HEADS)).to(DEVICE)
                m.load_state_dict(ckpt["state_dict"])
                m.eval()
                members.append(m)
                ps = []
                for data in test_loader:
                    data = data.to(DEVICE)
                    out = m(data.x, data.edge_index, data.batch)
                    ps.append(F.softmax(out, dim=1)[:, 1].cpu())
                probs_all.append(torch.cat(ps).numpy())
        targets = torch.cat([data.y[:, Y_COL].cpu() for data in test_loader]).numpy().flatten()
        members_auc = [round(float(roc_auc_score(targets, p)), 4) for p in probs_all]
        ens_prob = np.mean(probs_all, axis=0)
        ens_auc = round(float(roc_auc_score(targets, ens_prob)), 4)
        ens_ece = round(ece(ens_prob, targets), 4)
        results[conv] = {"n_members": len(members), "members_AUC": members_auc,
                         "ensemble_AUC": ens_auc, "ensemble_ECE": ens_ece}
        print(f"{conv}×Clintox 集成: 成员AUC={members_auc} 集成AUC={ens_auc} ECE={ens_ece}")

    out = os.path.join(ROOT, "reports", "clintox_ensemble.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
