# -*- coding: utf-8 -*-
"""修正基线分类 AUC：用 softmax 概率而非 argmax 标签计算（标准做法），并同步更新 comparison_results.json"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from data_prep import TASK_TYPE, load
from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

CLASS_DATASETS = ["BBBP"]


def load_model(conv_type, name):
    ckpt = torch.load(os.path.join(MODEL_DIR, f"{conv_type}_{name}.pt"), weights_only=False, map_location="cpu")
    out_dim = 2
    model = build_model(conv_type, ckpt["in_dim"], ckpt["hidden"], out_dim, ckpt["layers"], heads=ckpt.get("heads", 4))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def prob_auc(model, loader):
    probs, targets = [], []
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.batch)
        probs.append(F.softmax(out, dim=1)[:, 1].cpu())
        targets.append(data.y[:, 0].cpu())
    probs = torch.cat(probs).numpy()
    targets = torch.cat(targets).numpy()
    return float(roc_auc_score(targets, probs)), probs, targets


def main():
    result_path = os.path.join(REPORT_DIR, "comparison_results.json")
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    convs = ["GCN", "GAT", "GraphSAGE"]
    for name in CLASS_DATASETS:
        dataset, split = load(name, verbose=False)
        test_loader = DataLoader(dataset[[int(i) for i in split["test"]]], batch_size=BATCH_SIZE, shuffle=False)
        for conv in convs:
            model = load_model(conv, name)
            auc, _, _ = prob_auc(model, test_loader)
            old = data["results"][name][conv]["AUC"]
            data["results"][name][conv]["AUC"] = round(auc, 4)
            data["results"][name][conv]["AUC_old_argmax"] = old
            print(f"{conv}×{name}: AUC {old} -> {auc:.4f} (概率口径)")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("comparison_results.json 已更新（AUC 改为概率口径，原 argmax 值保留在 AUC_old_argmax）")


if __name__ == "__main__":
    main()
