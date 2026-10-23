# -*- coding: utf-8 -*-
"""
深度集成成员训练（方向 2）
==========================
对指定 (架构, 数据集) 组合用多个随机种子训练模型，保存为集成成员。
与 seed=42 的已有模型一起构成深度集成。

用法:
    python src/train_ensemble.py --convs GAT GraphSAGE GraphSAGE --names ESOL FreeSolv BBBP \
        --seeds 7 13 21 --epochs 300
"""
import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from data_prep import TASK_TYPE, load
from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
ENSEMBLE_DIR = os.path.join(ROOT, "models", "ensemble")
REPORT_DIR = os.path.join(ROOT, "reports")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATIENCE = 40
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 64
NUM_LAYERS = 3
HEADS = 4


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, optimizer=None, task_type="regression", target_col=0, class_weight=None):
    train = optimizer is not None
    model.train(train)
    total_loss, total = 0.0, 0
    preds, targets = [], []
    for data in loader:
        data = data.to(DEVICE)
        if train:
            optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        y = data.y[:, target_col].float()
        if task_type == "classification":
            loss = torch.nn.functional.cross_entropy(out, y.long(),
                                                     weight=class_weight.to(DEVICE) if class_weight is not None else None)
        else:
            loss = torch.nn.functional.mse_loss(out.squeeze(-1), y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(data.y)
        total += len(data.y)
        if task_type == "classification":
            preds.append(F.softmax(out, dim=1)[:, 1].cpu())
        else:
            preds.append(out.squeeze(-1).cpu())
        targets.append(y.cpu())
    return total_loss / max(total, 1), torch.cat(preds), torch.cat(targets)


def evaluate(model, loader, task_type, target_col=0, class_weight=None):
    model.eval()
    with torch.no_grad():
        _, preds, targets = run_epoch(model, loader, optimizer=None, task_type=task_type,
                                      target_col=target_col, class_weight=class_weight)
    if task_type == "regression":
        from sklearn.metrics import mean_squared_error, r2_score
        preds = preds.numpy().astype(np.float64)
        targets = targets.numpy().astype(np.float64)
        return {"RMSE": round(float(np.sqrt(mean_squared_error(targets, preds))), 4),
                "R2": round(float(r2_score(targets, preds)), 4)}
    from sklearn.metrics import roc_auc_score
    return {"AUC": round(float(roc_auc_score(targets.numpy(), preds.numpy().astype(np.float64))), 4)}


def train_one(conv_type, name, seed, epochs=300):
    set_seed(seed)
    dataset, split = load(name, verbose=False)
    # Clintox 双列标签：col0=FDA_APPROVED（近乎全正类），col1=CT_TOX（毒性，正类 7.6%）。
    # 注意：PyG 2.8 __getitem__ 返回克隆，对 dataset 元素重写 y 不生效，
    # 统一通过 run_epoch 的 target_col 直接取列（与 feat 数据口径一致）。
    is_clintox = name == "Clintox"
    class_weight = torch.tensor([1.0, 12.0]).to(DEVICE) if is_clintox else None
    target_col = 1 if is_clintox else 0
    train_loader = DataLoader(dataset[[int(i) for i in split["train"]]], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset[[int(i) for i in split["val"]]], batch_size=BATCH_SIZE, shuffle=False)
    task = TASK_TYPE[name]
    out_dim = 2 if task == "classification" else 1
    model = build_model(conv_type, dataset.num_features, HIDDEN_DIM, out_dim, NUM_LAYERS, heads=HEADS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val, best_state, bad = None, None, 0
    for epoch in range(1, epochs + 1):
        run_epoch(model, train_loader, optimizer, task, target_col=target_col, class_weight=class_weight)
        val = evaluate(model, val_loader, task, target_col=target_col, class_weight=class_weight)
        score = val["AUC"] if task == "classification" else -val["RMSE"]
        if best_val is None or score > best_val:
            best_val, best_state, bad = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    test = evaluate(model, DataLoader(dataset[[int(i) for i in split["test"]]], batch_size=BATCH_SIZE, shuffle=False),
                    task, target_col=target_col, class_weight=class_weight)
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    path = os.path.join(ENSEMBLE_DIR, f"{conv_type}_{name}_s{seed}.pt")
    torch.save({"state_dict": model.state_dict(), "metrics": test, "seed": seed,
                "task_type": task, "in_dim": dataset.num_features, "hidden": HIDDEN_DIM,
                "out_dim": out_dim, "layers": NUM_LAYERS, "heads": HEADS}, path)
    print(f"[成员] {conv_type}×{name} seed={seed} 测试集: {test}")
    return path, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", nargs="+", default=["GAT", "GraphSAGE", "GraphSAGE"])
    ap.add_argument("--names", nargs="+", default=["ESOL", "FreeSolv", "BBBP"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21])
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    assert len(args.convs) == len(args.names), "convs 与 names 需一一对应"
    for conv, name in zip(args.convs, args.names):
        for seed in args.seeds:
            train_one(conv, name, seed, epochs=args.epochs)
    print("深度集成成员训练完成。")


if __name__ == "__main__":
    main()
