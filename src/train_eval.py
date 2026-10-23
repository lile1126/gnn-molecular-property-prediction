# -*- coding: utf-8 -*-
"""
训练与评估模块
==============
- 训练循环（Adam + early stopping，保存最佳模型权重）
- 回归指标: RMSE / MAE / R²
- 分类指标: Accuracy / ROC-AUC / F1
- 三种架构（GCN / GAT / GraphSAGE）× 三个数据集（ESOL / FreeSolv / BBBP）对比实验

用法:
    python src/train_eval.py --convs GCN GAT GraphSAGE --names ESOL FreeSolv BBBP
"""
import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch_geometric.loader import DataLoader

from data_prep import SPLIT_ROOT, TASK_TYPE, load
from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 64
NUM_LAYERS = 3
HEADS = 4
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate_loaders(dataset, split, batch_size=BATCH_SIZE):
    idx_train, idx_val, idx_test = split["train"], split["val"], split["test"]
    train_set = dataset[[int(i) for i in idx_train]]
    val_set = dataset[[int(i) for i in idx_val]]
    test_set = dataset[[int(i) for i in idx_test]]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, optimizer=None, task_type="regression", class_weight=None, y_col=0):
    """一个 epoch。optimizer 为 None 时进入评估模式。返回 (loss, preds, targets)
    y_col: 多标签数据集（如 Clintox [n,2]）取标签列，Clintox 传 1（CT_TOX）。"""
    train = optimizer is not None
    model.train(train)
    total_loss, total = 0.0, 0
    preds, targets = [], []
    for data in loader:
        data = data.to(DEVICE)
        if train:
            optimizer.zero_grad()
        out = model.forward_data(data)
        y = data.y[:, y_col].float()
        if task_type == "classification":
            loss = torch.nn.functional.cross_entropy(out, y.long(), weight=class_weight)
        else:
            loss = torch.nn.functional.mse_loss(out.squeeze(-1), y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(data.y)
        total += len(data.y)
        if task_type == "classification":
            preds.append(torch.softmax(out, dim=-1)[:, 1].cpu())  # 概率口径（与权威基线一致）
        else:
            preds.append(out.squeeze(-1).cpu())
        targets.append(y.cpu())
    return total_loss / max(total, 1), torch.cat(preds), torch.cat(targets)


def evaluate(model, loader, task_type="regression", y_col=0):
    model.eval()
    with torch.no_grad():
        _, preds, targets = run_epoch(model, loader, optimizer=None, task_type=task_type, y_col=y_col)
    return compute_metrics(preds, targets, task_type)


def compute_metrics(preds, targets, task_type):
    preds = preds.numpy().astype(np.float64)
    targets = targets.numpy().astype(np.float64)
    if task_type == "regression":
        return {
            "RMSE": round(float(np.sqrt(mean_squared_error(targets, preds))), 4),
            "MAE": round(float(mean_absolute_error(targets, preds)), 4),
            "R2": round(float(r2_score(targets, preds)), 4),
        }
    # classification
    pred_labels = (preds > 0.5).astype(int)
    try:
        auc = float(roc_auc_score(targets, preds))
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": round(float(accuracy_score(targets, pred_labels)), 4),
        "AUC": round(auc, 4),
        "F1": round(float(f1_score(targets, pred_labels)), 4),
    }


def train_model(model, train_loader, val_loader, task_type, epochs=DEFAULT_EPOCHS, lr=LR,
                weight_decay=WEIGHT_DECAY, verbose=True, class_weight=None, y_col=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, bad_epochs = None, None, 0
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss, _, _ = run_epoch(model, train_loader, optimizer, task_type, class_weight=class_weight, y_col=y_col)
        val_metrics = evaluate(model, val_loader, task_type, y_col=y_col)
        score = val_metrics["AUC"] if task_type == "classification" else -val_metrics["RMSE"]
        history.append({"epoch": epoch, "train_loss": round(loss, 4), **val_metrics})
        if best_val is None or score > best_val:
            best_val, best_state, bad_epochs = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                if verbose:
                    print(f"  early stop @ epoch {epoch}, best_val={best_val:.4f}")
                break
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d} loss={loss:.4f} val={val_metrics}")
    model.load_state_dict(best_state)
    if verbose:
        print(f"  训练完成, 用时 {time.time() - t0:.1f}s, 最佳验证指标 {best_val:.4f}")
    return model, history


def run_one(conv_type, name, epochs=DEFAULT_EPOCHS, verbose=True):
    """训练单个 (架构, 数据集) 组合并返回测试集指标"""
    dataset, split = load(name, verbose=verbose)
    train_loader, val_loader, test_loader = collate_loaders(dataset, split)
    out_dim = 2 if TASK_TYPE[name] == "classification" else 1
    model = build_model(conv_type, dataset.num_features, HIDDEN_DIM, out_dim, NUM_LAYERS, heads=HEADS).to(DEVICE)
    if verbose:
        print(f"=== {conv_type} × {name} ===")
    model, history = train_model(model, train_loader, val_loader, TASK_TYPE[name], epochs=epochs, verbose=verbose)
    test_metrics = evaluate(model, test_loader, TASK_TYPE[name])
    if verbose:
        print(f"  {conv_type} × {name} 测试集: {test_metrics}")
    # 保存最佳模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{conv_type}_{name}.pt")
    torch.save({"state_dict": model.state_dict(), "metrics": test_metrics,
                "task_type": TASK_TYPE[name], "in_dim": dataset.num_features,
                "hidden": HIDDEN_DIM, "out_dim": out_dim, "layers": NUM_LAYERS, "heads": HEADS}, path)
    return conv_type, name, test_metrics, history


def run_comparison(convs=None, names=None, epochs=DEFAULT_EPOCHS):
    convs = convs or ["GCN", "GAT", "GraphSAGE"]
    names = names or ["ESOL", "FreeSolv", "BBBP"]
    results = {}
    for name in names:
        results[name] = {}
        for conv in convs:
            _, _, metrics, _ = run_one(conv, name, epochs=epochs)
            results[name][conv] = metrics
    # 保存对比结果
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "comparison_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "config": {
            "hidden": HIDDEN_DIM, "layers": NUM_LAYERS, "heads": HEADS,
            "epochs": epochs, "batch_size": BATCH_SIZE, "seed": SEED,
            "device": str(DEVICE)}}, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存: {path}")
    print("汇总:")
    for name in names:
        row = " | ".join(f"{conv}: {results[name][conv]}" for conv in convs)
        print(f"  [{name}] {row}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", nargs="+", default=["GCN", "GAT", "GraphSAGE"])
    ap.add_argument("--names", nargs="+", default=["ESOL", "FreeSolv", "BBBP"])
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = ap.parse_args()
    set_seed()
    run_comparison(convs=args.convs, names=args.names, epochs=args.epochs)


if __name__ == "__main__":
    main()
