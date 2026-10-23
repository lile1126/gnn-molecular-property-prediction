# -*- coding: utf-8 -*-
"""
预训练权重微调与对比实验
========================
用自监督预训练的 convs 权重初始化下游模型（输出头重新随机初始化），
在三个数据集上微调，与无预训练基线（reports/comparison_results.json）对比。

用法:
    python src/finetune_pretrained.py --convs GCN GAT GraphSAGE --names ESOL FreeSolv BBBP \
        --pattern full --epochs 300
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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch_geometric.loader import DataLoader

from data_prep import SPLIT_ROOT, TASK_TYPE, load
from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
PRETRAIN_DIR = os.path.join(ROOT, "models", "pretrained")
REPORT_DIR = os.path.join(ROOT, "reports")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_loader = DataLoader(dataset[[int(i) for i in split["train"]]], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[[int(i) for i in split["val"]]], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[[int(i) for i in split["test"]]], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, optimizer=None, task_type="regression"):
    train = optimizer is not None
    model.train(train)
    total_loss, total = 0.0, 0
    preds, targets = [], []
    for data in loader:
        data = data.to(DEVICE)
        if train:
            optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        y = data.y[:, 0].float()
        loss = torch.nn.functional.cross_entropy(out, y.long()) if task_type == "classification" \
            else torch.nn.functional.mse_loss(out.squeeze(-1), y)
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


def evaluate(model, loader, task_type="regression"):
    model.eval()
    with torch.no_grad():
        _, preds, targets = run_epoch(model, loader, optimizer=None, task_type=task_type)
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


def load_pretrained(model, pretrain_path):
    ckpt = torch.load(pretrain_path, weights_only=False)
    state = ckpt["state_dict"]
    current = model.state_dict()
    missing, unexpected = [], []
    for k, v in state.items():
        if k in current and current[k].shape == v.shape:
            current[k] = v
        else:
            unexpected.append(k)
    model.load_state_dict(current)
    print(f"  加载预训练权重: {len(state)} 组, 未匹配 {len(unexpected)} 组 -> {unexpected[:4]}")
    return model


def train_model(model, train_loader, val_loader, task_type, epochs=300, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val, best_state, bad_epochs = None, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss, _, _ = run_epoch(model, train_loader, optimizer, task_type)
        val_metrics = evaluate(model, val_loader, task_type)
        score = val_metrics["AUC"] if task_type == "classification" else -val_metrics["RMSE"]
        if best_val is None or score > best_val:
            best_val, best_state, bad_epochs = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)
    if verbose:
        print(f"  微调完成 {time.time() - t0:.0f}s, best_val={best_val:.4f}")
    return model


def run_one(conv_type, name, pretrain_path, epochs=300, verbose=True):
    dataset, split = load(name, verbose=verbose)
    train_loader, val_loader, test_loader = collate_loaders(dataset, split)
    out_dim = 2 if TASK_TYPE[name] == "classification" else 1
    model = build_model(conv_type, dataset.num_features, HIDDEN_DIM, out_dim, NUM_LAYERS, heads=HEADS).to(DEVICE)
    model = load_pretrained(model, pretrain_path)
    if verbose:
        print(f"=== {conv_type} × {name}（预训练初始化微调）===")
    model = train_model(model, train_loader, val_loader, TASK_TYPE[name], epochs=epochs, verbose=verbose)
    test_metrics = evaluate(model, test_loader, TASK_TYPE[name])
    if verbose:
        print(f"  {conv_type} × {name} 测试集: {test_metrics}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{conv_type}_{name}_pretrained.pt")
    torch.save({"state_dict": model.state_dict(), "metrics": test_metrics,
                "task_type": TASK_TYPE[name], "in_dim": dataset.num_features,
                "hidden": HIDDEN_DIM, "out_dim": out_dim, "layers": NUM_LAYERS,
                "heads": HEADS, "pretrained": os.path.basename(pretrain_path)}, path)
    return conv_type, name, test_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", nargs="+", default=["GCN", "GAT", "GraphSAGE"])
    ap.add_argument("--names", nargs="+", default=["ESOL", "FreeSolv", "BBBP"])
    ap.add_argument("--pattern", default="full")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    set_seed()
    # 读取基线（无预训练）
    base_path = os.path.join(REPORT_DIR, "comparison_results.json")
    baseline = {}
    if os.path.exists(base_path):
        baseline = json.load(open(base_path, encoding="utf-8"))["results"]
        print("已加载无预训练基线。")
    results = {}
    for name in args.names:
        results[name] = {}
        for conv in args.convs:
            pretrain_path = os.path.join(PRETRAIN_DIR, f"{conv}_pretrain_{args.pattern}.pt")
            if not os.path.exists(pretrain_path):
                print(f"!! 缺少预训练权重: {pretrain_path}，跳过")
                continue
            _, _, m = run_one(conv, name, pretrain_path, epochs=args.epochs)
            row = {"pretrained": m}
            if name in baseline and conv in baseline[name]:
                b = baseline[name][conv]
                row["baseline"] = b
                row["delta"] = {k: round(float(m[k]) - float(b[k]), 4) for k in b
                                if k in m and not k.endswith("_old_argmax")}
            results[name][conv] = row
    out_path = os.path.join(REPORT_DIR, "pretrain_finetune_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": {"pattern": args.pattern, "epochs": args.epochs, "seed": SEED,
                              "device": str(DEVICE)}, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n微调结果已保存: {out_path}")
    print("汇总（baseline -> pretrained, Δ）:")
    for name in args.names:
        for conv in args.convs:
            if name in results and conv in results[name]:
                r = results[name][conv]
                if "delta" in r:
                    print(f"  [{name}/{conv}] baseline={r['baseline']} pretrained={r['pretrained']} Δ={r['delta']}")
                else:
                    print(f"  [{name}/{conv}] pretrained={r['pretrained']}")


if __name__ == "__main__":
    main()
