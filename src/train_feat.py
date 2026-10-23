# -*- coding: utf-8 -*-
"""
特征工程对比实验（选题深挖 · 方向1）
====================================
对照选题"分子特征工程（分子指纹、图构建）"：
  feat_a : 原子特征增强（RDKit 27 维，对比基线默认 9 维）
  feat_b : 原子 + 键特征（键型/共轭/环内/立体 7 维）
  feat_c : 原子 + 键 + MACCS 指纹（166 位，GNN 表征与指纹融合）

实验矩阵：
  GCN  × {a, b, c} × {ESOL, FreeSolv, BBBP, Clintox}
  GAT / GraphSAGE × {c} × 4 数据集

统一超参与基线一致（hidden=64/layers=3/heads=4/batch=32/lr=1e-3/wd=5e-4/
patience=40/epochs=300/seed=42/8:1:1），复用同一划分，保证与基线可比。

用法:
    python src/train_feat.py
    python src/train_feat.py --convs GCN --names ESOL --levels a
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

import torch
from torch_geometric.loader import DataLoader

from data_prep import SPLIT_ROOT, TASK_TYPE, load as load_raw
from featurize import load_featurized
from models import build_model
from train_eval import (BATCH_SIZE, DEVICE, HIDDEN_DIM, HEADS, LR, NUM_LAYERS,
                        PATIENCE, SEED, WEIGHT_DECAY, compute_metrics, evaluate,
                        run_epoch, set_seed, train_model)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_MODEL_DIR = os.path.join(ROOT, "models", "feat")
REPORT_DIR = os.path.join(ROOT, "reports")


def collate(data_list, idx, batch_size=BATCH_SIZE):
    return DataLoader([data_list[int(i)] for i in idx], batch_size=batch_size, shuffle=True)


def get_split(name):
    split_path = os.path.join(SPLIT_ROOT, f"{name}_split.json")
    if os.path.exists(split_path):
        with open(split_path, encoding="utf-8") as f:
            return json.load(f)
    # Clintox 等未划分数据集，用同一随机种子生成
    _, split = load_raw(name, verbose=False)
    return split


def run_feat_one(conv_type, name, level, epochs=300, verbose=True):
    data_list, meta = load_featurized(name, level=level, verbose=verbose)
    split = get_split(name)
    task_type = meta["task"]
    out_dim = 2 if task_type == "classification" else 1
    model = build_model(conv_type, meta["atom_dim"], HIDDEN_DIM, out_dim, NUM_LAYERS,
                        heads=HEADS, edge_dim=meta["bond_dim"], fp_dim=meta["fp_dim"]).to(DEVICE)
    if verbose:
        print(f"=== {conv_type} × {name} × feat_{level} "
              f"(atom={meta['atom_dim']}, bond={meta['bond_dim']}, fp={meta['fp_dim']}) ===")
    train_loader = collate(data_list, split["train"])
    val_loader = collate(data_list, split["val"])
    test_loader = collate(data_list, split["test"])
    model, history = train_model(model, train_loader, val_loader, task_type,
                                 epochs=epochs, verbose=verbose)
    test_metrics = evaluate(model, test_loader, task_type)
    if verbose:
        print(f"  {conv_type} × {name} × feat_{level} 测试集: {test_metrics}")
    os.makedirs(FEAT_MODEL_DIR, exist_ok=True)
    path = os.path.join(FEAT_MODEL_DIR, f"{conv_type}_{name}_feat{level}.pt")
    torch.save({"state_dict": model.state_dict(), "metrics": test_metrics,
                "task_type": task_type, "in_dim": meta["atom_dim"],
                "edge_dim": meta["bond_dim"], "fp_dim": meta["fp_dim"],
                "hidden": HIDDEN_DIM, "out_dim": out_dim, "layers": NUM_LAYERS,
                "heads": HEADS}, path)
    return conv_type, name, level, test_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", nargs="+", default=["GCN", "GAT", "GraphSAGE"])
    ap.add_argument("--names", nargs="+", default=["ESOL", "FreeSolv", "BBBP", "Clintox"])
    ap.add_argument("--levels", nargs="+", default=["a", "b", "c"])
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    set_seed(SEED)
    # 实验矩阵：GCN 跑全部特征档，其余架构只跑最高档（c）
    matrix = []
    for name in args.names:
        for conv in args.convs:
            if conv == "GCN":
                levels = [l for l in args.levels]
            else:
                levels = ["c"] if "c" in args.levels else [args.levels[-1]]
            for level in levels:
                matrix.append((conv, name, level))

    results = {"baseline_ref": "reports/comparison_results.json（默认 9 维特征，无键特征/指纹）",
               "config": {"hidden": HIDDEN_DIM, "layers": NUM_LAYERS, "heads": HEADS,
                          "epochs": args.epochs, "batch_size": BATCH_SIZE, "seed": SEED,
                          "device": str(DEVICE)}}
    t0 = time.time()
    skipped = 0
    for conv, name, level in matrix:
        ckpt = os.path.join(FEAT_MODEL_DIR, f"{conv}_{name}_feat{level}.pt")
        if os.path.exists(ckpt):
            skipped += 1
            print(f"  跳过（已存在）: {conv} × {name} × feat{level}")
            continue
        try:
            _, _, _, metrics = run_feat_one(conv, name, level, epochs=args.epochs)
            results.setdefault(name, {})[f"{conv}_feat{level}"] = metrics
            print(f"  累计用时 {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"  [ERROR] {conv} × {name} × {level}: {e}")
            results.setdefault(name, {})[f"{conv}_feat{level}"] = {"error": str(e)}
    print(f"（跳过已存在 {skipped} 个）")
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "feat_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n特征工程对比结果已保存: {path}")
    for name in results:
        if name in ("baseline_ref", "config"):
            continue
        print(f"  [{name}] {results[name]}")


if __name__ == "__main__":
    main()
