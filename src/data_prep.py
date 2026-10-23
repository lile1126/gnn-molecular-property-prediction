# -*- coding: utf-8 -*-
"""
数据准备模块
============
下载 MoleculeNet 数据集（ESOL / FreeSolv / BBBP），用 PyG 处理为分子图数据，
划分 train/val/test 并保存划分索引，输出数据集统计信息。

用法:
    python src/data_prep.py            # 下载并处理全部三个数据集
    python src/data_prep.py --names ESOL
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from torch_geometric.datasets import MoleculeNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data")
SPLIT_ROOT = os.path.join(ROOT, "reports", "splits")

# 任务类型: regression / classification
TASK_TYPE = {
    "ESOL": "regression",     # 水溶解度 logS
    "FreeSolv": "regression", # 水合自由能
    "BBBP": "classification", # 血脑屏障渗透（二分类）
    "Clintox": "classification",  # 临床毒性 CT_TOX（二分类，正类 7.6%，选题"溶解度或毒性"）
}

# 训练/验证/测试 比例
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1


def make_splits(dataset, name, seed=42):
    """按固定随机种子划分索引并保存"""
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    idx_train = perm[:n_train].tolist()
    idx_val = perm[n_train : n_train + n_val].tolist()
    idx_test = perm[n_train + n_val :].tolist()
    os.makedirs(SPLIT_ROOT, exist_ok=True)
    split = {
        "name": name,
        "n": n,
        "train": idx_train,
        "val": idx_val,
        "test": idx_test,
        "seed": seed,
    }
    path = os.path.join(SPLIT_ROOT, f"{name}_split.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False)
    return split


def stats(dataset, name):
    """输出数据集统计信息"""
    ys = torch.cat([d.y for d in dataset], dim=0)
    if TASK_TYPE[name] == "regression":
        y = ys[:, 0]
        info = {
            "samples": len(dataset),
            "num_features": dataset.num_features,
            "target_mean": round(float(y.mean()), 4),
            "target_std": round(float(y.std()), 4),
            "target_min": round(float(y.min()), 4),
            "target_max": round(float(y.max()), 4),
            "num_edge_features": dataset.num_edge_features,
        }
    else:
        y = ys[:, 0].long()
        pos = int((y == 1).sum())
        info = {
            "samples": len(dataset),
            "num_features": dataset.num_features,
            "pos_class": pos,
            "neg_class": int(len(y) - pos),
            "pos_ratio": round(float(pos) / len(y), 4),
            "num_edge_features": dataset.num_edge_features,
        }
    return info


def load(name, verbose=True):
    if name not in TASK_TYPE:
        raise ValueError(f"未知数据集 {name}，可选: {list(TASK_TYPE)}")
    if verbose:
        print(f"[{name}] 下载/加载 MoleculeNet ...")
    dataset = MoleculeNet(root=DATA_ROOT, name=name)
    if verbose:
        info = stats(dataset, name)
        print(f"[{name}] 样本数={info['samples']}, 节点特征={info['num_features']}, 边特征={info['num_edge_features']}")
        if TASK_TYPE[name] == "regression":
            print(f"[{name}] 目标 mean={info['target_mean']}, std={info['target_std']}, 范围=[{info['target_min']}, {info['target_max']}]")
        else:
            print(f"[{name}] 正类={info['pos_class']}, 负类={info['neg_class']}, 正类占比={info['pos_ratio']}")
    split = make_splits(dataset, name)
    if verbose:
        print(f"[{name}] 划分: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])} -> {os.path.join(SPLIT_ROOT, name + '_split.json')}")
    return dataset, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["ESOL", "FreeSolv", "BBBP"])
    args = ap.parse_args()
    for name in args.names:
        load(name)
    print("数据准备完成。")


if __name__ == "__main__":
    main()
