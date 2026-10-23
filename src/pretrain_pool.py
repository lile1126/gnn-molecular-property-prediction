# -*- coding: utf-8 -*-
"""
预训练分子池组装
================
从 MoleculeNet 其他数据集（ClinTox / BACE / SIDER / Tox21）与下游三任务训练集
组装无标签分子池，供图对比学习自监督预训练使用。

设计要点：
  - 下游任务仅取训练集分子（避免验证/测试样本以无标签形式进入预训练池，保证评估严谨）
  - 自监督不使用任何标签，只利用分子图结构
  - 输出统一为 PyG 图列表，特征维度与下游一致（num_features=9, edge_features=3）

用法:
    python src/pretrain_pool.py
"""
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
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")

# 非下游数据集（全部用于预训练）
POOL_SOURCES = ["ClinTox", "BACE", "SIDER", "Tox21"]
# 下游任务：仅取训练集
DOWNSTREAM = ["ESOL", "FreeSolv", "BBBP"]


def load_split(name):
    path = os.path.join(REPORT_DIR, "splits", f"{name}_split.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_pool():
    graphs = []
    stats = {}
    for name in POOL_SOURCES:
        ds = MoleculeNet(root=DATA_ROOT, name=name)
        graphs.extend(ds)
        stats[name] = {"samples": len(ds), "used": len(ds)}
        print(f"[池] {name}: {len(ds)} 分子")
    for name in DOWNSTREAM:
        ds = MoleculeNet(root=DATA_ROOT, name=name)
        split = load_split(name)
        idx = [int(i) for i in split["train"]]
        sub = ds[idx]
        graphs.extend(sub)
        stats[name] = {"samples": len(ds), "used": len(idx)}
        print(f"[池] {name}(train): {len(idx)} 分子")
    print(f"预训练分子池合计: {len(graphs)} 分子")
    return graphs, stats


def main():
    graphs, stats = build_pool()
    # 过滤空图（节点数为 0 的异常样本）
    valid = [d for d in graphs if d.x.shape[0] > 0]
    print(f"过滤空图后: {len(valid)} 分子")
    os.makedirs(os.path.join(DATA_ROOT, "pretrain_pool"), exist_ok=True)
    path = os.path.join(DATA_ROOT, "pretrain_pool", "pool.pt")
    torch.save(valid, path)
    meta = {"n": len(valid), "num_features": valid[0].num_features,
            "num_edge_features": valid[0].num_edge_features, "sources": stats}
    meta_path = os.path.join(DATA_ROOT, "pretrain_pool", "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"已保存: {path}")
    print(f"元信息: {meta['n']} 分子, {meta['num_features']} 节点特征")


if __name__ == "__main__":
    main()
