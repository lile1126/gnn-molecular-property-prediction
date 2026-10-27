# -*- coding: utf-8 -*-
"""
数据准备模块
============
下载 MoleculeNet 数据集（ESOL / FreeSolv / BBBP / Clintox），用 PyG 处理为
分子图数据，划分 train/val/test 并保存划分索引，输出数据集统计信息。

划分文件采用“存在即复用”的策略，并通过数据指纹防止预处理变化后静默沿用
不匹配的实验划分。

用法:
    python src/data_prep.py
    python src/data_prep.py --names ESOL
    python src/data_prep.py --recreate-splits
"""
import argparse
import hashlib
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from torch_geometric.datasets import MoleculeNet

from task_config import TASK_TYPE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data")
SPLIT_ROOT = os.path.join(ROOT, "reports", "splits")
SPLIT_VERSION = 1

# 训练/验证/测试 比例
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1


def dataset_fingerprint(dataset):
    """Stable fingerprint of dataset ordering and SMILES content."""
    digest = hashlib.sha256()
    for data in dataset:
        smiles = getattr(data, "smiles", "")
        if isinstance(smiles, (list, tuple)):
            smiles = ".".join(str(x) for x in smiles)
        digest.update(str(smiles).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def split_path(name):
    return os.path.join(SPLIT_ROOT, f"{name}_split.json")


def validate_split(split, n, name=None):
    """Validate split integrity before using an existing file."""
    required = {"train", "val", "test"}
    missing = required.difference(split)
    if missing:
        raise ValueError(f"划分文件缺少字段: {sorted(missing)}")

    if int(split.get("n", -1)) != n:
        raise ValueError(
            f"{name or split.get('name', '?')} 划分样本数 {split.get('n')} "
            f"与当前数据集 {n} 不一致；如确认重建，请使用 --recreate-splits"
        )

    splits = {key: [int(i) for i in split[key]] for key in required}
    all_indices = [i for key in ("train", "val", "test") for i in splits[key]]
    if len(all_indices) != len(set(all_indices)):
        raise ValueError(f"{name or split.get('name', '?')} 划分中存在重复索引")
    if sorted(all_indices) != list(range(n)):
        raise ValueError(f"{name or split.get('name', '?')} 划分未完整覆盖 0..{n - 1}")


def write_split(split, path):
    """Atomically write a split so interrupted runs cannot corrupt it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def make_splits(dataset, name, seed=42):
    """按固定随机种子划分索引并保存。"""
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    split = {
        "name": name,
        "n": n,
        "train": perm[:n_train].tolist(),
        "val": perm[n_train:n_train + n_val].tolist(),
        "test": perm[n_train + n_val:].tolist(),
        "seed": seed,
        "version": SPLIT_VERSION,
        "dataset_hash": dataset_fingerprint(dataset),
    }
    write_split(split, split_path(name))
    return split


def load_or_make_split(dataset, name, seed=42, recreate=False):
    """Reuse a valid existing split; only create/replace it when necessary."""
    path = split_path(name)
    if os.path.exists(path) and not recreate:
        with open(path, encoding="utf-8") as f:
            split = json.load(f)
        validate_split(split, len(dataset), name=name)

        expected_hash = split.get("dataset_hash")
        if expected_hash:
            actual_hash = dataset_fingerprint(dataset)
            if expected_hash != actual_hash:
                raise ValueError(
                    f"{name} 数据指纹与已保存划分不一致；如确认重建，请使用 --recreate-splits"
                )
        return split
    return make_splits(dataset, name, seed=seed)


def stats(dataset, name):
    """输出数据集统计信息。"""
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


def load(name, verbose=True, recreate_split=False):
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
    split = load_or_make_split(dataset, name, recreate=recreate_split)
    if verbose:
        print(f"[{name}] 划分: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])} -> {split_path(name)}")
    return dataset, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["ESOL", "FreeSolv", "BBBP"])
    ap.add_argument("--recreate-splits", action="store_true",
                    help="重新生成并覆盖已有划分文件")
    args = ap.parse_args()
    for name in args.names:
        load(name, recreate_split=args.recreate_splits)
    print("数据准备完成。")


if __name__ == "__main__":
    main()