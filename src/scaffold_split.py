# -*- coding: utf-8 -*-
"""
Murcko 骨架划分（scaffold split）
==================================
随机划分会让同骨架分子同时出现在 train/test，造成骨架泄漏、指标偏乐观。
本脚本按 Bemis-Murcko 骨架分组做 8:1:1 划分：同一骨架的分子只进一个集合，
用于检验模型（及融合策略）对"新骨架"的真实泛化能力。

算法（MoleculeNet / DeepChem 标准确定性口径）：
  1. RDKit 提取每个分子的 Murcko 骨架 SMILES（无环分子骨架为空串，归为一组）
  2. 骨架组按大小降序排列（同大小按骨架串字典序，保证确定性）
  3. 依次装填：能放进 train(80%) 就放 train，否则尝试 val(10%)，否则 test

索引口径：基于 featurize.load_featurized(level='a') 的 data_list，
与 train_feat.py 训练时使用的索引完全一致；不改动现有随机划分文件。

用法:
    python src/scaffold_split.py
    python src/scaffold_split.py --names ESOL BBBP

输出:
    reports/splits/{name}_scaffold_split.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from data_prep import SPLIT_ROOT, dataset_fingerprint, write_split
from featurize import load_featurized
from task_config import task_type

NAMES = ["ESOL", "FreeSolv", "BBBP", "Clintox"]


def scaffold_of(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core)  # 无环分子 -> ""


def make_scaffold_split(data_list, name):
    n = len(data_list)
    groups = defaultdict(list)
    for i, d in enumerate(data_list):
        s = scaffold_of(d.smiles)
        if s is None:
            raise ValueError(f"{name} 第 {i} 个分子 SMILES 解析失败: {d.smiles}")
        groups[s].append(i)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    n_train, n_val = int(n * 0.8), int(n * 0.1)
    train, val, test = [], [], []
    for _, idx in ordered:
        if len(train) + len(idx) <= n_train:
            train.extend(idx)
        elif len(val) + len(idx) <= n_val:
            val.extend(idx)
        else:
            test.extend(idx)

    split = {
        "name": name,
        "n": n,
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "seed": None,
        "version": 1,
        "split_type": "scaffold",
        "scaffold_method": "murcko_deterministic_size_desc",
        "n_scaffolds": len(groups),
        "largest_scaffold_group": max(len(v) for v in groups.values()),
        "n_singleton_scaffolds": sum(1 for v in groups.values() if len(v) == 1),
        "dataset_hash": dataset_fingerprint(data_list),
    }
    return split, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=NAMES)
    args = ap.parse_args()
    os.makedirs(SPLIT_ROOT, exist_ok=True)
    for name in args.names:
        data_list, meta = load_featurized(name, level="a", verbose=False)
        # 与现有随机划分做样本数一致性检查（索引口径必须一致）
        rand_path = os.path.join(SPLIT_ROOT, f"{name}_split.json")
        if os.path.exists(rand_path):
            with open(rand_path, encoding="utf-8") as f:
                rand = json.load(f)
            if int(rand["n"]) != len(data_list):
                print(f"[警告] {name}: 随机划分 n={rand['n']} 与特征化数据 n={len(data_list)} 不一致，"
                      f"骨架划分按特征化数据口径生成，请核查！")
        split, groups = make_scaffold_split(data_list, name)
        path = os.path.join(SPLIT_ROOT, f"{name}_scaffold_split.json")
        write_split(split, path)
        print(f"[{name}] n={split['n']} 骨架数={split['n_scaffolds']} "
              f"最大组={split['largest_scaffold_group']} 单例骨架={split['n_singleton_scaffolds']}")
        print(f"  train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} -> {path}")
        if task_type(name) == "classification":
            for key in ("train", "val", "test"):
                ys = [float(data_list[i].y[0]) for i in split[key]]
                pos = sum(ys)
                print(f"  {key}: 正类 {int(pos)}/{len(ys)} ({pos / max(len(ys), 1):.1%})")
    print("骨架划分完成（未改动现有随机划分）。")


if __name__ == "__main__":
    main()