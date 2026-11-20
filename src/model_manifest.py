# -*- coding: utf-8 -*-
"""
模型清单生成（P0-5 的"模型版本信息"数据源）
===========================================
扫描 models/ 目录，统计 checkpoint 数量、种子集合与校验和，写出
reports/model_manifest.json，供 Web 的 /api/meta 展示模型版本。

用法：
    python src/model_manifest.py --tag v1.0-freeze-144ckpt
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.path.join(ROOT, "models")
OUT_PATH = os.path.join(ROOT, "reports", "model_manifest.json")

GROUPS = {
    "feat": "随机划分主模型（seed42，9 配置 × 4 数据集）",
    "feat_ensemble": "随机划分额外种子（7/13/21，用于同构/异质集成）",
    "feat_scaffold": "Murcko 骨架划分模型（seed42 + 多种子）",
    "ensemble": "基线特征深度集成成员",
    "hybrid": "HybridAgg 架构级融合（负结果）",
    "distill": "蒸馏学生模型（负结果）",
    "pretrained": "自监督预训练权重",
}


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan():
    groups = {}
    total = 0
    for sub, desc in GROUPS.items():
        d = os.path.join(MODEL_ROOT, sub)
        if not os.path.isdir(d):
            groups[sub] = {"description": desc, "n_files": 0, "files": {}}
            continue
        files = sorted(p for p in os.listdir(d) if p.endswith(".pt"))
        info = {"description": desc, "n_files": len(files),
                "files": {p: {"size_bytes": os.path.getsize(os.path.join(d, p)),
                              "sha256": sha256_of(os.path.join(d, p))[:16]} for p in files}}
        groups[sub] = info
        total += len(files)
    return groups, total


def infer_structure(groups):
    """从文件名推断数据集 / 架构 / 特征档 / 种子集合。"""
    import re
    convs, datasets, levels, seeds = set(), set(), set(), set()
    pattern = re.compile(r"^(?P<conv>GCN|GAT|GraphSAGE)_(?P<ds>[A-Za-z]+)_feat(?P<lv>[abc])(?:_s(?P<seed>\d+))?\.pt$")
    for sub in ("feat", "feat_ensemble", "feat_scaffold"):
        for fn in groups.get(sub, {}).get("files", {}):
            m = pattern.match(fn)
            if not m:
                continue
            convs.add(m.group("conv"))
            datasets.add(m.group("ds"))
            levels.add(m.group("lv"))
            seeds.add(int(m.group("seed")) if m.group("seed") else 42)
    return {"convs": sorted(convs), "datasets": sorted(datasets),
            "feat_levels": sorted(levels), "seeds": sorted(seeds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1.0-freeze-144ckpt")
    ap.add_argument("--skip-hash", action="store_true", help="只统计数量，不计算校验和（更快）")
    args = ap.parse_args()

    groups, total = scan()
    if args.skip_hash:
        for g in groups.values():
            g["files"] = {k: v["size_bytes"] for k, v in g.get("files", {}).items()}
    structure = infer_structure(groups)
    manifest = {
        "tag": args.tag,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_checkpoints": total,
        "n_main_144": groups.get("feat", {}).get("n_files", 0) + groups.get("feat_ensemble", {}).get("n_files", 0),
        "structure": structure,
        "groups": {k: {"description": v["description"], "n_files": v["n_files"]}
                   for k, v in groups.items()},
        "checksums": {k: v["files"] for k, v in groups.items()},
        "note": "144 = feat(36) + feat_ensemble(108)；其余为骨架划分、基线与负结果复现权重",
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"模型清单已写入 {OUT_PATH}")
    print(f"总 checkpoint: {total}；主实验 144 个: {manifest['n_main_144']}")
    print(f"结构: {structure}")


if __name__ == "__main__":
    main()
