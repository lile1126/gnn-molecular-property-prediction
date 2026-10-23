# -*- coding: utf-8 -*-
"""
特征模型深度集成成员训练（方向 9）
==================================
对特征工程全组合训练多 seed 集成成员：
  矩阵：GCN × {a,b,c} × 4 数据集 + GAT/GraphSAGE × {c} × 4 数据集 = 20 组
  seeds：7 / 13 / 21（与基线集成一致）
成员与 seed42 特征模型（models/feat/*.pt）共同构成 4 成员深度集成，
使 Web「深度集成」支持特征工程等级（feat_a/b/c），不再仅限基线。

Clintox 特征数据 y 已取 col1（featurize.py），类别加权 [1,12] 补偿 7.6% 正类。

用法:
    python src/train_feat_ensemble.py            # 全部 60 个成员
    python src/train_feat_ensemble.py --jobs 3   # 并行度
"""
import argparse
import json
import multiprocessing
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data_prep import SPLIT_ROOT, TASK_TYPE, load as load_raw
from featurize import load_featurized
from models import build_model
from train_eval import (BATCH_SIZE, DEVICE, HIDDEN_DIM, HEADS, LR, NUM_LAYERS,
                        PATIENCE, WEIGHT_DECAY, compute_metrics, evaluate,
                        run_epoch, set_seed, train_model)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_ENS_DIR = os.path.join(ROOT, "models", "feat_ensemble")
FEAT_MODEL_DIR = os.path.join(ROOT, "models", "feat")
REPORT_DIR = os.path.join(ROOT, "reports")

DATASETS = ["ESOL", "FreeSolv", "BBBP", "Clintox"]
SEEDS = [7, 13, 21]
EPOCHS = 300


def _matrix():
    """全组合矩阵：3 架构 × {a,b,c} × 4 数据集 = 36 组（Web 所有可选特征组合）"""
    tasks = []
    for name in DATASETS:
        for lv in ("a", "b", "c"):
            tasks.append(("GCN", name, lv))
            tasks.append(("GAT", name, lv))
            tasks.append(("GraphSAGE", name, lv))
    return tasks


def get_split(name):
    split_path = os.path.join(SPLIT_ROOT, f"{name}_split.json")
    if os.path.exists(split_path):
        with open(split_path, encoding="utf-8") as f:
            return json.load(f)
    _, split = load_raw(name, verbose=False)
    return split


def collate(data_list, idx, shuffle):
    return DataLoader([data_list[int(i)] for i in idx], batch_size=BATCH_SIZE, shuffle=shuffle)


def train_one_member(conv, name, level, seed, epochs=EPOCHS):
    """训练单个特征模型集成成员（顶层函数，multiprocessing spawn 可 pickle）"""
    set_seed(seed)
    data_list, meta = load_featurized(name, level=level, verbose=False)
    split = get_split(name)
    task_type = meta["task"]
    out_dim = 2 if task_type == "classification" else 1
    model = build_model(conv, meta["atom_dim"], HIDDEN_DIM, out_dim, NUM_LAYERS,
                        heads=HEADS, edge_dim=meta["bond_dim"], fp_dim=meta["fp_dim"]).to(DEVICE)
    is_clintox = name == "Clintox"
    class_weight = torch.tensor([1.0, 12.0]).to(DEVICE) if is_clintox else None
    train_loader = collate(data_list, split["train"], shuffle=True)
    val_loader = collate(data_list, split["val"], shuffle=False)
    test_loader = collate(data_list, split["test"], shuffle=False)
    model, _ = train_model(model, train_loader, val_loader, task_type,
                           epochs=epochs, verbose=False,
                           class_weight=class_weight, y_col=0)
    test = evaluate(model, test_loader, task_type, y_col=0)
    os.makedirs(FEAT_ENS_DIR, exist_ok=True)
    path = os.path.join(FEAT_ENS_DIR, f"{conv}_{name}_feat{level}_s{seed}.pt")
    torch.save({"state_dict": model.state_dict(), "metrics": test, "seed": seed,
                "task_type": task_type, "in_dim": meta["atom_dim"],
                "edge_dim": meta["bond_dim"], "fp_dim": meta["fp_dim"],
                "hidden": HIDDEN_DIM, "out_dim": out_dim, "layers": NUM_LAYERS,
                "heads": HEADS}, path)
    return conv, name, level, seed, test


def _worker(args):
    conv, name, level, seed, epochs = args
    try:
        conv, name, level, seed, test = train_one_member(conv, name, level, seed, epochs)
        print(f"[成员] {conv}×{name}×feat{level} seed={seed}: {test}", flush=True)
        return {"ok": True, "key": f"{conv}|{name}|{level}|{seed}", "metrics": test}
    except Exception as e:
        print(f"[ERROR] {conv}×{name}×feat{level} seed={seed}: {e}", flush=True)
        return {"ok": False, "key": f"{conv}|{name}|{level}|{seed}", "error": str(e)}


def eval_ensemble(conv, name, level, task_type):
    """集成评估：seed42 特征模型（baseline 成员）+ 3 个新成员，测试集平均"""
    import torch.nn.functional as F
    members = []
    base_path = os.path.join(FEAT_MODEL_DIR, f"{conv}_{name}_feat{level}.pt")
    if not os.path.exists(base_path):
        return None
    ckpt = torch.load(base_path, map_location=DEVICE, weights_only=False)
    out_dim = 2 if task_type == "classification" else 1
    m0 = build_model(conv, ckpt["in_dim"], ckpt["hidden"], out_dim, ckpt["layers"],
                     heads=ckpt.get("heads", 4), edge_dim=ckpt.get("edge_dim", 0),
                     fp_dim=ckpt.get("fp_dim", 0)).to(DEVICE)
    m0.load_state_dict(ckpt["state_dict"]); m0.eval()
    members.append(m0)
    for seed in SEEDS:
        p = os.path.join(FEAT_ENS_DIR, f"{conv}_{name}_feat{level}_s{seed}.pt")
        if not os.path.exists(p):
            continue
        c = torch.load(p, map_location=DEVICE, weights_only=False)
        m = build_model(conv, c["in_dim"], c["hidden"], out_dim, c["layers"],
                        heads=c.get("heads", 4), edge_dim=c.get("edge_dim", 0),
                        fp_dim=c.get("fp_dim", 0)).to(DEVICE)
        m.load_state_dict(c["state_dict"]); m.eval()
        members.append(m)
    if len(members) < 2:
        return None
    data_list, meta = load_featurized(name, level=level, verbose=False)
    split = get_split(name)
    test_loader = collate(data_list, split["test"], shuffle=False)
    preds, targets = [], None
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            x = data.x.float()
            edge_attr = getattr(data, "edge_attr", None)
            fp = getattr(data, "fp", None)
            batch = data.batch if data.batch is not None else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            outs = []
            for m in members:
                out = m(x, data.edge_index, batch, edge_attr=edge_attr, fp=fp).cpu()
                outs.append(F.softmax(out, dim=1)[:, 1] if task_type == "classification" else out.squeeze(-1))
            preds.append(torch.stack(outs).mean(0))
            if targets is None:
                targets = data.y[:, 0].float().cpu()
            else:
                targets = torch.cat([targets, data.y[:, 0].float().cpu()])
    preds = torch.cat(preds)
    ens_metrics = compute_metrics(preds, targets, task_type)
    member_metrics = [ckpt["metrics"]]
    for seed in SEEDS:
        p = os.path.join(FEAT_ENS_DIR, f"{conv}_{name}_feat{level}_s{seed}.pt")
        if os.path.exists(p):
            c = torch.load(p, map_location=DEVICE, weights_only=False)
            member_metrics.append(c["metrics"])
    member_files = [os.path.basename(os.path.join(FEAT_MODEL_DIR, f"{conv}_{name}_feat{level}.pt"))]
    member_files += [os.path.basename(os.path.join(FEAT_ENS_DIR, f"{conv}_{name}_feat{level}_s{seed}.pt"))
                     for seed in SEEDS if os.path.exists(os.path.join(FEAT_ENS_DIR, f"{conv}_{name}_feat{level}_s{seed}.pt"))]
    return {
        "members": member_metrics,
        "ensemble": ens_metrics,
        "n_members": len(members),
        "member_files": member_files,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3, help="并行训练进程数（GPU 显存允许时）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--eval-only", action="store_true", help="只评估集成（跳过训练）")
    args = ap.parse_args()

    tasks = []
    for conv, name, level in _matrix():
        for seed in SEEDS:
            p = os.path.join(FEAT_ENS_DIR, f"{conv}_{name}_feat{level}_s{seed}.pt")
            if os.path.exists(p):
                print(f"  跳过（已存在）: {conv}×{name}×feat{level} s{seed}")
                continue
            tasks.append((conv, name, level, seed, args.epochs))

    if tasks and not args.eval_only:
        t0 = time.time()
        print(f"待训练成员: {len(tasks)} 个（并行度 {args.jobs}）")
        with multiprocessing.Pool(args.jobs) as pool:
            results = pool.map(_worker, tasks)
        ok = sum(1 for r in results if r["ok"])
        print(f"训练完成: 成功 {ok}/{len(results)}，用时 {time.time() - t0:.0f}s")
    else:
        print("无需训练（全部已存在或 --eval-only）")

    # 集成评估
    report = {}
    for conv, name, level in _matrix():
        task_type = TASK_TYPE[name]
        r = eval_ensemble(conv, name, level, task_type)
        if r:
            report[f"{conv}|{name}|{level}"] = r
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "feat_ensemble_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"集成评估报告: {path}（{len(report)} 组）")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
