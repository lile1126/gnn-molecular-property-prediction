# -*- coding: utf-8 -*-
"""
骨架划分多种子实验（P0-2）
=========================
目的：把 Murcko 骨架划分下的 9 个「架构 × 特征档」配置补到 3 个以上训练种子，
      为「最强单模型 / 同构集成 / 异质 all9 集成」的对比提供均值±标准差与
      可做 bootstrap / 配对检验的逐样本预测。

设计：
  - 数据集：ESOL / FreeSolv / BBBP（Clintox 骨架 test 正样本仅 5 个，不纳入）
  - 配置：GCN/GAT/GraphSAGE × feat_a/b/c = 9 个
  - 种子：42（已有 models/feat_scaffold/*.pt）+ 7/13/21（本脚本补训）
  - 协议与 train_feat.py 完全一致：hidden=64 / layers=3 / heads=4 / batch=32 /
    lr=1e-3 / wd=5e-4 / patience=40 / epochs≤300，模型选择只看 val

产物：
  - models/feat_scaffold/{conv}_{name}_feat{level}_s{seed}.pt（seed≠42）
  - reports/scaffold_multiseed_results.json
      ├─ config：训练协议
      └─ datasets[name]：y_val / y_test、每个成员的 val/test 指标与逐样本预测
         （逐样本预测用于 analyze_scaffold_multiseed.py 的 bootstrap 与配对检验）

用法：
    python src/train_scaffold_multiseed.py --jobs 6
    python src/train_scaffold_multiseed.py --seeds 7 13 21 --names ESOL
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

from data_prep import SPLIT_ROOT
from featurize import load_featurized
from models import build_model
from task_config import compute_class_weight, TASK_TYPE
from train_eval import (BATCH_SIZE, DEVICE, HEADS, HIDDEN_DIM, LR, NUM_LAYERS, PATIENCE,
                        WEIGHT_DECAY, compute_metrics, evaluate, set_seed, train_model)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAF_DIR = os.path.join(ROOT, "models", "feat_scaffold")
REPORT_PATH = os.path.join(ROOT, "reports", "scaffold_multiseed_results.json")

NAMES = ["ESOL", "FreeSolv", "BBBP"]
CONVS = ["GCN", "GAT", "GraphSAGE"]
LEVELS = ["a", "b", "c"]
BASE_SEED = 42            # models/feat_scaffold/{conv}_{name}_feat{level}.pt 即 seed42
NEW_SEEDS = [7, 13, 21]   # 补训的 3 个种子 → 合计 4 个种子
EPOCHS = 300


def ckpt_path(conv, name, level, seed):
    if seed == BASE_SEED:
        return os.path.join(SCAF_DIR, f"{conv}_{name}_feat{level}.pt")
    return os.path.join(SCAF_DIR, f"{conv}_{name}_feat{level}_s{seed}.pt")


def get_scaffold_split(name):
    with open(os.path.join(SPLIT_ROOT, f"{name}_scaffold_split.json"), encoding="utf-8") as f:
        return json.load(f)


def _loaders(data_list, split, shuffle_train):
    return (
        DataLoader([data_list[int(i)] for i in split["train"]], batch_size=BATCH_SIZE, shuffle=shuffle_train),
        DataLoader([data_list[int(i)] for i in split["val"]], batch_size=BATCH_SIZE, shuffle=False),
        DataLoader([data_list[int(i)] for i in split["test"]], batch_size=BATCH_SIZE, shuffle=False),
    )


def train_member(conv, name, level, seed, epochs=EPOCHS):
    """训练单个成员并落盘（顶层函数，Windows spawn 可 pickle）。"""
    torch.set_num_threads(1)
    set_seed(seed)
    data_list, meta = load_featurized(name, level=level, verbose=False)
    split = get_scaffold_split(name)
    task_type = meta["task"]
    out_dim = 2 if task_type == "classification" else 1
    model = build_model(conv, meta["atom_dim"], HIDDEN_DIM, out_dim, NUM_LAYERS,
                        heads=HEADS, edge_dim=meta["bond_dim"], fp_dim=meta["fp_dim"]).to(DEVICE)
    train_loader, val_loader, test_loader = _loaders(data_list, split, shuffle_train=True)
    class_weight = compute_class_weight(name, data_list, split["train"], target_col=0)
    model, _ = train_model(model, train_loader, val_loader, task_type,
                           epochs=epochs, verbose=False, class_weight=class_weight, y_col=0)
    val_metrics = evaluate(model, val_loader, task_type, y_col=0)
    test_metrics = evaluate(model, test_loader, task_type, y_col=0)
    os.makedirs(SCAF_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": test_metrics,
                "val_metrics": val_metrics, "seed": seed, "task_type": task_type,
                "in_dim": meta["atom_dim"], "edge_dim": meta["bond_dim"], "fp_dim": meta["fp_dim"],
                "hidden": HIDDEN_DIM, "out_dim": out_dim, "layers": NUM_LAYERS,
                "heads": HEADS, "split": "scaffold"}, ckpt_path(conv, name, level, seed))
    return conv, name, level, seed, val_metrics, test_metrics


def _worker(args):
    conv, name, level, seed, epochs = args
    try:
        conv, name, level, seed, vm, tm = train_member(conv, name, level, seed, epochs)
        print(f"[成员] {conv}×{name}×feat{level} s{seed}: test={tm}", flush=True)
        return {"ok": True, "key": f"{conv}|{name}|{level}|{seed}", "val": vm, "test": tm}
    except Exception as e:
        print(f"[ERROR] {conv}×{name}×feat{level} s{seed}: {e}", flush=True)
        return {"ok": False, "key": f"{conv}|{name}|{level}|{seed}", "error": str(e)}


@torch.no_grad()
def _infer(model, loader):
    model.eval()
    preds, targets = [], []
    for data in loader:
        data = data.to(DEVICE)
        out = model.forward_data(data)
        if out.shape[-1] == 2:
            preds.append(torch.softmax(out, dim=-1)[:, 1].cpu())
        else:
            preds.append(out.squeeze(-1).cpu())
        targets.append(data.y[:, 0].float().cpu())
    return torch.cat(preds), torch.cat(targets)


def collect_dataset(name, seeds):
    """加载全部成员，重算 val/test 逐样本预测并汇总（含 seed42 主模型）。"""
    data_list, meta = load_featurized(name, level="a", verbose=False)
    split = get_scaffold_split(name)
    task_type = meta["task"]
    y_val = np.array([float(data_list[int(i)].y[0]) for i in split["val"]], dtype=np.float64)
    y_test = np.array([float(data_list[int(i)].y[0]) for i in split["test"]], dtype=np.float64)
    members = {}
    for level in LEVELS:
        dl, mt = load_featurized(name, level=level, verbose=False)
        _, val_loader, test_loader = _loaders(dl, split, shuffle_train=False)
        for conv in CONVS:
            for seed in seeds:
                path = ckpt_path(conv, name, level, seed)
                if not os.path.exists(path):
                    print(f"  [缺失] {os.path.basename(path)}")
                    continue
                ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
                model = build_model(conv, mt["atom_dim"], ckpt["hidden"], ckpt["out_dim"],
                                    ckpt["layers"], heads=ckpt.get("heads", HEADS),
                                    edge_dim=mt["bond_dim"], fp_dim=mt["fp_dim"]).to(DEVICE)
                model.load_state_dict(ckpt["state_dict"])
                pv, _ = _infer(model, val_loader)
                pt, _ = _infer(model, test_loader)
                key = f"{conv}_feat{level}_s{seed}"
                members[key] = {
                    "conv": conv, "level": level, "seed": seed,
                    "file": os.path.basename(path),
                    "val": compute_metrics(pv, torch.from_numpy(y_val).float(), task_type),
                    "test": compute_metrics(pt, torch.from_numpy(y_test).float(), task_type),
                    "val_pred": [round(float(v), 4) for v in pv.numpy()],
                    "test_pred": [round(float(v), 4) for v in pt.numpy()],
                }
                print(f"  {name} {key}: val={members[key]['val']} test={members[key]['test']}")
    return {
        "task": task_type,
        "n_train": len(split["train"]), "n_val": len(split["val"]), "n_test": len(split["test"]),
        "split_file": os.path.relpath(os.path.join(SPLIT_ROOT, f"{name}_scaffold_split.json"), ROOT),
        "y_val": [round(float(v), 4) for v in y_val],
        "y_test": [round(float(v), 4) for v in y_test],
        "members": members,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=NAMES)
    ap.add_argument("--convs", nargs="+", default=CONVS)
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=NEW_SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    tasks = []
    if not args.eval_only:
        for name in args.names:
            for conv in args.convs:
                for level in args.levels:
                    for seed in args.seeds:
                        if os.path.exists(ckpt_path(conv, name, level, seed)):
                            print(f"  跳过（已存在）: {conv}×{name}×feat{level} s{seed}")
                            continue
                        tasks.append((conv, name, level, seed, args.epochs))

    if tasks:
        t0 = time.time()
        print(f"待训练成员: {len(tasks)} 个（并行度 {args.jobs}）", flush=True)
        with multiprocessing.Pool(min(args.jobs, len(tasks))) as pool:
            results = pool.map(_worker, tasks)
        ok = sum(1 for r in results if r["ok"])
        print(f"训练完成: 成功 {ok}/{len(results)}，用时 {time.time() - t0:.0f}s")
    else:
        print("无需训练（全部已存在或 --eval-only）")

    seeds = sorted(set([BASE_SEED] + list(args.seeds)))
    report = {
        "config": {
            "split": "scaffold", "hidden": HIDDEN_DIM, "layers": NUM_LAYERS, "heads": HEADS,
            "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE, "epochs": args.epochs, "device": str(DEVICE),
            "seeds": seeds, "base_seed": BASE_SEED,
            "note": "seed42 复用 models/feat_scaffold 既有权重；7/13/21 由本脚本补训",
        },
    }
    for name in args.names:
        print(f"\n=== 汇总 {name} ===")
        report.setdefault("datasets", {})[name] = collect_dataset(name, seeds)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    old = {}
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, encoding="utf-8") as f:
            old = json.load(f)
    old = old.get("datasets", {})
    old.update(report["datasets"])
    report["datasets"] = old
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {REPORT_PATH}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
