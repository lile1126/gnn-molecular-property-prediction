# -*- coding: utf-8 -*-
"""
HybridAgg 架构级融合模型训练
==============================
方法创新点验证：每层自适应混合 GCN/GAT/GraphSAGE 三种聚合 + 多尺度读出。
训练协议与基线完全一致（hidden=64/layers=3/heads=4/batch=32/lr=1e-3/wd=5e-4/
patience=40/epochs=300/8:1:1），复用 train_eval.train_model，保证可比性。

用法:
    python src/train_hybrid.py                          # ESOL+BBBP, featc, 4种子, 随机划分
    python src/train_hybrid.py --names ESOL FreeSolv BBBP Clintox --split scaffold
    python src/train_hybrid.py --levels b c --seeds 42

输出:
    models/hybrid/HybridAgg_{name}_feat{level}[_scaf]_s{seed}.pt
    reports/hybrid_results.json   （成员/集成指标 + 每层混合权重 + 参照对比）
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
from torch_geometric.loader import DataLoader

from data_prep import SPLIT_ROOT
from featurize import load_featurized
from models import build_model
from task_config import compute_class_weight
from train_eval import (BATCH_SIZE, DEVICE, HIDDEN_DIM, HEADS, NUM_LAYERS,
                        compute_metrics, evaluate, set_seed, train_model)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "hybrid")
REPORT_DIR = os.path.join(ROOT, "reports")


def get_split(name, kind):
    suffix = "_scaffold_split.json" if kind == "scaffold" else "_split.json"
    with open(os.path.join(SPLIT_ROOT, f"{name}{suffix}"), encoding="utf-8") as f:
        return json.load(f)


def loader_of(data_list, idx, shuffle=False):
    return DataLoader([data_list[int(i)] for i in idx], batch_size=BATCH_SIZE, shuffle=shuffle)


def infer(model, data_list, idx):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader_of(data_list, idx):
            out = model.forward_data(batch)
            preds.append(torch.softmax(out, -1)[:, 1] if out.shape[-1] == 2 else out.squeeze(-1))
    return torch.cat(preds)


def ckpt_path(name, level, kind, seed):
    tag = "_scaf" if kind == "scaffold" else ""
    return os.path.join(MODEL_DIR, f"HybridAgg_{name}_feat{level}{tag}_s{seed}.pt")


def run_one(name, level, kind, seed, epochs):
    set_seed(seed)
    data_list, meta = load_featurized(name, level=level, verbose=False)
    split = get_split(name, kind)
    task = meta["task"]
    out_dim = 2 if task == "classification" else 1
    path = ckpt_path(name, level, kind, seed)
    if os.path.exists(path):
        ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
        model = build_model("HybridAgg", meta["atom_dim"], ckpt["hidden"], ckpt["out_dim"],
                            ckpt["layers"], heads=ckpt["heads"],
                            edge_dim=meta["bond_dim"], fp_dim=meta["fp_dim"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  [复用] {os.path.basename(path)}")
    else:
        model = build_model("HybridAgg", meta["atom_dim"], HIDDEN_DIM, out_dim, NUM_LAYERS,
                            heads=HEADS, edge_dim=meta["bond_dim"], fp_dim=meta["fp_dim"]).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"=== HybridAgg × {name} × feat{level} × {kind} × s{seed} (params={n_params}) ===")
        class_weight = compute_class_weight(name, data_list, split["train"], target_col=0)
        t0 = time.time()
        model, _ = train_model(
            model, loader_of(data_list, split["train"], shuffle=True),
            loader_of(data_list, split["val"]), task,
            epochs=epochs, verbose=True, class_weight=class_weight, y_col=0,
        )
        print(f"  用时 {time.time() - t0:.0f}s")
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "model_type": "HybridAgg",
                    "task_type": task, "in_dim": meta["atom_dim"],
                    "edge_dim": meta["bond_dim"], "fp_dim": meta["fp_dim"],
                    "hidden": HIDDEN_DIM, "out_dim": out_dim, "layers": NUM_LAYERS,
                    "heads": HEADS, "seed": seed, "split": kind,
                    "mix_weights": model.mix_weights()}, path)
    test_metrics = evaluate(model, loader_of(data_list, split["test"]), task, y_col=0)
    pv = infer(model, data_list, split["val"])
    pt = infer(model, data_list, split["test"])
    print(f"  s{seed} test: {test_metrics} mix={[[round(w, 3) for w in layer] for layer in model.mix_weights()]}")
    return test_metrics, pv, pt, model.mix_weights()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["ESOL", "BBBP"])
    ap.add_argument("--levels", nargs="+", default=["c"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 13, 21])
    ap.add_argument("--split", choices=["random", "scaffold"], default="random")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    ref_path = os.path.join(REPORT_DIR, "diversity", "fusion_preview.json")
    reference = {}
    if os.path.exists(ref_path) and args.split == "random":
        with open(ref_path, encoding="utf-8") as f:
            ref = json.load(f)
        for n, r in ref.items():
            reference[n] = {s: m["test"] for s, m in r["strategies"].items()
                            if s in ("best_homo_ensemble(val选)", "hetero_equal_all9")}

    out_path = os.path.join(REPORT_DIR, "hybrid_results.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)

    for name in args.names:
        for level in args.levels:
            key = f"{name}|feat{level}|{args.split}"
            members, pvs, pts, mixes = [], [], [], []
            for seed in args.seeds:
                m, pv, pt, mix = run_one(name, level, args.split, seed, args.epochs)
                members.append({"seed": seed, "test": m})
                pvs.append(pv.numpy())
                pts.append(pt.numpy())
                mixes.append(mix)
            data_list, meta = load_featurized(name, level=level, verbose=False)
            split = get_split(name, args.split)
            task = meta["task"]
            y_val = torch.tensor([float(data_list[int(i)].y[0]) for i in split["val"]])
            y_test = torch.tensor([float(data_list[int(i)].y[0]) for i in split["test"]])
            ens_val = compute_metrics(torch.from_numpy(np.mean(pvs, axis=0)), y_val, task)
            ens_test = compute_metrics(torch.from_numpy(np.mean(pts, axis=0)), y_test, task)
            mean_mix = np.mean(mixes, axis=0).round(4).tolist()
            results[key] = {
                "config": {"hidden": HIDDEN_DIM, "layers": NUM_LAYERS, "heads": HEADS,
                           "epochs": args.epochs, "split": args.split, "device": str(DEVICE)},
                "members": members,
                "ensemble_val": ens_val,
                "ensemble_test": ens_test,
                "mean_mix_weights_per_layer": mean_mix,
                "reference": reference.get(name, {}),
            }
            print(f"[{key}] 集成 test: {ens_test}")
            print(f"  平均混合权重(每层 GCN/GAT/SAGE): {mean_mix}")
            if name in reference:
                print(f"  参照: {reference[name]}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {out_path}")


if __name__ == "__main__":
    main()