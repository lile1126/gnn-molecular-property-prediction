# -*- coding: utf-8 -*-
"""
骨架划分下的异质集成验证
========================
目的：检验"异质等权集成的增益"是否只是随机划分下相似性泄漏的产物。
做法：在 Murcko 骨架划分（scaffold split）上重训 9 个配置（单种子 42），
      用 val 选专家、test 只评估一次，对比 最强单模型 / 异质等权(3专家) / 异质等权(all9)，
      并给出 scaffold 划分下的跨架构误差相关性，与随机划分对照。

Clintox 骨架 test 仅 5 个正样本，指标不可信，不纳入。

用法:
    python src/fusion_scaffold.py
"""
import json
import os
import sys

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
from train_eval import compute_metrics, set_seed, SEED
from train_feat import SCAF_MODEL_DIR, run_feat_one

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "reports")
DEVICE = torch.device("cpu")
NAMES = ["ESOL", "FreeSolv", "BBBP"]
CONVS = ["GCN", "GAT", "GraphSAGE"]
LEVELS = ["a", "b", "c"]


def get_scaffold_split(name):
    with open(os.path.join(SPLIT_ROOT, f"{name}_scaffold_split.json"), encoding="utf-8") as f:
        return json.load(f)


def infer(model, data_list, idx):
    loader = DataLoader([data_list[int(i)] for i in idx], batch_size=64, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            out = model.forward_data(batch)
            if out.shape[-1] == 2:
                preds.append(torch.softmax(out, dim=-1)[:, 1])
            else:
                preds.append(out.squeeze(-1))
    return torch.cat(preds).numpy().astype(np.float64)


def ensure_ckpt(conv, name, level, epochs=300):
    path = os.path.join(SCAF_MODEL_DIR, f"{conv}_{name}_feat{level}.pt")
    if os.path.exists(path):
        print(f"  已存在 {conv}_{name}_feat{level}，跳过训练")
        return path
    set_seed(SEED)
    run_feat_one(conv, name, level, epochs=epochs, verbose=True, split_kind="scaffold")
    return path


def load_and_infer(conv, name, level, data_list, meta, split):
    path = os.path.join(SCAF_MODEL_DIR, f"{conv}_{name}_feat{level}.pt")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    model = build_model(conv, meta["atom_dim"], ckpt["hidden"], ckpt["out_dim"],
                        ckpt["layers"], heads=ckpt["heads"],
                        edge_dim=meta["bond_dim"], fp_dim=meta["fp_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    key = f"{conv}_feat{level}"
    return key, infer(model, data_list, split["val"]), infer(model, data_list, split["test"])


def metrics_np(preds, y, task):
    return compute_metrics(torch.from_numpy(preds), torch.from_numpy(y), task)


def val_score(m, task):
    return m["AUC"] if task == "classification" else -m["RMSE"]


def analyze(name, experts_random):
    split = get_scaffold_split(name)
    task = "classification" if name == "BBBP" else "regression"
    print(f"\n=== {name} | scaffold | task={task} "
          f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} ===")

    val_pred, test_pred = {}, {}
    data_list = meta = y_val = y_test = None
    for level in LEVELS:
        dl, mt = load_featurized(name, level=level, verbose=False)
        if data_list is None:
            data_list, meta = dl, mt
            y_val = np.array([float(dl[int(i)].y[0]) for i in split["val"]])
            y_test = np.array([float(dl[int(i)].y[0]) for i in split["test"]])
        for conv in CONVS:
            ensure_ckpt(conv, name, level)
            got = load_and_infer(conv, name, level, dl, mt, split)
            if got is None:
                print(f"  [缺失] {conv}_{name}_feat{level}")
                continue
            k, pv, pt = got
            val_pred[k], test_pred[k] = pv, pt

    keys = sorted(val_pred)
    m_metrics = {k: {"val": metrics_np(val_pred[k], y_val, task),
                     "test": metrics_np(test_pred[k], y_test, task)} for k in keys}

    # 误差相关（test）
    err = np.stack([test_pred[k] - y_test for k in keys])
    corr = np.corrcoef(err)
    def group(k):
        c, rest = k.split("_", 1)
        return c, rest[4]
    n = len(keys)
    cross = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            if group(keys[i])[0] != group(keys[j])[0]:
                cross[i, j] = cross[j, i] = True
    cross_corr = float(np.mean(corr[cross])) if cross.any() else float("nan")

    # 策略（全部 val 选）
    best_single = max(keys, key=lambda k: val_score(m_metrics[k]["val"], task))
    experts = [e for e in experts_random if e in test_pred]
    strategies = {"best_single(val选)": test_pred[best_single],
                  "hetero_equal_3experts": np.mean([test_pred[e] for e in experts], axis=0),
                  "hetero_equal_all9": np.mean([test_pred[k] for k in keys], axis=0)}

    out = {"task": task, "n_train": len(split["train"]), "n_val": len(split["val"]),
           "n_test": len(split["test"]),
           "mean_corr_cross_architecture": round(cross_corr, 4),
           "best_single": best_single, "experts": experts,
           "model_metrics": m_metrics,
           "strategies": {s: metrics_np(p, y_test, task) for s, p in strategies.items()}}

    print(f"  跨架构误差相关 = {cross_corr:.4f}")
    for s, m in out["strategies"].items():
        print(f"  {s}: {m}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=NAMES)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "reports", "diversity", "fusion_preview.json"),
              encoding="utf-8") as f:
        preview = json.load(f)

    results = {}
    for name in args.names:
        results[name] = analyze(name, preview[name]["detail"]["experts_by_val"])

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "fusion_scaffold.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        old.update(results)
        results = old
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {path}")


if __name__ == "__main__":
    main()
