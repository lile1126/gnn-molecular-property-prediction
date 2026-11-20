# -*- coding: utf-8 -*-
"""
误差多样性分析 + 跨架构融合预实验（零训练成本）
================================================
目的：在投入任何新训练之前，用现有 144 个 checkpoint 回答两个问题：
  1. 跨架构 / 跨特征档 / 跨种子成员的预测误差相关性有多低？（融合是否有油水）
  2. 跨架构等权平均 / val 加权 stacking 能否超过最强单模型与同构集成？

口径：
  - 分类用 softmax 正类概率（与 feat_results.json 一致）
  - 回归用原始预测值；误差 = pred - y
  - 模型选择一律用 val 指标，test 只评估一次
  - Clintox 特征化数据 y 已是 CT_TOX 列（featurize.py 口径）

用法:
    python src/analyze_error_diversity.py

输出（reports/diversity/）:
    error_diversity.json   相关性统计 + 每模型 val/test 指标
    fusion_preview.json    各融合策略 test 指标对比
    preds_{name}.npz       全部成员 val/test 预测（供后续 stacking/蒸馏复用）
    heatmap_{name}.png     9 配置（种子集成后）误差相关热力图
"""
import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
from scipy.optimize import nnls
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

from data_prep import SPLIT_ROOT
from featurize import load_featurized
from models import build_model
from train_eval import compute_metrics, set_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(ROOT, "models", "feat")
ENS_DIR = os.path.join(ROOT, "models", "feat_ensemble")
OUT_DIR = os.path.join(ROOT, "reports", "diversity")

DEVICE = torch.device("cpu")  # 本分析只做推理，强制 CPU 保证可复现
NAMES = ["ESOL", "FreeSolv", "BBBP", "Clintox"]
CONVS = ["GCN", "GAT", "GraphSAGE"]
LEVELS = ["a", "b", "c"]
SEEDS = [7, 13, 21]  # feat_ensemble 中的额外种子；主模型为 seed42


def get_split(name):
    with open(os.path.join(SPLIT_ROOT, f"{name}_split.json"), encoding="utf-8") as f:
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


def load_and_infer(conv, name, level, suffix, data_list, meta, split):
    fname = f"{conv}_{name}_feat{level}{suffix}.pt"
    folder = FEAT_DIR if suffix == "" else ENS_DIR
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    model = build_model(conv, meta["atom_dim"], ckpt["hidden"], ckpt["out_dim"],
                        ckpt["layers"], heads=ckpt["heads"],
                        edge_dim=meta["bond_dim"], fp_dim=meta["fp_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    key = f"{conv}_feat{level}{suffix}"
    return key, infer(model, data_list, split["val"]), infer(model, data_list, split["test"])


def val_score(metrics, task):
    return metrics["AUC"] if task == "classification" else -metrics["RMSE"]


def metrics_np(preds, y, task):
    return compute_metrics(torch.from_numpy(preds), torch.from_numpy(y), task)


def mean_offdiag(corr, mask):
    vals = corr[mask]
    return float(np.mean(vals)) if vals.size else float("nan")


def analyze_dataset(name):
    split = get_split(name)
    task = "classification" if name in ("BBBP", "Clintox") else "regression"
    val_pred, test_pred = {}, {}
    y_val = y_test = None
    for level in LEVELS:
        data_list, meta = load_featurized(name, level=level, verbose=False)
        if y_val is None:
            y_val = np.array([float(data_list[int(i)].y[0]) for i in split["val"]])
            y_test = np.array([float(data_list[int(i)].y[0]) for i in split["test"]])
        for conv in CONVS:
            for suffix in [""] + [f"_s{s}" for s in SEEDS]:
                got = load_and_infer(conv, name, level, suffix, data_list, meta, split)
                if got is None:
                    print(f"  [缺失] {conv}_{name}_feat{level}{suffix}")
                    continue
                key, pv, pt = got
                val_pred[key], test_pred[key] = pv, pt
    keys = sorted(val_pred)
    print(f"[{name}] 成员数={len(keys)} val={len(y_val)} test={len(y_test)}")

    # ---- 种子集成：每个 (conv, level) 配置取 4 种子平均 ----
    cfg_val, cfg_test, cfg_members = {}, {}, {}
    for conv in CONVS:
        for level in LEVELS:
            members = [k for k in keys if k.startswith(f"{conv}_feat{level}")]
            if not members:
                continue
            cfg = f"{conv}_feat{level}"
            cfg_val[cfg] = np.mean([val_pred[k] for k in members], axis=0)
            cfg_test[cfg] = np.mean([test_pred[k] for k in members], axis=0)
            cfg_members[cfg] = members
    cfgs = sorted(cfg_val)

    # ---- 误差相关矩阵（test 残差）----
    err = np.stack([test_pred[k] - y_test for k in keys])
    corr = np.corrcoef(err)
    cfg_err = np.stack([cfg_test[c] - y_test for c in cfgs])
    cfg_corr = np.corrcoef(cfg_err)

    # 分组统计
    def group_of(k):
        conv, rest = k.split("_", 1)
        level = rest[4]  # featX...
        return conv, level

    n = len(keys)
    same_cfg = np.zeros((n, n), bool)   # 同架构同档（跨种子）
    cross_arch = np.zeros((n, n), bool)  # 跨架构
    cross_level = np.zeros((n, n), bool)  # 同架构跨档
    for i in range(n):
        for j in range(i + 1, n):
            ci, li = group_of(keys[i])
            cj, lj = group_of(keys[j])
            if ci == cj and li == lj:
                same_cfg[i, j] = same_cfg[j, i] = True
            if ci != cj:
                cross_arch[i, j] = cross_arch[j, i] = True
            if ci == cj and li != lj:
                cross_level[i, j] = cross_level[j, i] = True
    stats = {
        "n_members": n,
        "mean_corr_same_config_cross_seed": round(mean_offdiag(corr, same_cfg), 4),
        "mean_corr_cross_architecture": round(mean_offdiag(corr, cross_arch), 4),
        "mean_corr_same_arch_cross_level": round(mean_offdiag(corr, cross_level), 4),
        "mean_corr_config_level": round(mean_offdiag(cfg_corr, ~np.eye(len(cfgs), dtype=bool)), 4),
    }

    # ---- 每模型 / 每配置指标 ----
    model_metrics = {k: {"val": metrics_np(val_pred[k], y_val, task),
                         "test": metrics_np(test_pred[k], y_test, task)} for k in keys}
    cfg_metrics = {c: {"val": metrics_np(cfg_val[c], y_val, task),
                       "test": metrics_np(cfg_test[c], y_test, task),
                       "members": cfg_members[c]} for c in cfgs}

    # ---- 融合策略（选择全部基于 val）----
    best_single = max(keys, key=lambda k: val_score(model_metrics[k]["val"], task))
    best_homo = max(cfgs, key=lambda c: val_score(cfg_metrics[c]["val"], task))
    # 每架构按 val 选最优特征档（种子集成口径）
    chosen = {}
    for conv in CONVS:
        cand = [c for c in cfgs if c.startswith(conv)]
        chosen[conv] = max(cand, key=lambda c: val_score(cfg_metrics[c]["val"], task))
    experts = [chosen[c] for c in CONVS]

    strategies = {
        "best_single(val选)": test_pred[best_single],
        "best_homo_ensemble(val选)": cfg_test[best_homo],
        "hetero_equal(val选3专家)": np.mean([cfg_test[c] for c in experts], axis=0),
        "hetero_equal_all9": np.mean([cfg_test[c] for c in cfgs], axis=0),
        "hetero_equal_featc": np.mean([cfg_test[f"{c}_featc"] for c in CONVS
                                       if f"{c}_featc" in cfg_test], axis=0),
    }
    # val 加权 stacking（NNLS，权重非负、归一化；只在 val 上拟合）
    A_val = np.stack([cfg_val[c] for c in experts], axis=1)
    A_test = np.stack([cfg_test[c] for c in experts], axis=1)
    if task == "regression":
        w, _ = nnls(A_val, y_val)
    else:  # 分类：拟合正类概率
        w, _ = nnls(A_val, y_val)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(experts)) / len(experts)
    strategies["hetero_nnls(val拟合)"] = A_test @ w

    fusion = {c: {"test": metrics_np(p, y_test, task)} for c, p in strategies.items()}
    fusion_detail = {
        "best_single": best_single,
        "best_homo_ensemble": best_homo,
        "experts_by_val": experts,
        "nnls_weights": {c: round(float(wi), 4) for c, wi in zip(experts, w)},
    }

    # ---- 保存预测与热力图 ----
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, f"preds_{name}.npz"),
                        keys=np.array(keys), y_val=y_val, y_test=y_test,
                        val=np.stack([val_pred[k] for k in keys]),
                        test=np.stack([test_pred[k] for k in keys]))
    plot_heatmap(name, cfgs, cfg_corr, task)
    return {"task": task, "diversity": stats, "config_metrics": cfg_metrics,
            "fusion_preview": fusion, "fusion_detail": fusion_detail,
            "model_metrics": model_metrics}


def plot_heatmap(name, cfgs, corr, task):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=0, vmax=1, cmap="RdYlBu_r")
    ax.set_xticks(range(len(cfgs)), cfgs, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cfgs)), cfgs, fontsize=9)
    for i in range(len(cfgs)):
        for j in range(len(cfgs)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if corr[i, j] < 0.75 else "white")
    metric = "概率残差" if task == "classification" else "预测残差"
    ax.set_title(f"{name} 跨配置误差相关矩阵（test {metric}，种子集成后）\n越低 = 融合收益越大")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"heatmap_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  热力图 -> {path}")


def main():
    set_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results, all_fusion = {}, {}
    for name in NAMES:
        r = analyze_dataset(name)
        all_results[name] = r
        all_fusion[name] = {"detail": r["fusion_detail"], "strategies": r["fusion_preview"]}
        print(f"  多样性: {r['diversity']}")
        for s, m in r["fusion_preview"].items():
            print(f"  {s}: {m['test']}")
    with open(os.path.join(OUT_DIR, "error_diversity.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "fusion_preview.json"), "w", encoding="utf-8") as f:
        json.dump(all_fusion, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {OUT_DIR}")


if __name__ == "__main__":
    main()