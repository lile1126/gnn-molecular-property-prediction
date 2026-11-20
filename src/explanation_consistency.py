# -*- coding: utf-8 -*-
"""
跨模型解释一致性（P0-3）
========================
问题：同一个分子，换一种架构或换一档特征，GNNExplainer 指出的"关键原子/键"是否一致？
      如果 9 个配置各说各话，那么 Web 上的结构着色与 LLM 解释就不可信。

做法：
  1. 每个数据集从 test 集固定抽取 40 个代表性分子（回归按标签分位，分类按正负分层，seed=2026）；
  2. 用 9 个「架构 × 特征档」配置（均取 seed42 主模型，random split）分别跑 GNNExplainer，
     得到原子重要性 node_mask 与键重要性 edge_mask；
  3. 逐分子计算：
       - Top-5 / Top-10 原子集合的 Jaccard 重合度（36 个模型两两平均）
       - Top-5 / Top-10 化学键集合的 Jaccard 重合度
       - 原子/键重要性排序的 Spearman 秩相关（两两平均）
       - 9 个配置对同一分子的预测标准差（分歧度）
  4. 汇总为数据集级均值 ± 标准差，并给出高/中/低一致性等级；
  5. 输出 reports/explanation_consistency.json 与 reports/figures/explanation_consistency.png，
     逐分子平均重要性供 functional_group_explain.py 聚合到官能团。

用法：
    python src/explanation_consistency.py --jobs 9
    python src/explanation_consistency.py --names ESOL --num 20 --epochs 60
"""
import argparse
import itertools
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
from rdkit import Chem
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

from data_prep import TASK_TYPE, load
from featurize import load_featurized
from functional_group_explain import bond_importance_from_mask
from interpret import explain_one
from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(ROOT, "reports", "explanation_consistency.json")
FIG_PATH = os.path.join(ROOT, "reports", "figures", "explanation_consistency.png")
FEAT_DIR = os.path.join(ROOT, "models", "feat")

NAMES = ["ESOL", "FreeSolv", "BBBP"]
CONVS = ["GCN", "GAT", "GraphSAGE"]
LEVELS = ["a", "b", "c"]
N_PER_DATASET = 40
EXPLAIN_EPOCHS = 100
SELECT_SEED = 2026
EXPLAIN_SEED = 777  # GNNExplainer 掩码初始化种子（固定后可复现）
DEVICE = torch.device("cpu")

# 一致性等级阈值（在看到结果前预先设定，基于 Top-10 原子 Jaccard 均值）
LEVEL_RULES = {"high": 0.60, "medium": 0.35}


def select_molecules(name, num=N_PER_DATASET, seed=SELECT_SEED):
    """从 test 集抽取代表性分子：回归按标签分位取点，分类按正/负分层抽样。"""
    dataset, split = load(name, verbose=False)
    idx_test = [int(i) for i in split["test"]]
    task = TASK_TYPE[name]
    rng = np.random.default_rng(seed)
    if task == "regression":
        ys = np.array([float(dataset[i].y[0]) for i in idx_test])
        order = np.argsort(ys)
        n = min(num, len(idx_test))
        positions = np.unique(np.round(np.linspace(0, len(order) - 1, n)).astype(int))
        picked = [idx_test[order[p]] for p in positions]
        if len(picked) < n:  # 分位点重复时补随机样本
            remaining = [i for i in idx_test if i not in set(picked)]
            picked += list(rng.choice(remaining, size=n - len(picked), replace=False))
        return picked[:n]
    # 分类：正负各半
    pos = [i for i in idx_test if float(dataset[i].y[0]) >= 0.5]
    neg = [i for i in idx_test if float(dataset[i].y[0]) < 0.5]
    n_pos = min(len(pos), num // 2)
    n_neg = min(len(neg), num - n_pos)
    picked = list(rng.choice(pos, size=n_pos, replace=False)) + \
             list(rng.choice(neg, size=n_neg, replace=False))
    if len(picked) < num:  # 某一类不够时从另一类补
        rest = [i for i in idx_test if i not in set(picked)]
        picked += list(rng.choice(rest, size=min(num - len(picked), len(rest)), replace=False))
    return picked[:num]


def _load_model(conv, name, level):
    path = os.path.join(FEAT_DIR, f"{conv}_{name}_feat{level}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少特征模型 {path}，请先运行 train_feat.py / train_feat_ensemble.py")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = build_model(conv, ckpt["in_dim"], ckpt["hidden"], ckpt["out_dim"], ckpt["layers"],
                        heads=ckpt.get("heads", 4), edge_dim=ckpt.get("edge_dim", 0),
                        fp_dim=ckpt.get("fp_dim", 0)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _worker(task):
    """单个（数据集, 架构, 特征档）配置对全部代表分子的 GNNExplainer 归因。"""
    name, conv, level, idxs, epochs = task
    torch.set_num_threads(1)
    try:
        data_list, meta = load_featurized(name, level=level, verbose=False)
        model = _load_model(conv, name, level)
        task_type = meta["task"]
        items = {}
        for idx in idxs:
            data = data_list[int(idx)]
            # 每个（分子, 配置）用同一随机种子，保证解释结果可复现
            torch.manual_seed(EXPLAIN_SEED + int(idx))
            np.random.seed((EXPLAIN_SEED + int(idx)) % (2 ** 32))
            node_mask, edge_mask, pred = explain_one(model, data, task_type, epochs=epochs)
            items[str(idx)] = {
                "node_mask": [round(float(v), 4) for v in node_mask],
                "edge_mask": [round(float(v), 4) for v in edge_mask],
                "pred": round(float(pred), 4),
            }
        print(f"  [完成] {name} {conv}_feat{level}: {len(items)} 个分子", flush=True)
        return {"ok": True, "name": name, "conv": conv, "level": level, "items": items}
    except Exception as e:
        print(f"  [ERROR] {name} {conv}_feat{level}: {e}", flush=True)
        return {"ok": False, "name": name, "conv": conv, "level": level, "error": str(e)}


def _jaccard_topk(a, b, k):
    k = min(k, len(a), len(b))
    if k <= 0:
        return float("nan")
    ta = set(np.argsort(-a, kind="stable")[:k].tolist())
    tb = set(np.argsort(-b, kind="stable")[:k].tolist())
    return len(ta & tb) / len(ta | tb)


def _pairwise_stats(vectors, k_list):
    """vectors: [n_models, n_items]；返回每个 k 的两两 Jaccard 均值/标准差。"""
    out = {}
    pairs = list(itertools.combinations(range(len(vectors)), 2))
    for k in k_list:
        vals = [_jaccard_topk(vectors[i], vectors[j], k) for i, j in pairs]
        vals = [v for v in vals if not np.isnan(v)]
        out[f"jaccard_top{k}"] = {"mean": round(float(np.mean(vals)), 4) if vals else None,
                                  "std": round(float(np.std(vals)), 4) if vals else None}
    return out


def _pairwise_spearman(vectors):
    pairs = list(itertools.combinations(range(len(vectors)), 2))
    vals = []
    for i, j in pairs:
        a, b = np.asarray(vectors[i]), np.asarray(vectors[j])
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        rho = spearmanr(a, b).statistic
        if not np.isnan(rho):
            vals.append(float(rho))
    return {"mean": round(float(np.mean(vals)), 4) if vals else None,
            "std": round(float(np.std(vals)), 4) if vals else None}


def analyze_dataset(name, idxs, results, task_type):
    """合并 9 个配置的掩码，逐分子算一致性指标。"""
    models = sorted(results, key=lambda r: (r["conv"], r["level"]))
    keys = [f"{r['conv']}_feat{r['level']}" for r in models]
    molecules = []
    # 取 data 中的 smiles（各 feature 档一致，统一用 feat_c 的 smiles / 拓扑）
    data_list, _ = load_featurized(name, level="c", verbose=False)
    for idx in idxs:
        d = data_list[int(idx)]
        smiles = getattr(d, "smiles", None) or ""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            continue

        node_vecs, bond_vecs, preds = [], [], []
        for r in models:
            item = r["items"].get(str(idx))
            if item is None:
                break
            node_vecs.append(np.asarray(item["node_mask"], dtype=np.float64))
            bond_vecs.append(bond_importance_from_mask(mol, d.edge_index.numpy(),
                                                       np.asarray(item["edge_mask"])))
            preds.append(float(item["pred"]))
        if len(node_vecs) != len(models) or len(bond_vecs) != len(models):
            continue
        if any(len(v) != mol.GetNumAtoms() for v in node_vecs):
            print(f"  [跳过] {name} idx={idx} 原子数与掩码长度不一致")
            continue

        # 预测分歧度
        pred_mean = float(np.mean(preds))
        pred_std = float(np.std(preds))
        pred_rel_std = float(pred_std / (abs(pred_mean) + 1e-8))

        atom_stats = _pairwise_stats(node_vecs, [5, 10])
        bond_stats = _pairwise_stats(bond_vecs, [5, 10]) if mol.GetNumBonds() >= 5 else {}
        atom_rank = _pairwise_spearman(node_vecs)
        bond_rank = _pairwise_spearman(bond_vecs) if mol.GetNumBonds() >= 5 else {"mean": None, "std": None}

        # 供官能团聚合：跨配置平均重要性
        atom_mean = np.mean(node_vecs, axis=0)
        bond_mean = np.mean(bond_vecs, axis=0)
        top5_atoms = np.argsort(-atom_mean, kind="stable")[:5].tolist()
        molecules.append({
            "index": int(idx),
            "smiles": smiles,
            "y_true": round(float(d.y[0]), 4),
            "n_atoms": int(mol.GetNumAtoms()),
            "n_bonds": int(mol.GetNumBonds()),
            "pred_mean": round(pred_mean, 4),
            "pred_std": round(pred_std, 4),
            "pred_rel_std": round(pred_rel_std, 4),
            "atom": atom_stats,
            "bond": bond_stats,
            "atom_rank": atom_rank,
            "bond_rank": bond_rank,
            "top5_atoms": top5_atoms,
            "atom_importance_mean": [round(float(v), 4) for v in atom_mean],
            "bond_importance_mean": [round(float(v), 4) for v in bond_mean],
            "per_model": {k: {"pred": r["items"][str(idx)]["pred"],
                              "top5_atoms": np.argsort(-np.asarray(r["items"][str(idx)]["node_mask"]),
                                                       kind="stable")[:5].tolist()}
                          for k, r in zip(keys, models)},
        })

    # 汇总
    def agg(path):
        vals = []
        for m in molecules:
            cur = m
            for p in path:
                cur = cur.get(p) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if cur is not None:
                vals.append(float(cur))
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4), "n": len(vals)}

    summary = {
        "jaccard_top5_atom": agg(["atom", "jaccard_top5", "mean"]),
        "jaccard_top10_atom": agg(["atom", "jaccard_top10", "mean"]),
        "jaccard_top5_bond": agg(["bond", "jaccard_top5", "mean"]),
        "jaccard_top10_bond": agg(["bond", "jaccard_top10", "mean"]),
        "spearman_atom": agg(["atom_rank", "mean"]),
        "spearman_bond": agg(["bond_rank", "mean"]),
        "pred_std": agg(["pred_std"]),
        "pred_rel_std": agg(["pred_rel_std"]),
    }
    j10 = summary["jaccard_top10_atom"]["mean"]
    if j10 is None:
        level = "未知"
    elif j10 >= LEVEL_RULES["high"]:
        level = "高"
    elif j10 >= LEVEL_RULES["medium"]:
        level = "中"
    else:
        level = "低"
    return {
        "task": task_type,
        "n_molecules": len(molecules),
        "summary": {**summary, "level": level,
                    "level_rule": f"Top-10 原子 Jaccard ≥{LEVEL_RULES['high']} 高；"
                                  f"≥{LEVEL_RULES['medium']} 中；否则低"},
        "molecules": molecules,
    }


def plot_all(payload):
    datasets = [n for n in NAMES if n in payload]
    metrics = [("jaccard_top5_atom", "原子 Jaccard@5"), ("jaccard_top10_atom", "原子 Jaccard@10"),
               ("jaccard_top10_bond", "键 Jaccard@10"), ("spearman_atom", "原子排序 Spearman")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    width = 0.8 / len(metrics)
    x = np.arange(len(datasets))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for i, (key, label) in enumerate(metrics):
        means = [payload[d]["summary"][key]["mean"] or 0 for d in datasets]
        stds = [payload[d]["summary"][key]["std"] or 0 for d in datasets]
        ax.bar(x - 0.4 + width * (i + 0.5), means, width, yerr=stds, capsize=3,
               label=label, color=colors[i], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 1)
    ax.set_ylabel("一致性指标（越高越一致）")
    ax.set_title("跨模型解释一致性：9 配置两两平均")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    data, labels, level_colors = [], [], []
    for d in datasets:
        vals = [m["atom"]["jaccard_top10"]["mean"] for m in payload[d]["molecules"]
                if m["atom"]["jaccard_top10"]["mean"] is not None]
        data.append(vals)
        labels.append(f"{d}\n({payload[d]['summary']['level']})")
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, c in zip(bp["boxes"], ["#4C78A8", "#F58518", "#54A24B"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("#222222")
    ax.set_ylim(0, 1)
    ax.set_ylabel("单分子 Top-10 原子 Jaccard")
    ax.set_title("逐分子分布（箱线：中位数与四分位）")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Random split · seed42 主模型 · GNNExplainer epochs="
                 f"{payload.get('config', {}).get('epochs', EXPLAIN_EPOCHS)}", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存 -> {FIG_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=NAMES)
    ap.add_argument("--num", type=int, default=N_PER_DATASET)
    ap.add_argument("--epochs", type=int, default=EXPLAIN_EPOCHS)
    ap.add_argument("--jobs", type=int, default=9)
    args = ap.parse_args()

    t0 = time.time()
    payload = {"config": {
        "split": "random", "models": "9 配置（3 架构 × 3 特征档）seed42 主模型",
        "explainer": "GNNExplainer", "epochs": args.epochs, "select_seed": SELECT_SEED,
        "n_per_dataset": args.num, "device": "cpu",
        "note": "node_mask_type=object，键重要性按无向键折叠；两两比较覆盖 C(9,2)=36 对",
    }, "thresholds": LEVEL_RULES}

    for name in args.names:
        print(f"\n=== 解释一致性: {name} ===")
        idxs = select_molecules(name, num=args.num, seed=SELECT_SEED)
        print(f"  代表分子 {len(idxs)} 个")
        tasks = [(name, conv, level, idxs, args.epochs) for conv in CONVS for level in LEVELS]
        with multiprocessing.Pool(min(args.jobs, len(tasks))) as pool:
            results = pool.map(_worker, tasks)
        ok = [r for r in results if r["ok"]]
        if len(ok) != len(tasks):
            missing = [f"{r['conv']}_feat{r['level']}: {r.get('error')}" for r in results if not r["ok"]]
            print("  [警告] 部分配置失败:", missing)
        payload[name] = analyze_dataset(name, idxs, ok, TASK_TYPE[name])
        print(f"  {name} 汇总: {json.dumps(payload[name]['summary'], ensure_ascii=False)}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {REPORT_PATH}（用时 {time.time() - t0:.0f}s）")
    plot_all(payload)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
