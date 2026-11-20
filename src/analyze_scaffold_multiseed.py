# -*- coding: utf-8 -*-
"""
骨架划分多种子统计与显著性检验（P0-2 分析部分）
================================================
输入：reports/scaffold_multiseed_results.json（train_scaffold_multiseed.py 产出）
输出：reports/scaffold_multiseed_analysis.json + reports/figures/scaffold_multiseed.png

比较对象（每个数据集独立）：
  1. 最强单模型   ：每个种子按 val 选出 9 配置中的最优者，test 只评估一次；
  2. 同构集成     ：① 同配置多种子（同构-种子）；② 同架构 × 3 特征档 × 多种子（同构-架构）；
  3. 异质集成     ：① val 选 3 专家的异质集成；② 全部 9 配置的异质等权集成。

统计口径：
  - 单模型：4 个种子各一个值 → 均值 ± 标准差；
  - 集成：4 个"留一种子"重复（每次用其余 3 个种子建模）→ 均值 ± 标准差，
    另给全部种子（4 种子）部署版本的单一取值；
  - 95% bootstrap 置信区间：对 test 样本有放回重采样 2000 次（分类按正负分层），
    统计「异质 all9 − 最强单模型(seed42)」的指标差；
  - 配对显著性：① bootstrap 双侧 p 值；② 逐样本损失（回归 |误差|、分类 Brier）
    的 Wilcoxon 符号秩检验。

判定规则（与验收标准一致）：
  - CI 下限 > 0 且点估计 > 0 → "稳定提升"
  - CI 包含 0 但点估计 > 0 → "趋势性提升"
  - 点估计 ≤ 0 → "无提升"

用法：
    python src/analyze_scaffold_multiseed.py
    python src/analyze_scaffold_multiseed.py --bootstrap 5000
"""
import argparse
import itertools
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(ROOT, "reports", "scaffold_multiseed_results.json")
OUT_PATH = os.path.join(ROOT, "reports", "scaffold_multiseed_analysis.json")
FIG_PATH = os.path.join(ROOT, "reports", "figures", "scaffold_multiseed.png")

CONVS = ["GCN", "GAT", "GraphSAGE"]
LEVELS = ["a", "b", "c"]
PRIMARY = {"regression": "R2", "classification": "AUC"}


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def metrics(preds, y, task):
    preds = np.asarray(preds, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if task == "regression":
        return {"RMSE": float(np.sqrt(mean_squared_error(y, preds))),
                "MAE": float(mean_absolute_error(y, preds)),
                "R2": float(r2_score(y, preds))}
    labels = (preds > 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y, preds))
    except ValueError:
        auc = float("nan")
    return {"AUC": auc,
            "F1": float(f1_score(y.astype(int), labels, zero_division=0)),
            "Accuracy": float(np.mean(labels == y.astype(int)))}


def val_score(m, task):
    return m["AUC"] if task == "classification" else -m["RMSE"]


def primary_value(m, task):
    return m[PRIMARY[task]]


def mean_std(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return {"mean": None, "std": None, "n": 0, "values": []}
    return {"mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals, ddof=0)), 4),
            "n": len(vals),
            "values": [round(float(v), 4) for v in vals]}


# ---------------------------------------------------------------------------
# 集成构造
# ---------------------------------------------------------------------------
def member_key(conv, level, seed):
    return f"{conv}_feat{level}_s{seed}"


def ens_pred(members, keys, field):
    return np.mean([np.asarray(members[k][field], dtype=np.float64) for k in keys], axis=0)


def config_keys(name_members, conv, level, seeds):
    return [member_key(conv, level, s) for s in seeds
            if member_key(conv, level, s) in name_members]


def all_keys(name_members):
    return list(name_members.keys())


def evaluate_strategies(ds, seeds, task):
    """返回每个种子的单模型、每个留一重复的集成结果。"""
    members = ds["members"]
    y_val = np.asarray(ds["y_val"], dtype=np.float64)
    y_test = np.asarray(ds["y_test"], dtype=np.float64)
    out = {"per_seed": {}, "loo": {}}
    configs = [(c, l) for c in CONVS for l in LEVELS if config_keys(members, c, l, seeds)]

    # ---- 单模型：每个种子 val 选最优
    for s in seeds:
        cand = {}
        for c, l in configs:
            k = member_key(c, l, s)
            if k in members:
                cand[f"{c}_feat{l}"] = members[k]
        best = max(cand, key=lambda k: val_score(cand[k]["val"], task))
        out["per_seed"][s] = {"best_config": best,
                              "val": cand[best]["val"],
                              "test": cand[best]["test"],
                              "pred": np.asarray(cand[best]["test_pred"], dtype=np.float64)}
        # 所有配置在该 seed 的 test 指标
        out["per_seed"][s]["all_configs"] = {k: v["test"] for k, v in cand.items()}

    # ---- 留一种子重复
    for hold in seeds:
        used = [s for s in seeds if s != hold]
        rep = {"hold_out_seed": hold, "seeds_used": used}

        # 1) 同构-同配置（同配置多种子）
        homo_cfg = {}
        for c, l in configs:
            keys = config_keys(members, c, l, used)
            if len(keys) < 2:
                continue
            pv = ens_pred(members, keys, "val_pred")
            pt = ens_pred(members, keys, "test_pred")
            homo_cfg[f"{c}_feat{l}"] = {"val": metrics(pv, y_val, task),
                                        "test": metrics(pt, y_test, task),
                                        "pred": pt}
        best_cfg = max(homo_cfg, key=lambda k: val_score(homo_cfg[k]["val"], task)) if homo_cfg else None
        rep["homo_seed_ens"] = {"best_config": best_cfg,
                                "test": homo_cfg[best_cfg]["test"] if best_cfg else None,
                                "pred": homo_cfg[best_cfg]["pred"] if best_cfg else None,
                                "all": {k: v["test"] for k, v in homo_cfg.items()}}

        # 2) 同构-同架构（3 特征档 × 多种子）
        homo_arch = {}
        for c in CONVS:
            keys = [k for k in all_keys(members) if k.startswith(f"{c}_") and
                    any(k.endswith(f"_s{s}") for s in used)]
            if len(keys) < 3:
                continue
            homo_arch[c] = {"test": metrics(ens_pred(members, keys, "test_pred"), y_test, task),
                            "pred": ens_pred(members, keys, "test_pred"),
                            "n_members": len(keys),
                            "val": metrics(ens_pred(members, keys, "val_pred"), y_val, task)}
        best_arch = max(homo_arch, key=lambda k: val_score(homo_arch[k]["val"], task)) if homo_arch else None
        rep["homo_arch"] = {"best_conv": best_arch,
                            "test": homo_arch[best_arch]["test"] if best_arch else None,
                            "pred": homo_arch[best_arch]["pred"] if best_arch else None,
                            "n_members": homo_arch[best_arch]["n_members"] if best_arch else 0,
                            "all": {k: v["test"] for k, v in homo_arch.items()}}

        # 3) 异质 3 专家（按集成的 val 选 top3 配置）
        cfg_scores = {k: val_score(homo_cfg[k]["val"], task) for k in homo_cfg}
        experts = sorted(cfg_scores, key=lambda k: -cfg_scores[k])[:3]
        keys3 = []
        for e in experts:
            c, lv = e.split("_feat")
            keys3 += config_keys(members, c, lv, used)
        if keys3:
            rep["hetero_3experts"] = {"experts": experts,
                                      "n_members": len(keys3),
                                      "test": metrics(ens_pred(members, keys3, "test_pred"), y_test, task),
                                      "pred": ens_pred(members, keys3, "test_pred")}

        # 4) 异质 all9
        k9 = [k for k in all_keys(members) if any(k.endswith(f"_s{s}") for s in used)]
        if k9:
            rep["hetero_all9"] = {"n_members": len(k9),
                                  "n_configs": len(configs),
                                  "test": metrics(ens_pred(members, k9, "test_pred"), y_test, task),
                                  "pred": ens_pred(members, k9, "test_pred")}
        out["loo"][hold] = rep
    return out


def deployed_snapshot(ds, seeds, task):
    """全部种子（部署版本）的集成与基线。"""
    members = ds["members"]
    y_val = np.asarray(ds["y_val"], dtype=np.float64)
    y_test = np.asarray(ds["y_test"], dtype=np.float64)
    k_all = all_keys(members)
    y_val_arr = y_val

    hetero = {"n_members": len(k_all), "n_configs": len(CONVS) * len(LEVELS),
              "test": metrics(ens_pred(members, k_all, "test_pred"), y_test, task),
              "pred": ens_pred(members, k_all, "test_pred"),
              "val": metrics(ens_pred(members, k_all, "val_pred"), y_val_arr, task)}

    # 同构-同架构（每架构 3 档 × 全部种子）
    homo_arch = {}
    for c in CONVS:
        keys = [k for k in k_all if k.startswith(f"{c}_")]
        if keys:
            homo_arch[c] = {"n_members": len(keys),
                            "test": metrics(ens_pred(members, keys, "test_pred"), y_test, task),
                            "pred": ens_pred(members, keys, "test_pred"),
                            "val": metrics(ens_pred(members, keys, "val_pred"), y_val_arr, task)}
    best_arch = max(homo_arch, key=lambda k: val_score(homo_arch[k]["val"], task)) if homo_arch else None

    # 同构-同配置（4 种子）
    homo_cfg = {}
    for c in CONVS:
        for l in LEVELS:
            keys = config_keys(members, c, l, seeds)
            if len(keys) >= 2:
                homo_cfg[f"{c}_feat{l}"] = {"n_members": len(keys),
                                            "test": metrics(ens_pred(members, keys, "test_pred"), y_test, task),
                                            "pred": ens_pred(members, keys, "test_pred"),
                                            "val": metrics(ens_pred(members, keys, "val_pred"), y_val_arr, task)}
    best_cfg = max(homo_cfg, key=lambda k: val_score(homo_cfg[k]["val"], task)) if homo_cfg else None

    # 最强单模型：seed42（冻结版本的 canonical 单模型）与跨种子均值
    cand42 = {k: m for k, m in members.items() if k.endswith("_s42")}
    if cand42:
        b = max(cand42, key=lambda k: val_score(cand42[k]["val"], task))
        best42 = {"config": b, "val": cand42[b]["val"], "test": cand42[b]["test"],
                  "pred": np.asarray(cand42[b]["test_pred"], dtype=np.float64)}

    return {"hetero_all9": hetero,
            "homo_arch": {"best_conv": best_arch,
                          "test": homo_arch[best_arch]["test"] if best_arch else None,
                          "pred": homo_arch[best_arch]["pred"] if best_arch else None,
                          "n_members": homo_arch[best_arch]["n_members"] if best_arch else 0,
                          "all": {k: v["test"] for k, v in homo_arch.items()}},
            "homo_seed_ens": {"best_config": best_cfg,
                              "test": homo_cfg[best_cfg]["test"] if best_cfg else None,
                              "pred": homo_cfg[best_cfg]["pred"] if best_cfg else None,
                              "n_members": homo_cfg[best_cfg]["n_members"] if best_cfg else 0,
                              "all": {k: v["test"] for k, v in homo_cfg.items()}},
            "best_single_seed42": best42}


# ---------------------------------------------------------------------------
# 统计检验
# ---------------------------------------------------------------------------
def _boot_indices(y, task, rng):
    if task == "classification":
        pos = np.where(np.asarray(y) >= 0.5)[0]
        neg = np.where(np.asarray(y) < 0.5)[0]
        return np.concatenate([rng.choice(pos, len(pos), replace=True),
                               rng.choice(neg, len(neg), replace=True)])
    return rng.integers(0, len(y), len(y))


def bootstrap_delta(pred_a, pred_b, y, task, B=2000, seed=2027):
    """a = 集成，b = 基线；返回差值分布与 95% CI。"""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = _boot_indices(y, task, rng)
        if task == "classification" and len(np.unique((np.asarray(y)[idx] >= 0.5))) < 2:
            continue
        diffs.append(primary_value(metrics(np.asarray(pred_a)[idx], np.asarray(y)[idx], task), task)
                     - primary_value(metrics(np.asarray(pred_b)[idx], np.asarray(y)[idx], task), task))
    diffs = np.asarray(diffs)
    if diffs.size == 0:
        return {"delta": None, "ci95": [None, None], "p_boot": None, "n_boot": 0}
    point = primary_value(metrics(pred_a, y, task), task) - primary_value(metrics(pred_b, y, task), task)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return {"delta": round(float(point), 4),
            "delta_mean_boot": round(float(diffs.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_boot": round(min(1.0, p), 4),
            "n_boot": int(diffs.size)}


def wilcoxon_test(pred_a, pred_b, y, task):
    """逐样本配对损失差异的 Wilcoxon 符号秩检验。"""
    a = np.asarray(pred_a, dtype=np.float64)
    b = np.asarray(pred_b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if task == "regression":
        loss_a, loss_b = np.abs(y - a), np.abs(y - b)
    else:
        loss_a, loss_b = (a - y) ** 2, (b - y) ** 2
    d = loss_a - loss_b
    d = d[np.abs(d) > 1e-12]
    if len(d) < 10:
        return {"stat": None, "p": None, "n": int(len(d))}
    res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"stat": round(float(res.statistic), 3), "p": round(float(res.pvalue), 4), "n": int(len(d))}


def verdict(delta, ci):
    if delta is None or ci[0] is None:
        return "无法判定"
    if delta <= 0:
        return "无提升"
    if ci[0] > 0:
        return "稳定提升"
    return "趋势性提升"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def analyze_dataset(name, ds, B):
    task = ds["task"]
    members = ds["members"]
    seeds = sorted({int(k.split("_s")[-1]) for k in members})
    y_test = np.asarray(ds["y_test"], dtype=np.float64)

    # 单配置：均值 ± 标准差
    metric_names = ("RMSE", "MAE", "R2") if task == "regression" else ("AUC", "F1", "Accuracy")
    per_config = {}
    for c in CONVS:
        for l in LEVELS:
            keys = config_keys(members, c, l, seeds)
            if not keys:
                continue
            per_config[f"{c}_feat{l}"] = {
                "seeds": seeds,
                "test": {m: mean_std([members[k]["test"][m] for k in keys]) for m in metric_names},
                "n_members": len(keys),
            }

    strat = evaluate_strategies(ds, seeds, task)
    dep = deployed_snapshot(ds, seeds, task)

    # 集成策略的均值 ± 标准差（留一种子重复）
    def loo_stat(key):
        return mean_std([strat["loo"][h][key]["test"][PRIMARY[task]]
                         if strat["loo"][h].get(key) and strat["loo"][h][key].get("test") else None
                         for h in seeds])

    strategies = {
        "best_single": {
            "metric": PRIMARY[task],
            "values": [primary_value(strat["per_seed"][s]["test"], task) for s in seeds],
            **mean_std([primary_value(strat["per_seed"][s]["test"], task) for s in seeds]),
            "best_config_per_seed": {s: strat["per_seed"][s]["best_config"] for s in seeds},
            "test_per_seed": {s: strat["per_seed"][s]["test"] for s in seeds},
        },
        "homo_seed_ens": {"metric": PRIMARY[task], **loo_stat("homo_seed_ens"),
                          "best_config_per_loo": {h: strat["loo"][h]["homo_seed_ens"]["best_config"] for h in seeds},
                          "deployed": dep["homo_seed_ens"]["test"]},
        "homo_arch": {"metric": PRIMARY[task], **loo_stat("homo_arch"),
                      "best_conv_per_loo": {h: strat["loo"][h]["homo_arch"]["best_conv"] for h in seeds},
                      "deployed": dep["homo_arch"]["test"]},
        "hetero_3experts": {"metric": PRIMARY[task], **loo_stat("hetero_3experts"),
                            "experts_per_loo": {h: strat["loo"][h].get("hetero_3experts", {}).get("experts") for h in seeds}},
        "hetero_all9": {"metric": PRIMARY[task], **loo_stat("hetero_all9"),
                        "deployed": dep["hetero_all9"]["test"]},
    }

    # 主比较：部署版 hetero all9 (36 成员) vs seed42 最强单模型
    comparison = {}
    base = dep["best_single_seed42"]
    if base:
        comparison["hetero_all9_vs_best_single"] = {
            "baseline_config": base["config"],
            "baseline_test": base["test"],
            **bootstrap_delta(dep["hetero_all9"]["pred"], base["pred"], y_test, task, B=B),
            "wilcoxon": wilcoxon_test(dep["hetero_all9"]["pred"], base["pred"], y_test, task),
        }
        comparison["hetero_all9_vs_best_single"]["verdict"] = verdict(
            comparison["hetero_all9_vs_best_single"]["delta"],
            comparison["hetero_all9_vs_best_single"]["ci95"])

        # 稳健性：与每个种子的 val 最优单模型比较
        per_seed_cmp = {}
        for s in seeds:
            per_seed_cmp[s] = bootstrap_delta(dep["hetero_all9"]["pred"],
                                             strat["per_seed"][s]["pred"], y_test, task, B=B)
        comparison["hetero_all9_vs_each_seed_best"] = per_seed_cmp

        # 同构-架构 与 单模型
        comparison["homo_arch_vs_best_single"] = {
            **bootstrap_delta(dep["homo_arch"]["pred"], base["pred"], y_test, task, B=B),
        }
        comparison["homo_arch_vs_best_single"]["verdict"] = verdict(
            comparison["homo_arch_vs_best_single"]["delta"],
            comparison["homo_arch_vs_best_single"]["ci95"])
        comparison["hetero_all9_vs_homo_arch"] = {
            **bootstrap_delta(dep["hetero_all9"]["pred"], dep["homo_arch"]["pred"], y_test, task, B=B),
        }
        comparison["hetero_all9_vs_homo_arch"]["verdict"] = verdict(
            comparison["hetero_all9_vs_homo_arch"]["delta"],
            comparison["hetero_all9_vs_homo_arch"]["ci95"])

    return {
        "task": task, "n_test": int(len(y_test)), "seeds": seeds, "n_members": len(members),
        "per_config": per_config,
        "strategies": strategies,
        "deployed": {"hetero_all9": dep["hetero_all9"]["test"],
                     "homo_arch": dep["homo_arch"]["test"],
                     "homo_seed_ens": dep["homo_seed_ens"]["test"],
                     "best_single_seed42": base["test"] if base else None,
                     "best_single_config_seed42": base["config"] if base else None},
        "comparison": comparison,
    }


def plot_all(report):
    names = [n for n in ["ESOL", "FreeSolv", "BBBP"] if n in report["datasets"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    ax = axes[0]
    keys = ["best_single", "homo_seed_ens", "homo_arch", "hetero_3experts", "hetero_all9"]
    labels = ["最强单模型", "同构（同配置多种子）", "同构（同架构×特征档）", "异质（3 专家）", "异质（all9）"]
    colors = ["#9E9E9E", "#72B7B2", "#4C78A8", "#F58518", "#54A24B"]
    width = 0.8 / len(keys)
    x = np.arange(len(names))
    for i, (k, lb, c) in enumerate(zip(keys, labels, colors)):
        means = [report["datasets"][n]["strategies"][k]["mean"] for n in names]
        stds = [report["datasets"][n]["strategies"][k]["std"] or 0 for n in names]
        ax.bar(x - 0.4 + width * (i + 0.5), means, width, yerr=stds, capsize=2.5,
               label=lb, color=c, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, names)
    ax.set_ylabel("主指标（回归 R² ↑ / 分类 AUC ↑）")
    ax.set_title("骨架划分 · 多种子（4 seeds）均值 ± 标准差")
    ax.legend(fontsize=7.2, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for i, n in enumerate(names):
        cmp = report["datasets"][n]["comparison"]["hetero_all9_vs_best_single"]
        d = cmp["delta"]
        lo, hi = cmp["ci95"]
        ax.errorbar(i - 0.12, d, yerr=[[d - lo], [hi - d]], fmt="o", color="#4C78A8",
                    capsize=5, markersize=7, label="异质 all9" if i == 0 else None)
        ax.text(i + 0.34, d, f"{d:+.3f}\n[{lo:+.3f},{hi:+.3f}]\n{cmp['verdict']}", fontsize=7.5, va="center")
        c2 = report["datasets"][n]["comparison"]["homo_arch_vs_best_single"]
        d2 = c2["delta"]
        lo2, hi2 = c2["ci95"]
        ax.errorbar(i + 0.12, d2, yerr=[[d2 - lo2], [hi2 - d2]], fmt="s", color="#F58518",
                    capsize=5, markersize=6, label="同构 架构×特征" if i == 0 else None)
    ax.axhline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.25, len(names) - 0.3)
    ax.set_ylabel("Δ 主指标（相对最强单模型）")
    ax.set_title("95% bootstrap 置信区间（跨 0 即只能称趋势性提升）")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存 -> {FIG_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    with open(IN_PATH, encoding="utf-8") as f:
        src = json.load(f)
    report = {"config": src.get("config", {}), "bootstrap": args.bootstrap,
              "primary_metric": PRIMARY, "datasets": {}}
    for name, ds in src.get("datasets", {}).items():
        print(f"\n=== 分析 {name}（{ds['task']}，test={len(ds['y_test'])}）===")
        report["datasets"][name] = analyze_dataset(name, ds, args.bootstrap)
        s = report["datasets"][name]
        print(f"  单模型: {s['strategies']['best_single']['mean']} ± {s['strategies']['best_single']['std']}")
        print(f"  同构-种子: {s['strategies']['homo_seed_ens']['mean']} ± {s['strategies']['homo_seed_ens']['std']}")
        print(f"  同构-架构: {s['strategies']['homo_arch']['mean']} ± {s['strategies']['homo_arch']['std']}")
        print(f"  异质 all9: {s['strategies']['hetero_all9']['mean']} ± {s['strategies']['hetero_all9']['std']} "
              f"(deployed {s['deployed']['hetero_all9']})")
        cmp = s["comparison"]["hetero_all9_vs_best_single"]
        print(f"  Δ={cmp['delta']} CI={cmp['ci95']} p_boot={cmp['p_boot']} "
              f"wilcoxon_p={cmp['wilcoxon']['p']} → {cmp['verdict']}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {OUT_PATH}")
    plot_all(report)


if __name__ == "__main__":
    main()
