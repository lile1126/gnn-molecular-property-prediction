# -*- coding: utf-8 -*-
"""生成"特征工程收益"图（答辩 4 张核心图之一）。

数据来源：
  reports/comparison_results.json  基线 9 维特征（BASE）三种架构
  reports/feat_results.json        FEAT_A/B/C 各配置
口径：
  回归 = 相对基线最强 RMSE 的下降百分比（越大越好）
  分类 = 相对基线最强 AUC 的提升点数（越大越好）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
FIG = os.path.join(REP, "figures", "feature_gain.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def best_base(ds, metric, mode):
    path = os.path.join(REP, "comparison_results.json")
    payload = json.load(open(path, encoding="utf-8"))["results"]
    if ds not in payload:  # Clintox 基线单独存放
        payload = json.load(open(os.path.join(REP, "clintox_baseline.json"), encoding="utf-8"))
    rows = payload[ds]
    vals = [v[metric] for v in rows.values()]
    return min(vals) if mode == "min" else max(vals)


def best_feat(ds, metric, mode):
    rows = json.load(open(os.path.join(REP, "feat_results.json"), encoding="utf-8"))[ds]
    best_name, best_val = None, None
    for name, m in rows.items():
        if metric not in m:
            continue
        v = m[metric]
        if best_val is None or (v < best_val if mode == "min" else v > best_val):
            best_name, best_val = name, v
    return best_name, best_val


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # 回归：RMSE 相对下降
    ax = axes[0]
    names = ["ESOL", "FreeSolv"]
    base = [best_base(d, "RMSE", "min") for d in names]
    feat_best = [best_feat(d, "RMSE", "min") for d in names]
    gains = [(b - f[1]) / b * 100 for b, f in zip(base, feat_best)]
    bars = ax.bar(names, gains, color=["#4C78A8", "#54A24B"], width=0.5)
    for bar, b, f, g in zip(bars, base, feat_best, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, g + 0.6,
                f"−{g:.1f}%\n{b:.3f}→{f[1]:.3f}\n({f[0]})", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(gains) * 1.35)
    ax.set_ylabel("RMSE 相对基线下降（%）")
    ax.set_title("回归任务：特征工程收益")
    ax.grid(axis="y", alpha=0.25)

    # 分类：AUC 提升点数
    ax = axes[1]
    names = ["BBBP", "Clintox"]
    base = [best_base(d, "AUC", "max") for d in names]
    feat_best = [best_feat(d, "AUC", "max") for d in names]
    gains = [(f[1] - b) * 100 for b, f in zip(base, feat_best)]
    bars = ax.bar(names, gains, color=["#F58518", "#B279A2"], width=0.5)
    for bar, b, f, g in zip(bars, base, feat_best, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, g + 0.4,
                f"+{g:.1f} 点\n{b:.4f}→{f[1]:.4f}\n({f[0]})", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(gains) * 1.35)
    ax.set_ylabel("AUC 相对基线提升（点）")
    ax.set_title("分类任务：特征工程收益")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("特征工程收益（基线 = 9 维原子特征下最强架构；标注为最优特征档配置）", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("图已保存 ->", FIG)


if __name__ == "__main__":
    main()
