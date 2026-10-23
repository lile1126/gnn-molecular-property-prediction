# -*- coding: utf-8 -*-
"""
生成实验结果对比图表（reports/figures/*.png）
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CONVS = ["GCN", "GAT", "GraphSAGE"]
COLORS = ["#4a9fe8", "#2e9e5b", "#e8a13c"]


def load_results():
    path = os.path.join(ROOT, "reports", "comparison_results.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["results"]


def group_bar(ax, datasets, metric, ylabel, title, lower_better=False):
    x = np.arange(len(datasets))
    width = 0.26
    all_vals = []
    for i, conv in enumerate(CONVS):
        vals = [results[d][conv][metric] for d in datasets]
        all_vals.extend(vals)
        bars = ax.bar(x + (i - 1) * width, vals, width, label=conv, color=COLORS[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if not lower_better else 0.02),
                    f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    if lower_better:
        ax.set_ylim(0, max(all_vals) * 1.22)
    else:
        ax.set_ylim(0, 1.1)


results = load_results()

# 图1：回归 RMSE（越低越好）
fig, ax = plt.subplots(figsize=(7.5, 4.2))
group_bar(ax, ["ESOL", "FreeSolv"], "RMSE", "RMSE", "回归任务 RMSE 对比（越低越好）", lower_better=True)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "rmse_comparison.png"), dpi=140)
plt.close(fig)

# 图2：回归 R²（越高越好）
fig, ax = plt.subplots(figsize=(7.5, 4.2))
group_bar(ax, ["ESOL", "FreeSolv"], "R2", "R²", "回归任务 R² 对比（越高越好）")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "r2_comparison.png"), dpi=140)
plt.close(fig)

# 图3：BBBP 分类 AUC 与 Accuracy
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(2)
width = 0.26
for i, conv in enumerate(CONVS):
    vals = [results["BBBP"][conv]["AUC"], results["BBBP"][conv]["Accuracy"]]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=conv, color=COLORS[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["AUC", "Accuracy"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("指标")
ax.set_title("BBBP 血脑屏障分类对比（越高越好）")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "bbbp_comparison.png"), dpi=140)
plt.close(fig)

print("图表已生成:")
for f in sorted(os.listdir(FIG_DIR)):
    print(" ", os.path.join(FIG_DIR, f))
