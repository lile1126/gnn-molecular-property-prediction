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


# ---------------- 图4：融合策略 × 数据划分 对比 ----------------
def plot_fusion_compare():
    """随机划分 vs Murcko 骨架划分，单模型 / 3专家 / all9 三方对比。"""
    with open(os.path.join(ROOT, "reports", "diversity", "fusion_preview.json"),
              encoding="utf-8") as f:
        preview = json.load(f)
    with open(os.path.join(ROOT, "reports", "fusion_scaffold.json"), encoding="utf-8") as f:
        scaffold = json.load(f)

    cases = [("ESOL", "R2", "R² ↑"), ("FreeSolv", "R2", "R² ↑"), ("BBBP", "AUC", "AUC ↑")]
    labels = ["最强单模型", "3 专家等权", "all9 等权"]
    colors = ["#8c8c8c", "#4a9fe8", "#e8563c"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (name, metric, ylab) in zip(axes, cases):
        rand = preview[name]["strategies"]
        scaf = scaffold[name]["strategies"]
        rand_vals = [rand["best_single(val选)"]["test"][metric],
                     rand["hetero_equal(val选3专家)"]["test"][metric],
                     rand["hetero_equal_all9"]["test"][metric]]
        scaf_vals = [scaf["best_single(val选)"][metric],
                     scaf["hetero_equal_3experts"][metric],
                     scaf["hetero_equal_all9"][metric]]

        x = np.arange(2)
        width = 0.24
        for i, (lab, col) in enumerate(zip(labels, colors)):
            vals = [rand_vals[i], scaf_vals[i]]
            bars = ax.bar(x + (i - 1) * width, vals, width, label=lab, color=col)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                        f"{v:.3f}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["随机划分", "骨架划分"])
        ax.set_ylabel(ylab)
        ax.set_title(name, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        lo = min(rand_vals + scaf_vals)
        ax.set_ylim(max(0, lo - 0.12), 1.02)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("融合策略在两种数据划分下的对比（test 集，模型选择全部基于 val）",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    path = os.path.join(FIG_DIR, "fusion_split_compare.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(" 融合对比 ->", path)


plot_fusion_compare()
