# -*- coding: utf-8 -*-
"""打印报告所需的统计表（多种子 + 解释一致性 + 官能团），便于逐项核对。"""
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")


def load(name):
    with open(os.path.join(REP, name), encoding="utf-8") as f:
        return json.load(f)


def multiseed():
    d = load("scaffold_multiseed_analysis.json")["datasets"]
    for name, v in d.items():
        met = "R2" if v["task"] == "regression" else "AUC"
        print(f"\n== {name} ({met}, test={v['n_test']}) ==")
        for s in ["best_single", "homo_seed_ens", "homo_arch", "hetero_3experts", "hetero_all9"]:
            x = v["strategies"][s]
            print(f"  {s:16s} {x['mean']:.4f} ± {x['std']:.4f}")
        c = v["comparison"]["hetero_all9_vs_best_single"]
        print(f"  all9 - single: {c['delta']:+.4f} CI{c['ci95']} p_boot={c['p_boot']} "
              f"wilcoxon={c['wilcoxon']['p']} -> {c['verdict']}")
        c2 = v["comparison"]["homo_arch_vs_best_single"]
        print(f"  homo_arch - single: {c2['delta']:+.4f} CI{c2['ci95']} -> {c2['verdict']}")
        c3 = v["comparison"]["hetero_all9_vs_homo_arch"]
        print(f"  all9 - homo_arch: {c3['delta']:+.4f} CI{c3['ci95']} -> {c3['verdict']}")
        rows = sorted(((k, x["test"][met]["mean"], x["test"][met]["std"]) for k, x in v["per_config"].items()),
                      key=lambda r: -r[1])
        print("  top3:", [(k, round(m, 4), round(s, 4)) for k, m, s in rows[:3]])
        print("  bottom3:", [(k, round(m, 4)) for k, m, s in rows[-3:]])


def consistency():
    d = load("explanation_consistency.json")
    print("\n== 解释一致性（40 分子 × 9 配置 × 36 对）==")
    for name in ("ESOL", "FreeSolv", "BBBP"):
        s = d[name]["summary"]
        print(f"  {name}: Top10原子Jaccard {s['jaccard_top10_atom']['mean']:.3f} "
              f"Top5原子 {s['jaccard_top5_atom']['mean']:.3f} "
              f"Top10键 {s['jaccard_top10_bond']['mean']:.3f} "
              f"Spearman原子 {s['spearman_atom']['mean']:.3f} "
              f"预测std {s['pred_std']['mean']:.3f} → {s['level']}")


def groups():
    d = load("functional_group_explanations.json")["datasets"]
    print("\n== 官能团解释 ==")
    for name in ("ESOL", "FreeSolv", "BBBP"):
        ms = d[name]["molecules"]
        n_groups = [len(m["all_groups"]) for m in ms]
        top3 = [len(m["top3"]) for m in ms]
        zero = sum(1 for n in n_groups if n == 0)
        cover = st.mean(n_groups) if n_groups else 0
        print(f"  {name}: 分子 {len(ms)}，平均检出 {cover:.1f} 个官能团，"
              f"Top-3 满额 {sum(1 for t in top3 if t == 3)}，零检出 {zero}")


if __name__ == "__main__":
    multiseed()
    consistency()
    groups()
