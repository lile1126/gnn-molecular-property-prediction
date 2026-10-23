# -*- coding: utf-8 -*-
"""从三份微调日志解析并合并 9 组 (架构×任务) 预训练微调对比结果"""
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "reports")
LOGS = {"GCN": "finetune_gcn_full2.txt", "GAT": "finetune_gat_full.txt", "GraphSAGE": "finetune_sage_full.txt"}

# 基线（comparison_results.json，AUC 已修正为概率口径）
base_path = os.path.join(REPORT_DIR, "comparison_results.json")
baseline = json.load(open(base_path, encoding="utf-8"))["results"]

# 任务指标键
TASK_KEY = {"ESOL": "RMSE", "FreeSolv": "RMSE", "BBBP": "AUC"}


def parse_log(path):
    """从日志提取 {任务: {指标: 值}}（预训练微调测试集结果）"""
    text = open(os.path.join(REPORT_DIR, path), encoding="utf-8", errors="replace").read()
    out = {}
    # 匹配形如: GCN × BBBP 测试集: {'Accuracy': 0.8049, 'AUC': 0.8953, 'F1': 0.8765}
    pattern = re.compile(r"(\w+) × (\w+) 测试集: (\{.*?\})")
    for m in pattern.finditer(text):
        task, metrics_str = m.group(2), m.group(3)
        try:
            metrics = json.loads(metrics_str.replace("'", '"'))
        except Exception:
            continue
        out[task] = metrics
    return out


def main():
    merged = {}
    for conv, log in LOGS.items():
        parsed = parse_log(log)
        for task, metrics in parsed.items():
            merged.setdefault(task, {})[conv] = {
                "pretrained": metrics,
                "baseline": baseline.get(task, {}).get(conv, {}),
                "delta": {k: round(float(metrics[k]) - float(baseline[task][conv][k]), 4)
                          for k in metrics if k in baseline.get(task, {}).get(conv, {})
                          and not k.endswith("_old_argmax")},
            }
    out_path = os.path.join(REPORT_DIR, "pretrain_finetune_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": {"pattern": "full", "pretrain_epochs": 150, "finetune_epochs": 300},
                   "results": merged}, f, ensure_ascii=False, indent=2)
    print("合并完成:", out_path)
    for task in merged:
        for conv in merged[task]:
            r = merged[task][conv]
            print(f"[{task}/{conv}] baseline={r['baseline'].get(TASK_KEY[task])} "
                  f"pretrained={r['pretrained'].get(TASK_KEY[task])} Δ={r['delta'].get(TASK_KEY[task])}")


if __name__ == "__main__":
    main()
