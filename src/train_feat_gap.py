# -*- coding: utf-8 -*-
"""
补齐 Web 全组合缺口：GAT / GraphSAGE × {feat_a, feat_b} × 4 数据集 的 seed42 特征模型。
只补训缺失模型，已有结果不动；新结果合并进 reports/feat_results.json。
"""
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from train_feat import FEAT_MODEL_DIR, REPORT_DIR, run_feat_one

GAP_MATRIX = []
for name in ["ESOL", "FreeSolv", "BBBP", "Clintox"]:
    for lv in ["a", "b"]:
        GAP_MATRIX.append(("GAT", name, lv))
        GAP_MATRIX.append(("GraphSAGE", name, lv))


def main():
    report_path = os.path.join(REPORT_DIR, "feat_results.json")
    results = json.load(open(report_path, encoding="utf-8")) if os.path.exists(report_path) else {}
    t0 = time.time()
    skipped = 0
    trained = 0
    for conv, name, level in GAP_MATRIX:
        ckpt = os.path.join(FEAT_MODEL_DIR, f"{conv}_{name}_feat{level}.pt")
        if os.path.exists(ckpt):
            skipped += 1
            print(f"  跳过（已存在）: {conv} × {name} × feat{level}")
            continue
        try:
            _, _, _, metrics = run_feat_one(conv, name, level, epochs=300, verbose=True)
            results.setdefault(name, {})[f"{conv}_feat{level}"] = metrics
            trained += 1
            print(f"  累计用时 {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"  [ERROR] {conv} × {name} × {level}: {e}")
            results.setdefault(name, {})[f"{conv}_feat{level}"] = {"error": str(e)}
    print(f"（跳过 {skipped} 个，新训练 {trained} 个）")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"合并完成: {report_path}")


if __name__ == "__main__":
    main()
