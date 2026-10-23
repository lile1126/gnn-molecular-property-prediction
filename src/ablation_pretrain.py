# -*- coding: utf-8 -*-
"""
增强策略消融实验（方向 1 阶段 3）
================================
对比不同增强组合的预训练效果（GCN × ESOL 为代表任务）：
  - mask    : 仅原子特征掩码
  - perturb : 仅边扰动
  - drop    : 仅节点删除
  - full    : mask+perturb 与 mask+drop 双视图
每组用相同预训练预算（--pt-epochs，默认 60）→ 同配置微调 ESOL → 对比 RMSE。

用法:
    python src/ablation_pretrain.py --pt-epochs 60 --ft-epochs 300
"""
import argparse
import json
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
REPORT_DIR = os.path.join(ROOT, "reports")
PY = sys.executable

PATTERNS = ["mask", "perturb", "drop", "full"]
CONV = "GCN"
NAME = "ESOL"


def run(cmd):
    r = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("CMD FAILED:", " ".join(cmd))
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(1)
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-epochs", type=int, default=60)
    ap.add_argument("--ft-epochs", type=int, default=300)
    args = ap.parse_args()
    results = {}
    for pattern in PATTERNS:
        print(f"\n### 预训练 pattern={pattern} ({args.pt_epochs} epochs) ###")
        run([PY, "contrastive_pretrain.py", "--convs", CONV, "--pattern", pattern,
             "--epochs", str(args.pt_epochs)])
        print(f"### 微调 ESOL (预训练 {pattern}) ###")
        run([PY, "finetune_pretrained.py", "--convs", CONV, "--names", NAME,
             "--pattern", pattern, "--epochs", str(args.ft_epochs)])
        path = os.path.join(REPORT_DIR, "pretrain_finetune_results.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results[pattern] = data["results"][NAME][CONV]
        print(f"[消融] {pattern}: {results[pattern]}")
    # 汇总
    summary = {"task": f"{CONV}×{NAME}", "pretrain_epochs": args.pt_epochs,
               "finetune_epochs": args.ft_epochs, "results": results}
    out = os.path.join(REPORT_DIR, "ablation_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n消融实验完成: {out}")
    print("汇总:")
    for pattern, r in results.items():
        pt = r.get("pretrained", {})
        base = r.get("baseline", {})
        key = "RMSE" if "RMSE" in pt else "AUC"
        print(f"  {pattern:8s} baseline={base.get(key)} pretrained={pt.get(key)} Δ={r.get('delta', {}).get(key)}")


if __name__ == "__main__":
    main()
