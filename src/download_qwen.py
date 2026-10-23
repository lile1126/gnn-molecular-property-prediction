# -*- coding: utf-8 -*-
"""从 ModelScope 下载 Qwen2.5-3B-Instruct 到本地 models/Qwen2.5-3B-Instruct"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from modelscope import snapshot_download

ROOT = r"D:\桌面\工程实践\GNN分子性质预测"
target = os.path.join(ROOT, "models", "Qwen2.5-3B-Instruct")
os.makedirs(target, exist_ok=True)

print("开始下载 Qwen2.5-3B-Instruct ...")
path = snapshot_download(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir=target,
)
print(f"下载完成: {path}")
