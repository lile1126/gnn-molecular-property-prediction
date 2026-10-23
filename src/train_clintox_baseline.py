# -*- coding: utf-8 -*-
"""Clintox 毒性任务基线训练（选题深挖 · 方向2）：3 架构 × 默认特征。
标签统一取 CT_TOX 毒性列（col1，正类 7.6%）。类别加权 [1,12] 补偿不平衡。
注意：PyG 2.8 的 InMemoryDataset.__getitem__ 返回克隆，对 dataset 元素重写 y 不生效，
故通过 train_eval.run_epoch 的 y_col 参数在 DataLoader 中直接取列（与 feat 数据口径一致）。
结果单独存 reports/clintox_baseline.json。"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from data_prep import load
from models import build_model
from train_eval import (BATCH_SIZE, DEVICE, HIDDEN_DIM, HEADS, LR, NUM_LAYERS,
                        WEIGHT_DECAY, evaluate, set_seed, train_model)
from torch_geometric.loader import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Y_COL = 1  # Clintox col1 = CT_TOX 毒性


def main():
    set_seed()
    dataset, split = load("Clintox", verbose=True)
    task_type = "classification"
    out_dim = 2
    idx_train, idx_val, idx_test = split["train"], split["val"], split["test"]
    loaders = [DataLoader(dataset[[int(i) for i in idx]], batch_size=BATCH_SIZE, shuffle=(k == 0))
               for k, idx in enumerate([idx_train, idx_val, idx_test])]
    results = {}
    pos_w = torch.tensor([1.0, 12.0]).to(DEVICE)  # 正类 7.6% -> 类别权重补偿不平衡
    for conv in ["GCN", "GAT", "GraphSAGE"]:
        model = build_model(conv, dataset.num_features, HIDDEN_DIM, out_dim, NUM_LAYERS,
                            heads=HEADS).to(DEVICE)
        print(f"=== {conv} × Clintox (CT_TOX col1, class_weight=[1,12]) ===")
        model, _ = train_model(model, loaders[0], loaders[1], task_type, verbose=True,
                               class_weight=pos_w, y_col=Y_COL)
        metrics = evaluate(model, loaders[2], task_type, y_col=Y_COL)
        print(f"  {conv} × Clintox 测试集: {metrics}")
        results[conv] = metrics
        # 保存模型（Web 演示 / 深度集成成员复用，与 load_trained 兼容）
        path = os.path.join(ROOT, "models", f"{conv}_Clintox.pt")
        torch.save({"state_dict": model.state_dict(), "metrics": metrics,
                    "task_type": task_type, "in_dim": dataset.num_features,
                    "hidden": HIDDEN_DIM, "out_dim": out_dim, "layers": NUM_LAYERS,
                    "heads": HEADS}, path)
        print(f"  模型已保存: {path}")
    out = os.path.join(ROOT, "reports", "clintox_baseline.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"Clintox": results,
                   "task": "classification (CT_TOX col1, pos 7.6%, n=1480, split 8:1:1 seed42)",
                   "config": "hidden=64 layers=3 heads=4 seed=42, class_weight=[1,12]"},
                  f, ensure_ascii=False, indent=2)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
