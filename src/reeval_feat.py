# -*- coding: utf-8 -*-
"""重评估脚本（AUC 概率口径统一）：对已训好的特征工程模型与 Clintox 基线
重新计算测试集指标（分类用 softmax 概率 AUC，与 comparison_results.json 口径一致），
汇总到 reports/feat_results.json 与 reports/clintox_baseline.json。"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from data_prep import SPLIT_ROOT
from featurize import load_featurized
from models import build_model
from train_eval import DEVICE, HIDDEN_DIM, HEADS, NUM_LAYERS, evaluate, set_seed
from torch_geometric.loader import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_MODEL_DIR = os.path.join(ROOT, "models", "feat")
REPORT_DIR = os.path.join(ROOT, "reports")


def get_split(name):
    with open(os.path.join(SPLIT_ROOT, f"{name}_split.json"), encoding="utf-8") as f:
        return json.load(f)


def make_loader(data_list, idx, batch_size=32):
    return DataLoader([data_list[int(i)] for i in idx], batch_size=batch_size, shuffle=False)


def reeval_ckpt(ckpt_path):
    fname = os.path.basename(ckpt_path).replace(".pt", "")  # GCN_ESOL_feata
    conv, name, level = fname.rsplit("_", 2)
    level = level[-1]
    ckpt = torch.load(ckpt_path, weights_only=False)
    data_list, meta = load_featurized(name, level=level, verbose=False)
    split = get_split(name)
    task_type = meta["task"]
    model = build_model(conv, meta["atom_dim"], HIDDEN_DIM,
                        ckpt["out_dim"], ckpt["layers"], heads=ckpt["heads"],
                        edge_dim=meta["bond_dim"], fp_dim=meta["fp_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    test_loader = make_loader(data_list, split["test"])
    metrics = evaluate(model, test_loader, task_type)
    return conv, name, level, metrics, fname


def main():
    set_seed(42)
    feat_results = {"baseline_ref": "reports/comparison_results.json（默认 9 维特征，AUC 概率口径）",
                    "metric_note": "分类 AUC 用 softmax 概率口径（与权威基线一致）"}
    for p in sorted(glob.glob(os.path.join(FEAT_MODEL_DIR, "*.pt"))):
        conv, name, level, metrics, fname = reeval_ckpt(p)
        feat_results.setdefault(name, {})[f"{conv}_feat{level}"] = metrics
        print(f"{fname}: {metrics}")
    out1 = os.path.join(REPORT_DIR, "feat_results.json")
    with open(out1, "w", encoding="utf-8") as f:
        json.dump(feat_results, f, ensure_ascii=False, indent=2)
    print(f"feat_results.json -> {out1}")

    # Clintox 基线重评估（3 架构默认特征）
    clintox = {}
    for conv in ["GCN", "GAT", "GraphSAGE"]:
        bp = os.path.join(ROOT, "models", f"{conv}_Clintox.pt")
        if not os.path.exists(bp):
            print(f"缺失基线模型 {bp}")
            continue
        ckpt = torch.load(bp, weights_only=False)
        from data_prep import load
        dataset, split = load("Clintox", verbose=False)
        for d in dataset:
            d.y = d.y[:, 1:2].float()
        model = build_model(conv, dataset.num_features, ckpt["hidden"], ckpt["out_dim"],
                            ckpt["layers"], heads=ckpt["heads"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        test_loader = make_loader(dataset, split["test"])
        clintox[conv] = evaluate(model, test_loader, "classification")
        print(f"Clintox × {conv}: {clintox[conv]}")
    out2 = os.path.join(REPORT_DIR, "clintox_baseline.json")
    with open(out2, "w", encoding="utf-8") as f:
        json.dump({"Clintox": clintox, "task": "CT_TOX col1, pos 7.6%",
                   "metric_note": "AUC 概率口径"}, f, ensure_ascii=False, indent=2)
    print(f"clintox_baseline.json -> {out2}")


if __name__ == "__main__":
    main()
