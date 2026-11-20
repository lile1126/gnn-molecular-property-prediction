# -*- coding: utf-8 -*-
"""
异质集成知识蒸馏
================
教师：由误差多样性分析在验证集上选出的 3 个专家（各 4 种子集成）等权组合；
学生：单个 GNN（架构与特征档取自该数据集 val 最优单模型配置），推理只需一次前向。

损失：
  回归  L = MSE(student, y) + λ·MSE(student, teacher)
  分类  L = CE(student, y) + λ·T²·KL(softmax(teacher/T) ‖ log_softmax(student/T))

协议与基线一致：hidden=64/layers=3/heads=4/batch=32/lr=1e-3/wd=5e-4/patience=40/epochs≤300，
早停按验证集真实标签指标（与基线同口径），test 只评估一次。

用法:
    python src/distill_ensemble.py --names ESOL BBBP

输出:
    models/distill/Student_{name}_{conv}_feat{level}_s{seed}.pt
    reports/distill_results.json
"""
import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from data_prep import SPLIT_ROOT
from featurize import load_featurized
from models import build_model
from task_config import compute_class_weight
from train_eval import (BATCH_SIZE, DEVICE, HIDDEN_DIM, HEADS, LR, NUM_LAYERS,
                        PATIENCE, WEIGHT_DECAY, compute_metrics, set_seed)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(ROOT, "models", "feat")
ENS_DIR = os.path.join(ROOT, "models", "feat_ensemble")
STU_DIR = os.path.join(ROOT, "models", "distill")
REPORT_DIR = os.path.join(ROOT, "reports")
DIV_JSON = os.path.join(REPORT_DIR, "diversity", "error_diversity.json")
SEEDS = [42, 7, 13, 21]
EXTRA_SEEDS = [7, 13, 21]


def get_split(name):
    with open(os.path.join(SPLIT_ROOT, f"{name}_split.json"), encoding="utf-8") as f:
        return json.load(f)


def parse_spec(spec):
    """'GraphSAGE_featb' -> ('GraphSAGE', 'b')"""
    conv, level = spec.rsplit("_feat", 1)
    return conv, level


def infer(model, data_list, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in DataLoader([data_list[int(i)] for i in idx],
                                batch_size=64, shuffle=False):
            o = model.forward_data(batch)
            out.append(torch.softmax(o, -1)[:, 1] if o.shape[-1] == 2 else o.squeeze(-1))
    return torch.cat(out)


def load_member(conv, name, level, suffix, meta):
    folder = FEAT_DIR if suffix == "" else ENS_DIR
    fname = f"{conv}_{name}_feat{level}{suffix}.pt"
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    model = build_model(conv, meta["atom_dim"], ckpt["hidden"], ckpt["out_dim"],
                        ckpt["layers"], heads=ckpt["heads"],
                        edge_dim=meta["bond_dim"], fp_dim=meta["fp_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    return model


def teacher_preds(name, experts, split):
    """三位专家 × 4 种子的等权平均教师输出（train/val/test）。"""
    acc = {k: [] for k in ("train", "val", "test")}
    for spec in experts:
        conv, level = parse_spec(spec)
        data_list, meta = load_featurized(name, level=level, verbose=False)
        for suffix in [""] + [f"_s{s}" for s in EXTRA_SEEDS]:
            model = load_member(conv, name, level, suffix, meta)
            if model is None:
                raise FileNotFoundError(f"缺少成员 {conv}_{name}_feat{level}{suffix}.pt")
            for k in acc:
                acc[k].append(infer(model, data_list, split[k]).numpy())
        print(f"    专家 {spec}: 4 成员已推理")
    return {k: torch.tensor(np.mean(v, axis=0), dtype=torch.float32) for k, v in acc.items()}


def build_loader(data_list, idx, batch_size=BATCH_SIZE, shuffle=False):
    return DataLoader([data_list[int(i)] for i in idx], batch_size=batch_size, shuffle=shuffle)


def distill_one(name, spec, seed, teacher, epochs, lam, T):
    conv, level = parse_spec(spec)
    data_list, meta = load_featurized(name, level=level, verbose=False)
    split = get_split(name)
    task = meta["task"]
    out_dim = 2 if task == "classification" else 1
    set_seed(seed)
    model = build_model(conv, meta["atom_dim"], HIDDEN_DIM, out_dim, NUM_LAYERS,
                        heads=HEADS, edge_dim=meta["bond_dim"], fp_dim=meta["fp_dim"]).to(DEVICE)
    class_weight = compute_class_weight(name, data_list, split["train"], target_col=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_idx = [int(i) for i in split["train"]]
    y_train = torch.tensor([float(data_list[i].y[0]) for i in train_idx])
    t_train = teacher["train"]
    best_score, best_state, bad = None, None, 0
    val_loader = build_loader(data_list, split["val"])
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_idx)).tolist()
        ep_loader = DataLoader([data_list[train_idx[i]] for i in order],
                               batch_size=BATCH_SIZE, shuffle=False)
        total, seen = 0.0, 0
        for bi, batch in enumerate(ep_loader):
            optimizer.zero_grad()
            out = model.forward_data(batch)
            b = len(batch.y)
            s = bi * BATCH_SIZE
            y = y_train[order[s:s + b]].to(DEVICE)
            t = t_train[order[s:s + b]].to(DEVICE)
            if task == "classification":
                w = class_weight.to(DEVICE) if class_weight is not None else None
                hard = F.cross_entropy(out, y.long(), weight=w)
                soft_t = torch.stack([1 - t, t], dim=1).clamp(1e-4, 1)
                kl = F.kl_div(F.log_softmax(out / T, dim=1), soft_t,
                              reduction="batchmean") * (T ** 2)
                loss = hard + lam * kl
            else:
                hard = F.mse_loss(out.squeeze(-1), y)
                distill = F.mse_loss(out.squeeze(-1), t)
                loss = hard + lam * distill
            loss.backward()
            optimizer.step()
            total += loss.item() * b
            seen += b
        model.eval()
        with torch.no_grad():
            _, pv, yv = _eval(model, val_loader)
        val_metrics = compute_metrics(pv, yv, task)
        score = val_metrics["AUC"] if task == "classification" else -val_metrics["RMSE"]
        if best_score is None or score > best_score:
            best_score, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if ep % 20 == 0:
            print(f"    ep{ep:3d} loss={total / max(seen, 1):.4f} val={val_metrics}")

    model.load_state_dict(best_state)
    test_metrics = compute_metrics(*_eval(model, build_loader(data_list, split["test"]))[1:], task)
    os.makedirs(STU_DIR, exist_ok=True)
    path = os.path.join(STU_DIR, f"Student_{name}_{conv}_feat{level}_s{seed}.pt")
    torch.save({"state_dict": model.state_dict(), "model_type": conv, "task_type": task,
                "in_dim": meta["atom_dim"], "edge_dim": meta["bond_dim"],
                "fp_dim": meta["fp_dim"], "hidden": HIDDEN_DIM, "out_dim": out_dim,
                "layers": NUM_LAYERS, "heads": HEADS, "seed": seed,
                "distill": {"teacher": "hetero_ensemble", "lam": lam, "T": T}},
               path)
    print(f"  s{seed} test={test_metrics} 用时 {time.time() - t0:.0f}s")
    return test_metrics


def _eval(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            o = model.forward_data(batch)
            preds.append(torch.softmax(o, -1)[:, 1] if o.shape[-1] == 2 else o.squeeze(-1))
            targets.append(batch.y[:, 0].float())
    return None, torch.cat(preds), torch.cat(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["ESOL", "BBBP"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lam", type=float, default=0.5, help="蒸馏损失权重")
    ap.add_argument("--T", type=float, default=2.0, help="分类蒸馏温度")
    args = ap.parse_args()

    with open(DIV_JSON, encoding="utf-8") as f:
        div = json.load(f)
    out_path = os.path.join(REPORT_DIR, "distill_results.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)

    for name in args.names:
        if name not in div:
            print(f"[跳过] {name} 缺少多样性分析结果")
            continue
        task = div[name]["task"]
        experts = div[name]["fusion_detail"]["experts_by_val"]
        # 学生配置 = val 最优单模型配置（保证与最强基线同架构同档）
        cfg_metrics = div[name]["config_metrics"]
        best_cfg = max(cfg_metrics, key=lambda c: (
            cfg_metrics[c]["val"]["AUC"] if task == "classification"
            else -cfg_metrics[c]["val"]["RMSE"]))
        print(f"\n=== {name} | 教师={experts} | 学生={best_cfg} ===")
        split = get_split(name)
        teacher = teacher_preds(name, experts, split)
        members = []
        for seed in args.seeds:
            m = distill_one(name, best_cfg, seed, teacher, args.epochs, args.lam, args.T)
            members.append({"seed": seed, "test": m})

        # 同配置单模型基线（来自已有结果）
        base_test = []
        for seed in args.seeds:
            key = best_cfg if seed == 42 else f"{best_cfg}_s{seed}"
            mm = div[name]["model_metrics"].get(key)
            if mm:
                base_test.append(mm["test"])
        keys = ["RMSE", "R2", "AUC", "F1"]
        base_mean = None
        if base_test:
            base_mean = {}
            for k in keys:
                vals = [m[k] for m in base_test if k in m]
                if vals:
                    base_mean[k] = round(float(np.mean(vals)), 4)
        stu_mean = {}
        for k in keys:
            if k in members[0]["test"]:
                stu_mean[k] = round(float(np.mean([m["test"][k] for m in members])), 4)
        results[name] = {
            "task": task, "teacher_experts": experts, "student_config": best_cfg,
            "distill": {"lam": args.lam, "T": args.T},
            "members": members,
            "student_mean_test": stu_mean,
            "single_baseline_mean_test": base_mean,
            "teacher_reference": div[name]["fusion_preview"]["hetero_equal(val选3专家)"]["test"],
        }
        print(f"  诚实结论：学生均值 {stu_mean} | 单模型基线 {base_mean} | 教师 {results[name]['teacher_reference']}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {out_path}")


if __name__ == "__main__":
    main()