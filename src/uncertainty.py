# -*- coding: utf-8 -*-
"""
不确定性量化（方向 2）
======================
- MC-Dropout：推理时保持 dropout 开启，T 次前向采样得到预测分布（均值 ± 标准差）
- 深度集成：多随机种子模型平均（可选，需先训练 ensemble 成员）
- 校准评估：
    * 回归：90% 预测区间覆盖率（PI coverage）、平均区间宽度、不确定性-绝对误差 Spearman 相关
    * 分类：期望校准误差（ECE）、熵-误差相关性
- 输出：reports/uncertainty_report.json

用法:
    python src/uncertainty.py                        # 对 9 个已有模型做 MC-Dropout 校准评估
    python src/uncertainty.py --mc --n-samples 50
    python src/uncertainty.py --ensemble             # 使用 models/ensemble/* 多 seed 模型
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from data_prep import TASK_TYPE, load
from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
ENSEMBLE_DIR = os.path.join(ROOT, "models", "ensemble")
REPORT_DIR = os.path.join(ROOT, "reports")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = 64
NUM_LAYERS = 3
HEADS = 4
BATCH_SIZE = 32
DEFAULT_T = 30


def set_dropout_train(model, active=True):
    """切换 dropout 训练模式（MC-Dropout 用）"""
    model.train(active)
    # 只开启 conv/head 里的 dropout，不更新任何参数
    for p in model.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def mc_predict(model, loader, task_type, T=DEFAULT_T):
    """MC-Dropout 采样：返回 (mean, std/entropy, preds, targets)"""
    set_dropout_train(model, True)
    all_preds = [[] for _ in range(T)]
    targets = []
    for data in loader:
        data = data.to(DEVICE)
        y = data.y[:, 0]
        for t in range(T):
            out = model(data.x, data.edge_index, data.batch)
            if task_type == "classification":
                all_preds[t].append(F.softmax(out, dim=1)[:, 1].cpu())
            else:
                all_preds[t].append(out.squeeze(-1).cpu())
        targets.append(y.cpu())
    targets = torch.cat(targets).numpy()
    stack = torch.stack([torch.cat(p) for p in all_preds]).numpy()  # [T, N]
    if task_type == "classification":
        mean = stack.mean(axis=0)
        entropy = -np.sum(mean * np.log(np.clip(mean, 1e-12, 1.0)) + (1 - mean) * np.log(np.clip(1 - mean, 1e-12, 1.0)), axis=0)
        return mean, entropy, stack, targets
    std = stack.std(axis=0)
    return stack.mean(axis=0), std, stack, targets


@torch.no_grad()
def ensemble_predict(models, loader, task_type):
    """多模型集成预测（同架构不同 seed）"""
    for m in models:
        m.eval()
    preds_list = []
    targets = None
    for data in loader:
        data = data.to(DEVICE)
        outs = []
        for m in models:
            out = m(data.x, data.edge_index, data.batch)
            outs.append(out.cpu())
        outs = torch.stack(outs)  # [M, B, ...]
        if task_type == "classification":
            probs = torch.stack([F.softmax(o, dim=1)[:, 1] for o in outs])
            preds_list.append(probs)
        else:
            preds_list.append(outs.squeeze(-1))
        if targets is None:
            targets = data.y[:, 0].cpu()
        else:
            targets = torch.cat([targets, data.y[:, 0].cpu()])
    stack = torch.cat(preds_list, dim=1).numpy()  # [M, N]
    targets = targets.numpy()
    if task_type == "classification":
        mean = stack.mean(axis=0)
        entropy = -np.sum(mean * np.log(np.clip(mean, 1e-12, 1.0)) + (1 - mean) * np.log(np.clip(1 - mean, 1e-12, 1.0)), axis=0)
        return mean, entropy, stack, targets
    return stack.mean(axis=0), stack.std(axis=0), stack, targets


def regression_calibration(pred_mean, pred_std, targets, level=0.9):
    """回归校准：90% 预测区间覆盖率、区间宽度、不确定性-误差相关性"""
    z = 1.645  # 90% 双侧
    lo, hi = pred_mean - z * pred_std, pred_mean + z * pred_std
    coverage = float(np.mean((targets >= lo) & (targets <= hi)))
    avg_width = float(np.mean(hi - lo))
    abs_err = np.abs(targets - pred_mean)
    corr = float(spearmanr(pred_std, abs_err).statistic)
    return {"pi_coverage_90": round(coverage, 4), "avg_interval_width": round(avg_width, 4),
            "uncertainty_err_corr": round(corr, 4)}


def classification_calibration(prob, targets, n_bins=10):
    """分类校准：期望校准误差 ECE + 熵-误差相关性"""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        idx = np.where(bin_ids == b)[0]
        if len(idx) == 0:
            continue
        conf = prob[idx].mean()
        acc = (targets[idx] == 1).mean()
        ece += len(idx) / len(targets) * abs(conf - acc)
    entropy = -prob * np.log(np.clip(prob, 1e-12, 1)) - (1 - prob) * np.log(np.clip(1 - prob, 1e-12, 1))
    err = (prob > 0.5).astype(int) != targets.astype(int)
    corr = float(spearmanr(entropy, err).statistic)
    return {"ECE": round(float(ece), 4), "uncertainty_err_corr": round(corr, 4)}


def load_one_model(conv_type, name, task_type):
    path = os.path.join(MODEL_DIR, f"{conv_type}_{name}.pt")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    out_dim = 2 if task_type == "classification" else 1
    model = build_model(conv_type, ckpt["in_dim"], ckpt["hidden"], out_dim,
                        ckpt["layers"], heads=ckpt.get("heads", 4))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    return model


def run_mc(convs, names, n_samples=DEFAULT_T):
    report = {}
    for name in names:
        dataset, split = load(name, verbose=False)
        test_loader = DataLoader(dataset[[int(i) for i in split["test"]]], batch_size=BATCH_SIZE, shuffle=False)
        task = TASK_TYPE[name]
        report[name] = {}
        for conv in convs:
            model = load_one_model(conv, name, task)
            if model is None:
                continue
            mean, unc, stack, targets = mc_predict(model, test_loader, task, T=n_samples)
            if task == "regression":
                calib = regression_calibration(mean, unc, targets)
            else:
                calib = classification_calibration(mean, targets)
                try:
                    auc = float(roc_auc_score(targets, mean))
                except ValueError:
                    auc = float("nan")
                calib["AUC"] = round(auc, 4)
            report[name][conv] = calib
            print(f"[MC] {conv}×{name}: {calib}")
    return report


def run_ensemble(name, n_seeds=None):
    """对指定数据集用 models/ensemble/{conv}_{name}_s{seed}.pt 做集成"""
    task = TASK_TYPE[name]
    dataset, split = load(name, verbose=False)
    test_loader = DataLoader(dataset[[int(i) for i in split["test"]]], batch_size=BATCH_SIZE, shuffle=False)
    results = {}
    for conv in ["GCN", "GAT", "GraphSAGE"]:
        paths = sorted([p for p in os.listdir(ENSEMBLE_DIR)
                        if p.startswith(f"{conv}_{name}_s") and p.endswith(".pt")]) if os.path.isdir(ENSEMBLE_DIR) else []
        if not paths:
            continue
        models = []
        for p in paths:
            ckpt = torch.load(os.path.join(ENSEMBLE_DIR, p), weights_only=False, map_location="cpu")
            out_dim = 2 if task == "classification" else 1
            m = build_model(conv, ckpt["in_dim"], ckpt["hidden"], out_dim, ckpt["layers"], heads=ckpt.get("heads", 4))
            m.load_state_dict(ckpt["state_dict"])
            m.to(DEVICE)
            models.append(m)
        mean, unc, stack, targets = ensemble_predict(models, test_loader, task)
        from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
        if task == "regression":
            rmse = float(np.sqrt(mean_squared_error(targets, mean)))
            r2 = float(r2_score(targets, mean))
            calib = regression_calibration(mean, unc, targets)
            calib.update({"RMSE": round(rmse, 4), "R2": round(r2, 4)})
            members_rmse = [round(float(np.sqrt(mean_squared_error(targets, stack[i]))), 4) for i in range(len(models))]
            calib["member_RMSE"] = members_rmse
        else:
            calib = classification_calibration(mean, targets)
            try:
                auc = float(roc_auc_score(targets, mean))
            except ValueError:
                auc = float("nan")
            calib["AUC"] = round(auc, 4)
            members_auc = [round(float(roc_auc_score(targets, stack[i])), 4) for i in range(len(models))]
            calib["member_AUC"] = members_auc
        results[conv] = {"n_members": len(models), **calib}
        print(f"[ENS] {conv}×{name} ({len(models)} 成员): {calib}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", nargs="+", default=["GCN", "GAT", "GraphSAGE"])
    ap.add_argument("--names", nargs="+", default=["ESOL", "FreeSolv", "BBBP"])
    ap.add_argument("--n-samples", type=int, default=DEFAULT_T)
    ap.add_argument("--mc", action="store_true", default=True)
    ap.add_argument("--ensemble", action="store_true")
    args = ap.parse_args()

    report = {}
    if args.mc:
        report["mc_dropout"] = run_mc(args.convs, args.names, args.n_samples)
        report["config"] = {"method": "MC-Dropout", "n_samples": args.n_samples,
                            "device": str(DEVICE)}
    if args.ensemble:
        ens = {}
        for name in args.names:
            r = run_ensemble(name)
            if r:
                ens[name] = r
        report["ensemble"] = ens
        report["config_ensemble"] = {"n_members": None, "device": str(DEVICE)}
    out_path = os.path.join(REPORT_DIR, "uncertainty_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"不确定性评估完成: {out_path}")


if __name__ == "__main__":
    main()
