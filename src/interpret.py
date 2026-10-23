# -*- coding: utf-8 -*-
"""
可解释性模块
============
使用 GNNExplainer 对测试集中的代表性分子生成：
  - 原子重要性（node mask）
  - 化学键重要性（edge mask）
并用 RDKit 将分子结构按重要性着色输出 PNG 可视化。

用法:
    python src/interpret.py --conv GCN --name ESOL --num 4
    python src/interpret.py --conv GCN --names ESOL FreeSolv BBBP --num 4
"""
import argparse
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
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

from data_prep import TASK_TYPE, load
from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")
OUT_DIR = os.path.join(REPORT_DIR, "interpret")
HIDDEN_DIM = 64
NUM_LAYERS = 3
HEADS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained(conv, name):
    path = os.path.join(MODEL_DIR, f"{conv}_{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到已训练模型 {path}，请先运行 train_eval.py")
    ckpt = torch.load(path, map_location=DEVICE)
    out_dim = 2 if TASK_TYPE[name] == "classification" else 1
    model = build_model(conv, ckpt["in_dim"], ckpt["hidden"], out_dim, ckpt["layers"], heads=ckpt.get("heads", HEADS)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def select_examples(dataset, split, task_type, num=4, seed=0):
    """挑选代表性分子：回归取预测误差最大/最小的样本，分类取置信度最高的正/负样本。
    返回 (data, smiles) 列表。此处只做数据挑选，预测在 explain_one 中完成。
    """
    import random
    random.seed(seed)
    idx_test = [int(i) for i in split["test"]]
    idx_sel = random.sample(idx_test, min(num, len(idx_test)))
    out = []
    for i in idx_sel:
        d = dataset[i]
        smi = getattr(d, "smiles", None)
        if smi is None or Chem.MolFromSmiles(smi) is None:
            continue
        out.append((i, d, smi))
    return out[:num]


def explain_one(model, data, task_type, epochs=300, edge_attr=None, fp=None):
    """对单个分子运行 GNNExplainer，返回 (node_mask, edge_mask, pred_value)

    edge_attr/fp：特征模型（feat_b/c）额外输入，会转发给模型 forward；
    基线模型保持 None。
    """
    mode = "regression" if task_type == "regression" else "binary_classification"
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=0.05),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=ModelConfig(mode=mode, task_level="graph", return_type="raw"),
    )
    x = data.x.to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)
    batch = data.batch.to(DEVICE) if data.batch is not None else None
    if edge_attr is None and hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_attr = data.edge_attr.to(DEVICE)
    if fp is None and hasattr(data, "fp") and data.fp is not None:
        fp = data.fp.to(DEVICE)
    explanation = explainer(x, edge_index, batch=batch, edge_attr=edge_attr, fp=fp)
    node_mask = explanation.node_mask.detach().cpu().numpy().flatten().astype(np.float64)
    edge_mask = explanation.edge_mask.detach().cpu().numpy().flatten().astype(np.float64)
    with torch.no_grad():
        pred = model(x, edge_index, batch, edge_attr=edge_attr, fp=fp)
    if task_type == "regression":
        pred_val = float(pred.squeeze(-1).item())
    else:
        pred_val = float(torch.softmax(pred, dim=-1)[0, 1].item())
    return node_mask, edge_mask, pred_val


def draw_molecule(smiles, node_mask, edge_mask, edge_index, out_path, title=""):
    """按重要性着色绘制分子，输出 PNG。
    实现：RDKit 计算 2D 坐标，matplotlib 绘制节点/边并按 mask 着色（RdYlBu_r：红=重要，蓝=次要）。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(n_atoms)]
    )

    def norm(v):
        v = np.asarray(v, dtype=np.float64)
        r = v.max() - v.min()
        return (v - v.min()) / r if r > 1e-9 else np.zeros_like(v)

    nm = norm(node_mask)
    em = norm(edge_mask)

    bond_map = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond_map[(i, j)] = b.GetIdx()
        bond_map[(j, i)] = b.GetIdx()
    bond_imp = {}
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        if (i, j) in bond_map:
            bid = bond_map[(i, j)]
            bond_imp.setdefault(bid, []).append(em[e])
    bond_val = {bid: float(np.mean(v)) for bid, v in bond_imp.items()}

    cmap = plt.get_cmap("RdYlBu_r")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    # 键
    for bid, v in bond_val.items():
        b = mol.GetBondWithIdx(bid)
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        c = cmap(v)
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color=c, linewidth=1.2 + 3.2 * v, zorder=1,
        )
    # 原子
    for i in range(n_atoms):
        v = nm[i] if i < len(nm) else 0.0
        c = cmap(v)
        ax.scatter(coords[i, 0], coords[i, 1], s=720, color=c,
                   edgecolors="#333333", linewidths=1.2, zorder=2)
        ax.text(coords[i, 0], coords[i, 1], mol.GetAtomWithIdx(i).GetSymbol(),
                ha="center", va="center", fontsize=13, fontweight="bold", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.62, pad=0.03)
    cbar.set_label("重要性（红=关键）", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def explain_dataset(conv, name, num=4, epochs=300):
    dataset, split = load(name, verbose=False)
    task_type = TASK_TYPE[name]
    model, ckpt = load_trained(conv, name)
    examples = select_examples(dataset, split, task_type, num=num)
    results = []
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== 可解释性分析: {conv} × {name}（{len(examples)} 个分子）===")
    for i, data, smiles in examples:
        node_mask, edge_mask, pred = explain_one(model, data, task_type, epochs=epochs)
        tag = f"{conv}_{name}_idx{i}"
        img_path = os.path.join(OUT_DIR, f"{tag}.png")
        ok = draw_molecule(smiles, node_mask, edge_mask, data.edge_index.cpu().numpy(), img_path)
        top_atoms = [int(a) for a in np.argsort(-node_mask)[:5]]
        top_bonds = [int(b) for b in np.argsort(-edge_mask)[:5]]
        res = {
            "conv": conv, "dataset": name, "index": i, "smiles": smiles,
            "pred": round(pred, 4),
            "top5_atoms": top_atoms,
            "top5_edges": top_bonds,
            "image": os.path.relpath(img_path, ROOT),
            "drawn": ok,
        }
        results.append(res)
        print(f"  idx={i} SMILES={smiles} 预测={pred:.4f} 图片={'OK' if ok else 'FAIL'} top原子={top_atoms}")
    path = os.path.join(OUT_DIR, f"{conv}_{name}_explain.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="GCN")
    ap.add_argument("--names", nargs="+", default=["ESOL", "FreeSolv", "BBBP"])
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    for name in args.names:
        explain_dataset(args.conv, name, num=args.num, epochs=args.epochs)


if __name__ == "__main__":
    main()
