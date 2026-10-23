# -*- coding: utf-8 -*-
"""
注意力机制可视化模块
====================
对 GAT 模型提取各层边注意力系数 alpha（PyG GATConv return_attention_weights），
并按化学键在分子 2D 结构图上可视化：键粗细/颜色编码注意力权重（红=注意力高）。

与 GNNExplainer（掩码视角）互补：GNNExplainer 回答"模型依赖哪些结构"，
注意力可视化回答"模型在传播时如何分配邻居权重"。

用法:
    python src/attention_viz.py --names ESOL FreeSolv BBBP --num 4 --layer 0
    python src/attention_viz.py --names BBBP --num 2 --layer 0 --both
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
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

from data_prep import TASK_TYPE, load
from interpret import load_trained, select_examples

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "reports", "attention")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_attention(model, data, layer=0):
    """逐层前向提取 GAT 边注意力系数。

    返回:
        alphas: list[Tensor]，每层 [E_l, H_l]（H_l=多头数；中间层 heads，末层 1）
        edge_indices: list[Tensor]，每层含自环的 [2, E_l]
    """
    x = data.x.to(DEVICE).float()
    edge_index = data.edge_index.to(DEVICE)
    alphas, edge_indices = [], []
    for i, conv in enumerate(model.convs):
        out, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
        alphas.append(alpha.detach().cpu())      # [E, H]
        edge_indices.append(ei.detach().cpu())
        x = out
        if i < len(model.convs) - 1:
            x = F.relu(x)
            x = F.dropout(x, p=model.dropout, training=False)
    if layer < 0 or layer >= len(alphas):
        raise ValueError(f"layer {layer} 超出范围（共 {len(alphas)} 层）")
    return alphas, edge_indices


def draw_attention_molecule(smiles, edge_index, alpha, out_path=None, title="", atom_color=None, as_buffer=False):
    """按边注意力权重着色绘制分子：键粗细/颜色编码注意力（红=高）。
    out_path 为空且 as_buffer=False 时仅返回 False；as_buffer=True 返回 PNG bytes。
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

    # 多头平均 -> 每边标量；过滤自环
    a = alpha.numpy()
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = np.asarray(a, dtype=np.float64)

    bond_map = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond_map[(i, j)] = b.GetIdx()
        bond_map[(j, i)] = b.GetIdx()
    bond_att = {}
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        if i == j:
            continue
        if (i, j) in bond_map:
            bid = bond_map[(i, j)]
            bond_att.setdefault(bid, []).append(a[e])
    if not bond_att:
        return False
    bond_val = {bid: float(np.mean(v)) for bid, v in bond_att.items()}

    # 归一化到 [0,1]（键级）
    vals = np.array(list(bond_val.values()))
    r = vals.max() - vals.min()
    norm = {bid: (v - vals.min()) / r if r > 1e-9 else 0.5 for bid, v in bond_val.items()}

    # 原子色：浅灰（若传入 atom_color 则按其着色）
    if atom_color is None:
        atom_c = ["#d8dde6"] * n_atoms
    else:
        ac = np.asarray(atom_color, dtype=np.float64)
        rr = ac.max() - ac.min()
        acn = (ac - ac.min()) / rr if rr > 1e-9 else np.zeros_like(ac)
        cmap_a = plt.get_cmap("RdYlBu_r")
        atom_c = [cmap_a(v) for v in acn]

    cmap = plt.get_cmap("RdYlBu_r")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    # 键
    for bid, v in bond_val.items():
        b = mol.GetBondWithIdx(bid)
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        c = cmap(norm[bid])
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color=c, linewidth=1.2 + 3.2 * norm[bid], zorder=1,
        )
    # 原子
    for i in range(n_atoms):
        ax.scatter(coords[i, 0], coords[i, 1], s=720, color=atom_c[i],
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
    cbar.set_label("注意力权重（红=高）", fontsize=11)
    fig.tight_layout()
    if as_buffer:
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    if out_path is None:
        plt.close(fig)
        return True
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def draw_per_head(smiles, edge_index, alpha, out_path=None, title="", as_buffer=False):
    """逐头注意力对比：2×2 并排绘制每头的键注意力（选题深挖 · 方向3）。
    要求 alpha [E, H]（layer0 通常 H=4）。
    out_path 为空且 as_buffer=False 时仅返回 False；as_buffer=True 返回 PNG bytes。
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
    a = alpha.numpy()
    heads = a.shape[1] if a.ndim > 1 else 1
    bond_map = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond_map[(i, j)] = b.GetIdx()
        bond_map[(j, i)] = b.GetIdx()
    cmap = plt.get_cmap("RdYlBu_r")

    ncols, nrows = (2, (heads + 1) // 2) if heads > 1 else (1, 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 5.2 * nrows),
                             squeeze=False)
    for h in range(heads):
        ax = axes[h // ncols][h % ncols]
        ah = a[:, h]
        # 每头内独立归一化（对比该头最关注的键）
        vals = ah[ah >= 0]
        r = vals.max() - vals.min() if len(vals) else 0
        bond_att = {}
        for e in range(edge_index.shape[1]):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            if i == j:
                continue
            if (i, j) in bond_map:
                bid = bond_map[(i, j)]
                bond_att.setdefault(bid, []).append(ah[e])
        if not bond_att:
            continue
        bv = {bid: float(np.mean(v)) for bid, v in bond_att.items()}
        vv = np.array(list(bv.values()))
        rr = vv.max() - vv.min()
        norm = {bid: (v - vv.min()) / rr if rr > 1e-9 else 0.5 for bid, v in bv.items()}
        for bid, v in bv.items():
            b = mol.GetBondWithIdx(bid)
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            c = cmap(norm[bid])
            ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                    color=c, linewidth=1.2 + 3.2 * norm[bid], zorder=1)
        for i in range(n_atoms):
            ax.scatter(coords[i, 0], coords[i, 1], s=560, color="#d8dde6",
                       edgecolors="#333333", linewidths=1.0, zorder=2)
            ax.text(coords[i, 0], coords[i, 1], mol.GetAtomWithIdx(i).GetSymbol(),
                    ha="center", va="center", fontsize=12, fontweight="bold", zorder=3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Head {h + 1}", fontsize=13)
    for h in range(heads, nrows * ncols):
        axes[h // ncols][h % ncols].axis("off")
    if title:
        fig.suptitle(title, fontsize=14, y=1.0)
    fig.tight_layout()
    if as_buffer:
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    if out_path is None:
        plt.close(fig)
        return True
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def attention_dataset(name, conv="GAT", num=4, layer=0, both=False, seed=0, per_head=False):
    dataset, split = load(name, verbose=False)
    task_type = TASK_TYPE[name]
    model, ckpt = load_trained(conv, name)
    examples = select_examples(dataset, split, task_type, num=num, seed=seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    print(f"=== 注意力可视化: {conv} × {name}（layer={layer}，{len(examples)} 个分子，per_head={per_head}）===")
    for i, data, smiles in examples:
        alphas, edge_indices = extract_attention(model, data, layer=layer)
        alpha = alphas[layer]
        ei = edge_indices[layer]
        # 预测（同前向）
        with torch.no_grad():
            x = data.x.to(DEVICE).float()
            out = model(x, data.edge_index.to(DEVICE), data.batch.to(DEVICE) if data.batch is not None else None)
        if task_type == "regression":
            pred = float(out.squeeze(-1).item())
        else:
            pred = float(torch.softmax(out, dim=-1)[0, 1].item())
        tag = f"{conv}_{name}_idx{i}"
        img_path = os.path.join(OUT_DIR, f"{tag}_attn_layer{layer}.png")
        ok = draw_attention_molecule(smiles, ei, alpha, img_path)
        ph_ok = False
        per_head_images = []
        if per_head and alpha.ndim > 1:
            ph_dir = os.path.join(OUT_DIR, "perhead")
            os.makedirs(ph_dir, exist_ok=True)
            n_h = alpha.shape[1]
            for h in range(n_h):
                hp = os.path.join(ph_dir, f"{tag}_head{h + 1}.png")
                draw_attention_molecule(smiles, ei, alpha[:, h], hp, title=f"Head {h + 1}")
                per_head_images.append(os.path.relpath(hp, ROOT))
            combo = os.path.join(ph_dir, f"{tag}_allheads.png")
            ph_ok = draw_per_head(smiles, ei, alpha, combo, title=f"{name} idx{i} 逐头注意力对比")
            per_head_images.append(os.path.relpath(combo, ROOT))
        # top 键
        a1 = alpha.numpy()
        if a1.ndim > 1:
            a1 = a1.mean(axis=1)
        top = []
        for e in np.argsort(-a1)[:8]:
            i0, j0 = int(ei[0, e]), int(ei[1, e])
            if i0 == j0:
                continue
            top.append({"edge": [i0, j0], "alpha": round(float(a1[e]), 4)})
        res = {
            "conv": conv, "dataset": name, "index": i, "smiles": smiles,
            "pred": round(pred, 4), "layer": layer,
            "heads": int(alpha.shape[1]) if alpha.ndim > 1 else 1,
            "top8_edges": top,
            "image": os.path.relpath(img_path, ROOT),
            "per_head_images": per_head_images,
            "drawn": ok,
        }
        results.append(res)
        print(f"  idx={i} SMILES={smiles} 预测={pred:.4f} 图片={'OK' if ok else 'FAIL'} "
              f"逐头={'OK' if ph_ok else '-'}")
    path = os.path.join(OUT_DIR, f"{conv}_{name}_attention_layer{layer}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="GAT")
    ap.add_argument("--names", nargs="+", default=["ESOL", "FreeSolv", "BBBP"])
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--layer", type=int, default=0, help="提取第几层注意力（0=第一层，最贴近化学键）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-head", action="store_true", help="逐头注意力对比（每头单独图 + 2x2 并排）")
    args = ap.parse_args()
    for name in args.names:
        attention_dataset(name, conv=args.conv, num=args.num, layer=args.layer,
                          seed=args.seed, per_head=args.per_head)


if __name__ == "__main__":
    main()
