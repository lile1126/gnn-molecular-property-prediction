# -*- coding: utf-8 -*-
"""
分子特征工程模块（选题深挖 · 方向1）
====================================
对照选题要求"分子特征工程（分子指纹、图构建）"，在 MoleculeNet 数据基础上
重算三档特征级别：

  feat_a : 原子特征增强（RDKit 原子性质，24 维）
  feat_b : 原子 + 键特征（键型/共轭/环内/立体，7 维）
  feat_c : 原子 + 键 + 分子指纹（MACCS 166 位）

图拓扑复用 MoleculeNet 的 edge_index（保证与基线可比），只替换节点/键/分子级特征。
Clintox 任务取 CT_TOX 毒性列（col 1）。

用法:
    python src/featurize.py --names ESOL FreeSolv BBBP Clintox --level c
"""
import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data")
FEAT_DIR = os.path.join(DATA_ROOT, "featurized")

# ---------------- 原子特征 ----------------
ATOM_TYPES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B", "other"]


def atom_features(atom):
    """RDKit 原子 -> 特征向量（24 维）"""
    sym = atom.GetSymbol()
    t = ATOM_TYPES.index(sym) if sym in ATOM_TYPES else ATOM_TYPES.index("other")
    deg = min(atom.GetDegree(), 5)
    hyb = atom.GetHybridization()
    h = {Chem.rdchem.HybridizationType.SP: 0, Chem.rdchem.HybridizationType.SP2: 1,
         Chem.rdchem.HybridizationType.SP3: 2}.get(hyb, 3)
    return np.array([
        *[1.0 if i == t else 0.0 for i in range(len(ATOM_TYPES))],
        *[1.0 if deg == i else 0.0 for i in range(6)],
        *[1.0 if h == i else 0.0 for i in range(3)],
        float(atom.GetChiralTag() != Chem.rdchem.CHI_UNSPECIFIED),  # 手性
        float(atom.GetIsAromatic()),                                 # 芳香
        float(atom.IsInRing()),                                      # 环内
        float(atom.GetTotalNumHs() > 0),                             # H 供体近似
        float(atom.GetNoImplicit() and atom.GetFormalCharge() < 0),  # 受体近似
        float(atom.GetFormalCharge()),                               # 形式电荷
    ], dtype=np.float32)


# ---------------- 键特征 ----------------
def bond_features(bond):
    bt = bond.GetBondType()
    b = {Chem.rdchem.BondType.SINGLE: 0, Chem.rdchem.BondType.DOUBLE: 1,
         Chem.rdchem.BondType.TRIPLE: 2, Chem.rdchem.BondType.AROMATIC: 3}.get(bt, 0)
    return np.array([
        *[1.0 if b == i else 0.0 for i in range(4)],
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE),
    ], dtype=np.float32)


def build_featurized(dataset, name, level="c"):
    """为 MoleculeNet 数据集重算特征，返回 feature dict 列表与统计。"""
    items = []
    n_skip = 0
    for i, d in enumerate(dataset):
        smi = getattr(d, "smiles", None)
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            n_skip += 1
            continue
        x = np.stack([atom_features(a) for a in mol.GetAtoms()]).astype(np.float32)
        if level in ("b", "c"):
            e_idx = d.edge_index.numpy()
            n_edges = e_idx.shape[1]
            ea = np.zeros((n_edges, 7), dtype=np.float32)
            bmap = {}
            for b in mol.GetBonds():
                bmap[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = b
                bmap[(b.GetEndAtomIdx(), b.GetBeginAtomIdx())] = b
            for e in range(n_edges):
                i0, j0 = int(e_idx[0, e]), int(e_idx[1, e])
                if i0 == j0:
                    continue  # 自环无键特征
                b = bmap.get((i0, j0))
                if b is not None:
                    ea[e] = bond_features(b)
            edge_attr = torch.tensor(ea)
        else:
            edge_attr = None
        fp = None
        if level == "c":
            fp = MACCSkeys.GenMACCSKeys(mol).ToBitString()
            fp = np.array([float(c) for c in fp], dtype=np.float32)  # 167 位（含首位 pad）
            fp = fp[1:].astype(np.float32)  # 166 位，2D 存储 [1,166] 便于 PyG Batch 拼接
        y = d.y[:, 1].float() if name == "Clintox" else d.y[:, 0].float()
        items.append(Data(
            x=torch.tensor(x),
            edge_index=d.edge_index,
            edge_attr=edge_attr,
            fp=torch.tensor(fp).unsqueeze(0) if fp is not None else None,
            y=y.unsqueeze(0),
            smiles=smi,
        ))
    if n_skip:
        print(f"  跳过 {n_skip} 个解析失败分子")
    return items


def featurize_smiles(smiles, level="c"):
    """Web 用：对任意 SMILES 构建特征 Data（与训练口径一致）。
    base -> 返回 None（走 PyG from_smiles 默认 9 维特征）。"""
    if level == "base":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 解析失败: {smiles}")
    x = np.stack([atom_features(a) for a in mol.GetAtoms()]).astype(np.float32)
    # 图拓扑对所有非 base 档位都要构建（与训练口径一致：feat_a 保留边、无键特征）
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    edge_attr = None
    if level in ("b", "c"):
        ea = np.zeros((edge_index.shape[1], 7), dtype=np.float32)
        bmap = {}
        for b in mol.GetBonds():
            bmap[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = b
            bmap[(b.GetEndAtomIdx(), b.GetBeginAtomIdx())] = b
        for e in range(edge_index.shape[1]):
            b = bmap.get((int(edge_index[0, e]), int(edge_index[1, e])))
            if b is not None:
                ea[e] = bond_features(b)
        edge_attr = torch.tensor(ea)
    fp = None
    if level == "c":
        fp = MACCSkeys.GenMACCSKeys(mol).ToBitString()
        fp = np.array([float(c) for c in fp], dtype=np.float32)[1:]
        fp = torch.tensor(fp).unsqueeze(0)
    return Data(x=torch.tensor(x), edge_index=edge_index, edge_attr=edge_attr, fp=fp)


def load_featurized(name, level="c", verbose=True):
    """加载（或缓存构建）特征工程数据集，返回 (data_list, meta)"""
    os.makedirs(FEAT_DIR, exist_ok=True)
    cache = os.path.join(FEAT_DIR, f"{name}_feat{level}.pt")
    if os.path.exists(cache):
        if verbose:
            print(f"[{name}/feat{level}] 使用缓存 {cache}")
        obj = torch.load(cache, weights_only=False)
        return obj["data"], obj["meta"]
    if verbose:
        print(f"[{name}/feat{level}] 构建特征（MoleculeNet + RDKit）...")
    ds = MoleculeNet(root=DATA_ROOT, name=name)
    items = build_featurized(ds, name, level=level)
    meta = {
        "name": name, "level": level, "n": len(items),
        "atom_dim": items[0].x.shape[1],
        "bond_dim": getattr(items[0], "edge_attr", None).shape[1]
        if getattr(items[0], "edge_attr", None) is not None else 0,
        "fp_dim": getattr(items[0], "fp", None).shape[1]
        if getattr(items[0], "fp", None) is not None else 0,
        "task": "classification" if name in ("BBBP", "Clintox") else "regression",
    }
    torch.save({"data": items, "meta": meta}, cache)
    if verbose:
        print(f"[{name}/feat{level}] 完成 n={meta['n']} atom_dim={meta['atom_dim']} bond_dim={meta['bond_dim']} fp_dim={meta['fp_dim']} -> {cache}")
    return items, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["ESOL", "FreeSolv", "BBBP", "Clintox"])
    ap.add_argument("--levels", nargs="+", default=["a", "b", "c"])
    args = ap.parse_args()
    for name in args.names:
        for level in args.levels:
            load_featurized(name, level=level)
    print("特征工程数据准备完成。")


if __name__ == "__main__":
    main()
