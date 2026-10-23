# -*- coding: utf-8 -*-
"""验证特征对齐（临时）：edge_attr 非零率 + 抽查分子"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from rdkit import Chem

obj = torch.load(r"D:\桌面\工程实践\GNN分子性质预测\data\featurized\ESOL_featc.pt", weights_only=False)
data = obj["data"]
# 统计边特征全零比例
tot = zeros = 0
for d in data[:200]:
    if d["edge_attr"] is None:
        continue
    ea = d["edge_attr"]
    tot += ea.shape[0]
    zeros += int((ea.abs().sum(dim=1) == 0).sum())
print(f"前200分子：边总数 {tot}，全零边 {zeros}（{zeros / max(tot, 1) * 100:.2f}%）")

# 抽查：分子 5 的 edge_attr 前 3 条 vs RDKit 手算
d = data[5]
smi = d["smiles"]
mol = Chem.MolFromSmiles(smi)
print("smiles:", smi)
print("x dim:", d["x"].shape, "edge_attr dim:", d["edge_attr"].shape, "fp dim:", d["fp"].shape)
ea = d["edge_attr"][:6]
print("edge_attr 前6行:\n", ea)
bmap = {}
for b in mol.GetBonds():
    bmap[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = b
    bmap[(b.GetEndAtomIdx(), b.GetBeginAtomIdx())] = b
ei = d["edge_index"]
for e in range(min(6, ei.shape[1])):
    i0, j0 = int(ei[0, e]), int(ei[1, e])
    b = bmap.get((i0, j0))
    btype = str(b.GetBondType()) if b else "SELF/None"
    print(f"  edge {i0}-{j0}: {btype} | feat={ea[e].tolist()}")
