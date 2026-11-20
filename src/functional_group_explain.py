# -*- coding: utf-8 -*-
"""
官能团级解释（P0-4）
====================
把 GNNExplainer 的**原子重要性**与**化学键重要性**聚合到化学官能团，
输出每个分子的 Top-3 重要官能团。

三层职责必须严格区分（报告中同样按此表述）：
  1. 模型归因（GNNExplainer）：原子/键重要性，来自 9 个「架构 × 特征档」配置的平均；
  2. 化学识别（RDKit）：SMARTS / 环感知判定官能团，只回答"分子里有什么"，
     不参与"哪个重要"的排序；
  3. 自然语言（规则模板或 LLM）：只负责把 1+2 的结果讲成人话，
     **不允许 LLM 自行决定哪个官能团重要**。

聚合口径：
  - 每个重原子最多归属一个官能团（按优先级从特异到通用分配，避免羧基与羟基重复计数）；
  - 官能团重要性 = 组内原子重要性均值（同时给出最大值与占全分子重要性的份额）；
  - 化学键重要性 = 组内键（两端原子都在组内）的均值。

用法：
    python src/functional_group_explain.py --names ESOL FreeSolv BBBP
    python src/functional_group_explain.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --dataset ESOL
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
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSISTENCY_PATH = os.path.join(ROOT, "reports", "explanation_consistency.json")
REPORT_PATH = os.path.join(ROOT, "reports", "functional_group_explanations.json")

NAMES = ["ESOL", "FreeSolv", "BBBP"]

# (优先级, 名称, SMARTS)；优先级数字越小越先分配，已分配原子不再被后续组占用
GROUPS = [
    (1, "羧基", "C(=O)[OX2H1]"),
    (2, "酯基", "[#6][CX3](=[OX1])[OX2H0][#6]"),
    (3, "酰胺", "[NX3][CX3](=[OX1])"),
    (4, "磺酰基", "S(=O)(=O)"),
    (5, "硝基", "[NX3](=O)=O"),
    (6, "氰基", "[C-]#[N+]"),
    (7, "羰基", "[CX3]=[OX1]"),
    (8, "羟基", "[OX2H]"),
    (9, "氨基", "[NX3;H2,H1;!$(NC=O)]"),
    (10, "醚键", "[OD2]([#6])[#6]"),
    (11, "三氟甲基", "[CX4](F)(F)F"),
    (12, "硫醚", "[#16X2H0]"),
]

HALOGENS = {"F": "卤素(F)", "Cl": "卤素(Cl)", "Br": "卤素(Br)", "I": "卤素(I)"}

LAYER_NOTE = {
    "model_attribution": "GNNExplainer（9 配置 seed42 主模型掩码平均，reports/explanation_consistency.json）",
    "chemical_recognition": "RDKit SMARTS + 环感知，仅判定存在性，不参与排序",
    "natural_language": "规则模板（离线降级）或 LLM 复述；LLM 不得决定官能团重要性",
}


def bond_importance_from_mask(mol, edge_index, edge_mask):
    """把有向边掩码折算到 RDKit 的无向键（双向取均值），返回长度 = 键数的数组。"""
    import numpy as _np
    edge_index = _np.asarray(edge_index)
    n_bonds = mol.GetNumBonds()
    acc = [[] for _ in range(n_bonds)]
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        bond = mol.GetBondBetweenAtoms(u, v)
        if bond is not None:
            acc[bond.GetIdx()].append(float(edge_mask[e]))
    return _np.array([float(_np.mean(a)) if a else 0.0 for a in acc])


def detect_groups(mol):
    """RDKit 识别官能团，并按优先级做原子唯一分配。返回 (groups, unassigned_atoms)。"""
    n_atoms = mol.GetNumAtoms()
    assigned = [None] * n_atoms
    groups = []

    for _, gname, smarts in GROUPS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt, uniquify=True)
        if not matches:
            continue
        keep = []
        for match in matches:
            atoms = set(int(a) for a in match)
            if any(assigned[a] is not None for a in atoms):
                # 允许部分重叠（如羧基中的羰基碳已分配），但至少一半原子空闲才收为新组
                free = [a for a in atoms if assigned[a] is None]
                if len(free) < max(1, len(atoms) // 2):
                    continue
            for a in atoms:
                assigned[a] = gname
            keep.append(atoms)
        if keep:
            union = set().union(*keep)
            groups.append({"name": gname, "atoms": sorted(union),
                           "instances": [sorted(s) for s in keep]})

    # 卤素：逐元素聚合成一个组
    for sym, label in HALOGENS.items():
        atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == sym and assigned[a.GetIdx()] is None]
        if atoms:
            for a in atoms:
                assigned[a] = label
            groups.append({"name": label, "atoms": atoms, "instances": [atoms]})

    # 芳香环：RDKit 环感知（覆盖苯环/吡啶/五元杂芳环）。
    # 环是骨架，允许与取代基在连接原子处重叠（聚合时按组内均值计算，不影响排序）。
    rings = []
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(int(a)).GetIsAromatic() for a in ring):
            atoms = [int(a) for a in ring]
            for a in atoms:
                if assigned[a] is None:
                    assigned[a] = "芳香环"
            rings.append(atoms)
    if rings:
        union = set().union(*[set(r) for r in rings])
        groups.append({"name": "芳香环", "atoms": sorted(union), "instances": rings})

    unassigned = [i for i in range(n_atoms) if assigned[i] is None]
    return groups, unassigned


def aggregate(smiles, atom_importance, bond_importance=None):
    """把原子/键重要性聚合到官能团。返回全部组与 Top-3。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 无法解析: {smiles}")
    atom_importance = np.asarray(atom_importance, dtype=np.float64).flatten()
    if atom_importance.shape[0] != mol.GetNumAtoms():
        raise ValueError(f"原子重要性长度 {atom_importance.shape[0]} 与分子原子数 {mol.GetNumAtoms()} 不一致")
    bond_importance = None if bond_importance is None else np.asarray(bond_importance, dtype=np.float64).flatten()

    groups, unassigned = detect_groups(mol)
    total = float(np.sum(np.abs(atom_importance))) + 1e-12
    bond_map = {}
    if bond_importance is not None and bond_importance.shape[0] == mol.GetNumBonds():
        for b in mol.GetBonds():
            bond_map[b.GetIdx()] = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())

    rows = []
    for g in groups:
        idx = g["atoms"]
        vals = atom_importance[idx]
        bvals = []
        if bond_importance is not None:
            for bid, (u, v) in bond_map.items():
                if u in idx and v in idx:
                    bvals.append(float(bond_importance[bid]))
        rows.append({
            "group": g["name"],
            "n_atoms": len(idx),
            "atoms": idx,
            "instances": g["instances"],
            "atom_importance_mean": round(float(np.mean(vals)), 4),
            "atom_importance_max": round(float(np.max(vals)), 4),
            "importance_share": round(float(np.sum(np.abs(vals)) / total), 4),
            "bond_importance_mean": round(float(np.mean(bvals)), 4) if bvals else None,
        })
    rows.sort(key=lambda r: (-r["atom_importance_mean"], -r["atom_importance_max"]))
    top3 = rows[:3]
    assigned_share = 1.0 - len(unassigned) / max(1, mol.GetNumAtoms())
    return {
        "smiles": smiles,
        "n_atoms": int(mol.GetNumAtoms()),
        "n_bonds": int(mol.GetNumBonds()),
        "groups_detected": len(rows),
        "atom_assignment_coverage": round(float(assigned_share), 3),
        "top3": top3,
        "all_groups": rows,
        "unassigned_atoms": unassigned,
        "layers": LAYER_NOTE,
    }


def explain_from_consistency(name, payload):
    molecules = payload.get(name, {}).get("molecules", [])
    out = []
    for m in molecules:
        try:
            res = aggregate(m["smiles"], m["atom_importance_mean"], m["bond_importance_mean"])
        except ValueError as e:
            print(f"  [跳过] {name} idx={m['index']}: {e}")
            continue
        out.append({
            "index": m["index"],
            "smiles": m["smiles"],
            "y_true": m["y_true"],
            "pred_mean": m["pred_mean"],
            "pred_std": m["pred_std"],
            "top3": res["top3"],
            "all_groups": res["all_groups"],
            "groups_detected": res["groups_detected"],
            "atom_assignment_coverage": res["atom_assignment_coverage"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=NAMES)
    ap.add_argument("--smiles", default=None, help="单分子演示（需配合 --dataset 的预计算掩码，否则只打印识别结果）")
    ap.add_argument("--dataset", default="ESOL")
    args = ap.parse_args()

    if args.smiles:
        mol = Chem.MolFromSmiles(args.smiles)
        if mol is None:
            raise SystemExit("SMILES 无法解析")
        groups, unassigned = detect_groups(mol)
        print(json.dumps({"smiles": args.smiles, "groups": groups, "unassigned": unassigned},
                         ensure_ascii=False, indent=2))
        return

    if not os.path.exists(CONSISTENCY_PATH):
        raise SystemExit(f"缺少 {CONSISTENCY_PATH}，请先运行 python src/explanation_consistency.py")
    with open(CONSISTENCY_PATH, encoding="utf-8") as f:
        payload = json.load(f)

    report = {"config": {
        "attribution_source": "reports/explanation_consistency.json",
        "aggregation": "原子唯一归属 + 组内均值/最大值/份额；键为组内键均值",
        "priority_order": [g[1] for g in GROUPS] + ["卤素", "芳香环"],
    }, "layers": LAYER_NOTE, "datasets": {}}
    for name in args.names:
        if name not in payload:
            print(f"  [跳过] {name} 不在 explanation_consistency.json 中")
            continue
        rows = explain_from_consistency(name, payload)
        report["datasets"][name] = {"n_molecules": len(rows), "molecules": rows}
        # 统计最常见 Top-1 官能团
        counts = {}
        for r in rows:
            if r["top3"]:
                counts[r["top3"][0]["group"]] = counts.get(r["top3"][0]["group"], 0) + 1
        report["datasets"][name]["top1_group_counts"] = counts
        print(f"  {name}: {len(rows)} 个分子；Top-1 官能团分布 {counts}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
