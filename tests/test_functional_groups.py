# -*- coding: utf-8 -*-
"""官能团识别与重要性聚合的回归测试。"""
import sys
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from functional_group_explain import (aggregate, bond_importance_from_mask,
                                       detect_groups)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
ETHANOL = "CCO"


class FunctionalGroupTests(unittest.TestCase):
    def test_aspirin_groups(self):
        mol = Chem.MolFromSmiles(ASPIRIN)
        groups, unassigned = detect_groups(mol)
        names = {g["name"] for g in groups}
        self.assertIn("羧基", names)
        self.assertIn("酯基", names)
        self.assertIn("芳香环", names)
        self.assertLessEqual(len(unassigned), mol.GetNumAtoms())

    def test_carboxyl_not_double_counted_as_hydroxyl(self):
        mol = Chem.MolFromSmiles(ASPIRIN)
        groups, _ = detect_groups(mol)
        carboxyl = next(g for g in groups if g["name"] == "羧基")
        # 羧基的 OH 氧不应再出现在独立的羟基组里（原子唯一归属）
        self.assertFalse(any(g["name"] == "羟基" and set(g["atoms"]) & set(carboxyl["atoms"])
                             for g in groups))

    def test_ethanol_hydroxyl(self):
        mol = Chem.MolFromSmiles(ETHANOL)
        groups, _ = detect_groups(mol)
        names = {g["name"] for g in groups}
        self.assertIn("羟基", names)

    def test_aggregate_orders_by_importance(self):
        mol = Chem.MolFromSmiles(IBUPROFEN)
        n = mol.GetNumAtoms()
        mask = np.zeros(n)
        carboxyl = next(g for g in detect_groups(mol)[0] if g["name"] == "羧基")
        mask[carboxyl["atoms"]] = 1.0
        res = aggregate(IBUPROFEN, mask)
        self.assertEqual(res["top3"][0]["group"], "羧基")
        scores = [g["atom_importance_mean"] for g in res["top3"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(res["atom_assignment_coverage"] > 0, True)

    def test_bond_importance_mapping(self):
        mol = Chem.MolFromSmiles(ASPIRIN)
        edge_index = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edge_index += [[i, j], [j, i]]
        edge_index = np.array(edge_index).T
        edge_mask = np.arange(edge_index.shape[1], dtype=float)
        out = bond_importance_from_mask(mol, edge_index, edge_mask)
        self.assertEqual(len(out), mol.GetNumBonds())

    def test_layers_are_explicit(self):
        res = aggregate(ETHANOL, np.array([0.1, 0.9, 0.2]))
        self.assertIn("model_attribution", res["layers"])
        self.assertIn("chemical_recognition", res["layers"])
        self.assertIn("natural_language", res["layers"])


if __name__ == "__main__":
    unittest.main()
