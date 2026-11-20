# -*- coding: utf-8 -*-
"""解释一致性度量与报告的回归测试。"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explanation_consistency import (_jaccard_topk, _pairwise_spearman,
                                     _pairwise_stats)

REPORT = ROOT / "reports" / "explanation_consistency.json"


class ExplanationConsistencyTests(unittest.TestCase):
    def test_jaccard_topk_identical_and_disjoint(self):
        a = np.array([5, 4, 3, 2, 1, 0], dtype=float)
        self.assertEqual(_jaccard_topk(a, a.copy(), 5), 1.0)
        b = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        # top5: {0,1,2,3,4} vs {5,4,3,2,1} → 重合 4，并集 6
        self.assertAlmostEqual(_jaccard_topk(a, b, 5), 4 / 6, places=6)

    def test_pairwise_stats_shapes(self):
        vecs = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 0]), np.array([0, 2, 3, 4, 5])]
        stats = _pairwise_stats(vecs, [5, 10])
        self.assertIn("jaccard_top5", stats)
        self.assertGreaterEqual(stats["jaccard_top5"]["mean"], 0)
        self.assertLessEqual(stats["jaccard_top5"]["mean"], 1)

    def test_spearman_constant_is_skipped(self):
        stats = _pairwise_spearman([np.ones(5), np.arange(5)])
        self.assertIsNone(stats["mean"])  # 常数向量不参与相关

    @unittest.skipUnless(REPORT.exists(), "缺少 explanation_consistency.json")
    def test_report_structure_and_level_rule(self):
        with open(REPORT, encoding="utf-8") as f:
            payload = json.load(f)
        thresholds = payload["thresholds"]
        for name in ("ESOL", "FreeSolv", "BBBP"):
            ds = payload[name]
            self.assertEqual(ds["n_molecules"], len(ds["molecules"]))
            self.assertGreaterEqual(ds["n_molecules"], 30)
            self.assertLessEqual(ds["n_molecules"], 50)
            j10 = ds["summary"]["jaccard_top10_atom"]["mean"]
            level = ds["summary"]["level"]
            if j10 >= thresholds["high"]:
                self.assertEqual(level, "高")
            elif j10 >= thresholds["medium"]:
                self.assertEqual(level, "中")
            else:
                self.assertEqual(level, "低")
            for m in ds["molecules"]:
                self.assertEqual(len(m["atom_importance_mean"]), m["n_atoms"])
                self.assertEqual(len(m["bond_importance_mean"]), m["n_bonds"])


if __name__ == "__main__":
    unittest.main()
