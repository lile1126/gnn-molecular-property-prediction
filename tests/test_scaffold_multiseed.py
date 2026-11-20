# -*- coding: utf-8 -*-
"""骨架划分多种子结果与统计检验的回归测试。"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_scaffold_multiseed import bootstrap_delta, mean_std, verdict

ANALYSIS = ROOT / "reports" / "scaffold_multiseed_analysis.json"
RAW = ROOT / "reports" / "scaffold_multiseed_results.json"


class ScaffoldMultiseedTests(unittest.TestCase):
    def test_mean_std(self):
        stats = mean_std([0.7, 0.8, 0.9])
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mean"], 0.8, places=4)
        self.assertAlmostEqual(stats["std"], 0.0816, places=3)

    def test_bootstrap_identical_predictions(self):
        y = np.linspace(0, 1, 40)
        pred = y.copy()
        res = bootstrap_delta(pred, pred, y, "regression", B=200)
        self.assertAlmostEqual(res["delta"], 0.0, places=6)
        self.assertEqual(res["ci95"], [0.0, 0.0])

    def test_verdict_rule(self):
        self.assertEqual(verdict(0.02, [0.01, 0.03]), "稳定提升")
        self.assertEqual(verdict(0.02, [-0.01, 0.04]), "趋势性提升")
        self.assertEqual(verdict(-0.01, [-0.03, 0.01]), "无提升")

    @unittest.skipUnless(RAW.exists(), "缺少 scaffold_multiseed_results.json")
    def test_raw_report_covers_three_datasets_and_seeds(self):
        with open(RAW, encoding="utf-8") as f:
            payload = json.load(f)
        for name in ("ESOL", "FreeSolv", "BBBP"):
            self.assertIn(name, payload["datasets"])
            ds = payload["datasets"][name]
            seeds = {int(k.split("_s")[-1]) for k in ds["members"]}
            self.assertGreaterEqual(len(seeds), 3, f"{name} 种子不足 3 个: {seeds}")
            for k, m in ds["members"].items():
                self.assertEqual(len(m["test_pred"]), len(ds["y_test"]), k)
                self.assertEqual(len(m["val_pred"]), len(ds["y_val"]), k)

    @unittest.skipUnless(ANALYSIS.exists(), "缺少 scaffold_multiseed_analysis.json")
    def test_analysis_structure_and_verdict_consistency(self):
        with open(ANALYSIS, encoding="utf-8") as f:
            payload = json.load(f)
        for name in ("ESOL", "FreeSolv", "BBBP"):
            ds = payload["datasets"][name]
            for key in ("best_single", "homo_seed_ens", "homo_arch", "hetero_all9"):
                self.assertIn(key, ds["strategies"])
                self.assertEqual(ds["strategies"][key]["metric"],
                                 "R2" if ds["task"] == "regression" else "AUC")
            cmp = ds["comparison"]["hetero_all9_vs_best_single"]
            self.assertIn(cmp["verdict"], ("稳定提升", "趋势性提升", "无提升", "无法判定"))
            lo, hi = cmp["ci95"]
            if cmp["verdict"] == "稳定提升":
                self.assertGreater(lo, 0, f"{name}: CI 跨 0 却写成稳定提升")
            if cmp["delta"] <= 0:
                self.assertEqual(cmp["verdict"], "无提升")
            self.assertGreaterEqual(cmp["n_boot"], 1000)


if __name__ == "__main__":
    unittest.main()
