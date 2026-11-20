# -*- coding: utf-8 -*-
"""LLM 解释降级与一致性校验的回归测试（不调用真实 API / 不跑 GNNExplainer）。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from torch_geometric.utils import from_smiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import llm_explain


class LlmFallbackTests(unittest.TestCase):
    def setUp(self):
        self.structure = {
            "smiles": "CCO", "heavy_atoms": 3,
            "key_atoms": ["O", "C"], "key_bonds": ["C-O(单)"],
            "functional_groups": ["羟基"],
            "top_groups": [{"group": "羟基", "score": 0.82, "share": 0.55, "atoms": [2], "n_atoms": 1}],
        }

    def test_rule_template_mentions_evidence_only(self):
        text = llm_explain.rule_based_explanation(self.structure, "ESOL", -0.4, "regression")
        self.assertIn("规则模板", text)
        self.assertIn("羟基", text)
        self.assertIn("GNNExplainer", text)
        self.assertIn("RDKit", text)
        self.assertNotIn("DeepSeek", text)

    def test_consistency_detects_undetected_group(self):
        text = "分子中的羧基提高了水溶性。"
        res = llm_explain.check_consistency(text, self.structure)
        self.assertIn("羧基", res["mentioned_undetected"])
        self.assertEqual(res["mentioned_detected"], [])  # 文本没有提到已检出的羟基

    def test_generate_explanation_falls_back_without_api(self):
        data = from_smiles("CCO")
        node_mask = np.array([0.1, 0.1, 0.9])
        edge_mask = np.zeros(data.edge_index.shape[1]) + 0.4
        with patch.object(llm_explain, "explain_one", return_value=(node_mask, edge_mask, 0.3)), \
                patch.object(llm_explain, "_call_llm_deepseek", side_effect=RuntimeError("未配置 API Key")):
            result = llm_explain.generate_explanation("CCO", dataset="ESOL", model=object(), data=data)
        self.assertEqual(result["provider"], "rule_template")
        self.assertIn("规则模板", result["explanation"])
        self.assertIn("未配置", result["fallback_reason"])
        self.assertEqual(result["structure"]["top_groups"][0]["group"], "羟基")


if __name__ == "__main__":
    unittest.main()
