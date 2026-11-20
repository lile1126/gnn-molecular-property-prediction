# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch_geometric.data import Data

import app as webapp
from models import build_model


class WebUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webapp.app.config.update(TESTING=True)
        cls.client = webapp.app.test_client()

    def test_index_and_static_assets(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("MolGNN", html)
        self.assertIn("/static/css/app.css", html)
        self.assertIn("/static/js/app.js", html)
        self.assertIn("/static/js/batch.js", html)
        self.assertIn("批量预测", html)
        self.assertIn("解释一致性", html)
        css_response = self.client.get("/static/css/app.css")
        js_response = self.client.get("/static/js/app.js")
        batch_response = self.client.get("/static/js/batch.js")
        self.assertEqual(css_response.status_code, 200)
        self.assertEqual(js_response.status_code, 200)
        self.assertEqual(batch_response.status_code, 200)
        for resp in (css_response, js_response, batch_response):
            resp.close()

    def test_meta_exposes_version_consistency_and_limits(self):
        response = self.client.get("/api/meta")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        for key in ("model_version", "explanation_consistency", "max_smiles_len", "max_batch_rows"):
            self.assertIn(key, payload)
        self.assertLessEqual(payload["max_smiles_len"], 512)
        self.assertIn("level", payload["explanation_consistency"]["ESOL"])
        self.assertEqual(payload["llm_available"] in (True, False), True)

    def test_smiles_validation(self):
        valid = self.client.post("/api/validate", json={"smiles": "CCCO"})
        self.assertEqual(valid.status_code, 200)
        self.assertTrue(valid.get_json()["ok"])

        invalid = self.client.post("/api/validate", json={"smiles": "C1=CC=CC(C)C"})
        self.assertEqual(invalid.status_code, 200)
        self.assertFalse(invalid.get_json()["ok"])

    def test_predict_with_small_model(self):
        model = build_model("GCN", 9, 8, 1, num_layers=1).to(webapp.DEVICE).eval()
        with patch.dict(webapp.MODELS, {("GCN", "ESOL"): model}, clear=False):
            response = self.client.post("/api/predict", json={
                "smiles": "CCCO",
                "dataset": "ESOL",
                "conv": "GCN",
                "method": "single",
                "feat": "base",
                "explain": False,
            })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["task_type"], "regression")
        self.assertIn("prediction", payload)

    def test_mc_prediction_restores_model_state(self):
        model = build_model("GCN", 9, 8, 1, num_layers=1, dropout=0.4).to(webapp.DEVICE).eval()
        before = [parameter.requires_grad for parameter in model.parameters()]
        data = Data(
            x=torch.randn(4, 9),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            batch=torch.zeros(4, dtype=torch.long),
        )
        mean, std = webapp.mc_predict_single(model, data, "regression", T=3)
        self.assertTrue(torch.isfinite(torch.tensor(mean)))
        self.assertTrue(torch.isfinite(torch.tensor(std)))
        self.assertFalse(model.training)
        self.assertEqual(before, [parameter.requires_grad for parameter in model.parameters()])

    def test_graphsage_edge_features_forward(self):
        model = build_model(
            "GraphSAGE", 9, 8, 1, num_layers=2, edge_dim=3,
        ).to(webapp.DEVICE).eval()
        x = torch.randn(4, 9, device=webapp.DEVICE)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long,
            device=webapp.DEVICE,
        )
        edge_attr = torch.randn(4, 3, device=webapp.DEVICE)
        with torch.no_grad():
            output = model(
                x, edge_index, batch=torch.zeros(4, dtype=torch.long, device=webapp.DEVICE),
                edge_attr=edge_attr,
            )
        self.assertEqual(tuple(output.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()