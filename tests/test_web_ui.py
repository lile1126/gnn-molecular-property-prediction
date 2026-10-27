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
        css_response = self.client.get("/static/css/app.css")
        js_response = self.client.get("/static/js/app.js")
        self.assertEqual(css_response.status_code, 200)
        self.assertEqual(js_response.status_code, 200)
        css_response.close()
        js_response.close()

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