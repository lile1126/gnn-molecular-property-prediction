# -*- coding: utf-8 -*-
"""批量 CSV 上传、批量预测与 CSV 导出的回归测试。"""
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import app as webapp
from models import build_model


class BatchCsvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webapp.app.config.update(TESTING=True)
        cls.client = webapp.app.test_client()

    def _patch_models(self):
        models = {("GCN", "ESOL"): build_model("GCN", 9, 8, 1, num_layers=1).to(webapp.DEVICE).eval(),
                  ("GCN", "BBBP"): build_model("GCN", 9, 8, 2, num_layers=1).to(webapp.DEVICE).eval()}
        return patch.dict(webapp.MODELS, models, clear=False)

    def test_batch_json_list(self):
        with self._patch_models():
            response = self.client.post("/api/batch", json={
                "smiles_list": ["CCCO", "CCO"],
                "dataset": "ESOL", "conv": "GCN", "feat": "base", "method": "single",
            })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["n_ok"], 2)
        for row in payload["results"]:
            self.assertEqual(row["status"], "ok")
            self.assertIn("prediction", row)

    def test_batch_rejects_invalid_and_long_smiles(self):
        with self._patch_models():
            response = self.client.post("/api/batch", json={
                "smiles_list": ["C1=CC=CC(C)C", "C" * 301],
                "dataset": "ESOL", "conv": "GCN", "feat": "base", "method": "single",
            })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["n_ok"], 0)
        self.assertEqual(payload["n_error"], 2)
        messages = " ".join(row["error"] for row in payload["errors"])
        self.assertIn("过长", messages)

    def test_batch_limits(self):
        with self._patch_models():
            too_many = self.client.post("/api/batch", json={
                "smiles_list": ["CCO"] * 201, "dataset": "ESOL", "conv": "GCN",
                "feat": "base", "method": "single",
            })
        self.assertEqual(too_many.status_code, 400)
        empty = self.client.post("/api/batch", json={
            "smiles_list": [], "dataset": "ESOL", "conv": "GCN", "feat": "base", "method": "single",
        })
        self.assertEqual(empty.status_code, 400)

    def test_csv_upload(self):
        csv_bytes = "smiles,id\nCCCO,p1\nCCO,p2\n".encode("utf-8")
        with self._patch_models():
            response = self.client.post("/api/batch", data={
                "file": (io.BytesIO(csv_bytes), "mols.csv"),
                "dataset": "ESOL", "conv": "GCN", "feat": "base", "method": "single",
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["n_ok"], 2)
        self.assertEqual([r["id"] for r in payload["results"]], ["p1", "p2"])

    def test_export_csv(self):
        response = self.client.post("/api/export", json={
            "dataset": "ESOL",
            "results": [{"row": 2, "id": "p1", "smiles": "CCCO", "status": "ok",
                         "prediction": -0.5, "std": 0.1, "ci90": [-0.66, -0.34],
                         "unit": "logS", "method": "单模型", "task_type": "regression"}],
            "errors": [{"row": 3, "id": "bad", "smiles": "X", "error": "SMILES 解析失败"}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers["Content-Type"])
        self.assertIn("attachment", response.headers["Content-Disposition"])
        text = response.data.decode("utf-8")
        self.assertTrue(text.startswith("\ufeff"))
        self.assertIn("CCCO", text)
        self.assertIn("SMILES 解析失败", text)

    def test_export_requires_results(self):
        response = self.client.post("/api/export", json={"results": [], "errors": []})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
