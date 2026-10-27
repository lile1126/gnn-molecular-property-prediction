# -*- coding: utf-8 -*-
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_prep
from task_config import compute_class_weight


def fake_dataset(smiles_values):
    return [SimpleNamespace(smiles=value) for value in smiles_values]


class SplitTests(unittest.TestCase):
    def test_existing_split_is_reused(self):
        dataset = fake_dataset(["CC", "CCC", "CCCC", "CCCCC", "CCCCCC"])
        with tempfile.TemporaryDirectory() as tmp, patch.object(data_prep, "SPLIT_ROOT", tmp):
            first = data_prep.load_or_make_split(dataset, "ESOL")
            path = Path(tmp) / "ESOL_split.json"
            before = path.read_text(encoding="utf-8")
            second = data_prep.load_or_make_split(dataset, "ESOL")
            after = path.read_text(encoding="utf-8")
        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertIn("dataset_hash", second)
        self.assertEqual(second["version"], data_prep.SPLIT_VERSION)

    def test_changed_dataset_is_rejected(self):
        original = fake_dataset(["CC", "CCC", "CCCC"])
        changed = fake_dataset(["CC", "CCC", "CCO"])
        with tempfile.TemporaryDirectory() as tmp, patch.object(data_prep, "SPLIT_ROOT", tmp):
            data_prep.load_or_make_split(original, "ESOL")
            with self.assertRaises(ValueError):
                data_prep.load_or_make_split(changed, "ESOL")

    def test_recreate_split_is_explicit(self):
        dataset = fake_dataset(["CC", "CCC", "CCCC"])
        with tempfile.TemporaryDirectory() as tmp, patch.object(data_prep, "SPLIT_ROOT", tmp):
            first = data_prep.load_or_make_split(dataset, "ESOL")
            second = data_prep.load_or_make_split(dataset, "ESOL", recreate=True)
        self.assertEqual(first["train"], second["train"])


class TaskConfigTests(unittest.TestCase):
    def test_clintox_weight_comes_from_training_labels(self):
        dataset = [
            SimpleNamespace(y=torch.tensor([[1.0, 1.0]])),
            SimpleNamespace(y=torch.tensor([[1.0, 0.0]])),
            SimpleNamespace(y=torch.tensor([[1.0, 0.0]])),
            SimpleNamespace(y=torch.tensor([[1.0, 0.0]])),
            SimpleNamespace(y=torch.tensor([[1.0, 0.0]])),
        ]
        weight = compute_class_weight("Clintox", dataset, range(len(dataset)))
        self.assertAlmostEqual(float(weight[0]), 1.0)
        self.assertAlmostEqual(float(weight[1]), 4.0)

    def test_non_clintox_is_unweighted(self):
        dataset = [SimpleNamespace(y=torch.tensor([[1.0]]))]
        self.assertIsNone(compute_class_weight("BBBP", dataset, [0], target_col=0))


if __name__ == "__main__":
    unittest.main()