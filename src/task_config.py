# -*- coding: utf-8 -*-
"""Single source of truth for dataset task semantics."""
from typing import Iterable, Optional, Sequence

import torch


TASK_TYPE = {
    "ESOL": "regression",
    "FreeSolv": "regression",
    "BBBP": "classification",
    "Clintox": "classification",
}

# MoleculeNet's Clintox has two labels. CT_TOX is column 1; other supported
# datasets use column 0.
TARGET_COL = {
    "ESOL": 0,
    "FreeSolv": 0,
    "BBBP": 0,
    "Clintox": 1,
}


def task_type(name: str) -> str:
    """Return the task type for a supported dataset."""
    try:
        return TASK_TYPE[name]
    except KeyError as exc:
        raise ValueError(f"未知数据集 {name}，可选: {list(TASK_TYPE)}") from exc


def target_column(name: str) -> int:
    """Return the target column used by all training/evaluation paths."""
    try:
        return TARGET_COL[name]
    except KeyError as exc:
        raise ValueError(f"未知数据集 {name}，可选: {list(TASK_TYPE)}") from exc


def compute_class_weight(
    name: str,
    dataset: Sequence,
    indices: Iterable[int],
    target_col: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Compute inverse-frequency binary class weights from training data.

    Only Clintox opts into class weighting. ``target_col`` can be overridden for
    featurized datasets, whose y has already been reduced to one target column.
    """
    if task_type(name) != "classification" or name != "Clintox":
        return None

    col = target_column(name) if target_col is None else target_col
    labels = torch.cat([
        dataset[int(i)].y[:, col].reshape(-1).long()
        for i in indices
    ])
    n_pos = int((labels == 1).sum())
    n_neg = int((labels != 1).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"{name} 训练集必须同时包含正负样本，当前 n_pos={n_pos}, n_neg={n_neg}"
        )
    return torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)