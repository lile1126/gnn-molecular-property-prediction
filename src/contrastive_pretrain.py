# -*- coding: utf-8 -*-
"""
图对比学习自监督预训练（GraphCL 风格）
======================================
在无标签分子池上，用增强视图对 + InfoNCE 损失训练三种架构（GCN / GAT / GraphSAGE）的图编码器。

增强策略（batch 级别，向量化实现）：
  - attr_mask   : 按节点随机掩码原子特征（置零）
  - edge_perturb: 按比例随机删边 / 加同图随机边
  - node_drop   : 按比例随机删除节点（含关联边），图内重新编号

视图组合（主实验 full）：
  - view1 = attr_mask(0.2) + edge_perturb(0.15)
  - view2 = attr_mask(0.2) + node_drop(0.2)
消融可单独启用其中一种增强（--pattern mask / perturb / drop / full）。

训练后仅保存 convs 权重（prefix "convs."），微调时加载并重新初始化输出头。

用法:
    python src/contrastive_pretrain.py --convs GCN GAT GraphSAGE --pattern full --epochs 150
"""
import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import build_model
import compat_patch  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models", "pretrained")
REPORT_DIR = os.path.join(ROOT, "reports")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = 64
NUM_LAYERS = 3
HEADS = 4
PROJ_DIM = 32
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------- 增强（batch 级别） ----------------

def attr_mask(x, p=0.2):
    """按节点掩码特征（分子图节点特征为 one-hot，置零即丢失原子类型信息）"""
    keep = torch.rand(x.shape[0], 1, device=x.device) > p
    return x * keep


def edge_perturb(edge_index, batch, p=0.15):
    """按比例删边 + 在同图内随机生成等量新边"""
    n_edges = edge_index.size(1)
    keep = torch.rand(n_edges, device=edge_index.device) >= p
    ei = edge_index[:, keep]
    n_add = max(int(p * n_edges), 0)
    if n_add > 0:
        num_graphs = int(batch.max().item()) + 1
        cnt = torch.bincount(batch)
        off = torch.cat([torch.zeros(1, dtype=torch.long, device=batch.device), torch.cumsum(cnt, 0)[:-1]])
        g = torch.randint(0, num_graphs, (n_add,), device=batch.device)
        size_g = cnt[g]
        # 避免自环：u != v
        u_local = torch.rand(n_add, device=batch.device) * size_g
        v_local = torch.rand(n_add, device=batch.device) * size_g
        u = (off[g] + u_local).long()
        v = (off[g] + v_local).long()
        good = u != v
        if good.any():
            new_edges = torch.stack([u[good], v[good]])
            ei = torch.cat([ei, new_edges], dim=1)
    return ei


def node_drop(x, edge_index, batch, p=0.2):
    """删除节点并重编号（保证每图至少保留 1 个节点，避免视图间 batch 大小不一致）"""
    n = x.shape[0]
    keep = torch.rand(n, device=x.device) > p
    if keep.all():
        return x, edge_index, batch
    # 每图至少保留一个节点
    num_graphs = int(batch.max().item()) + 1
    cnt_keep = torch.zeros(num_graphs, dtype=torch.long, device=x.device)
    cnt_keep.scatter_add_(0, batch, keep.long())
    empty_graphs = torch.nonzero(cnt_keep == 0).squeeze(1)
    if empty_graphs.numel() > 0:
        for g in empty_graphs.tolist():
            idx = (batch == g).nonzero()[0].item()
            keep[idx] = True
    mapping = -torch.ones(n, dtype=torch.long, device=x.device)
    mapping[keep] = torch.arange(int(keep.sum()), device=x.device)
    x2 = x[keep]
    b2 = batch[keep]
    keep_edges = keep[edge_index[0]] & keep[edge_index[1]]
    ei2 = mapping[edge_index[:, keep_edges]]
    return x2, ei2, b2


# ---------------- 数据集与 collate ----------------

class PairAugDataset(Dataset):
    """对每个分子返回两个增强视图"""

    def __init__(self, graphs, pattern="full"):
        self.graphs = graphs
        self.pattern = pattern

    def __len__(self):
        return len(self.graphs)

    def _aug(self, d, view):
        x = d.x.float().clone()
        ei = d.edge_index.clone()
        b = d.batch.clone() if d.batch is not None else torch.zeros(d.x.shape[0], dtype=torch.long)
        if self.pattern == "mask":
            x = attr_mask(x, 0.2)
        elif self.pattern == "perturb":
            ei = edge_perturb(ei, b, 0.15)
        elif self.pattern == "drop":
            x, ei, b = node_drop(x, ei, b, 0.2)
        else:  # full
            if view == 1:
                x = attr_mask(x, 0.2)
                ei = edge_perturb(ei, b, 0.15)
            else:
                x = attr_mask(x, 0.2)
                x, ei, b = node_drop(x, ei, b, 0.2)
        out = type(d)()
        out.x = x
        out.edge_index = ei
        out.batch = b
        return out

    def __getitem__(self, i):
        d = self.graphs[i]
        return self._aug(d, 1), self._aug(d, 2)


def pair_collate(batch):
    a = [p[0] for p in batch]
    b = [p[1] for p in batch]
    return collate_list(a), collate_list(b)


def collate_list(items):
    xs, eis, bs = [], [], []
    n_off = 0
    for it in items:
        xs.append(it.x)
        eis.append(it.edge_index + n_off)
        bs.append(torch.full((it.x.shape[0],), len(bs), dtype=torch.long))
        n_off += it.x.shape[0]
    out = type(items[0])()
    out.x = torch.cat(xs, dim=0)
    out.edge_index = torch.cat(eis, dim=1)
    out.batch = torch.cat(bs, dim=0)
    return out


# ---------------- 模型 ----------------

def graph_repr(model, data):
    """图级表示：convs + global mean pooling（不经输出头）"""
    x = data.x.float()
    if x.dtype != torch.float:
        x = x.float()
    for i, conv in enumerate(model.convs):
        x = conv(x, data.edge_index)
        if i < len(model.convs) - 1:
            x = F.relu(x)
            x = F.dropout(x, p=model.dropout, training=model.training)
    return global_mean_pool_impl(x, data.batch)


def global_mean_pool_impl(x, batch):
    from torch_geometric.nn import global_mean_pool
    return global_mean_pool(x, batch)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=HIDDEN_DIM, out_dim=PROJ_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, z):
        return F.normalize(self.mlp(z), dim=-1)


def infonce_loss(z1, z2, tau=0.5):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.mm(z1, z2.t()) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    return loss


def train_one(conv_type, pool, pattern, epochs, batch_size=256, lr=1e-3):
    print(f"=== 预训练 {conv_type} (pattern={pattern}, epochs={epochs}) ===")
    model = build_model(conv_type, 9, HIDDEN_DIM, 1, NUM_LAYERS, heads=HEADS).to(DEVICE)
    proj = ProjectionHead().to(DEVICE)
    params = list(model.convs.parameters()) + list(proj.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    ds = PairAugDataset(pool, pattern=pattern)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=pair_collate,
                        num_workers=0, drop_last=True)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        proj.train()
        total_loss, total = 0.0, 0
        for b1, b2 in loader:
            b1 = b1.to(DEVICE)
            b2 = b2.to(DEVICE)
            optimizer.zero_grad()
            z1 = proj(graph_repr(model, b1))
            z2 = proj(graph_repr(model, b2))
            loss = infonce_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * z1.size(0)
            total += z1.size(0)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} loss={total_loss / max(total, 1):.4f} ({time.time() - t0:.0f}s)")
    # 保存 encoder（convs 部分）
    os.makedirs(MODEL_DIR, exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items() if k.startswith("convs.")}
    path = os.path.join(MODEL_DIR, f"{conv_type}_pretrain_{pattern}.pt")
    torch.save({"state_dict": state, "pattern": pattern, "epochs": epochs,
                "hidden": HIDDEN_DIM, "layers": NUM_LAYERS, "heads": HEADS,
                "in_dim": 9, "seed": SEED}, path)
    print(f"  已保存: {path} (convs only, {len(state)} 个参数组)")
    return path, total_loss / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", nargs="+", default=["GCN", "GAT", "GraphSAGE"])
    ap.add_argument("--pattern", default="full", choices=["mask", "perturb", "drop", "full"])
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    set_seed()
    pool_path = os.path.join(DATA_ROOT, "pretrain_pool", "pool.pt")
    pool = torch.load(pool_path, weights_only=False)
    print(f"加载预训练分子池: {len(pool)} 分子 (device={DEVICE})")
    results = {}
    for conv in args.convs:
        path, final_loss = train_one(conv, pool, args.pattern, args.epochs)
        results[conv] = {"path": path, "final_loss": round(final_loss, 4), "pattern": args.pattern}
    out_path = os.path.join(REPORT_DIR, "pretrain_run.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": {"pattern": args.pattern, "epochs": args.epochs,
                              "hidden": HIDDEN_DIM, "layers": NUM_LAYERS, "heads": HEADS,
                              "proj_dim": PROJ_DIM, "device": str(DEVICE), "pool_size": len(pool)},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"预训练完成，记录: {out_path}")


if __name__ == "__main__":
    main()
