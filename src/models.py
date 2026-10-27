# -*- coding: utf-8 -*-
"""
模型定义模块
============
实现三种用于分子性质预测的图神经网络架构：
  - GCNNet   : 图卷积网络 (Kipf & Welling, 2017)
  - GATNet   : 图注意力网络 (Veličković et al., 2018)
  - SAGENet  : GraphSAGE (Hamilton et al., 2017)

共同结构：若干层图卷积 + 全局平均池化 + MLP 输出头。
回归任务输出 1 维，分类任务输出 num_classes 维。
"""
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

import compat_patch  # noqa: F401  应用 PyG/torch 兼容补丁

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool


class SAGEConvEdge(SAGEConv):
    """GraphSAGE + 键特征：邻居消息加入边特征投影。

    边特征通过 thread-local 暂存，不同请求线程之间不会共享临时状态，
    可用于 Flask 多线程推理。
    """

    def __init__(self, in_channels, out_channels, edge_dim=0, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.edge_dim = edge_dim
        self.fuse = False  # 禁用融合路径，保证 message 被调用
        if edge_dim > 0:
            src = in_channels[0] if isinstance(in_channels, tuple) else in_channels
            self.edge_lin = nn.Linear(edge_dim, src, bias=False)
        else:
            self.edge_lin = None
        self._edge_attr_local = threading.local()

    def forward(self, x, edge_index, edge_attr=None, size=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        # PyG 2.8 的 SAGEConv.propagate 不接收额外的 edge_attr 参数；这里使用
        # thread-local 暂存，避免 Flask 多线程请求之间共享临时状态。
        self._edge_attr_local.value = edge_attr
        try:
            out = self.propagate(edge_index, x=x, size=size)
        finally:
            del self._edge_attr_local.value
        out = self.lin_l(out)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def message(self, x_j):
        edge_attr = getattr(self._edge_attr_local, "value", None)
        if edge_attr is not None and self.edge_lin is not None:
            return x_j + self.edge_lin(edge_attr)
        return x_j


class BaseGNN(nn.Module):
    """GNN 公共基类：提供通用 forward 骨架（子类需实现 build_convs）。
    可选键特征（edge_dim>0）与分子指纹（fp_dim>0）为特征工程扩展，
    默认 0 时行为与旧版完全一致（旧权重兼容）。
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.3,
                 edge_dim=0, fp_dim=0):
        super().__init__()
        assert num_layers >= 1
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.fp_dim = fp_dim
        self.convs = self.build_convs()
        if edge_dim > 0:
            self.edge_lin = nn.Linear(edge_dim, 1)  # GCN 键特征加权（sigmoid 边权重）
        if fp_dim > 0:
            self.fp_lin = nn.Sequential(nn.Linear(fp_dim, hidden_dim), nn.ReLU())
        head_in = hidden_dim + (hidden_dim if fp_dim > 0 else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def build_convs(self):
        raise NotImplementedError

    def forward(self, x, edge_index, batch=None, edge_attr=None, fp=None):
        """统一前向接口：训练与 GNNExplainer 共用。
        回归输出 [N, 1]，分类输出 [N, num_classes]。
        """
        if x.dtype != torch.float:
            x = x.float()  # MoleculeNet/from_smiles 的原子特征为 int64 one-hot
        for i, conv in enumerate(self.convs):
            kw = {}
            if edge_attr is not None and self.edge_dim > 0:
                if isinstance(conv, GATConv):
                    kw["edge_attr"] = edge_attr
                elif isinstance(conv, GCNConv):
                    kw["edge_weight"] = torch.sigmoid(self.edge_lin(edge_attr)).squeeze(-1)
                elif isinstance(conv, SAGEConvEdge):
                    kw["edge_attr"] = edge_attr
            x = conv(x, edge_index, **kw)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if batch is None:
            x = x.mean(dim=0, keepdim=True)
        else:
            x = global_mean_pool(x, batch)
        if fp is not None and self.fp_dim > 0:
            x = torch.cat([x, self.fp_lin(fp)], dim=-1)
        return self.head(x)

    def forward_data(self, data):
        """兼容旧式 data 对象调用（含 edge_attr / fp 属性）"""
        return self.forward(
            data.x, data.edge_index, data.batch,
            getattr(data, "edge_attr", None), getattr(data, "fp", None),
        )


class GCNNet(BaseGNN):
    """图卷积网络"""

    def build_convs(self):
        convs = nn.ModuleList()
        for i in range(self.num_layers):
            in_ch = self.in_dim if i == 0 else self.hidden_dim
            convs.append(GCNConv(in_ch, self.hidden_dim))
        return convs


class SAGENet(BaseGNN):
    """GraphSAGE"""

    def build_convs(self):
        convs = nn.ModuleList()
        for i in range(self.num_layers):
            in_ch = self.in_dim if i == 0 else self.hidden_dim
            convs.append(SAGEConvEdge(in_ch, self.hidden_dim, edge_dim=self.edge_dim))
        return convs


class GATNet(BaseGNN):
    """图注意力网络：中间层 multi-head，最后一层单头收敛到 hidden_dim"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.3,
                 heads=4, edge_dim=0, fp_dim=0):
        self.heads = heads
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout,
                         edge_dim=edge_dim, fp_dim=fp_dim)

    def build_convs(self):
        convs = nn.ModuleList()
        for i in range(self.num_layers):
            in_ch = self.in_dim if i == 0 else self.hidden_dim * self.heads
            edim = self.edge_dim if self.edge_dim > 0 else None
            if i < self.num_layers - 1:
                convs.append(GATConv(in_ch, self.hidden_dim, heads=self.heads, dropout=self.dropout,
                                     edge_dim=edim))
            else:
                convs.append(GATConv(in_ch, self.hidden_dim, heads=1, concat=False, dropout=self.dropout,
                                     edge_dim=edim))
        return convs


MODEL_REGISTRY = {
    "GCN": GCNNet,
    "GAT": GATNet,
    "GraphSAGE": SAGENet,
}


def build_model(conv_type, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.3,
                heads=4, edge_dim=0, fp_dim=0):
    if conv_type not in MODEL_REGISTRY:
        raise ValueError(f"未知模型类型 {conv_type}，可选: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[conv_type]
    if conv_type == "GAT":
        return cls(in_dim, hidden_dim, out_dim, num_layers=num_layers, dropout=dropout,
                   heads=heads, edge_dim=edge_dim, fp_dim=fp_dim)
    return cls(in_dim, hidden_dim, out_dim, num_layers=num_layers, dropout=dropout,
               edge_dim=edge_dim, fp_dim=fp_dim)
