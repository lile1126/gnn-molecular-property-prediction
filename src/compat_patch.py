# -*- coding: utf-8 -*-
"""
PyG 2.8.x + torch 2.14 兼容补丁
================================
torch 2.14 禁止 LongTensor 上的 in-place 幂运算类型提升，导致 PyG 的
gcn_norm 中 `deg.pow_(-0.5)` 报错（result type Float can't be cast to Long）。

此处复刻 gcn_norm（仅修正 deg 的 dtype 处理），并在导入时自动应用。
"""
import torch
from typing import Optional


def _patch_gcn_norm():
    try:
        import torch_geometric.nn.conv.gcn_conv as _mod
    except ImportError:
        return False

    from torch_geometric.typing import Adj, OptTensor, SparseTensor, torch_sparse
    from torch_geometric.utils import add_remaining_self_loops
    from torch_geometric.utils import add_self_loops as add_self_loops_fn
    from torch_geometric.utils import is_torch_sparse_tensor, scatter, to_edge_index
    from torch_geometric.utils.num_nodes import maybe_num_nodes
    from torch_geometric.utils.sparse import set_sparse_value

    def gcn_norm(
        edge_index: Adj,
        edge_weight: OptTensor = None,
        num_nodes: Optional[int] = None,
        improved: bool = False,
        add_self_loops: bool = True,
        flow: str = "source_to_target",
        dtype: Optional[torch.dtype] = None,
    ):
        fill_value = 2.0 if improved else 1.0

        if isinstance(edge_index, SparseTensor):
            adj_t = edge_index
            if not adj_t.has_value():
                adj_t = adj_t.fill_value(1.0, dtype=dtype)
            if add_self_loops:
                adj_t = torch_sparse.fill_diag(adj_t, fill_value)
            deg = torch_sparse.sum(adj_t, dim=1)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
            adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))
            return adj_t

        if is_torch_sparse_tensor(edge_index):
            assert edge_index.size(0) == edge_index.size(1)
            if edge_index.layout == torch.sparse_csc:
                raise NotImplementedError(
                    "Sparse CSC matrices are not yet supported in 'gcn_norm'"
                )
            adj_t = edge_index
            if add_self_loops:
                adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)
            edge_index, value = to_edge_index(adj_t)
            col, row = edge_index[0], edge_index[1]
            deg = scatter(value, col, 0, dim_size=num_nodes, reduce="sum")
            if deg.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                deg = deg.float()
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
            value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]
            return set_sparse_value(adj_t, value), None

        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if add_self_loops:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes
            )

        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce="sum")
        if deg.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            deg = deg.float()  # 修复：LongTensor 的 in-place 幂运算被 torch 2.14 拒绝
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    _mod.gcn_norm = gcn_norm
    return True


_APPLIED = _patch_gcn_norm()
