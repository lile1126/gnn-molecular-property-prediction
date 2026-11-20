# -*- coding: utf-8 -*-
"""
LLM 化学解释增强（方向 3）
==========================
流程：SMILES + GNNExplainer 原子/键重要性掩码 → RDKit 提取关键结构特征 →
LLM 生成化学语言解释 → 官能团一致性校验。

唯一 LLM 通道：DeepSeek API（deepseek-chat，需 DEEPSEEK_API_KEY 或 deepseek_config.json）。
（本地 Qwen 4bit 通道已移除：需要 CUDA GPU，项目运行环境为纯 CPU。）

关键点：
  - 掩码→结构：top-k 重要原子 + 高重要性键 → RDKit 官能团/子结构匹配
  - 提示词注入结构事实，LLM 只做"基于证据的化学解释"，防止幻觉
  - 归因模型与 Web 推理模式一一对应：单模型 / MC / 同构集成主成员 / 异质集成 9 配置主成员

用法:
    python src/llm_explain.py --smiles "CCCO" --dataset ESOL --conv GCN
    python src/llm_explain.py --smiles "CCCO" --dataset BBBP --method hetero
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.utils import from_smiles

from data_prep import TASK_TYPE
from interpret import explain_one, load_trained

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "reports")

# DeepSeek API（OpenAI 兼容接口，标准库 urllib 调用，无额外依赖）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK_LABEL = {"ESOL": "水溶解度 logS", "FreeSolv": "水合自由能",
              "BBBP": "血脑屏障渗透", "Clintox": "药物毒性（CT_TOX）"}

# 常见官能团 SMARTS（用于结构特征提取与一致性校验）
FUNCTIONAL_GROUPS = [
    ("羟基", "[OX2H]"),
    ("羧基", "C(=O)[OX2H1]"),
    ("羰基", "[CX3]=[OX1]"),
    ("氨基", "[NX3;H2,H1;!$(NC=O)]"),
    ("酰胺", "[NX3][CX3](=[OX1])"),
    ("酯基", "[#6][CX3](=O)[OX2H0][#6]"),
    ("苯环", "c1ccccc1"),
    ("吡啶环", "c1ccncc1"),
    ("卤素(F)", "[F]"),
    ("卤素(Cl)", "[Cl]"),
    ("卤素(Br)", "[Br]"),
    ("醚键", "[OD2]([#6])[#6]"),
    ("硝基", "[NX3](=O)=O"),
    ("氰基", "[C-]#[N+]"),
    ("双键", "C=C"),
    ("三键", "C#C"),
    ("硫醚", "[#16X2H0]"),
    ("磺酰基", "S(=O)(=O)"),
]


def extract_structure(smiles, node_mask, edge_mask, data, top_k=8):
    """从掩码提取关键结构特征（RDKit）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "SMILES 无法解析"}
    node_mask = np.asarray(node_mask)
    top_idx = np.argsort(node_mask)[::-1][:min(top_k, len(node_mask))].tolist()
    # 高重要性键（边）
    edge_mask = np.asarray(edge_mask)
    top_edges = np.argsort(edge_mask)[::-1][:min(6, len(edge_mask))].tolist()
    ei = data.edge_index.numpy()
    key_bonds = []
    for e in top_edges:
        u, v = int(ei[0][e]), int(ei[1][e])
        bond = mol.GetBondBetweenAtoms(u, v)
        if bond is not None:
            key_bonds.append(f"{mol.GetAtomWithIdx(u).GetSymbol()}-{mol.GetAtomWithIdx(v).GetSymbol()}({'单' if bond.GetBondType().name == 'SINGLE' else '双' if bond.GetBondType().name == 'DOUBLE' else bond.GetBondType().name})")
    # 关键原子类型
    key_atoms = []
    for i in top_idx:
        atom = mol.GetAtomWithIdx(i)
        key_atoms.append(f"{atom.GetSymbol()}{'(芳香)' if atom.GetIsAromatic() else ''}")
    # 官能团匹配
    groups = []
    for name, smarts in FUNCTIONAL_GROUPS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            groups.append(name)
    return {
        "key_atoms": key_atoms,
        "key_bonds": key_bonds,
        "functional_groups": groups,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "smiles": smiles,
    }


def build_prompt(structure, dataset, model_note, pred, task_type, task_label):
    """基于结构证据构造提示词"""
    sys_prompt = ("你是计算化学与机器学习交叉领域的助手。请基于给定的分子结构证据，"
                  "用简洁、准确的中文解释图神经网络（GNN）为何做出该预测。"
                  "只使用提供的证据，不要编造不存在的官能团或性质；不超过 3 句话。"
                  "注意化学常识：羟基、羧基、氨基等极性基团增强水溶性（logS 更大）；"
                  "长烷基链、芳环、卤素降低水溶性。不要写出自相矛盾的化学逻辑。")
    task_prior = {
        "ESOL": "水溶解度 logS：数值越大越易溶于水。极性基团（羟基/羧基/氨基）增溶，疏水基团（烷基/芳环/卤素）减溶。",
        "FreeSolv": "水合自由能：数值越负表示溶剂化越稳定、越易溶。",
        "BBBP": "血脑屏障渗透概率：接近 1 表示预测可渗透。渗透性与脂溶性、分子量、氢键供体数相关。",
        "Clintox": "药物毒性（CT_TOX）：接近 1 表示预测有毒。毒性常与亲电/反应性基团（硝基、芳香胺、卤素取代等）及代谢活化位点相关。",
    }[dataset]
    user_prompt = (
        f"预测任务：{task_label}。归因模型：{model_note}。"
        f"预测结果：{pred:.4f}（{'数值越大性质越强' if task_type == 'classification' else '数值即目标属性'}）。\n"
        f"任务化学先验：{task_prior}\n"
        f"分子 SMILES：{structure['smiles']}（含 {structure['heavy_atoms']} 个重原子）。\n"
        f"GNNExplainer 识别的高重要性原子（按重要度排序）：{', '.join(structure['key_atoms'])}。\n"
        f"高重要性化学键：{', '.join(structure['key_bonds'])}。\n"
        f"分子含有的官能团：{', '.join(structure['functional_groups']) if structure['functional_groups'] else '未匹配到常见官能团'}。\n"
        "请结合上述证据解释该预测结果的化学依据。"
    )
    return sys_prompt, user_prompt


def check_consistency(text, structure):
    """一致性校验：LLM 文本中提及的官能团是否在 RDKit 检测集合内"""
    mentioned = []
    fake = []
    for name in structure["functional_groups"]:
        if name in text:
            mentioned.append(name)
    # 常见但未检出却被 LLM 提及的官能团
    for name, _ in FUNCTIONAL_GROUPS:
        if name not in structure["functional_groups"] and name in text:
            fake.append(name)
    return {"mentioned_detected": mentioned, "mentioned_undetected": fake}


def _mean_masks(models, datas, task_type, epochs=100):
    """对多个归因模型分别求 GNNExplainer 掩码，再按样本平均。

    models 与 datas 一一对应（不同架构/特征档的输入维度不同，需各自的 data）。
    多模型时默认 epochs=100（单模型 300）：9 配置串行归因，300 会到 30 秒量级。
    返回 (node_mask, edge_mask, pred_mean)
    """
    node_masks, edge_masks, preds = [], [], []
    for model, data in zip(models, datas):
        nm, em, p = explain_one(model, data, task_type, epochs=epochs)
        node_masks.append(np.asarray(nm, dtype=np.float64))
        edge_masks.append(np.asarray(em, dtype=np.float64))
        preds.append(float(p))
    n_len = node_masks[0].shape[0]
    node_mask = np.mean([m for m in node_masks if m.shape[0] == n_len], axis=0)
    # 同一分子各特征档的边集一致；架构不同不影响边数
    e_len = edge_masks[0].shape[0]
    same = [m for m in edge_masks if m.shape[0] == e_len]
    edge_mask = np.mean(same, axis=0) if same else edge_masks[0]
    return node_mask, edge_mask, float(np.mean(preds))


def generate_explanation(smiles, dataset="ESOL", conv="GCN", top_k=8, model=None, data=None,
                         models=None, datas=None, method="single", model_note=None,
                         prediction=None):
    """完整流程：掩码 → 结构 → LLM 解释 → 一致性校验

    model/data      : 单模型归因（method=single / mc / ensemble / attention）
    models/datas    : 多模型掩码平均（method=hetero，9 配置 seed42 主成员）
    prediction      : 界面已显示的推理结果；传入时解释文本引用它，保证与推理模式一致
    """
    name = dataset
    task_type = TASK_TYPE[name]
    if models:
        # epochs=60 与 Web 端异质集成结构图（HETERO_MASK_EPOCHS）保持一致，
        # 保证解释文本与图上高重要性原子来自同一批掩码
        node_mask, edge_mask, pred = _mean_masks(models, datas, task_type, epochs=60)
        first_data = datas[0]
    else:
        if model is None:
            model, _ = load_trained(conv, name)
        if data is None:
            data = from_smiles(smiles)
        node_mask, edge_mask, pred = explain_one(model, data, task_type)
        first_data = data
    if prediction is not None:
        pred = float(prediction)
    structure = extract_structure(smiles, node_mask, edge_mask, first_data, top_k=top_k)
    if "error" in structure:
        return {"error": structure["error"]}
    note = model_note or f"{conv} × 单模型"
    sys_prompt, user_prompt = build_prompt(structure, dataset, note, pred, task_type,
                                           TASK_LABEL[name])
    llm_text = _call_llm_deepseek(sys_prompt, user_prompt)
    consistency = check_consistency(llm_text, structure)
    return {
        "smiles": smiles,
        "prediction": round(float(pred), 4),
        "structure": structure,
        "explanation": llm_text,
        "consistency": consistency,
        "provider": "deepseek",
        "method": method,
        "model_note": note,
    }


def _load_deepseek_key():
    """读取 DeepSeek API Key：优先环境变量 DEEPSEEK_API_KEY，其次项目根 deepseek_config.json"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    cfg_path = os.path.join(ROOT, "deepseek_config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return str(json.load(f).get("api_key", "")).strip()
        except Exception:
            pass
    return ""


def _call_llm_deepseek(sys_prompt, user_prompt, max_new_tokens=160, timeout=120):
    """调用 DeepSeek API（OpenAI 兼容 /chat/completions，标准库实现，无额外依赖）"""
    import urllib.error
    import urllib.request
    api_key = _load_deepseek_key()
    if not api_key:
        raise RuntimeError(
            "未配置 DeepSeek API Key：请设置环境变量 DEEPSEEK_API_KEY，"
            "或在项目根目录创建 deepseek_config.json（{\"api_key\": \"sk-...\"}）"
        )
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.6,
        "top_p": 0.9,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        raise RuntimeError(f"DeepSeek API 请求失败（HTTP {e.code}）：{detail or e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"DeepSeek API 请求异常：{e}") from e
    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"DeepSeek API 返回格式异常：{json.dumps(body, ensure_ascii=False)[:300]}"
        ) from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", default="CCCO")
    ap.add_argument("--dataset", default="ESOL", choices=["ESOL", "FreeSolv", "BBBP", "Clintox"])
    ap.add_argument("--conv", default="GCN", choices=["GCN", "GAT", "GraphSAGE"])
    ap.add_argument("--method", default="single", choices=["single", "mc", "ensemble", "attention"],
                    help="归因模式（hetero 需经 Web 端调用，CLI 仅支持单模型归因）")
    args = ap.parse_args()
    result = generate_explanation(args.smiles, args.dataset, args.conv, method=args.method,
                                  model_note=f"{args.conv} × 单模型（{args.method}）")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
