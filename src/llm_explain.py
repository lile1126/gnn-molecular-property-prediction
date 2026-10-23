# -*- coding: utf-8 -*-
"""
LLM 化学解释增强（方向 3）
==========================
流程：SMILES + GNNExplainer 原子/键重要性掩码 → RDKit 提取关键结构特征 →
本地 Qwen2.5-3B-Instruct 生成化学语言解释 → 官能团一致性校验。

关键点：
  - 完全本地部署（Qwen2.5-3B-Instruct，4bit 量化运行在 GTX 1660 Ti 6GB 上）
  - 掩码→结构：top-k 重要原子 + 高重要性键 → RDKit 官能团/子结构匹配
  - 提示词注入结构事实，LLM 只做"基于证据的化学解释"，防止幻觉

用法:
    python src/llm_explain.py --smiles "CCCO" --dataset ESOL --conv GCN
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
from rdkit.Chem import Draw, AllChem
from torch_geometric.utils import from_smiles

from data_prep import TASK_TYPE
from interpret import explain_one, load_trained

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-3B-Instruct", "models",
                         "Qwen--Qwen2.5-3B-Instruct", "snapshots", "master")
REPORT_DIR = os.path.join(ROOT, "reports")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def build_prompt(structure, dataset, conv, pred, task_type, task_label):
    """基于结构证据构造提示词（Qwen chat 模板）"""
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
        f"预测任务：{task_label}。模型：{conv} 图神经网络。"
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


def generate_explanation(smiles, dataset="ESOL", conv="GCN", top_k=8, model=None, data=None):
    """完整流程：掩码 → 结构 → LLM 解释 → 一致性校验

    model/data 可外部传入（复用 Web 已加载的模型与特征数据，避免显存重复占用）；
    未传时自动加载基线模型。
    """
    name = dataset
    task_type = TASK_TYPE[name]
    if model is None:
        model, _ = load_trained(conv, name)
    if data is None:
        data = from_smiles(smiles)
    node_mask, edge_mask, pred = explain_one(model, data, task_type)
    structure = extract_structure(smiles, node_mask, edge_mask, data, top_k=top_k)
    if "error" in structure:
        return {"error": structure["error"]}
    sys_prompt, user_prompt = build_prompt(structure, dataset, conv, pred, task_type,
                                           {"ESOL": "水溶解度 logS", "FreeSolv": "水合自由能",
                                            "BBBP": "血脑屏障渗透", "Clintox": "药物毒性（CT_TOX）"}[name])
    llm_text = _call_llm(sys_prompt, user_prompt)
    consistency = check_consistency(llm_text, structure)
    return {
        "smiles": smiles,
        "prediction": round(float(pred), 4),
        "structure": structure,
        "explanation": llm_text,
        "consistency": consistency,
    }


_LLM = {"pipe": None, "tokenizer": None}


def _call_llm(sys_prompt, user_prompt, max_new_tokens=160):
    """调用本地 Qwen2.5-3B（懒加载 + 4bit 量化）"""
    if _LLM["pipe"] is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.float16, trust_remote_code=True)
        _LLM["pipe"] = mdl
        _LLM["tokenizer"] = tok
    mdl = _LLM["pipe"]
    tok = _LLM["tokenizer"]
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                           temperature=0.6, top_p=0.9, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", default="CCCO")
    ap.add_argument("--dataset", default="ESOL", choices=["ESOL", "FreeSolv", "BBBP"])
    ap.add_argument("--conv", default="GCN", choices=["GCN", "GAT", "GraphSAGE"])
    args = ap.parse_args()
    result = generate_explanation(args.smiles, args.dataset, args.conv)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
