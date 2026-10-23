# -*- coding: utf-8 -*-
"""
Web 演示服务
============
输入 SMILES 分子式 → 选择数据集与模型 → 返回预测值 + 分子结构图（原子/键重要性着色）。

用法:
    python src/app.py     # 启动后访问 http://127.0.0.1:5000
"""
import base64
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from torch_geometric.utils import from_smiles

from data_prep import TASK_TYPE
from interpret import draw_molecule, explain_one, load_trained

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENSEMBLE_DIR = os.path.join(ROOT, "models", "ensemble")

CONVS = ["GCN", "GAT", "GraphSAGE"]
DATASETS = ["ESOL", "FreeSolv", "BBBP", "Clintox"]
FEAT_LEVELS = ["base", "a", "b", "c"]
MC_T = 30

app = Flask(__name__)
MODELS = {}
ENSEMBLES = {}
FEAT_MODELS = {}

FEAT_LABEL = {"base": "基线特征（9 维）", "a": "feat_a 原子增强（27 维）",
              "b": "feat_b +键特征（+7 维）", "c": "feat_c +MACCS 指纹（+166 位）"}

TASK_LABEL = {"ESOL": "水溶解度 logS（回归）", "FreeSolv": "水合自由能（回归）",
              "BBBP": "血脑屏障渗透（二分类）", "Clintox": "临床毒性 CT_TOX（二分类）"}
UNIT_LABEL = {"ESOL": "logS", "FreeSolv": "kcal/mol", "BBBP": "P(渗透) 概率", "Clintox": "P(毒性) 概率"}


def load_all():
    for conv in CONVS:
        for name in DATASETS:
            model, _ = load_trained(conv, name)
            MODELS[(conv, name)] = model
    print("已加载模型:", list(MODELS.keys()))


def _load_ensemble_members(conv, name, level="base"):
    """懒加载深度集成成员（同架构多 seed，含 seed=42 基线模型）。
    level=base: models/{conv}_{name}.pt + models/ensemble/{conv}_{name}_s*.pt
    level=a/b/c: models/feat/{conv}_{name}_feat{level}.pt + models/feat_ensemble/{conv}_{name}_feat{level}_s*.pt"""
    key = (conv, name, level)
    if key in ENSEMBLES:
        return ENSEMBLES[key]
    from models import build_model
    members = []
    if level == "base":
        base = os.path.join(ROOT, "models", f"{conv}_{name}.pt")
        candidates = [base] if os.path.exists(base) else []
        if os.path.isdir(ENSEMBLE_DIR):
            candidates += [os.path.join(ENSEMBLE_DIR, p) for p in sorted(os.listdir(ENSEMBLE_DIR))
                           if p.startswith(f"{conv}_{name}_s") and p.endswith(".pt")]
    else:
        base = os.path.join(ROOT, "models", "feat", f"{conv}_{name}_feat{level}.pt")
        candidates = [base] if os.path.exists(base) else []
        fe_dir = os.path.join(ROOT, "models", "feat_ensemble")
        if os.path.isdir(fe_dir):
            candidates += [os.path.join(fe_dir, p) for p in sorted(os.listdir(fe_dir))
                           if p.startswith(f"{conv}_{name}_feat{level}_s") and p.endswith(".pt")]
    for path in candidates:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        out_dim = 2 if TASK_TYPE[name] == "classification" else 1
        m = build_model(conv, ckpt["in_dim"], ckpt["hidden"], out_dim, ckpt["layers"],
                        heads=ckpt.get("heads", 4), edge_dim=ckpt.get("edge_dim", 0),
                        fp_dim=ckpt.get("fp_dim", 0)).to(DEVICE)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        members.append(m)
    ENSEMBLES[key] = members
    return members


def _load_feat_model(conv, name, level):
    """懒加载特征工程模型（选题深挖 · 方向1：feat_a/b/c）"""
    key = (conv, name, level)
    if key in FEAT_MODELS:
        return FEAT_MODELS[key]
    from models import build_model
    path = os.path.join(ROOT, "models", "feat", f"{conv}_{name}_feat{level}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"该组合未训练（实验矩阵：GCN×{{a,b,c}}、GAT/GraphSAGE×c，{conv}×{name}×feat{level} 不在其中）")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m = build_model(conv, ckpt["in_dim"], ckpt["hidden"], ckpt["out_dim"], ckpt["layers"],
                    heads=ckpt.get("heads", 4), edge_dim=ckpt.get("edge_dim", 0),
                    fp_dim=ckpt.get("fp_dim", 0)).to(DEVICE)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    FEAT_MODELS[key] = m
    return m


def _build_data(smiles, level):
    """按特征等级构建图数据：base 用 PyG from_smiles（9 维），feat_a/b/c 用 RDKit 特征。"""
    if level == "base":
        return from_smiles(smiles)
    from featurize import featurize_smiles
    return featurize_smiles(smiles, level)


def _feat_info(level):
    return {"feat": level, "feat_label": FEAT_LABEL[level]}


# ---------------------------------------------------------------------------
# 交互逻辑优化：/api/meta（组合可用性矩阵）+ /api/validate（SMILES 即时校验）
# ---------------------------------------------------------------------------
def _scan_availability():
    """扫描 models 目录，返回真实可用的 (conv, dataset, level) 组合与集成可用集合。
    ensemble_ok 为三元组：feat 组合在 feat_ensemble 目录存在成员时也支持深度集成。"""
    trained = set()
    for conv in CONVS:
        for name in DATASETS:
            base = os.path.join(ROOT, "models", f"{conv}_{name}.pt")
            if os.path.exists(base):
                trained.add((conv, name, "base"))
            for lv in FEAT_LEVELS[1:]:
                if os.path.exists(os.path.join(ROOT, "models", "feat", f"{conv}_{name}_feat{lv}.pt")):
                    trained.add((conv, name, lv))
    ensemble_ok = set()
    fe_dir = os.path.join(ROOT, "models", "feat_ensemble")
    for conv in CONVS:
        for name in DATASETS:
            # 基线集成：基线模型 + ensemble/ 下 s 成员
            cnt = 0
            if os.path.exists(os.path.join(ROOT, "models", f"{conv}_{name}.pt")):
                cnt += 1
            if os.path.isdir(ENSEMBLE_DIR):
                cnt += sum(1 for p in os.listdir(ENSEMBLE_DIR)
                           if p.startswith(f"{conv}_{name}_s") and p.endswith(".pt"))
            if cnt >= 2:
                ensemble_ok.add((conv, name, "base"))
            # 特征集成：seed42 特征模型 + feat_ensemble/ 下 s 成员
            for lv in FEAT_LEVELS[1:]:
                cnt2 = 0
                if os.path.exists(os.path.join(ROOT, "models", "feat", f"{conv}_{name}_feat{lv}.pt")):
                    cnt2 += 1
                if os.path.isdir(fe_dir):
                    cnt2 += sum(1 for p in os.listdir(fe_dir)
                                if p.startswith(f"{conv}_{name}_feat{lv}_s") and p.endswith(".pt"))
                if cnt2 >= 2:
                    ensemble_ok.add((conv, name, lv))
    return trained, ensemble_ok


TRAINED, ENSEMBLE_OK = _scan_availability()


@app.route("/api/meta", methods=["GET"])
def api_meta():
    """返回模型可用性矩阵，前端据此联动禁用/灰化不可用选项，避免 400。"""
    trained = sorted(f"{c}|{d}|{lv}" for c, d, lv in TRAINED)
    ens = sorted(f"{c}|{d}|{lv}" for c, d, lv in ENSEMBLE_OK)
    return jsonify({
        "convs": CONVS, "datasets": DATASETS, "feat_levels": FEAT_LEVELS,
        "feat_labels": FEAT_LABEL, "task_labels": TASK_LABEL, "unit_labels": UNIT_LABEL,
        "trained": trained, "ensemble_ok": ens,
        "mc_t": MC_T, "ensemble_n": 4,
    })


@app.route("/api/validate", methods=["POST"])
def api_validate():
    """SMILES 合法性即时校验（前端防抖调用，红色边框/绿色边框提示）。"""
    body = request.get_json(force=True) or {}
    smiles = (body.get("smiles") or "").strip()
    if not smiles:
        return jsonify({"ok": False, "error": "SMILES 为空"})
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({"ok": False, "error": "RDKit 无法解析该 SMILES（语法/价键错误）"})
        canon = Chem.MolToSmiles(mol)
        return jsonify({"ok": True, "canonical": canon})
    except Exception as e:
        return jsonify({"ok": False, "error": f"解析异常: {e}"})


@torch.no_grad()
def mc_predict_single(model, data, task_type, T=MC_T):
    """MC-Dropout 单分子采样：返回 (mean, std) 或 (prob_mean, entropy)"""
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    data = data.to(DEVICE)
    x = data.x.float()
    batch = data.batch if data.batch is not None else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    edge_attr = getattr(data, "edge_attr", None)
    fp = getattr(data, "fp", None)
    preds = []
    for _ in range(T):
        out = model(x, data.edge_index, batch, edge_attr=edge_attr, fp=fp)
        if task_type == "classification":
            preds.append(F.softmax(out, dim=1)[:, 1].item())
        else:
            preds.append(out.squeeze(-1).item())
    arr = np.array(preds)
    if task_type == "classification":
        p = float(arr.mean())
        ent = float(-(p * np.log(max(p, 1e-12)) + (1 - p) * np.log(max(1 - p, 1e-12))))
        return p, ent
    return float(arr.mean()), float(arr.std())


@torch.no_grad()
def ensemble_predict_single(models, data, task_type):
    """深度集成单分子预测：返回 (mean, std) 或 (prob_mean, entropy)"""
    data = data.to(DEVICE)
    x = data.x.float()
    batch = data.batch if data.batch is not None else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    edge_attr = getattr(data, "edge_attr", None)
    fp = getattr(data, "fp", None)
    outs = []
    for m in models:
        m.eval()
        out = m(x, data.edge_index, batch, edge_attr=edge_attr, fp=fp).cpu()
        if task_type == "classification":
            outs.append(F.softmax(out, dim=1)[:, 1].item())
        else:
            outs.append(out.squeeze(-1).item())
    arr = np.array(outs)
    if task_type == "classification":
        p = float(arr.mean())
        ent = float(-(p * np.log(max(p, 1e-12)) + (1 - p) * np.log(max(1 - p, 1e-12))))
        return p, ent
    return float(arr.mean()), float(arr.std())


@app.route("/")
def index():
    return PAGE_HTML


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True) or {}
    smiles = (body.get("smiles") or "").strip()
    conv = body.get("conv", "GCN")
    name = body.get("dataset", "ESOL")
    method = body.get("method", "single")
    level = body.get("feat", "base")
    if conv not in CONVS or name not in DATASETS:
        return jsonify({"error": "无效的模型参数"}), 400
    if level not in FEAT_LEVELS:
        return jsonify({"error": "无效的特征等级参数"}), 400
    if method not in ("single", "mc", "ensemble"):
        return jsonify({"error": "无效的方法参数"}), 400
    if not smiles:
        return jsonify({"error": "SMILES 为空"}), 400
    try:
        data = _build_data(smiles, level)
    except Exception as e:
        return jsonify({"error": f"SMILES 解析失败: {e}"}), 400
    task_type = TASK_TYPE[name]
    explain = bool(body.get("explain", True))
    if level != "base":
        try:
            model = _load_feat_model(conv, name, level)
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 400
    else:
        model = MODELS[(conv, name)]
    node_mask = edge_mask = pred = None
    if explain:
        try:
            node_mask, edge_mask, pred = explain_one(model, data, task_type)
        except Exception as e:
            return jsonify({"error": f"解释计算失败: {e}"}), 500
    resp = {
        "task_type": task_type,
        "task_label": TASK_LABEL[name],
        "smiles": smiles,
        "unit": UNIT_LABEL[name],
    }
    resp.update(_feat_info(level))
    if node_mask is not None:
        resp["node_mask"] = [round(float(v), 4) for v in node_mask]
        resp["edge_mask"] = [round(float(v), 4) for v in edge_mask]
    if method == "mc":
        mean, unc = mc_predict_single(model, data, task_type)
        resp["method"] = "MC-Dropout"
        resp["mc_samples"] = MC_T
    elif method == "ensemble":
        members = _load_ensemble_members(conv, name, level)
        if len(members) < 2:
            return jsonify({"error": "深度集成成员不足（需要至少 2 个 seed 模型，请先运行 train_ensemble.py / train_feat_ensemble.py）"}), 400
        mean, unc = ensemble_predict_single(members, data, task_type)
        resp["method"] = f"深度集成（{len(members)} 成员）"
        resp["n_members"] = len(members)
    else:
        resp["method"] = "单模型"
        if pred is None:
            with torch.no_grad():
                x = data.x.to(DEVICE).float()
                out = model(x, data.edge_index.to(DEVICE),
                            data.batch.to(DEVICE) if data.batch is not None else None,
                            edge_attr=getattr(data, "edge_attr", None).to(DEVICE)
                            if getattr(data, "edge_attr", None) is not None else None,
                            fp=getattr(data, "fp", None).to(DEVICE)
                            if getattr(data, "fp", None) is not None else None)
            pred = float(out.squeeze(-1).item()) if task_type == "regression" \
                else float(torch.softmax(out, dim=-1)[0, 1].item())
        resp["prediction"] = round(float(pred), 4)
        if node_mask is not None:
            img_b64 = _render_image(smiles, conv, name, data, node_mask, edge_mask)
            resp["image"] = img_b64
        return jsonify(resp)

    # mc / ensemble：给出均值 + 不确定性
    if task_type == "regression":
        resp["prediction"] = round(float(mean), 4)
        resp["std"] = round(float(unc), 4)
        resp["ci90"] = [round(float(mean - 1.645 * unc), 4), round(float(mean + 1.645 * unc), 4)]
    else:
        resp["prediction"] = round(float(mean), 4)
        resp["entropy"] = round(float(unc), 4)
    if node_mask is not None:
        img_b64 = _render_image(smiles, conv, name, data, node_mask, edge_mask)
        resp["image"] = img_b64
    return jsonify(resp)


@app.route("/api/attention", methods=["POST"])
def attention_viz():
    """GAT 注意力机制可视化：提取第一层多头注意力系数，在分子图上按键权重着色。
    per_head=true 时额外返回 4 张单头图 + 1 张 2x2 逐头对比图（选题深挖 · 方向3）。"""
    from attention_viz import draw_per_head, extract_attention, draw_attention_molecule
    body = request.get_json(force=True) or {}
    smiles = (body.get("smiles") or "").strip()
    name = body.get("dataset", "BBBP")
    layer = int(body.get("layer", 0))
    per_head = bool(body.get("per_head", False))
    if name not in DATASETS:
        return jsonify({"error": "无效的数据集参数"}), 400
    if not smiles:
        return jsonify({"error": "SMILES 为空"}), 400
    if ("GAT", name) not in MODELS:
        return jsonify({"error": "GAT 模型未加载，请稍候重试"}), 500
    try:
        data = from_smiles(smiles)
    except Exception as e:
        return jsonify({"error": f"SMILES 解析失败: {e}"}), 400
    model = MODELS[("GAT", name)]
    task_type = TASK_TYPE[name]
    try:
        alphas, edge_indices = extract_attention(model, data, layer=layer)
        alpha = alphas[layer]
        ei = edge_indices[layer]
        png = draw_attention_molecule(smiles, ei, alpha, title="", as_buffer=True)
        b64 = base64.b64encode(png).decode("ascii")
        heads_imgs = []
        combo_b64 = None
        if per_head and alpha.ndim > 1:
            for h in range(alpha.shape[1]):
                hp = draw_attention_molecule(smiles, ei, alpha[:, h], title=f"Head {h + 1}", as_buffer=True)
                if hp:
                    heads_imgs.append(base64.b64encode(hp).decode("ascii"))
            cp = draw_per_head(smiles, ei, alpha, title=f"{name} 逐头注意力对比", as_buffer=True)
            if cp:
                combo_b64 = base64.b64encode(cp).decode("ascii")
        with torch.no_grad():
            x = data.x.to(DEVICE).float()
            out = model(x, data.edge_index.to(DEVICE),
                        data.batch.to(DEVICE) if data.batch is not None else None)
        if task_type == "regression":
            pred = round(float(out.squeeze(-1).item()), 4)
        else:
            pred = round(float(torch.softmax(out, dim=-1)[0, 1].item()), 4)
        a1 = alpha.numpy()
        if a1.ndim > 1:
            a1 = a1.mean(axis=1)
        top = []
        for e in np.argsort(-a1)[:8]:
            i0, j0 = int(ei[0, e]), int(ei[1, e])
            if i0 == j0:
                continue
            top.append({"edge": [i0, j0], "alpha": round(float(a1[e]), 4)})
        return jsonify({
            "image": b64, "prediction": pred, "task_label": TASK_LABEL[name],
            "layer": layer, "heads": int(alpha.shape[1]) if alpha.ndim > 1 else 1,
            "n_edges": int(ei.shape[1]), "top8_edges": top,
            "per_head_images": heads_imgs, "combo_image": combo_b64,
            "method": f"GAT 注意力可视化（layer {layer}，多头平均" + (" + 逐头对比" if per_head else "") + "）",
        })
    except Exception as e:
        return jsonify({"error": f"注意力计算失败: {e}"}), 500


@app.route("/api/explain", methods=["POST"])
def explain_llm():
    """LLM 化学解释：GNNExplainer 掩码 → 结构证据 → 本地 Qwen2.5-3B 生成解释"""
    body = request.get_json(force=True) or {}
    smiles = (body.get("smiles") or "").strip()
    conv = body.get("conv", "GCN")
    name = body.get("dataset", "ESOL")
    if conv not in CONVS or name not in DATASETS:
        return jsonify({"error": "无效的模型参数"}), 400
    if not smiles:
        return jsonify({"error": "SMILES 为空"}), 400
    try:
        from llm_explain import generate_explanation
        # 复用 Web 已加载的基线模型与特征数据（避免重复加载模型导致显存超限）
        model = MODELS.get((conv, name))
        if model is None:
            from interpret import load_trained
            model, _ = load_trained(conv, name)
        data = _build_data(smiles, "base")
        result = generate_explanation(smiles, dataset=name, conv=conv, model=model, data=data)
    except Exception as e:
        return jsonify({"error": f"LLM 解释失败: {e}"}), 500
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify({
        "explanation": result["explanation"],
        "prediction": result["prediction"],
        "functional_groups": result["structure"]["functional_groups"],
        "key_atoms": result["structure"]["key_atoms"],
        "consistency": result["consistency"],
        "method": "本地 Qwen2.5-3B-Instruct (4bit)",
    })


def _render_image(smiles, conv, name, data, node_mask, edge_mask):
    img_b64 = None
    try:
        tmp = os.path.join(ROOT, "reports", "interpret", f"_web_{conv}_{name}.png")
        ok = draw_molecule(smiles, node_mask, edge_mask, data.edge_index.cpu().numpy(), tmp, title=f"{conv} | {name}")
        if ok and os.path.exists(tmp):
            with open(tmp, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        print("绘图失败:", e)
    return img_b64


PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>GNN 分子性质预测</title>
<style>
  body { font-family: 'Microsoft YaHei', system-ui, sans-serif; margin: 0; background: #f4f6fa; color: #222; }
  .wrap { max-width: 960px; margin: 0 auto; padding: 28px 20px 40px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color: #666; font-size: 13px; margin-bottom: 22px; }
  .card { background: #fff; border-radius: 10px; padding: 18px 20px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 18px; }
  label { font-size: 13px; color: #444; display: block; margin: 10px 0 4px; }
  input[type=text], select { box-sizing: border-box; padding: 9px 10px; font-size: 14px; border: 1px solid #ccd3dd; border-radius: 6px; }
  input[type=text] { width: 100%; }
  .inp-wrap { position: relative; }
  .inp-wrap.ok input { border-color: #2e9e5b; }
  .inp-wrap.bad input { border-color: #d33; }
  #smilesStatus { display: inline-block; margin-left: 8px; font-size: 12px; color: #999; }
  .ok-msg { color: #2e9e5b; } .bad-msg { color: #d33; }
  select { width: 100%; }
  .row { display: flex; gap: 4%; }
  .btn { margin-top: 14px; background: #2e6be6; color: #fff; border: 0; padding: 10px 26px; font-size: 14px; border-radius: 6px; cursor: pointer; }
  .btn:hover { background: #245ac4; }
  .btn:disabled { background: #9db4e0; cursor: wait; }
  .result { margin-top: 16px; display: none; }
  .pred { font-size: 26px; font-weight: 700; color: #2e6be6; }
  .imgbox img { max-width: 100%; border: 1px solid #e2e7ee; border-radius: 8px; }
  .err { color: #d33; margin-top: 12px; font-size: 14px; white-space: pre-wrap; }
  .hint { font-size: 12px; color: #999; margin-top: 6px; }
  .legend { font-size: 12px; color: #777; margin-top: 8px; }
  .sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -1px; margin: 0 4px 0 12px; }
  .chip { display:inline-block; margin:8px 6px 0 0; padding:5px 12px; font-size:12px; color:#2e6be6; background:#eef3fd; border:1px solid #d5e2fa; border-radius:14px; cursor:pointer; }
  .chip:hover { background:#dfeafc; }
  .opt { display:flex; align-items:center; gap:6px; margin:0; font-weight:400; font-size:13px; cursor:pointer; padding:6px 10px; border:1px solid #e2e7ee; border-radius:8px; }
  .opt.on { border-color:#2e6be6; background:#eef3fd; }
  .opt input { margin:0; }
  .hint2 { font-size:12px; color:#999; margin:8px 0 0; }
  .sw-on { color:#2e6be6; }
  .result-card { margin-top:14px; border:1px solid #e2e7ee; border-radius:10px; padding:14px 16px; background:#fbfcfe; display:none; }
  .kpi { font-size:13px; color:#555; margin:4px 0; }
  .kpi b { color:#222; }
  .tag { display:inline-block; padding:2px 10px; border-radius:10px; font-size:12px; background:#eef3fd; color:#2e6be6; margin-right:8px; }
  .llm-box { margin-top:14px; border-left:3px solid #6a4fc4; border-radius:6px; padding:10px 14px; background:#faf9fd; display:none; font-size:13px; color:#444; line-height:1.7; }
  .btn-w { background:#6a4fc4; }
  .btn-w:hover { background:#5a3fb0; }
  .btn-w:disabled { background:#b3a7d8; }
  /* 交互逻辑优化新增 */
  .req-hint { margin-top:10px; padding:8px 12px; font-size:12px; border-radius:6px; display:none; }
  .req-hint.warn { background:#fff7e8; color:#9a6b00; border:1px solid #f0d9a8; }
  .req-hint.err { background:#fdf0f0; color:#b3261e; border:1px solid #f0c8c4; }
  .req-hint.ok { background:#eef7f1; color:#1e7a45; border:1px solid #cfe8d8; }
  select:disabled { background:#f1f3f6; color:#9aa4b2; }
  .hist-card .ht { font-size:13px; font-weight:700; margin:0 0 8px; color:#444; }
  .htable { width:100%; border-collapse:collapse; font-size:12px; }
  .htable th, .htable td { border-bottom:1px solid #e6ebf2; padding:6px 8px; text-align:left; }
  .htable th { color:#888; font-weight:600; background:#f7f9fc; }
  .htable td .play { color:#2e6be6; cursor:pointer; text-decoration:none; }
  .htable td .play:hover { text-decoration:underline; }
  .hist-empty { font-size:12px; color:#aaa; padding:8px 0; }
  .hbox { max-height:220px; overflow:auto; }
  .btn-s { margin-top:8px; padding:4px 12px; font-size:12px; background:#fff; color:#555; border:1px solid #ccd3dd; border-radius:6px; cursor:pointer; }
  .btn-s:hover { background:#f4f6fa; }
</style>
</head>
<body>
<div class="wrap">
  <h1>GNN 分子性质预测</h1>
  <div class="sub">基于图神经网络（GCN / GAT / GraphSAGE）的分子性质预测与可解释性演示 · 工程实践课题</div>
  <div class="card">
    <label>SMILES 分子式 <span id="smilesStatus"></span></label>
    <div class="inp-wrap" id="inpWrap">
      <input type="text" id="smiles" placeholder="例如：CCCO（正丙醇）、c1ccccc1（苯）、CC(=O)Oc1ccccc1C(=O)O（阿司匹林）" value="CCCO" oninput="debValidate()" onkeydown="onEnter(event)">
    </div>
    <div>
      <span class="chip" onclick="setSmiles('CCCO')">正丙醇</span>
      <span class="chip" onclick="setSmiles('c1ccccc1')">苯</span>
      <span class="chip" onclick="setSmiles('CC(=O)Oc1ccccc1C(=O)O')">阿司匹林</span>
      <span class="chip" onclick="setSmiles('C(C(F)(F)F)Cl')">氟烷</span>
      <span class="chip" onclick="setSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')">咖啡因</span>
      <span class="chip" onclick="setSmiles('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O')">布洛芬</span>
    </div>
    <div class="row" style="margin-top:8px;">
      <div style="flex:1;">
        <label>预测任务</label>
        <select id="dataset" onchange="refreshUI()">
          <option value="ESOL">ESOL 水溶解度（回归）</option>
          <option value="FreeSolv">FreeSolv 水合自由能（回归）</option>
          <option value="BBBP">BBBP 血脑屏障渗透（分类）</option>
          <option value="Clintox">Clintox 临床毒性（分类）</option>
        </select>
      </div>
      <div style="flex:1;">
        <label>模型架构 <span id="convNote" style="display:none;color:#0f766e;">（注意力仅基于 GAT）</span></label>
        <select id="conv" onchange="refreshUI()">
          <option value="GCN">GCN</option>
          <option value="GAT">GAT</option>
          <option value="GraphSAGE">GraphSAGE</option>
        </select>
      </div>
      <div style="flex:1;">
        <label>特征工程等级 <span id="featNote" style="display:none;color:#0f766e;"></span></label>
        <select id="feat" onchange="featChanged()">
          <option value="base">基线特征（9 维）</option>
          <option value="a">feat_a 原子增强（27 维）</option>
          <option value="b">feat_b +键特征（+7 维）</option>
          <option value="c">feat_c +MACCS 指纹（+166 位）</option>
        </select>
      </div>
    </div>
    <label>分析模式</label>
    <div class="row" style="gap:10px;margin-top:4px;">
      <label class="opt on" style="flex:1;justify-content:center;"><input type="radio" name="method" value="single" checked> 单模型</label>
      <label class="opt" style="flex:1;justify-content:center;"><input type="radio" name="method" value="mc"> MC-Dropout</label>
      <label class="opt" style="flex:1;justify-content:center;"><input type="radio" name="method" value="ensemble"> 深度集成</label>
      <label class="opt" style="flex:1;justify-content:center;"><input type="radio" name="method" value="attention"> 注意力可视化</label>
    </div>
    <div class="hint2" id="methodHint">单模型直接推理，输出预测值与 GNNExplainer 分子解释图。</div>
    <label style="display:flex;align-items:center;gap:8px;margin:12px 0 0;font-weight:400;cursor:pointer;">
      <input type="checkbox" id="withExplain" checked> 生成分子解释图（GNNExplainer，较慢）
    </label>
    <div id="reqHint" class="req-hint"></div>
    <div style="margin-top:14px;">
      <button class="btn" id="btn" onclick="run()">运行分析</button>
      <button class="btn btn-w" id="btn2" onclick="runLlm()">LLM 化学解释</button>
    </div>
    <div class="err" id="err"></div>
  </div>

  <div class="result-card" id="result">
    <div><span class="tag" id="tag">方法</span><span class="kpi">任务：<b id="kTask"></b>｜SMILES：<b id="kSmiles"></b></span></div>
    <div class="kpi">预测结果：<span class="pred" id="pred"></span> <span id="unit" style="font-size:14px;color:#555;"></span></div>
    <div class="unc" id="unc" style="margin-top:6px;font-size:14px;color:#555;"></div>
    <div class="imgbox" id="imgbox" style="display:none;"><img id="img" alt="分子图"></div>
    <div id="perHeadBox" style="display:none;margin-top:12px;display:none;"></div>
    <div class="legend" id="legendGnn" style="display:none;">
      <span class="sw" style="background:#8f8fd9;"></span>原子重要性低
      <span class="sw" style="background:#e04040;"></span>原子重要性高（红色越深越关键）
    </div>
    <div class="legend" id="legendAttn" style="display:none;">
      <span class="sw" style="background:#3a5bd0;"></span>注意力低
      <span class="sw" style="background:#d63838;"></span>注意力高（键越红模型越关注）
    </div>
    <div class="hint" id="hint"></div>
  </div>

  <div class="llm-box" id="llmBox">
    <div class="kpi" style="margin:0 0 6px;"><b>LLM 化学解释（本地 Qwen2.5-3B）</b></div>
    <div id="llmText"></div>
  </div>

  <div class="card hist-card">
    <div class="ht">预测历史（对比多分子 / 多配置）<span style="font-weight:400;color:#999;font-size:12px;">点击行回放该配置</span></div>
    <div class="hbox">
      <table class="htable" id="histTable">
        <thead><tr><th>#</th><th>SMILES</th><th>任务</th><th>架构</th><th>特征</th><th>方法</th><th>预测值</th><th>不确定性</th><th></th></tr></thead>
        <tbody id="histBody"></tbody>
      </table>
      <div class="hist-empty" id="histEmpty">暂无预测记录——运行一次分析后自动加入。</div>
    </div>
    <button class="btn-s" onclick="clearHistory()">清空历史</button>
  </div>
</div>
<script>
var META = null;
var HISTORY = [];
var VALIDATED = null;   // 最近一次校验的 SMILES -> {ok, canonical}
var _vTimer = null;

function setSmiles(s) {
  document.getElementById('smiles').value = s;
  document.getElementById('inpWrap').className = 'inp-wrap';
  document.getElementById('smilesStatus').textContent = '';
  debValidate();
}
function sel(el, on) { if (on) { el.classList.add('on'); } else { el.classList.remove('on'); } }
function curMethod() { return document.querySelector('input[name="method"]:checked').value; }
function curFeat() { return document.getElementById('feat').value; }
function onEnter(e) { if (e.key === 'Enter') { e.preventDefault(); run(); } }

/* ---------- 组合可用性联动（/api/meta） ---------- */
function loadMeta() {
  fetch('/api/meta').then(function (r) { return r.json(); }).then(function (d) {
    META = d;
    refreshUI();
  }).catch(function () { META = null; });
}
function trainedKey(c, d, lv) { return c + '|' + d + '|' + lv; }
function isTrained(c, d, lv) { return META && META.trained.indexOf(trainedKey(c, d, lv)) >= 0; }
function isEnsembleOk(c, d, lv) { return META && META.ensemble_ok.indexOf(c + '|' + d + '|' + lv) >= 0; }
function featOpts() {
  var conv = document.getElementById('conv').value;
  var ds = document.getElementById('dataset').value;
  var out = [];
  ['base', 'a', 'b', 'c'].forEach(function (lv) {
    out.push({ lv: lv, ok: isTrained(conv, ds, lv) });
  });
  return out;
}
function refreshUI() {
  var m = curMethod();
  var conv = document.getElementById('conv').value;
  var ds = document.getElementById('dataset').value;
  var featSel = document.getElementById('feat');
  var note = document.getElementById('featNote');
  var convNote = document.getElementById('convNote');
  var hint = document.getElementById('methodHint');
  var withExp = document.getElementById('withExplain');

  /* 1) feat 选项按真实训练矩阵灰化 */
  var opts = featOpts();
  Array.prototype.forEach.call(featSel.options, function (o) {
    var lv = o.value;
    var item = opts.filter(function (x) { return x.lv === lv; })[0];
    o.disabled = !item.ok;
    if (o.disabled && o.selected) { o.selected = false; }
  });
  /* 当前 feat 不可用 → 切到最近的可用档位 */
  if (curFeat() === '' || !isTrained(conv, ds, curFeat())) {
    var fallback = 'base';
    for (var i = 0; i < opts.length; i++) { if (opts[i].ok) { fallback = opts[i].lv; break; } }
    featSel.value = fallback;
  }

  /* 2) 方法联动 */
  if (m === 'attention') {
    document.getElementById('conv').disabled = true;
    convNote.style.display = 'inline';
    featSel.disabled = true;
    hint.textContent = '提取 GAT 多头注意力系数，按键权重着色；仅基于 GAT 模型与基线特征。';
    withExp.closest('label').style.display = 'none';
    reqHintOk('注意力可视化仅支持 GAT × 基线特征。');
  } else {
    document.getElementById('conv').disabled = false;
    convNote.style.display = 'none';
    featSel.disabled = false;
    withExp.closest('label').style.display = '';
    var fm = curFeat();
    if (fm !== 'base') {
      note.style.display = 'inline';
      note.textContent = '特征模型支持单模型 / MC-Dropout / 深度集成（解释图基于基线）';
    } else {
      note.style.display = 'none';
    }
    if (m === 'mc') { hint.textContent = 'MC-Dropout 蒙特卡洛采样 T=' + (META ? META.mc_t : 30) + ' 次，输出 90% 置信区间与标准差 σ。'; }
    else if (m === 'ensemble') { hint.textContent = isEnsembleOk(conv, ds, fm) ? '同架构多 seed（' + (META ? META.ensemble_n : 4) + ' 成员）预测取平均，输出成员分歧（σ / 熵）作为不确定性。' : '该组合集成成员未训练，请先运行 train_ensemble.py / train_feat_ensemble.py。'; }
    else { hint.textContent = '单模型直接推理，输出预测值与 GNNExplainer 分子解释图。'; }
    /* 3) 组合可用性总提示 */
    if (!isTrained(conv, ds, 'base')) {
      reqHintErr('该组合（' + conv + ' × ' + ds + '）基线模型缺失，请先运行训练脚本。');
    } else if (m === 'ensemble' && !isEnsembleOk(conv, ds, fm)) {
      reqHintWarn('深度集成：该组合缺少集成成员，可用单模型 / MC-Dropout。');
    } else if (fm !== 'base' && !isTrained(conv, ds, fm)) {
      reqHintErr('该组合未训练特征模型（实验矩阵：GCN×{a,b,c}、GAT/GraphSAGE×c），已自动切换到可用档位。');
    } else {
      reqHintOk('当前组合可用：' + conv + ' × ' + ds + ' × ' + (META ? META.feat_labels[fm] : fm));
    }
  }
}
function reqHintOk(t) { reqHint('ok', t); }
function reqHintWarn(t) { reqHint('warn', t); }
function reqHintErr(t) { reqHint('err', t); }
function reqHint(kind, t) {
  var el = document.getElementById('reqHint');
  el.className = 'req-hint ' + kind;
  el.textContent = t;
  el.style.display = 'block';
}
function featChanged() {
  var m = curMethod();
  var fm = curFeat();
  var note = document.getElementById('featNote');
  var withExp = document.getElementById('withExplain');
  if (m === 'attention') { return; }
  if (fm !== 'base') {
    note.style.display = 'inline';
    note.textContent = '特征模型支持单模型 / MC-Dropout / 深度集成（解释图基于基线）';
  } else {
    note.style.display = 'none';
  }
  refreshUI();
}

document.querySelectorAll('input[name="method"]').forEach(function (r) {
  r.addEventListener('change', function () {
    document.querySelectorAll('input[name="method"]').forEach(function (x) {
      sel(x.parentElement, x.checked);
    });
    refreshUI();
  });
});

/* ---------- SMILES 即时校验（/api/validate，防抖 300ms） ---------- */
function debValidate() {
  clearTimeout(_vTimer);
  _vTimer = setTimeout(validate, 300);
}
function validate() {
  var s = document.getElementById('smiles').value.trim();
  var wrap = document.getElementById('inpWrap');
  var st = document.getElementById('smilesStatus');
  if (s.length < 2) {
    wrap.className = 'inp-wrap';
    st.className = ''; st.textContent = '';
    VALIDATED = null;
    return;
  }
  if (VALIDATED && VALIDATED.input === s) { return; }
  fetch('/api/validate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smiles: s })
  }).then(function (r) { return r.json(); }).then(function (d) {
    VALIDATED = { input: s, ok: d.ok, canonical: d.canonical };
    if (d.ok) {
      wrap.className = 'inp-wrap ok';
      st.className = 'ok-msg';
      st.textContent = '✔ 合法' + (d.canonical && d.canonical !== s ? '（规范化：' + d.canonical + '）' : '');
    } else {
      wrap.className = 'inp-wrap bad';
      st.className = 'bad-msg';
      st.textContent = '✘ ' + (d.error || '无法解析');
    }
  }).catch(function () { wrap.className = 'inp-wrap'; st.textContent = ''; });
}

/* ---------- 运行与渲染 ---------- */
function setBusy(b, label) {
  var b1 = document.getElementById('btn');
  var b2 = document.getElementById('btn2');
  b1.disabled = b; b2.disabled = b;
  b1.textContent = b ? (label || '计算中...') : '运行分析';
  b2.textContent = b ? '等待中...' : 'LLM 化学解释';
}
function showErr(msg) {
  var e = document.getElementById('err');
  e.textContent = msg || '';
  e.style.display = msg ? 'block' : 'none';
}
function renderBase(d) {
  document.getElementById('tag').textContent = d.method || '';
  document.getElementById('kTask').textContent = d.task_label || '';
  document.getElementById('kSmiles').textContent = d.smiles || '';
  document.getElementById('pred').textContent = d.prediction;
  document.getElementById('unit').textContent = d.unit || '';
}

function run() {
  var m = curMethod();
  var smiles = document.getElementById('smiles').value.trim();
  var dataset = document.getElementById('dataset').value;
  showErr('');
  document.getElementById('result').style.display = 'none';
  if (!smiles) { showErr('请输入 SMILES 分子式'); return; }
  var withExplain = document.getElementById('withExplain').checked;
  if (m === 'attention') {
    var ac2 = new AbortController();
    var tm2 = setTimeout(function () { ac2.abort(); }, 120000);
    fetch('/api/attention', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles: smiles, dataset: dataset, per_head: true }),
      signal: ac2.signal
    }).then(function (r) { return r.json(); }).then(function (d) {
      clearTimeout(tm2);
      setBusy(false);
      if (d.error) { showErr(d.error); return; }
      renderAttention(d);
      pushHistory({ smiles: smiles, dataset: dataset, conv: 'GAT', feat: 'base', method: '注意力', pred: d.prediction, unc: d.heads + '头' });
    }).catch(function (e) {
      clearTimeout(tm2);
      setBusy(false);
      showErr('请求失败：' + (e && e.name === 'AbortError' ? '注意力渲染超时（大分子较慢），请稍后重试' : e));
    });
    setBusy(true, '渲染注意力图（大分子约需 1-2 分钟）...');
    return;
  }
  var ac = new AbortController();
  var tm = setTimeout(function () { ac.abort(); }, 180000);
  fetch('/api/predict', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      smiles: smiles, dataset: dataset, conv: document.getElementById('conv').value,
      method: m, explain: withExplain, feat: curFeat()
    }),
    signal: ac.signal
  }).then(function (r) { return r.json(); }).then(function (d) {
    clearTimeout(tm);
    setBusy(false);
    if (d.error) { showErr(d.error); return; }
    renderPrediction(d);
    var unc = d.ci90 ? ('[' + d.ci90[0] + ', ' + d.ci90[1] + ']') : (d.entropy !== undefined ? ('H=' + d.entropy) : (d.std !== undefined ? ('σ=' + d.std) : '—'));
    pushHistory({ smiles: smiles, dataset: dataset, conv: document.getElementById('conv').value, feat: curFeat(), method: d.method, pred: d.prediction, unc: unc });
  }).catch(function (e) {
    clearTimeout(tm);
    setBusy(false);
    showErr('请求失败：' + (e && e.name === 'AbortError' ? '分析超时（解释图计算较慢），请稍后重试' : e));
  });
  setBusy(true, withExplain ? '计算中（含 GNNExplainer 解释图）...' : '计算中...');
}

function renderPrediction(d) {
  renderBase(d);
  document.getElementById('tag').textContent = (d.method || '') + (d.feat_label ? '｜' + d.feat_label : '');
  var unc = document.getElementById('unc');
  if (d.ci90) {
    unc.style.display = 'block';
    unc.textContent = '90% 置信区间：[' + d.ci90[0] + ', ' + d.ci90[1] + ']｜标准差 σ = ' + d.std + '｜MC 采样 T=' + d.mc_samples;
  } else if (d.entropy !== undefined) {
    unc.style.display = 'block';
    unc.textContent = '预测熵 H = ' + d.entropy + '（越大不确定性越高）｜成员数 = ' + d.n_members;
  } else if (d.std !== undefined) {
    unc.style.display = 'block';
    unc.textContent = '标准差 σ = ' + d.std + '（成员分歧，越大不确定性越高）｜成员数 = ' + d.n_members;
  } else {
    unc.style.display = 'none';
  }
  var imgbox = document.getElementById('imgbox');
  var img = document.getElementById('img');
  var legendGnn = document.getElementById('legendGnn');
  var legendAttn = document.getElementById('legendAttn');
  if (d.image) {
    img.src = 'data:image/png;base64,' + d.image;
    imgbox.style.display = 'block';
    legendGnn.style.display = 'block';
    legendAttn.style.display = 'none';
  } else {
    img.src = ''; imgbox.style.display = 'none';
    legendGnn.style.display = 'none'; legendAttn.style.display = 'none';
  }
  var hint = document.getElementById('hint');
  hint.textContent = d.image ? '着色基于 GNNExplainer 原子/键重要性（红色越深对预测越关键）。' : '未生成解释图（已关闭）。';
  document.getElementById('result').style.display = 'block';
}

function renderAttention(d) {
  renderBase(d);
  document.getElementById('unit').textContent = '（GAT 注意力视图）';
  var unc = document.getElementById('unc');
  unc.style.display = 'block';
  unc.textContent = '多头注意力：' + d.heads + ' 头 × layer ' + d.layer + '（多头平均 + 逐头对比）｜边数：' + d.n_edges;
  var img = document.getElementById('img');
  img.src = d.image ? 'data:image/png;base64,' + d.image : '';
  document.getElementById('imgbox').style.display = d.image ? 'block' : 'none';
  document.getElementById('legendGnn').style.display = 'none';
  document.getElementById('legendAttn').style.display = d.image ? 'block' : 'none';
  var tops = (d.top8_edges || []).map(function (t) {
    return '原子 ' + t.edge.join('-') + '（α=' + t.alpha + '）';
  }).join('；');
  var hintText = '键着色 = 注意力权重（红=高）；Top 注意力键：' + tops + '。';
  if (d.per_head_images && d.per_head_images.length) {
    hintText += ' 下方为 ' + d.per_head_images.length + ' 个注意力头各自的关注视图（逐头对比）。';
  }
  document.getElementById('hint').textContent = hintText;
  var ph = document.getElementById('perHeadBox');
  if (d.per_head_images && d.per_head_images.length) {
    ph.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:12px;';
    ph.innerHTML = '';
    d.per_head_images.forEach(function (b64, k) {
      var w = document.createElement('div');
      w.style.cssText = 'flex:1;min-width:130px;';
      w.innerHTML = '<div style="font-size:12px;color:#666;margin-bottom:4px;">Head ' + (k + 1) + '</div>' +
        '<img src="data:image/png;base64,' + b64 + '" style="width:100%;border:1px solid #e2e7ee;border-radius:6px;">';
      ph.appendChild(w);
    });
    if (d.combo_image) {
      var cb = document.createElement('div');
      cb.style.cssText = 'flex:0 0 100%;margin-top:10px;';
      cb.innerHTML = '<div style="font-size:12px;color:#666;margin-bottom:4px;">逐头注意力对比（2×2）</div>' +
        '<img src="data:image/png;base64,' + d.combo_image + '" style="max-width:100%;border:1px solid #e2e7ee;border-radius:6px;">';
      ph.appendChild(cb);
    }
  } else {
    ph.style.cssText = 'display:none;';
  }
  document.getElementById('result').style.display = 'block';
}

/* ---------- 预测历史对比 ---------- */
function pushHistory(h) {
  HISTORY.unshift(h);
  if (HISTORY.length > 12) { HISTORY.pop(); }
  renderHistory();
}
function renderHistory() {
  var body = document.getElementById('histBody');
  var empty = document.getElementById('histEmpty');
  body.innerHTML = '';
  if (!HISTORY.length) { empty.style.display = 'block'; return; }
  empty.style.display = 'none';
  HISTORY.forEach(function (h, i) {
    var tr = document.createElement('tr');
    tr.innerHTML = '<td>' + (i + 1) + '</td>' +
      '<td style="max-width:150px;word-break:break-all;">' + h.smiles + '</td>' +
      '<td>' + h.dataset + '</td><td>' + h.conv + '</td>' +
      '<td>' + (h.feat === 'base' ? '基线' : 'feat_' + h.feat) + '</td>' +
      '<td>' + h.method + '</td><td><b>' + h.pred + '</b></td>' +
      '<td>' + h.unc + '</td>' +
      '<td><a class="play" onclick="replay(' + i + ')">回放</a></td>';
    body.appendChild(tr);
  });
}
function replay(i) {
  var h = HISTORY[i];
  document.getElementById('smiles').value = h.smiles;
  document.getElementById('dataset').value = h.dataset;
  document.getElementById('conv').value = h.conv;
  document.getElementById('feat').value = h.feat;
  if (h.method === '注意力') {
    document.querySelector('input[name="method"][value="attention"]').checked = true;
  } else if (h.method.indexOf('MC') >= 0) {
    document.querySelector('input[name="method"][value="mc"]').checked = true;
  } else if (h.method.indexOf('深度集成') >= 0) {
    document.querySelector('input[name="method"][value="ensemble"]').checked = true;
  } else {
    document.querySelector('input[name="method"][value="single"]').checked = true;
  }
  document.querySelectorAll('input[name="method"]').forEach(function (x) { sel(x.parentElement, x.checked); });
  document.getElementById('result').style.display = 'none';
  refreshUI();
  run();
}
function clearHistory() { HISTORY = []; renderHistory(); }

/* ---------- LLM 化学解释 ---------- */
function runLlm() {
  var smiles = document.getElementById('smiles').value.trim();
  var dataset = document.getElementById('dataset').value;
  var conv = curMethod() === 'attention' ? 'GAT' : document.getElementById('conv').value;
  showErr('');
  if (!smiles) { showErr('请输入 SMILES 分子式'); return; }
  setBusy(true, 'LLM 解释生成中（约 1-2 分钟，请勿重复操作）...');
  var ctl = new AbortController();
  var timer = setTimeout(function () { ctl.abort(); }, 330000);
  fetch('/api/explain', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smiles: smiles, dataset: dataset, conv: conv }),
    signal: ctl.signal
  }).then(function (r) { return r.json(); }).then(function (d) {
    clearTimeout(timer);
    setBusy(false);
    if (d.error) { showErr(d.error); return; }
    var warn = '';
    if (d.consistency && d.consistency.mentioned_undetected && d.consistency.mentioned_undetected.length) {
      warn = '\\n【一致性校验】LLM 提及但分子中未检出：' + d.consistency.mentioned_undetected.join('、') + '（已标记，请以 RDKit 检出为准）';
    }
    document.getElementById('llmText').textContent =
      d.explanation + '｜检出官能团：' + (d.functional_groups || []).join('、') + '（' + d.method + '）' + warn;
    document.getElementById('llmBox').style.display = 'block';
  }).catch(function (e) {
    clearTimeout(timer);
    setBusy(false);
    showErr('请求失败：' + (e && e.name === 'AbortError' ? '等待超时（本地 LLM 首次加载较慢，请稍后重试）' : e));
  });
}
loadMeta();
</script>
</body>
</html>"""


if __name__ == "__main__":
    load_all()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
