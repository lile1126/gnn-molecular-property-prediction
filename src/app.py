# -*- coding: utf-8 -*-
"""
Web 演示服务
============
输入 SMILES 分子式 → 选择数据集与模型 → 返回预测值 + 分子结构图（原子/键重要性着色）。

用法:
    python src/app.py     # 启动后访问 http://127.0.0.1:5000
"""
import base64
import io
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
from flask import Flask, jsonify, render_template, request
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
    """MC-Dropout 单分子采样，并在结束后恢复模型的原始模式。"""
    was_training = model.training
    model.train()
    try:
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
    finally:
        model.train(was_training)

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
    return render_template("index.html")


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
    """LLM 化学解释：GNNExplainer 掩码 → 结构证据 → LLM 生成解释

    provider: local=本地 Qwen2.5-3B-Instruct（4bit），deepseek=DeepSeek API
    """
    body = request.get_json(force=True) or {}
    smiles = (body.get("smiles") or "").strip()
    conv = body.get("conv", "GCN")
    name = body.get("dataset", "ESOL")
    provider = body.get("provider", "local")
    if conv not in CONVS or name not in DATASETS:
        return jsonify({"error": "无效的模型参数"}), 400
    if provider not in ("local", "deepseek"):
        return jsonify({"error": "无效的 LLM 通道（可选 local / deepseek）"}), 400
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
        result = generate_explanation(smiles, dataset=name, conv=conv, model=model, data=data,
                                      provider=provider)
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
        "provider": result.get("provider", provider),
        "method": "DeepSeek API (deepseek-chat)" if provider == "deepseek"
                  else "本地 Qwen2.5-3B-Instruct (4bit)",
    })


def _render_image(smiles, conv, name, data, node_mask, edge_mask):
    img_b64 = None
    try:
        buf = io.BytesIO()
        ok = draw_molecule(
            smiles, node_mask, edge_mask, data.edge_index.cpu().numpy(), buf,
            title=f"{conv} | {name}",
        )
        if ok:
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print("绘图失败:", e)
    return img_b64





if __name__ == "__main__":
    load_all()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
