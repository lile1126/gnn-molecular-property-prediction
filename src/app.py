# -*- coding: utf-8 -*-
"""
Web 演示服务
============
输入 SMILES 分子式 → 选择数据集与模型 → 返回预测值 + 分子结构图（原子/键重要性着色）。

用法:
    python src/app.py     # 启动后访问 http://127.0.0.1:5000
"""
import base64
import csv
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
from flask import Flask, Response, jsonify, render_template, request
from torch_geometric.utils import from_smiles

from data_prep import TASK_TYPE
from interpret import draw_molecule, explain_one, load_trained
from llm_explain import _load_deepseek_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENSEMBLE_DIR = os.path.join(ROOT, "models", "ensemble")
MANIFEST_PATH = os.path.join(ROOT, "reports", "model_manifest.json")
CONSISTENCY_PATH = os.path.join(ROOT, "reports", "explanation_consistency.json")

CONVS = ["GCN", "GAT", "GraphSAGE"]
DATASETS = ["ESOL", "FreeSolv", "BBBP", "Clintox"]
FEAT_LEVELS = ["base", "a", "b", "c"]
MC_T = 30
MAX_SMILES_LEN = 300          # 超长分子拦截（正常药物分子 SMILES 远小于此值）
MAX_BATCH_ROWS = 200          # 批量上传上限，避免单次请求过长阻塞
LLM_TIMEOUT = 30              # Web 端 API 超时；失败立即降级为规则模板

app = Flask(__name__, static_url_path=os.environ.get("STATIC_URL_PATH", "/static"))
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
            f"{conv} × {name} × feat_{level} 的权重缺失"
            f"（完整矩阵为 3 架构 × 4 数据集 × 3 特征档 = 36 个，请检查 models/feat/ 是否齐全）")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m = build_model(conv, ckpt["in_dim"], ckpt["hidden"], ckpt["out_dim"], ckpt["layers"],
                    heads=ckpt.get("heads", 4), edge_dim=ckpt.get("edge_dim", 0),
                    fp_dim=ckpt.get("fp_dim", 0)).to(DEVICE)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    FEAT_MODELS[key] = m
    return m


def _build_data(smiles, level):
    """按特征等级构建图数据：base 用 PyG from_smiles（9 维），feat_a/b/c 用 RDKit 特征。

    注意：from_smiles 对非法 SMILES 不返回 None，而是返回 0 原子的空 Data，
    直接送入模型/GNNExplainer 会触发除零等内部错误，这里统一拦成 ValueError（端点转 400）。
    """
    if level == "base":
        data = from_smiles(smiles)
        if data is None or getattr(data, "x", None) is None or data.x.shape[0] == 0:
            raise ValueError(f"SMILES 无法解析为有效分子: {smiles}")
        return data
    from featurize import featurize_smiles
    data = featurize_smiles(smiles, level)
    if data is None or getattr(data, "x", None) is None or data.x.shape[0] == 0:
        raise ValueError(f"SMILES 无法解析为有效分子: {smiles}")
    return data


def _feat_info(level):
    return {"feat": level, "feat_label": FEAT_LABEL[level]}


# ---------------------------------------------------------------------------
# 模型版本 / 解释一致性 / 输入校验（P0-5）
# ---------------------------------------------------------------------------
_MANIFEST_CACHE = None
_CONSISTENCY_CACHE = None


def _model_version():
    """返回模型清单（reports/model_manifest.json，由 src/model_manifest.py 生成）。"""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as f:
                _MANIFEST_CACHE = json.load(f)
            return _MANIFEST_CACHE
        except Exception as e:
            print("读取模型清单失败:", e)
    _MANIFEST_CACHE = {"tag": "unknown", "generated_at": None,
                       "note": "未找到 MODEL_MANIFEST.json，请运行 python src/model_manifest.py"}
    return _MANIFEST_CACHE


def _consistency_info(name):
    """解释一致性等级（高/中/低）+ 关键指标，来自 explanation_consistency.json。"""
    global _CONSISTENCY_CACHE
    if _CONSISTENCY_CACHE is None:
        _CONSISTENCY_CACHE = {}
        if os.path.exists(CONSISTENCY_PATH):
            try:
                with open(CONSISTENCY_PATH, encoding="utf-8") as f:
                    raw = json.load(f)
                for ds, info in raw.items():
                    if isinstance(info, dict) and "summary" in info:
                        s = info["summary"]
                        _CONSISTENCY_CACHE[ds] = {
                            "level": s.get("level", "未评估"),
                            "jaccard_top10_atom": (s.get("jaccard_top10_atom") or {}).get("mean"),
                            "spearman_atom": (s.get("spearman_atom") or {}).get("mean"),
                            "pred_std": (s.get("pred_std") or {}).get("mean"),
                            "n_molecules": info.get("n_molecules"),
                            "note": "9 配置两两平均；等级阈值见 reports/explanation_consistency.json",
                        }
            except Exception as e:
                print("读取解释一致性失败:", e)
    return _CONSISTENCY_CACHE.get(name, {"level": "未评估", "note": "尚未运行 explanation_consistency.py"})


def _validate_smiles_input(smiles, field="SMILES"):
    """空输入 / 超长分子统一拦截。返回清洗后的字符串，非法时抛 ValueError。"""
    smiles = (smiles or "").strip()
    if not smiles:
        raise ValueError(f"{field} 为空，请输入分子结构（例如 CCCO）")
    if len(smiles) > MAX_SMILES_LEN:
        raise ValueError(f"{field} 过长（{len(smiles)} 字符，上限 {MAX_SMILES_LEN}）；"
                         f"请确认输入的是单个分子而非反应式/聚合物")
    return smiles


def _functional_group_payload(smiles, node_mask=None, edge_mask=None, data=None):
    """官能团级归因：有掩码则聚合法重要性，无掩码只做 RDKit 识别（importance_available=False）。"""
    try:
        from functional_group_explain import aggregate, detect_groups
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if node_mask is not None and edge_mask is not None and data is not None:
            from functional_group_explain import bond_importance_from_mask
            bond_imp = bond_importance_from_mask(mol, data.edge_index.cpu().numpy(), edge_mask)
            res = aggregate(smiles, node_mask, bond_imp)
            return {"top3": res["top3"], "groups_detected": res["groups_detected"],
                    "atom_assignment_coverage": res["atom_assignment_coverage"],
                    "importance_available": True, "layers": res["layers"],
                    "note": "重要性来自 GNNExplainer 原子掩码（组内均值）；官能团由 RDKit 识别"}
        groups, _ = detect_groups(mol)
        return {"top3": [], "groups_detected": len(groups),
                "detected": [g["name"] for g in groups],
                "importance_available": False,
                "note": "未生成结构掩码，仅展示 RDKit 识别的官能团（不做重要性排序）"}
    except Exception as e:
        print("官能团聚合失败:", e)
        return None


# ---------------------------------------------------------------------------
# 异质集成（最终方案）：9 配置 × 4 种子等权
# 依据 reports/fusion_summary.md —— 同构种子平均无增益，跨架构+跨特征档才有效
# ---------------------------------------------------------------------------
HETERO_LEVELS = ["a", "b", "c"]
# 异质集成结构归因的 GNNExplainer 迭代数：9 个模型串行，300 会让单次预测到 30 秒级
HETERO_MASK_EPOCHS = 60


_HETERO_READY = {}


def _feat_seed_count(name="ESOL"):
    """单配置的成员数：seed42 主模型 + feat_ensemble 里的 s 种子。"""
    fe_dir = os.path.join(ROOT, "models", "feat_ensemble")
    if not os.path.isdir(fe_dir):
        return 1
    conv, lv = CONVS[0], HETERO_LEVELS[0]
    return 1 + sum(1 for p in os.listdir(fe_dir)
                   if p.startswith(f"{conv}_{name}_feat{lv}_s") and p.endswith(".pt"))


def _hetero_ready(name, refresh=False):
    """目录级判断（不加载权重），/api/meta 与 /api/predict 共用；结果缓存避免重复 listdir。"""
    if not refresh and name in _HETERO_READY:
        return _HETERO_READY[name]
    fe_dir = os.path.join(ROOT, "models", "feat_ensemble")
    ok = os.path.isdir(fe_dir)
    if ok:
        for lv in HETERO_LEVELS:
            for conv in CONVS:
                if not os.path.exists(os.path.join(ROOT, "models", "feat", f"{conv}_{name}_feat{lv}.pt")):
                    ok = False
                    break
                n = sum(1 for p in os.listdir(fe_dir)
                        if p.startswith(f"{conv}_{name}_feat{lv}_s") and p.endswith(".pt"))
                if n < 3:
                    ok = False
                    break
            if not ok:
                break
    _HETERO_READY[name] = ok
    return ok


@torch.no_grad()
def hetero_predict(name, smiles, task_type):
    """异质等权集成：每个配置先做 4 种子平均，再对 9 个配置等权平均。

    与同构集成（`ensemble`）的区别：成员来自不同架构与不同特征档，
    test 残差相关从 0.92 降到 0.61~0.87，是唯一在全部数据集上稳定提升的方案。
    """
    per_cfg = []
    for lv in HETERO_LEVELS:
        try:
            data = _build_data(smiles, lv)
        except ValueError as e:  # 非法 SMILES 不能被当成"成员缺失"静默跳过
            raise ValueError(str(e)) from e
        except Exception:
            continue
        for conv in CONVS:
            members = _load_ensemble_members(conv, name, lv)
            if not members:
                continue
            mean, _ = ensemble_predict_single(members, data, task_type)
            per_cfg.append(mean)
    if not per_cfg:
        raise ValueError("异质集成成员缺失，请先运行 train_feat.py 与 train_feat_ensemble.py")
    arr = np.array(per_cfg)
    if task_type == "classification":
        p = float(arr.mean())
        ent = float(-(p * np.log(max(p, 1e-12)) + (1 - p) * np.log(max(1 - p, 1e-12))))
        return p, ent, len(per_cfg)
    return float(arr.mean()), float(arr.std()), len(per_cfg)


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
    hetero = [n for n in DATASETS if _hetero_ready(n)]
    n_seed = _feat_seed_count()
    version = _model_version()
    return jsonify({
        "convs": CONVS, "datasets": DATASETS, "feat_levels": FEAT_LEVELS,
        "feat_labels": FEAT_LABEL, "task_labels": TASK_LABEL, "unit_labels": UNIT_LABEL,
        "trained": trained, "ensemble_ok": ens,
        "hetero_ok": hetero, "hetero_n": len(HETERO_LEVELS) * len(CONVS) * n_seed,
        "mc_t": MC_T, "ensemble_n": 4,
        "llm_provider": "deepseek",
        "llm_available": bool(_load_deepseek_key()),
        "model_version": {"tag": version.get("tag", "unknown"),
                          "generated_at": version.get("generated_at"),
                          "n_checkpoints": version.get("n_checkpoints")},
        "explanation_consistency": {n: _consistency_info(n) for n in DATASETS},
        "max_smiles_len": MAX_SMILES_LEN,
        "max_batch_rows": MAX_BATCH_ROWS,
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
    conv = body.get("conv", "GCN")
    name = body.get("dataset", "ESOL")
    method = body.get("method", "single")
    level = body.get("feat", "base")
    if conv not in CONVS or name not in DATASETS:
        return jsonify({"error": "无效的模型参数"}), 400
    if level not in FEAT_LEVELS:
        return jsonify({"error": "无效的特征等级参数"}), 400
    if method not in ("single", "mc", "ensemble", "hetero"):
        return jsonify({"error": "无效的方法参数"}), 400
    try:
        smiles = _validate_smiles_input(body.get("smiles"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    task_type = TASK_TYPE[name]

    # 异质集成：9 配置 × 4 种子等权，与 conv / feat 选择无关（模型级融合的最终方案）
    if method == "hetero":
        if not _hetero_ready(name):
            return jsonify({"error": f"{name} 的异质集成成员不齐（需 9 配置 × 4 种子）"}), 400
        try:
            mean, unc, n_cfg = hetero_predict(name, smiles, task_type)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"异质集成推理失败: {e}"}), 400
        # 结构归因图：与 LLM 解释同源（9 配置 seed42 主成员掩码平均）
        node_mask = edge_mask = first_data = None
        n_attr = 0
        if bool(body.get("explain", True)):
            try:
                from llm_explain import _mean_masks
                models, datas = [], []
                for lv in HETERO_LEVELS:
                    d = _build_data(smiles, lv)
                    for c in CONVS:
                        mem = _load_ensemble_members(c, name, lv)
                        if mem:
                            models.append(mem[0])
                            datas.append(d)
                if models:
                    node_mask, edge_mask, _ = _mean_masks(models, datas, task_type,
                                                           epochs=HETERO_MASK_EPOCHS)
                    first_data, n_attr = datas[0], len(models)
            except Exception as e:
                print("异质集成归因失败:", e)
                node_mask = edge_mask = None
        resp = {
            "task_type": task_type,
            "task_label": TASK_LABEL[name],
            "smiles": smiles,
            "unit": UNIT_LABEL[name],
            "feat": "hetero",
            "feat_label": "跨特征档（a/b/c 全组合）",
            "method": f"异质集成（{n_cfg} 配置 × 4 种子等权）",
            "n_members": n_cfg * 4,
            "n_configs": n_cfg,
        }
        if task_type == "regression":
            resp["prediction"] = round(float(mean), 4)
            resp["std"] = round(float(unc), 4)
            resp["ci90"] = [round(float(mean - 1.645 * unc), 4),
                            round(float(mean + 1.645 * unc), 4)]
        else:
            resp["prediction"] = round(float(mean), 4)
            resp["entropy"] = round(float(unc), 4)
        if node_mask is not None:
            resp["node_mask"] = [round(float(v), 4) for v in node_mask]
            resp["edge_mask"] = [round(float(v), 4) for v in edge_mask]
            img = _render_image(smiles, "Hetero", name, first_data, node_mask, edge_mask)
            if img:
                resp["image"] = img
            resp["attribution"] = f"{n_attr} 个配置 seed42 主成员掩码平均"
        resp["consistency"] = _consistency_info(name)
        resp["model_version"] = _model_version().get("tag", "unknown")
        resp["functional_groups"] = _functional_group_payload(
            smiles, node_mask, edge_mask, first_data)
        return jsonify(resp)

    try:
        data = _build_data(smiles, level)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"分子预处理失败: {e}"}), 400
    explain = bool(body.get("explain", True))
    if level != "base":
        try:
            model = _load_feat_model(conv, name, level)
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 400
    else:
        model = MODELS.get((conv, name))
        if model is None:  # 兼容 flask run / gunicorn 等非 __main__ 启动方式（懒加载）
            from interpret import load_trained
            model, _ = load_trained(conv, name)
            MODELS[(conv, name)] = model
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
    resp["consistency"] = _consistency_info(name)
    resp["model_version"] = _model_version().get("tag", "unknown")
    resp["functional_groups"] = _functional_group_payload(
        smiles, node_mask, edge_mask, data if node_mask is not None else None)
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
        resp["method"] = f"同构集成（{len(members)} 成员）"
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
    name = body.get("dataset", "BBBP")
    layer = int(body.get("layer", 0))
    per_head = bool(body.get("per_head", False))
    if name not in DATASETS:
        return jsonify({"error": "无效的数据集参数"}), 400
    try:
        smiles = _validate_smiles_input(body.get("smiles"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if ("GAT", name) not in MODELS:
        from interpret import load_trained
        try:
            model, _ = load_trained("GAT", name)
        except Exception as e:
            return jsonify({"error": f"GAT 模型加载失败: {e}"}), 500
        MODELS[("GAT", name)] = model
    try:
        data = from_smiles(smiles)
        if data is None or getattr(data, "x", None) is None or data.x.shape[0] == 0:
            raise ValueError("SMILES 无法解析为有效分子")
    except Exception as e:
        return jsonify({"error": f"SMILES 解析失败: {e}"}), 400
    model = MODELS[("GAT", name)]
    task_type = TASK_TYPE[name]
    try:
        alphas, edge_indices = extract_attention(model, data, layer=layer)
        alpha = alphas[layer]
        ei = edge_indices[layer]
        png = draw_attention_molecule(smiles, ei, alpha, title="", as_buffer=True)
        # 单原子分子（如 "[Na+]"）没有化学键，draw_attention_molecule 返回 False
        b64 = base64.b64encode(png).decode("ascii") if png else None
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
            "warning": None if b64 else "该分子没有化学键，无法生成注意力着色图（预测值仍然有效）",
            "per_head_images": heads_imgs, "combo_image": combo_b64,
            "method": f"GAT 注意力可视化（layer {layer}，多头平均" + (" + 逐头对比" if per_head else "") + "）",
        })
    except Exception as e:
        return jsonify({"error": f"注意力计算失败: {e}"}), 500


# ---------------------------------------------------------------------------
# 批量预测与 CSV 导出（P0-5）
# ---------------------------------------------------------------------------
def _light_predict(smiles, name, conv, level, method):
    """批量场景的轻量推理：不跑 GNNExplainer，返回统一的预测字段。"""
    task_type = TASK_TYPE[name]
    unit = UNIT_LABEL[name]
    if method == "hetero":
        if not _hetero_ready(name):
            raise ValueError(f"{name} 的异质集成成员不齐（需 9 配置 × 4 种子）")
        mean, unc, n_cfg = hetero_predict(name, smiles, task_type)
        out = {"prediction": round(float(mean), 4),
               "method": f"异质集成（{n_cfg} 配置 × 4 种子）",
               "n_members": n_cfg * 4}
        if task_type == "regression":
            out["std"] = round(float(unc), 4)
            out["ci90"] = [round(float(mean - 1.645 * unc), 4), round(float(mean + 1.645 * unc), 4)]
        else:
            out["entropy"] = round(float(unc), 4)
        out.update({"unit": unit, "feat": "hetero"})
        return out

    data = _build_data(smiles, level)
    if level == "base":
        model = MODELS.get((conv, name))
        if model is None:
            model, _ = load_trained(conv, name)
            MODELS[(conv, name)] = model
    else:
        model = _load_feat_model(conv, name, level)

    if method == "mc":
        mean, unc = mc_predict_single(model, data, task_type)
        out = {"prediction": round(float(mean), 4), "method": f"MC-Dropout（{MC_T} 次采样）",
               "mc_samples": MC_T}
    elif method == "ensemble":
        members = _load_ensemble_members(conv, name, level)
        if len(members) < 2:
            raise ValueError("同构集成成员不足（至少需要 2 个种子模型）")
        mean, unc = ensemble_predict_single(members, data, task_type)
        out = {"prediction": round(float(mean), 4),
               "method": f"同构集成（{len(members)} 成员）", "n_members": len(members)}
    else:
        with torch.no_grad():
            d = data.to(DEVICE)
            pred = model.forward_data(d)
        if task_type == "regression":
            out = {"prediction": round(float(pred.squeeze(-1).item()), 4), "method": "单模型"}
        else:
            out = {"prediction": round(float(torch.softmax(pred, dim=-1)[0, 1].item()), 4),
                   "method": "单模型"}
    if method in ("mc", "ensemble"):
        if task_type == "regression":
            out["std"] = round(float(unc), 4)
            out["ci90"] = [round(float(mean - 1.645 * unc), 4), round(float(mean + 1.645 * unc), 4)]
        else:
            out["entropy"] = round(float(unc), 4)
    out.update({"unit": unit, "feat": level})
    return out


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    """批量 CSV 上传 / JSON 列表 → 逐行预测（不跑解释器，适合虚拟筛选）。"""
    import time as _time
    body = request.get_json(silent=True) or {}
    conv = request.form.get("conv") or body.get("conv", "GCN")
    name = request.form.get("dataset") or body.get("dataset", "ESOL")
    level = request.form.get("feat") or body.get("feat", "base")
    method = request.form.get("method") or body.get("method", "single")
    if conv not in CONVS or name not in DATASETS:
        return jsonify({"error": "无效的模型参数"}), 400
    if level not in FEAT_LEVELS:
        return jsonify({"error": "无效的特征等级参数"}), 400
    if method not in ("single", "mc", "ensemble", "hetero"):
        return jsonify({"error": "批量预测支持 single / mc / ensemble / hetero（attention 请单条运行）"}), 400

    rows = []
    if "file" in request.files:
        upload = request.files["file"]
        raw = upload.read()
        if len(raw) > 2 * 1024 * 1024:
            return jsonify({"error": "CSV 文件过大（上限 2 MB）"}), 400
        text = raw.decode("utf-8-sig", errors="replace")
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if not header:
            return jsonify({"error": "CSV 为空或缺少表头"}), 400
        lowered = [h.strip().lower() for h in header]
        idx_smi = next((i for i, h in enumerate(lowered)
                        if h in ("smiles", "smile", "canonical_smiles", "structure")), 0)
        idx_id = next((i for i, h in enumerate(lowered)
                       if h in ("id", "name", "molecule", "compound")), None)
        for r_i, row in enumerate(reader, start=2):
            if not row:
                continue
            smi = row[idx_smi].strip() if idx_smi < len(row) else ""
            if not smi:
                continue
            rid = row[idx_id].strip() if (idx_id is not None and idx_id < len(row)) else ""
            rows.append({"row": r_i, "id": rid, "smiles": smi})
    else:
        lst = body.get("smiles_list") or []
        if not isinstance(lst, list):
            return jsonify({"error": "smiles_list 必须是字符串数组"}), 400
        for i, smi in enumerate(lst, start=1):
            if isinstance(smi, dict):
                rows.append({"row": i, "id": str(smi.get("id", "")), "smiles": str(smi.get("smiles", "")).strip()})
            else:
                rows.append({"row": i, "id": "", "smiles": str(smi).strip()})

    rows = [r for r in rows if r["smiles"]]
    if not rows:
        return jsonify({"error": "没有可预测的 SMILES（请检查 CSV 的 smiles 列或空输入）"}), 400
    if len(rows) > MAX_BATCH_ROWS:
        return jsonify({"error": f"单次最多 {MAX_BATCH_ROWS} 条（当前 {len(rows)} 条），请拆分文件"}), 400

    # 模型可用性前置检查，避免每行重复报错
    if method != "hetero":
        if level == "base":
            if not os.path.exists(os.path.join(ROOT, "models", f"{conv}_{name}.pt")):
                return jsonify({"error": f"模型缺失：{conv} × {name} 基线权重不存在，请先运行 train_eval.py"}), 400
            if method == "ensemble":
                cnt = 1 + sum(1 for p in os.listdir(ENSEMBLE_DIR)
                              if p.startswith(f"{conv}_{name}_s") and p.endswith(".pt")) if os.path.isdir(ENSEMBLE_DIR) else 1
                if cnt < 2:
                    return jsonify({"error": f"模型缺失：{conv} × {name} 的同构集成成员不足"}), 400
        elif not os.path.exists(os.path.join(ROOT, "models", "feat", f"{conv}_{name}_feat{level}.pt")):
            return jsonify({"error": f"模型缺失：{conv} × {name} × feat_{level} 权重不存在，请先运行 train_feat.py"}), 400
        elif method == "ensemble":
            cnt = 1 + sum(1 for p in os.listdir(os.path.join(ROOT, "models", "feat_ensemble"))
                          if p.startswith(f"{conv}_{name}_feat{level}_s") and p.endswith(".pt")) \
                if os.path.isdir(os.path.join(ROOT, "models", "feat_ensemble")) else 1
            if cnt < 2:
                return jsonify({"error": f"模型缺失：{conv} × {name} × feat_{level} 的同构集成成员不足"}), 400
    elif not _hetero_ready(name):
        return jsonify({"error": f"{name} 的异质集成成员不齐（需 9 配置 × 4 种子）"}), 400

    t0 = _time.time()
    results, errors = [], []
    for r in rows:
        try:
            smi = _validate_smiles_input(r["smiles"])
            out = _light_predict(smi, name, conv, level, method)
            results.append({"row": r["row"], "id": r["id"], "smiles": smi,
                            "status": "ok", "task_type": TASK_TYPE[name], **out})
        except ValueError as e:
            errors.append({"row": r["row"], "id": r["id"], "smiles": r["smiles"],
                           "status": "error", "error": str(e)})
        except Exception as e:
            errors.append({"row": r["row"], "id": r["id"], "smiles": r["smiles"],
                           "status": "error", "error": f"预测失败: {e}"})
    return jsonify({
        "dataset": name, "conv": conv, "feat": level, "method": method,
        "task_label": TASK_LABEL[name], "unit": UNIT_LABEL[name],
        "results": results, "errors": errors,
        "n_ok": len(results), "n_error": len(errors),
        "elapsed_s": round(_time.time() - t0, 2),
        "model_version": _model_version().get("tag", "unknown"),
        "note": "批量模式不生成 GNNExplainer 结构解释；导出 CSV 可在 Excel 中直接打开",
    })


@app.route("/api/export", methods=["POST"])
def export_csv():
    """把批量预测结果导出为 CSV（UTF-8 BOM，Excel 中文不乱码）。"""
    body = request.get_json(force=True) or {}
    rows = body.get("results") or []
    errors = body.get("errors") or []
    if not rows and not errors:
        return jsonify({"error": "没有可导出的结果"}), 400
    fields = ["row", "id", "smiles", "status", "prediction", "std", "entropy",
              "ci90_low", "ci90_high", "unit", "method", "error", "task_type", "dataset", "model_version"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    version = _model_version().get("tag", "unknown")
    for r in rows:
        ci = r.get("ci90") or [None, None]
        writer.writerow({
            "row": r.get("row"), "id": r.get("id", ""), "smiles": r.get("smiles", ""),
            "status": r.get("status", "ok"), "prediction": r.get("prediction"),
            "std": r.get("std", ""), "entropy": r.get("entropy", ""),
            "ci90_low": ci[0] if ci else "", "ci90_high": ci[1] if len(ci) > 1 else "",
            "unit": r.get("unit", ""), "method": r.get("method", ""), "error": "",
            "task_type": r.get("task_type", ""), "dataset": body.get("dataset", ""),
            "model_version": version,
        })
    for e in errors:
        writer.writerow({"row": e.get("row"), "id": e.get("id", ""), "smiles": e.get("smiles", ""),
                         "status": "error", "error": e.get("error", ""),
                         "dataset": body.get("dataset", ""), "model_version": version})
    csv_text = "\ufeff" + buf.getvalue()
    fname = f"molgnn_batch_{body.get('dataset', 'result')}_{_model_version().get('tag', 'v')}.csv"
    return Response(csv_text.encode("utf-8"), mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.route("/api/explain", methods=["POST"])
def explain_llm():
    """LLM 化学解释：GNNExplainer 掩码 → 结构证据 → DeepSeek API 生成解释

    归因模型与推理模式一一对应：
      single/mc   → 当前 conv × feat 的 seed42 模型
      ensemble    → 该配置同构集成的 seed42 主成员
      hetero      → 9 配置 seed42 主成员，掩码平均
      attention   → GAT × feat_base
    body.prediction 若传入（界面已显示的推理结果），解释文本引用它，避免数值口径错位。
    """
    body = request.get_json(force=True) or {}
    conv = body.get("conv", "GCN")
    name = body.get("dataset", "ESOL")
    level = body.get("feat", "base")
    method = (body.get("method") or "single").strip()
    raw_pred = body.get("prediction")
    if conv not in CONVS or name not in DATASETS:
        return jsonify({"error": "无效的模型参数"}), 400
    if level not in FEAT_LEVELS:
        return jsonify({"error": "无效的特征等级参数"}), 400
    if method not in ("single", "mc", "ensemble", "hetero", "attention"):
        return jsonify({"error": "无效的推理模式参数"}), 400
    try:
        smiles = _validate_smiles_input(body.get("smiles"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    pred = None
    try:
        if raw_pred is not None:
            pred = float(raw_pred)
    except (TypeError, ValueError):
        pred = None

    try:
        from llm_explain import generate_explanation
        if method == "hetero":
            models, datas = [], []
            for lv in HETERO_LEVELS:
                d = _build_data(smiles, lv)
                for c in CONVS:
                    members = _load_ensemble_members(c, name, lv)
                    if members:
                        models.append(members[0])
                        datas.append(d)
            if not models:
                return jsonify({"error": f"{name} 的异质集成成员缺失"}), 400
            note = f"异质集成 9 配置 seed42 主成员（{len(models)} 个模型掩码平均）"
        elif method == "ensemble":
            data = _build_data(smiles, level)
            members = _load_ensemble_members(conv, name, level)
            if not members:
                return jsonify({"error": "同构集成成员缺失"}), 400
            models, datas = [members[0]], [data]
            note = f"{conv} · feat_{level} 同构集成主成员（共 {len(members)} 成员）"
        elif method == "attention":
            models, datas = [MODELS[("GAT", name)]], [from_smiles(smiles)]
            note = "GAT · feat_base 注意力模型"
        else:
            data = _build_data(smiles, level)
            if level == "base":
                model = MODELS.get((conv, name))
                if model is None:
                    from interpret import load_trained
                    model, _ = load_trained(conv, name)
                    MODELS[(conv, name)] = model
            else:
                model = _load_feat_model(conv, name, level)
            models, datas = [model], [data]
            note = f"{conv} · feat_{level} {'MC-Dropout' if method == 'mc' else '单模型'}"
        result = generate_explanation(smiles, dataset=name, conv=conv,
                                      models=models, datas=datas, method=method,
                                      model_note=note, prediction=pred, data=datas[0],
                                      llm_timeout=LLM_TIMEOUT)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"LLM 解释失败: {e}"}), 500
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify({
        "explanation": result["explanation"],
        "prediction": result["prediction"],
        "functional_groups": result["structure"]["functional_groups"],
        "top_groups": result["structure"].get("top_groups", []),
        "group_layers": result["structure"].get("group_layers"),
        "key_atoms": result["structure"]["key_atoms"],
        "consistency": result["consistency"],
        "feat": level,
        "method": method,
        "model_note": result.get("model_note"),
        "provider": result.get("provider", "deepseek"),
        "fallback_reason": result.get("fallback_reason"),
        "llm_method": ("DeepSeek API (deepseek-chat)" if result.get("provider") == "deepseek"
                       else "规则模板（离线降级）"),
        "model_version": _model_version().get("tag", "unknown"),
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
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5000")),
            debug=False, threaded=True)
