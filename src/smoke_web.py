# -*- coding: utf-8 -*-
"""Web 冒烟测试 v2（临时）：Clintox + 特征工程 + 逐头注意力 + 集成"""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:5000"


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode("utf-8")[:300]}


cases = [
    # (desc, path, payload)
    ("Clintox single GCN", "/api/predict",
     {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "conv": "GCN", "dataset": "Clintox", "method": "single", "explain": False}),
    ("ESOL feat_c single GCN", "/api/predict",
     {"smiles": "CCCO", "conv": "GCN", "dataset": "ESOL", "method": "single", "feat": "c", "explain": False}),
    ("ESOL feat_a single GAT", "/api/predict",
     {"smiles": "c1ccccc1", "conv": "GAT", "dataset": "ESOL", "method": "single", "feat": "a", "explain": False}),
    ("BBBP feat_c single SAGE", "/api/predict",
     {"smiles": "CN(C)c1cc(C2=NC(N)=NN2)ccn1", "conv": "GraphSAGE", "dataset": "BBBP", "method": "single", "feat": "c", "explain": False}),
    ("Clintox feat_b single GCN", "/api/predict",
     {"smiles": "CCCO", "conv": "GCN", "dataset": "Clintox", "method": "single", "feat": "b", "explain": False}),
    ("ESOL feat_c MC", "/api/predict",
     {"smiles": "CCCO", "conv": "GCN", "dataset": "ESOL", "method": "mc", "feat": "c", "explain": False}),
    ("ESOL feat_a single GCN", "/api/predict",
     {"smiles": "CCCO", "conv": "GCN", "dataset": "ESOL", "method": "single", "feat": "a", "explain": False}),
    ("ESOL feat_a ensemble GCN", "/api/predict",
     {"smiles": "CCCO", "conv": "GCN", "dataset": "ESOL", "method": "ensemble", "feat": "a", "explain": False}),
    ("ESOL feat_c ensemble GCN", "/api/predict",
     {"smiles": "CCCO", "conv": "GCN", "dataset": "ESOL", "method": "ensemble", "feat": "c", "explain": False}),
    ("GAT×ESOL feat_a ensemble GCN", "/api/predict",
     {"smiles": "CCCO", "conv": "GAT", "dataset": "ESOL", "method": "ensemble", "feat": "a", "explain": False}),
    ("Clintox ensemble(基线)", "/api/predict",
     {"smiles": "c1ccccc1", "conv": "GCN", "dataset": "Clintox", "method": "ensemble", "explain": False}),
    ("Attention per-head Clintox", "/api/attention",
     {"smiles": "CCCO", "dataset": "Clintox", "per_head": True}),
    ("Meta 组合可用性", "/api/meta", {}),
    ("Validate 合法 SMILES", "/api/validate", {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}),
    ("Validate 非法 SMILES", "/api/validate", {"smiles": "C1=CC=CC(C)C"}),
]

import urllib.request as _ur


def get(path):
    try:
        with _ur.urlopen(BASE + path, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"http_error": str(e)}


for desc, path, payload in cases:
    if path == "/api/meta":
        r = get(path)
        if "error" in r or "http_error" in r:
            print(f"[FAIL] {desc}: {r}")
        else:
            ok_feat = any(x.endswith("|c") for x in r.get("trained", []))
            ok_ens = any(x.startswith("GCN|") for x in r.get("ensemble_ok", []))
            print(f"[OK] {desc}: trained={len(r.get('trained') or [])} ensemble_ok={len(r.get('ensemble_ok') or [])} feat_c={'Y' if ok_feat else 'N'} ens_GCN={'Y' if ok_ens else 'N'}")
        continue
    if path == "/api/validate":
        r = post(path, payload)
        expect_fail = "非法" in desc
        if r.get("ok") is False and expect_fail:
            print(f"[OK] {desc}: 正确拒绝（{r.get('error')}）")
        elif r.get("ok") is True and not expect_fail:
            print(f"[OK] {desc}: ok=True canon={r.get('canonical')}")
        else:
            print(f"[FAIL] {desc}: {r}")
        continue
    r = post(path, payload)
    if "error" in r or "http_error" in r:
        print(f"[FAIL] {desc}: {r}")
    else:
        extra = ""
        if path.endswith("attention"):
            extra = f" heads={r.get('heads')} per_head={len(r.get('per_head_images') or [])} combo={'Y' if r.get('combo_image') else 'N'}"
        else:
            extra = f" feat={r.get('feat')} pred={r.get('prediction')} unit={r.get('unit')} method={r.get('method')}"
        print(f"[OK] {desc}: {extra}")
