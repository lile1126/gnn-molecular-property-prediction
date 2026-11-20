(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {};
  const state = { meta: null, file: null, data: null, busy: false };

  function cacheElements() {
    ["batchFile", "batchDrop", "batchFileName", "batchDataset", "batchConv", "batchFeat",
     "batchMethod", "batchRunBtn", "batchExportBtn", "batchClearBtn", "batchStatus",
     "batchBody", "batchEmpty", "modelVersionLine"].forEach((id) => { els[id] = $(id); });
  }

  function fillSelect(select, values, labeler) {
    if (!select) return;
    select.replaceChildren();
    (values || []).forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = labeler ? labeler(value) : value;
      select.appendChild(option);
    });
  }

  function setStatus(text, kind = "") {
    if (!els.batchStatus) return;
    els.batchStatus.textContent = text;
    els.batchStatus.className = `batch-status${kind ? ` is-${kind}` : ""}`;
  }

  function setBusy(busy) {
    state.busy = busy;
    els.batchRunBtn.disabled = busy;
    els.batchRunBtn.classList.toggle("is-loading", busy);
    els.batchRunBtn.textContent = busy ? "预测中..." : "运行批量预测";
  }

  async function loadMeta() {
    try {
      const response = await fetch("api/meta");
      state.meta = await response.json();
    } catch (error) {
      state.meta = { datasets: ["ESOL", "FreeSolv", "BBBP", "Clintox"], convs: ["GCN", "GAT", "GraphSAGE"],
        feat_levels: ["base", "a", "b", "c"], task_labels: {} };
    }
    const meta = state.meta;
    fillSelect(els.batchDataset, meta.datasets, (v) => meta.task_labels?.[v] || v);
    fillSelect(els.batchConv, meta.convs);
    fillSelect(els.batchFeat, meta.feat_levels,
      (v) => meta.feat_labels?.[v] || v);
    if (els.modelVersionLine && meta.model_version) {
      els.modelVersionLine.textContent =
        `模型版本 · ${meta.model_version.tag || "unknown"}${meta.model_version.n_checkpoints ? ` · ${meta.model_version.n_checkpoints} ckpt` : ""}`;
    }
  }

  async function runBatch() {
    if (state.busy) return;
    if (!state.file) { setStatus("请先选择 CSV 文件（需含 smiles 列）", "error"); return; }
    const form = new FormData();
    form.append("file", state.file);
    form.append("dataset", els.batchDataset.value);
    form.append("conv", els.batchConv.value);
    form.append("feat", els.batchFeat.value);
    form.append("method", els.batchMethod.value);
    setBusy(true);
    setStatus("正在批量预测（不生成解释图）...");
    try {
      const response = await fetch("api/batch", { method: "POST", body: form });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.error) throw new Error(data.error || `请求失败（HTTP ${response.status}）`);
      state.data = data;
      renderResults(data);
      setStatus(`完成：成功 ${data.n_ok} 条，失败 ${data.n_error} 条，用时 ${data.elapsed_s}s`,
        data.n_error ? "warn" : "ok");
    } catch (error) {
      setStatus(error.message, "error");
    } finally {
      setBusy(false);
    }
  }

  function renderResults(data) {
    els.batchBody.replaceChildren();
    const rows = [...(data.results || []), ...(data.errors || [])]
      .sort((a, b) => (a.row || 0) - (b.row || 0));
    const isClassification = (data.results || []).some((r) => r.task_type === "classification");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      const cells = [
        String(row.row ?? ""),
        row.id || "—",
        row.smiles || "",
        row.status === "ok"
          ? `${Number(row.prediction).toFixed(4)}${isClassification ? "" : ` ${row.unit || ""}`}`
          : "—",
        row.status === "ok"
          ? (row.std !== undefined ? `σ ${Number(row.std).toFixed(4)}`
            : row.entropy !== undefined ? `H ${Number(row.entropy).toFixed(4)}` : "—")
          : "—",
        row.status === "ok" ? "成功" : (row.error || "失败"),
      ];
      cells.forEach((text, index) => {
        const td = document.createElement("td");
        td.textContent = text;
        if (index === 5 && row.status !== "ok") td.className = "is-error-cell";
        tr.appendChild(td);
      });
      els.batchBody.appendChild(tr);
    });
    els.batchEmpty.hidden = rows.length > 0;
    els.batchExportBtn.disabled = rows.length === 0;
  }

  async function exportCsv() {
    if (!state.data) return;
    try {
      const response = await fetch("api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          results: state.data.results, errors: state.data.errors,
          dataset: state.data.dataset, method: state.data.method,
        }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `导出失败（HTTP ${response.status}）`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `molgnn_batch_${state.data.dataset || "result"}.csv`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      setStatus("已导出 CSV（UTF-8 BOM，Excel 可直接打开）", "ok");
    } catch (error) {
      setStatus(error.message, "error");
    }
  }

  function clearBatch() {
    state.file = null;
    state.data = null;
    if (els.batchFile) els.batchFile.value = "";
    if (els.batchFileName) els.batchFileName.textContent = "未选择文件";
    if (els.batchBody) els.batchBody.replaceChildren();
    if (els.batchEmpty) els.batchEmpty.hidden = false;
    if (els.batchExportBtn) els.batchExportBtn.disabled = true;
    setStatus("");
  }

  function init() {
    cacheElements();
    loadMeta();
    els.batchFile?.addEventListener("change", () => {
      state.file = els.batchFile.files?.[0] || null;
      els.batchFileName.textContent = state.file
        ? `${state.file.name}（${(state.file.size / 1024).toFixed(1)} KB）` : "未选择文件";
      setStatus("");
    });
    els.batchRunBtn?.addEventListener("click", runBatch);
    els.batchExportBtn?.addEventListener("click", exportCsv);
    els.batchClearBtn?.addEventListener("click", clearBatch);
    els.batchMethod?.addEventListener("change", () => {
      const isHetero = els.batchMethod.value === "hetero";
      [els.batchConv, els.batchFeat].forEach((node) => { if (node) node.disabled = isHetero; });
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
