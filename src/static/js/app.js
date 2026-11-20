(() => {
  "use strict";

  const STORAGE_KEY = "molgnn.history.v1";
  const MAX_HISTORY = 12;
  const state = {
    meta: null,
    validated: null,
    history: [],
    busy: false,
    lastPrediction: null,
    validationTimer: null,
    requestController: null,
  };

  const $ = (id) => document.getElementById(id);
  const els = {};

  const DEFAULT_META = {
    convs: ["GCN", "GAT", "GraphSAGE"],
    datasets: ["ESOL", "FreeSolv", "BBBP", "Clintox"],
    feat_levels: ["base", "a", "b", "c"],
    feat_labels: {
      base: "基线特征（9 维）",
      a: "原子增强（27 维）",
      b: "原子 + 键特征",
      c: "原子 + 键 + MACCS 指纹",
    },
    task_labels: {
      ESOL: "水溶解度 logS",
      FreeSolv: "水合自由能",
      BBBP: "血脑屏障渗透",
      Clintox: "临床毒性 CT_TOX",
    },
    unit_labels: {
      ESOL: "logS",
      FreeSolv: "kcal/mol",
      BBBP: "概率",
      Clintox: "概率",
    },
    trained: [],
    ensemble_ok: [],
    hetero_ok: [],
    hetero_n: 36,
    mc_t: 30,
    ensemble_n: 4,
  };

  const METHOD_HINTS = {
    single: "单模型直接推理，速度快，适合快速筛选。",
    hetero: "9 配置（3 架构 × 3 特征档）× 4 种子等权，跨架构互补，泛化最稳（课题最终方案）。",
    mc: "多次开启 Dropout 采样，估计预测离散程度与近似区间。",
    ensemble: "聚合同架构多随机种子模型，用成员分歧估计不确定性。",
    attention: "提取 GAT 多头注意力系数，观察模型关注的化学键。",
  };

  function cacheElements() {
    [
      "systemState", "systemStateText", "smiles", "smilesWrap", "smilesStatus",
      "validationMark", "charCount", "dataset", "conv", "feat", "featNote",
      "methodHint", "explainRow", "withExplain", "availability", "availabilityText",
      "runBtn", "llmBtn", "llmTag", "llmLoadingText",
      "errorMessage", "resultEmpty", "resultLoading",
      "loadingTitle", "loadingText", "resultContent", "resultHeadline", "methodBadge",
      "predictionValue", "predictionUnit", "predictionContext", "resultTask",
      "resultConv", "resultRuntime", "uncertaintyCard", "confidencePill",
      "uncertaintyRange", "uncertaintyMetricLabel", "uncertaintyMetric",
      "intervalTrack", "intervalFill", "intervalPoint", "uncertaintyNote",
      "structureCard", "structureImage", "structureNote", "attentionCard", "attentionImage",
      "attentionTag", "headGrid", "comboCard", "comboImage", "edgeCard", "edgeList",
      "llmCard", "llmLoading", "llmContent", "llmText", "functionalGroups", "llmModelNote",
      "consistencyWarning", "historyBody", "historyEmpty", "clearHistoryBtn",
      "explainMetaCard", "consistencyPill", "consistencyJaccard", "consistencySpearman",
      "consistencyStd", "functionalGroupList", "explainMetaNote", "modelVersionLine"
    ].forEach((id) => { els[id] = $(id); });
  }

  function selectedMethod() {
    return document.querySelector('input[name="method"]:checked')?.value || "single";
  }

  function taskType(dataset) {
    return dataset === "ESOL" || dataset === "FreeSolv" ? "regression" : "classification";
  }

  function metaLabel(collection, key, fallback) {
    return state.meta?.[collection]?.[key] || fallback || key;
  }

  function trainedKey(conv, dataset, feat) {
    return `${conv}|${dataset}|${feat}`;
  }

  function isTrained(conv, dataset, feat) {
    return Boolean(state.meta?.trained?.includes(trainedKey(conv, dataset, feat)));
  }

  function isEnsembleReady(conv, dataset, feat) {
    return Boolean(state.meta?.ensemble_ok?.includes(trainedKey(conv, dataset, feat)));
  }

  function setSystemState(text, kind = "") {
    els.systemStateText.textContent = text;
    els.systemState.className = `system-state${kind ? ` is-${kind}` : ""}`;
  }

  function setAvailability(kind, text) {
    els.availability.className = `availability${kind ? ` is-${kind}` : ""}`;
    els.availabilityText.textContent = text;
  }

  function showError(message = "") {
    els.errorMessage.textContent = message;
    els.errorMessage.hidden = !message;
  }

  function populateSelect(select, values, labeler) {
    const previous = select.value;
    select.replaceChildren();
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = labeler(value);
      select.appendChild(option);
    });
    if (values.includes(previous)) select.value = previous;
  }

  function populateFromMeta() {
    const meta = state.meta || DEFAULT_META;
    populateSelect(els.dataset, meta.datasets, (value) => meta.task_labels?.[value] || value);
    populateSelect(els.conv, meta.convs, (value) => value);
    populateSelect(els.feat, meta.feat_levels, (value) => meta.feat_labels?.[value] || value);
  }

  async function loadMeta() {
    try {
      state.meta = await fetchJSON("api/meta", undefined, 15000);
      populateFromMeta();
      const count = state.meta.trained?.length || 0;
      setSystemState(`${count} 个模型组合可用`, "ready");
      if (els.modelVersionLine) {
        const v = state.meta.model_version || {};
        els.modelVersionLine.textContent = `模型版本 · ${v.tag || "unknown"}${v.n_checkpoints ? ` · ${v.n_checkpoints} ckpt` : ""}`;
      }
    } catch (error) {
      state.meta = { ...DEFAULT_META };
      populateFromMeta();
      setSystemState("模型状态未知", "error");
      showError(`无法读取模型状态：${error.message}`);
    }
    refreshUI();
  }

  function currentFeat() {
    return els.feat.value || "base";
  }

  function refreshUI() {
    const method = selectedMethod();
    const dataset = els.dataset.value;
    const featureSelect = els.feat;
    const isAttention = method === "attention";
    const isHetero = method === "hetero";

    document.querySelectorAll(".method-option").forEach((option) => {
      const radio = option.querySelector('input[type="radio"]');
      option.classList.toggle("is-active", radio.checked);
    });

    if (isAttention) {
      els.conv.value = "GAT";
      els.conv.disabled = true;
      featureSelect.value = "base";
      featureSelect.disabled = true;
      els.explainRow.hidden = true;
      els.withExplain.checked = false;
    } else if (isHetero) {
      els.conv.disabled = true;
      featureSelect.disabled = true;
      featureSelect.value = "base";
      // 异质集成同样提供结构归因图（9 配置 seed42 主成员掩码平均），开关保留
      els.explainRow.hidden = false;
      els.withExplain.checked = true;
    } else {
      els.conv.disabled = false;
      featureSelect.disabled = false;
      els.explainRow.hidden = false;
      els.withExplain.checked = true;
    }

    const meta = state.meta || DEFAULT_META;
    Array.from(featureSelect.options).forEach((option) => {
      const enabled = (isAttention || isHetero)
        ? option.value === "base"
        : !state.meta || isTrained(els.conv.value, dataset, option.value);
      option.disabled = !enabled;
    });

    if (isAttention || isHetero) {
      featureSelect.value = "base";
    } else if (featureSelect.selectedOptions[0]?.disabled) {
      const fallback = Array.from(featureSelect.options).find((option) => !option.disabled);
      if (fallback) featureSelect.value = fallback.value;
    }

    const feat = currentFeat();
    els.featNote.textContent = isAttention ? "固定基线特征" : "";
    els.methodHint.textContent = METHOD_HINTS[method];

    if (!state.meta) {
      setAvailability("", "正在检查当前模型组合...");
      return;
    }

    if (isHetero) {
      const ok = (state.meta.hetero_ok || []).includes(dataset);
      setAvailability(
        ok ? "ok" : "warn",
        ok
          ? `${dataset} · 9 配置 × 4 种子 = 36 成员异质集成可用`
          : `${dataset} 的异质集成成员不全，请先训练全部 9 配置与 3 个额外种子`
      );
      return;
    }

    if (!isTrained(els.conv.value, dataset, "base")) {
      setAvailability("error", `${els.conv.value} × ${dataset} 的基线模型缺失。`);
    } else if (method === "ensemble" && !isEnsembleReady(els.conv.value, dataset, feat)) {
      setAvailability("warn", "该组合的集成成员不足，建议改用单模型或 MC-Dropout。");
    } else if (!isAttention && !isTrained(els.conv.value, dataset, feat)) {
      setAvailability("warn", "当前特征等级不可用，已自动切换到可用组合。");
    } else {
      const featureName = meta.feat_labels?.[feat] || feat;
      setAvailability("ok", `${els.conv.value} × ${dataset} × ${featureName} 可用`);
    }
  }

  function updateCharacterCount() {
    els.charCount.textContent = `${els.smiles.value.length} / 512`;
  }

  function resetValidation() {
    state.validated = null;
    els.smilesWrap.className = "smiles-control";
    els.smilesStatus.className = "field-status";
    els.smilesStatus.textContent = "";
  }

  function debounceValidation() {
    clearTimeout(state.validationTimer);
    state.validationTimer = setTimeout(validateSmiles, 320);
  }

  async function validateSmiles() {
    const smiles = els.smiles.value.trim();
    if (!smiles) {
      resetValidation();
      return;
    }
    if (smiles.length > 512) {
      state.validated = { input: smiles, ok: false };
      els.smilesWrap.className = "smiles-control is-invalid";
      els.smilesStatus.className = "field-status is-invalid";
      els.smilesStatus.textContent = "SMILES 过长";
      return;
    }
    if (state.validated?.input === smiles) return;

    try {
      const data = await fetchJSON("api/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
      }, 15000);
      state.validated = { input: smiles, ok: data.ok };
      if (data.ok) {
        els.smilesWrap.className = "smiles-control is-valid";
        els.smilesStatus.className = "field-status is-valid";
        els.smilesStatus.textContent = data.canonical && data.canonical !== smiles
          ? `合法 · ${data.canonical}` : "SMILES 合法";
      } else {
        els.smilesWrap.className = "smiles-control is-invalid";
        els.smilesStatus.className = "field-status is-invalid";
        els.smilesStatus.textContent = data.error || "无法解析";
      }
    } catch (error) {
      resetValidation();
    }
  }

  function setRunLoading(loading, label = "运行分析") {
    state.busy = loading;
    els.runBtn.disabled = loading;
    els.llmBtn.disabled = loading;
    els.runBtn.classList.toggle("is-loading", loading);
    els.runBtn.querySelector(".button-label").textContent = loading ? label : "运行分析";
  }

  function showLoading(title, text) {
    els.resultEmpty.hidden = true;
    els.resultContent.hidden = true;
    els.resultLoading.hidden = false;
    els.loadingTitle.textContent = title;
    els.loadingText.textContent = text;
  }

  function showEmpty() {
    els.resultLoading.hidden = true;
    els.resultContent.hidden = true;
    els.resultEmpty.hidden = false;
  }

  async function fetchJSON(url, options = {}, timeout = 180000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      let data;
      try {
        data = await response.json();
      } catch (error) {
        throw new Error(`服务返回了无法解析的响应（HTTP ${response.status}）`);
      }
      if (!response.ok) {
        throw new Error(data?.error || `请求失败（HTTP ${response.status}）`);
      }
      if (data?.error) throw new Error(data.error);
      return data;
    } finally {
      clearTimeout(timer);
    }
  }

  function formatNumber(value, digits = 4) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : "--";
  }

  function setImage(imageElement, base64) {
    if (!base64) return false;
    imageElement.src = `data:image/png;base64,${base64}`;
    return true;
  }

  function renderPrediction(data, elapsedMs) {
    const method = selectedMethod();
    const dataset = els.dataset.value;
    const feat = currentFeat();
    const isClassification = data.task_type === "classification";
    const value = Number(data.prediction);

    els.resultHeadline.textContent = `${dataset} 预测`;
    els.methodBadge.textContent = data.method || (
      method === "mc" ? "MC-Dropout"
        : method === "hetero" ? "异质集成"
          : method === "ensemble" ? "同构集成" : "单模型");
    els.predictionValue.textContent = formatNumber(value);
    els.predictionUnit.textContent = isClassification ? "probability" : (data.unit || metaLabel("unit_labels", dataset, ""));
    els.predictionContext.textContent = isClassification
      ? `${formatNumber(value * 100, 1)}% 阳性概率 · 阈值 0.50`
      : `${data.task_label || metaLabel("task_labels", dataset, dataset)} · ${data.unit || metaLabel("unit_labels", dataset, "")}`;
    els.resultTask.textContent = data.task_label || metaLabel("task_labels", dataset, dataset);
    els.resultConv.textContent = data.feat === "hetero"
      ? "3 架构 × 3 特征档（9 配置）"
      : `${els.conv.value} · ${metaLabel("feat_labels", feat, feat)}`;
    els.resultRuntime.textContent = `${(elapsedMs / 1000).toFixed(1)} s`;

    renderUncertainty(data, isClassification);
    renderStructure(data);
    renderExplainMeta(data);
    hideAttentionResult();
    els.llmCard.hidden = true;
    els.resultLoading.hidden = true;
    els.resultEmpty.hidden = true;
    els.resultContent.hidden = false;
    state.lastPrediction = data;
  }

  function renderUncertainty(data, isClassification) {
    if (data.ci90 || data.std !== undefined || data.entropy !== undefined) {
      els.uncertaintyCard.hidden = false;
      if (data.ci90) {
        const [low, high] = data.ci90;
        els.confidencePill.textContent = "90% 近似区间";
        els.uncertaintyRange.textContent = `[${formatNumber(low, 3)}, ${formatNumber(high, 3)}]`;
        els.uncertaintyMetricLabel.textContent = "标准差 σ";
        els.uncertaintyMetric.textContent = formatNumber(data.std, 4);
        els.uncertaintyNote.textContent = "基于采样离散度估计；区间覆盖率取决于数据分布与模型校准。";
        const span = Math.max(Number(high) - Number(low), 1e-9);
        const point = Math.min(100, Math.max(0, ((Number(data.prediction) - Number(low)) / span) * 100));
        els.intervalTrack.hidden = false;
        els.intervalFill.style.width = "90%";
        els.intervalFill.style.left = "5%";
        els.intervalPoint.style.left = `${5 + point * 0.9}%`;
      } else if (data.entropy !== undefined) {
        els.confidencePill.textContent = isClassification ? "分类不确定性" : "经验估计";
        els.uncertaintyRange.textContent = formatNumber(data.entropy, 4);
        els.uncertaintyMetricLabel.textContent = "方法";
        els.uncertaintyMetric.textContent = data.method || "MC-Dropout";
        els.uncertaintyNote.textContent = "熵越高，表示模型对当前分类判断越不确定。";
        els.intervalTrack.hidden = true;
      } else {
        els.confidencePill.textContent = "经验估计";
        els.uncertaintyRange.textContent = formatNumber(data.std, 4);
        els.uncertaintyMetricLabel.textContent = "方法";
        els.uncertaintyMetric.textContent = data.method || "同构集成";
        els.uncertaintyNote.textContent = "数值为模型成员或采样之间的预测离散度。";
        els.intervalTrack.hidden = true;
      }
    } else {
      els.uncertaintyCard.hidden = true;
    }
  }

  function renderStructure(data) {
    const hasImage = setImage(els.structureImage, data.image);
    els.structureCard.hidden = !hasImage;
    if (els.structureNote) {
      els.structureNote.textContent = data.attribution || "";
      els.structureNote.hidden = !data.attribution;
    }
  }

  function dashNumber(value, digits = 3) {
    return (value === null || value === undefined) ? "--" : formatNumber(value, digits);
  }

  function renderExplainMeta(data) {
    const consistency = data.consistency;
    const fg = data.functional_groups;
    if (!consistency && !fg) {
      els.explainMetaCard.hidden = true;
      return;
    }
    els.explainMetaCard.hidden = false;

    const level = consistency?.level || "未评估";
    els.consistencyPill.textContent = `${level}一致性`;
    els.consistencyPill.className = `consistency-pill ${
      level === "高" ? "is-high"
        : level === "中" ? "is-medium"
          : level === "低" ? "is-low" : "is-unknown"}`;
    els.consistencyJaccard.textContent = dashNumber(consistency?.jaccard_top10_atom, 3);
    els.consistencySpearman.textContent = dashNumber(consistency?.spearman_atom, 3);
    els.consistencyStd.textContent = dashNumber(consistency?.pred_std, 3);

    els.functionalGroupList.replaceChildren();
    const groups = (fg && fg.top3) || [];
    if (groups.length) {
      groups.forEach((group, index) => {
        const chip = document.createElement("span");
        chip.className = "fg-chip";
        const rank = document.createElement("b");
        rank.textContent = `#${index + 1}`;
        const name = document.createElement("span");
        name.textContent = group.group;
        const score = document.createElement("i");
        score.textContent = Number.isFinite(Number(group.atom_importance_mean))
          ? Number(group.atom_importance_mean).toFixed(3) : "--";
        chip.append(rank, name, score);
        els.functionalGroupList.appendChild(chip);
      });
      const caption = document.createElement("small");
      caption.textContent = "Top-3 由 GNNExplainer 归因排序；官能团由 RDKit 识别";
      els.functionalGroupList.appendChild(caption);
    } else if (fg && fg.detected?.length) {
      fg.detected.slice(0, 8).forEach((nameText) => {
        const chip = document.createElement("span");
        chip.className = "fg-chip is-muted";
        chip.textContent = nameText;
        els.functionalGroupList.appendChild(chip);
      });
      const caption = document.createElement("small");
      caption.textContent = "未生成掩码，仅展示 RDKit 识别的官能团（无重要性）";
      els.functionalGroupList.appendChild(caption);
    }
    els.explainMetaNote.textContent = (fg && fg.note) || (consistency && consistency.note) || "";
  }

  function hideAttentionResult() {
    els.attentionCard.hidden = true;
    els.headGrid.hidden = true;
    els.headGrid.replaceChildren();
    els.comboCard.hidden = true;
    els.edgeCard.hidden = true;
    els.edgeList.replaceChildren();
  }

  function renderAttention(data, elapsedMs) {
    const dataset = els.dataset.value;
    const method = "注意力可视化";
    const isClassification = taskType(dataset) === "classification";
    const value = Number(data.prediction);

    els.resultHeadline.textContent = `${dataset} 注意力分析`;
    els.methodBadge.textContent = method;
    els.predictionValue.textContent = formatNumber(value);
    els.predictionUnit.textContent = isClassification ? "probability" : metaLabel("unit_labels", dataset, "");
    els.predictionContext.textContent = isClassification
      ? `${formatNumber(value * 100, 1)}% 阳性概率 · GAT 注意力解释`
      : `${data.task_label || metaLabel("task_labels", dataset, dataset)} · GAT 注意力解释`;
    els.resultTask.textContent = data.task_label || metaLabel("task_labels", dataset, dataset);
    els.resultConv.textContent = `GAT · layer ${data.layer ?? 0}`;
    els.resultRuntime.textContent = `${(elapsedMs / 1000).toFixed(1)} s`;

    els.uncertaintyCard.hidden = true;
    els.structureCard.hidden = true;
    els.llmCard.hidden = true;
    els.explainMetaCard.hidden = true;
    const hasImage = setImage(els.attentionImage, data.image);
    els.attentionCard.hidden = !hasImage;
    if (!hasImage && data.warning) showError(data.warning);
    els.attentionTag.textContent = `${data.heads || 1} heads`;
    els.attentionImage.alt = `${dataset} GAT 注意力分子图`;

    els.headGrid.replaceChildren();
    (data.per_head_images || []).forEach((image, index) => {
      const card = document.createElement("div");
      card.className = "head-card";
      const label = document.createElement("span");
      label.textContent = `HEAD ${index + 1}`;
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${image}`;
      img.alt = `注意力头 ${index + 1}`;
      card.append(label, img);
      els.headGrid.appendChild(card);
    });
    els.headGrid.hidden = !els.headGrid.children.length;

    const hasCombo = setImage(els.comboImage, data.combo_image);
    els.comboCard.hidden = !hasCombo;

    els.edgeList.replaceChildren();
    (data.top8_edges || []).forEach((edge, index) => {
      if (!edge?.edge) return;
      const chip = document.createElement("span");
      chip.className = "edge-chip";
      chip.append(`#${index + 1}  ${edge.edge[0]} → ${edge.edge[1]}  `);
      const strong = document.createElement("b");
      strong.textContent = `α ${formatNumber(edge.alpha, 4)}`;
      chip.appendChild(strong);
      els.edgeList.appendChild(chip);
    });
    els.edgeCard.hidden = !els.edgeList.children.length;

    els.resultLoading.hidden = true;
    els.resultEmpty.hidden = true;
    els.resultContent.hidden = false;
  }

  function uncertaintySummary(data) {
    if (data.ci90) return `[${formatNumber(data.ci90[0], 3)}, ${formatNumber(data.ci90[1], 3)}]`;
    if (data.entropy !== undefined) return `H=${formatNumber(data.entropy, 3)}`;
    if (data.std !== undefined) return `σ=${formatNumber(data.std, 3)}`;
    return "—";
  }

  async function runAnalysis() {
    if (state.busy) return;
    const smiles = els.smiles.value.trim();
    const dataset = els.dataset.value;
    const method = selectedMethod();
    const conv = method === "attention" ? "GAT" : els.conv.value;
    const feat = (method === "attention" || method === "hetero") ? "base" : currentFeat();

    showError();
    if (!smiles) {
      showError("请输入 SMILES 分子式。");
      els.smiles.focus();
      return;
    }
    if (state.validated?.input === smiles && !state.validated.ok) {
      showError("当前 SMILES 未通过校验，请修正后再运行。");
      return;
    }

    const startedAt = performance.now();
    setRunLoading(true, method === "attention" ? "正在渲染..." : "正在分析...");
    showLoading(
      method === "attention" ? "正在提取注意力" : "正在分析分子",
      method === "attention"
        ? "将生成整体注意力图与逐头对比，大分子可能需要更久。"
        : method === "hetero"
          ? "正在加载 36 个成员模型并推理，首次调用会慢一些。"
          : "正在完成图神经网络推理、结构解释与结果整理。"
    );

    try {
      let data;
      if (method === "attention") {
        data = await fetchJSON("api/attention", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles, dataset, per_head: true, layer: 0 }),
        }, 150000);
      } else {
        data = await fetchJSON("api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            smiles,
            dataset,
            conv,
            method,
            feat,
            explain: els.withExplain.checked,
          }),
        }, 200000);
      }

      const elapsedMs = performance.now() - startedAt;
      if (method === "attention") {
        renderAttention(data, elapsedMs);
        pushHistory({
          smiles, dataset, conv: "GAT", feat: "base", method: "注意力",
          prediction: data.prediction, uncertainty: `${data.heads || 1} heads`,
        });
      } else {
        renderPrediction(data, elapsedMs);
        pushHistory({
          smiles, dataset, conv, feat, method: data.method || method,
          prediction: data.prediction, uncertainty: uncertaintySummary(data),
        });
      }
    } catch (error) {
      showEmpty();
      const message = error.name === "AbortError"
        ? "请求超时。解释计算较慢，可关闭结构解释图后重试。"
        : error.message;
      showError(message);
    } finally {
      setRunLoading(false);
    }
  }

  function prepareLlmPanel() {
    if (els.resultContent.hidden) {
      els.resultEmpty.hidden = true;
      els.resultLoading.hidden = true;
      els.resultContent.hidden = false;
      els.resultHeadline.textContent = "化学解释";
      els.methodBadge.textContent = "LLM";
      els.predictionValue.textContent = "--";
      els.predictionUnit.textContent = "";
      els.predictionContext.textContent = "由 GNNExplainer 结构证据驱动";
      els.resultTask.textContent = metaLabel("task_labels", els.dataset.value, els.dataset.value);
      els.resultConv.textContent = `${els.conv.value} · ${metaLabel("feat_labels", currentFeat(), currentFeat())}`;
      els.resultRuntime.textContent = "—";
      els.uncertaintyCard.hidden = true;
      els.structureCard.hidden = true;
      hideAttentionResult();
    }
    els.llmCard.hidden = false;
    els.llmLoading.hidden = false;
    els.llmContent.hidden = true;
  }

  async function runLlmExplanation() {
    const smiles = els.smiles.value.trim();
    if (!smiles || state.busy) return;
    const dataset = els.dataset.value;
    const method = selectedMethod();
    const conv = method === "attention" ? "GAT" : els.conv.value;
    const feat = (method === "attention" || method === "hetero") ? "base" : currentFeat();
    const shownPrediction = state.lastPrediction?.prediction;
    showError();
    prepareLlmPanel();
    state.busy = true;
    els.llmBtn.disabled = true;
    els.runBtn.disabled = true;
    const labelNode = els.llmBtn.querySelector("span:last-child");
    const originalLabel = labelNode?.textContent || "LLM 化学解释";
    if (labelNode) labelNode.textContent = "解释生成中...";

    try {
      const data = await fetchJSON("api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          smiles, dataset, conv, feat, method,
          prediction: typeof shownPrediction === "number" ? shownPrediction : undefined,
        }),
      }, 340000);

      if (!state.lastPrediction && data.prediction !== undefined) {
        els.predictionValue.textContent = formatNumber(data.prediction);
        els.predictionUnit.textContent = taskType(dataset) === "classification" ? "probability" : metaLabel("unit_labels", dataset, "");
        els.predictionContext.textContent = "预测值来自当前选择模型";
      }
      if (els.llmModelNote) {
        els.llmModelNote.textContent = data.model_note
          ? `归因依据：${data.model_note} · 预测 ${formatNumber(data.prediction)}${taskType(dataset) === "classification" ? "" : ` ${metaLabel("unit_labels", dataset, "")}`}`
          : "";
      }
      els.llmText.textContent = data.explanation || "未返回解释文本。";
      els.llmTag.textContent = data.provider === "rule_template"
        ? "规则模板 · 离线降级" : "DeepSeek · API";
      els.functionalGroups.replaceChildren();
      (data.top_groups || []).forEach((group) => {
        const chip = document.createElement("span");
        chip.className = "group-chip is-ranked";
        chip.textContent = `${group.group} · ${formatNumber(group.score, 3)}`;
        els.functionalGroups.appendChild(chip);
      });
      (data.functional_groups || []).forEach((group) => {
        const chip = document.createElement("span");
        chip.className = "group-chip is-muted";
        chip.textContent = group;
        els.functionalGroups.appendChild(chip);
      });
      const missing = data.consistency?.mentioned_undetected || [];
      els.consistencyWarning.hidden = !missing.length;
      els.consistencyWarning.textContent = missing.length
        ? `一致性校验：解释中提及但分子中未检出 ${missing.join("、")}，请以 RDKit 结果为准。`
        : "";
      if (data.provider === "rule_template" && data.fallback_reason) {
        els.consistencyWarning.hidden = false;
        els.consistencyWarning.textContent += `${els.consistencyWarning.textContent ? " " : ""}已降级为规则模板（${data.fallback_reason}）。`;
      }
      els.llmLoading.hidden = true;
      els.llmContent.hidden = false;
    } catch (error) {
      els.llmLoading.hidden = true;
      els.llmContent.hidden = false;
      els.llmText.textContent = "解释生成失败。";
      els.functionalGroups.replaceChildren();
      els.consistencyWarning.hidden = true;
      if (els.llmModelNote) els.llmModelNote.textContent = "";
      showError(error.name === "AbortError" ? "LLM 解释等待超时，请稍后重试。" : error.message);
    } finally {
      state.busy = false;
      els.llmBtn.disabled = false;
      els.runBtn.disabled = false;
      if (labelNode) labelNode.textContent = originalLabel || "LLM 化学解释";
    }
  }

  function loadHistory() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed.slice(0, MAX_HISTORY) : [];
    } catch (error) {
      return [];
    }
  }

  function storeHistory() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.history));
    } catch (error) {
      // History is a convenience; private browsing may disable localStorage.
    }
  }

  function pushHistory(entry) {
    state.history.unshift({ ...entry, timestamp: Date.now() });
    state.history = state.history.slice(0, MAX_HISTORY);
    storeHistory();
    renderHistory();
  }

  function methodShortName(method) {
    if (!method) return "—";
    if (method.includes("异质")) return "异质";
    if (method.includes("集成")) return "同构";
    if (method.includes("MC")) return "MC";
    if (method.includes("注意")) return "注意力";
    if (method.includes("LLM")) return "LLM";
    return "单模型";
  }

  function renderHistory() {
    els.historyBody.replaceChildren();
    els.historyEmpty.hidden = state.history.length > 0;
    state.history.forEach((entry, index) => {
      const row = document.createElement("tr");
      const values = [
        entry.smiles,
        entry.dataset,
        entry.conv,
        methodShortName(entry.method),
        formatNumber(entry.prediction, 4),
        entry.uncertainty || "—",
      ];
      values.forEach((value, column) => {
        const cell = document.createElement("td");
        if (column === 0) {
          cell.title = value;
          cell.textContent = value;
        } else if (column === 4) {
          const strong = document.createElement("strong");
          strong.textContent = value;
          cell.appendChild(strong);
        } else {
          cell.textContent = value;
        }
        row.appendChild(cell);
      });
      const actionCell = document.createElement("td");
      const replay = document.createElement("button");
      replay.type = "button";
      replay.className = "replay-button";
      replay.textContent = "回放";
      replay.addEventListener("click", () => replayHistory(index));
      actionCell.appendChild(replay);
      row.appendChild(actionCell);
      els.historyBody.appendChild(row);
    });
  }

  function replayHistory(index) {
    const entry = state.history[index];
    if (!entry) return;
    els.smiles.value = entry.smiles;
    els.dataset.value = entry.dataset;
    els.conv.value = entry.conv;
    const method = entry.method?.includes("注意") ? "attention"
      : entry.method?.includes("异质") ? "hetero"
        : entry.method?.includes("集成") ? "ensemble"
          : entry.method?.includes("MC") ? "mc" : "single";
    if (method !== "hetero"
        && !Array.from(els.feat.options).some((option) => option.value === entry.feat)) return;
    const radio = document.querySelector(`input[name="method"][value="${method}"]`);
    if (radio) radio.checked = true;
    refreshUI();
    if (!els.feat.disabled && method !== "hetero") els.feat.value = entry.feat;
    updateCharacterCount();
    validateSmiles();
    window.scrollTo({ top: 0, behavior: "smooth" });
    runAnalysis();
  }

  function bindEvents() {
    els.smiles.addEventListener("input", () => {
      updateCharacterCount();
      resetValidation();
      debounceValidation();
    });
    els.smiles.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.isComposing) {
        event.preventDefault();
        runAnalysis();
      }
    });
    document.querySelectorAll(".example-chip").forEach((button) => {
      button.addEventListener("click", () => {
        els.smiles.value = button.dataset.smiles || "";
        updateCharacterCount();
        resetValidation();
        validateSmiles();
        els.smiles.focus();
      });
    });
    [els.dataset, els.conv, els.feat].forEach((control) => {
      control.addEventListener("change", refreshUI);
    });
    document.querySelectorAll('input[name="method"]').forEach((radio) => {
      radio.addEventListener("change", refreshUI);
    });
    els.runBtn.addEventListener("click", runAnalysis);
    els.llmBtn.addEventListener("click", runLlmExplanation);
    els.clearHistoryBtn.addEventListener("click", () => {
      state.history = [];
      storeHistory();
      renderHistory();
    });
    document.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        runAnalysis();
      }
    });
  }

  function init() {
    cacheElements();
    state.history = loadHistory();
    populateFromMeta();
    bindEvents();
    updateCharacterCount();
    renderHistory();
    refreshUI();
    validateSmiles();
    loadMeta();
  }

  document.addEventListener("DOMContentLoaded", init);
})();