
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
      warn = '
【一致性校验】LLM 提及但分子中未检出：' + d.consistency.mentioned_undetected.join('、') + '（已标记，请以 RDKit 检出为准）';
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
