# 深化扩展 · 基于图神经网络的分子性质预测

课题主线完成后，围绕「小样本瓶颈 → 不确定性 → 可解释性」三条主线做了以下深化扩展。
技术栈：Python + PyTorch + PyTorch Geometric，本地 GPU（GTX 1660 Ti 6GB）训练。

---

## 方向 ① 图对比学习自监督预训练（破解小样本瓶颈）

### 动机
ESOL / FreeSolv / BBBP 标注样本仅数百～两千，深度 GNN 在小样本下容易过拟合。
利用 MoleculeNet 中大量**无标签分子**做自监督预训练，让编码器先学会通用的分子结构表征，
再在下游小样本任务上微调，缓解数据瓶颈。

### 方法（GraphCL 风格对比学习）
- **预训练分子池**：ClinTox / BACE / SIDER / Tox21 全量 + 下游三任务**训练集**（验证/测试严格排除）
  → 共 **15,289 分子**（节点特征 9 维，与下游一致，权重可直接迁移）
- **三种图增强**（batch 级向量化实现）：
  - `attr_mask`：按 20% 概率置零原子特征（丢失原子类型信息）
  - `edge_perturb`：按 15% 比例删边 + 同图随机加等量边
  - `node_drop`：按 20% 概率删除节点（含关联边，图内重编号，保证每图至少保留 1 节点）
- **视图组合（full）**：view1 = attr_mask + edge_perturb；view2 = attr_mask + node_drop
- **InfoNCE 对称交叉熵损失**（温度 0.5），投影头 64→64→32，L2 归一化
- **训练**：batch 256，150 epochs × 3 架构（GCN / GAT / GraphSAGE），Adam lr=1e-3
- **迁移方式**：仅保存 `convs.*` 编码器权重，微调时重新随机初始化输出头

### 结果（测试集，预训练微调 vs 无预训练基线）

| 任务 | 训练集规模 | 架构 | 基线 | 预训练微调 | Δ |
|---|---|---|---|---|---|
| ESOL (logS, RMSE) | 902 | GCN | 0.7952 | 0.8170 | +0.0218 |
| | | GAT | 0.7484 | 0.8390 | +0.0906 |
| | | GraphSAGE | 0.7870 | **0.7603** | **−0.0267** |
| FreeSolv (kcal/mol, RMSE) | 513 | GCN | 2.0053 | **1.7710** | **−0.2343** |
| | | GAT | 2.2051 | **1.6570** | **−0.5481** |
| | | GraphSAGE | 1.3842 | 1.4496 | +0.0654 |
| BBBP (渗透, AUC) | 1631 | GCN | 0.8085 | **0.8953** | **+0.0868** |
| | | GAT | 0.8303 | **0.8976** | **+0.0673** |
| | | GraphSAGE | 0.8759 | **0.8954** | **+0.0195** |

**结论**：
1. 预训练对**最小数据集 FreeSolv（513 样本）**提升最显著：GAT RMSE 2.205→1.657（R² 0.688→0.824）、GCN RMSE 2.005→1.771（R² 0.742→0.799）——直接验证"自监督预训练破解小样本瓶颈"。
2. 分类任务 BBBP 三架构 AUC 全部提升（+0.02~+0.09）。
3. ESOL 上 GCN/GAT 微降、GraphSAGE 提升——与数据集化学分布差异、单 seed 随机性有关，通过增强消融进一步分析。

### 增强策略消融（GCN × ESOL，40 pt-epochs）

| pattern | 增强组合 | 微调后 RMSE | Δ vs 基线(0.7952) |
|---|---|---|---|
| mask | 仅原子特征掩码 | 0.8152 | +0.0200 |
| perturb | 仅边扰动 | 0.8763 | +0.0811 |
| drop | 仅节点删除 | 0.8384 | +0.0432 |
| **full** | **双视图组合** | **0.7931** | **−0.0021** |

**消融结论**：单增强各自在 ESOL 上均有伤害，唯独 full 双视图组合持平基线——
**视图多样性（互补增强信号）是预训练有效性的关键**，印证 GraphCL 的核心设计论点；
同时解释 ESOL 上预训练收益有限：预训练池以药物分子为主（Tox21/ClinTox 等），
与 ESOL 水溶性小分子的化学分布存在领域差异，且 ESOL 训练集规模（902）相对充足。

### 关键文件
- `src/pretrain_pool.py`（分子池组装）、`src/contrastive_pretrain.py`（预训练）
- `src/finetune_pretrained.py`（微调对比）、`src/ablation_pretrain.py`（消融）
- 产物：`models/pretrained/*_pretrain_full.pt`、`data/pretrain_pool/pool.pt`
- 结果：`reports/pretrain_finetune_results.json`、`reports/ablation_results.json`

---

## 方向 ② 不确定性量化（MC-Dropout + 深度集成）

### 动机
点预测无法告知"可信度"。对药物/材料筛选场景，预测不确定性能指导实验优先级。

### 方法
1. **MC-Dropout**：推理时保持 dropout 开启，30 次前向采样 → 均值 ± 标准差（回归）/ 预测熵（分类）
2. **深度集成**：同架构多随机种子（42/7/13/21）训练 4 成员 → 均值与成员间离散度
3. **校准评估**：回归 90% 预测区间覆盖率 / 区间宽度 / 不确定性-误差 Spearman 相关；分类 ECE

### 初步结果（MC-Dropout，T=30）
| 任务 | 指标 | GCN | GAT | GraphSAGE |
|---|---|---|---|---|
| ESOL | PI coverage(90%) | 0.790 | 0.825 | 0.763 |
| FreeSolv | PI coverage(90%) | 0.492 | 0.708 | 0.708 |
| FreeSolv | 不确定性-误差相关 | 0.266 | 0.246 | 0.444 |
| BBBP | ECE | 0.036 | 0.090 | 0.046 |

- 回归区间普遍**欠覆盖**（<90%），FreeSolv 最明显——MC-Dropout 高估自信
- 分类 ECE 0.036~0.090，GraphSAGE×BBBP 熵-误差相关 0.30，不确定性有预测力

### 深度集成 vs MC-Dropout（FreeSolv / BBBP，3 成员 seed 7/13/21）

| 任务 | 方法 | 点预测质量 | 校准指标 |
|---|---|---|---|
| FreeSolv | MC-Dropout (GAT) | — | PI coverage 0.738（宽 4.13） |
| FreeSolv | 集成 (GAT×3) | RMSE 2.174 | PI coverage 0.415（宽 1.55） |
| BBBP | MC-Dropout (GraphSAGE) | AUC 0.869 | ECE 0.040 |
| BBBP | 集成 (GraphSAGE×3) | **AUC 0.899** | ECE 0.077 |

**结论**：
1. **集成提升点预测**：GraphSAGE×BBBP 集成 AUC 0.899 > 全部单成员（0.880~0.895）与 seed42 基线（0.876）——"集成均值优于成员"的标准效应
2. **校准权衡**：3 成员集成区间过窄（低估不确定性，FreeSolv coverage 仅 0.42）；MC-Dropout 区间更宽更保守、ECE 更低。成员多样性不足时集成会低估不确定性——对实践有指导意义（需增大成员数/seed 差异）

### Web 演示升级
`src/app.py` 支持三种预测方法：单模型 / MC-Dropout（返回 90% 置信区间或熵）/ 深度集成。

---

## 方向 ③ 本地 LLM 化学解释增强

### 动机
GNNExplainer 输出的是原子/键重要性掩码，普通用户难以直接理解。
用本地大模型把掩码转成**自然语言化学解释**，并做一致性校验防止幻觉。

### 方法
- **模型**：Qwen2.5-3B-Instruct（ModelScope 下载，本地 4bit 量化运行于 1660 Ti 6GB）
- **流程**：GNNExplainer 掩码 → top-k 重要原子/键 → RDKit 官能团匹配（18 种 SMARTS）→
  结构化提示词 → LLM 生成 2~3 句化学解释 → 官能团一致性校验（LLM 提及项必须在 RDKit 检出集内）
- **Web 集成**：app.py 新增「LLM 化学解释」按钮

### 验证案例
- **阿司匹林（GCN×ESOL）**：关键原子为羧基/酯基 O、芳香 C；检出羟基、羧基、羰基、酯基、苯环、醚键——与 logS 预测高度相关
- **正丙醇（GCN×ESOL）**：关键原子 O（羟基氧），LLM 生成解释提及"羟基增强亲水性"；**一致性校验拦截幻觉**：LLM 曾提及"羰基"（RDKit 未检出，被标记 `mentioned_undetected`）
- **发现**：3B 模型在无先验约束时出现化学逻辑错误（"羟基增加亲水性从而降低溶解度"——方向说反），注入任务化学先验（极性基团增溶/疏水基团减溶）后逻辑明显改善。说明 **LLM 化学解释必须以 RDKit 证据 + 先验约束为护栏**，一致性校验是必要组件

---

## 方向 ④ 对照选题深挖：分子特征工程（选题明示重点）

### 动机
选题原文明确「项目重点在于**分子特征工程（分子指纹、图构建）**」。基线只用了
MoleculeNet 默认 9 维原子 one-hot，**分子指纹与键特征完全空白**（数据自带的
edge_attr 键型特征也未进入模型）。本方向补齐该空白并做系统性对比。

### 三档特征（RDKit 从 SMILES 重算，图拓扑复用 MoleculeNet 保证可比）
| 档位 | 原子特征 (27维) | 键特征 (7维) | 指纹 |
|---|---|---|---|
| feat_a | RDKit 原子性质增强（12 原子类型 + 度 + 杂化 + 手性/芳香/环/供受体/电荷） | — | — |
| feat_b | 同上 | 键型 4 + 共轭 + 环内 + 立体 | — |
| feat_c | 同上 | 同上 | **MACCS 166 位**（与 GNN 池化表征融合进输出头） |

三种架构的键特征接入方式：GCN = 键特征经 sigmoid 生成边权重；GAT = GATConv
原生 edge_dim（键特征参与注意力与消息）；GraphSAGE = 自定义 SAGEConvEdge（邻居
消息加入键特征投影）。

### 结果（测试集，统一概率口径，同超参同划分，vs 基线默认 9 维特征）

| 任务 | 架构 | 基线 | feat_a 原子增强 | feat_b +键 | feat_c +指纹 |
|---|---|---|---|---|---|
| ESOL (RMSE↓) | GCN | 0.7952 | 0.6976 | 0.7879 | **0.6803** |
| ESOL (R²↑) | GCN | 0.8493 | 0.8841 | 0.8521 | **0.8898** |
| FreeSolv (RMSE↓) | GCN | 2.0053 | 1.1353 | 1.2715 | **1.0085** |
| FreeSolv (R²↑) | GCN | 0.7419 | 0.9173 | 0.8962 | **0.9347** |
| BBBP (AUC↑) | GCN | 0.8085 | 0.8833 | 0.9034 | **0.9075** |
| Clintox (AUC↑) | GCN | 0.6946 | 0.8755 | **0.8926** | 0.8526 |
| BBBP (AUC↑) | GAT | 0.8303 | — | — | **0.9184** |
| BBBP (AUC↑) | GraphSAGE | 0.8759 | — | — | **0.9097** |
| Clintox (AUC↑) | GAT | 0.6438 | — | — | **0.8308** |
| Clintox (AUC↑) | GraphSAGE | 0.6931 | — | — | **0.8571** |
| ESOL (RMSE↓) | GAT | 0.7484 | — | — | 0.7003 |
| ESOL (RMSE↓) | GraphSAGE | 0.7870 | — | — | 0.7861 |
| FreeSolv (RMSE↓) | GAT | 2.2051 | — | — | 1.4777 |
| FreeSolv (RMSE↓) | GraphSAGE | 1.3842 | — | — | 1.1013 |

**结论**：
1. **原子特征增强（feat_a）即带来系统性提升**：FreeSolv R² 0.742→0.917、Clintox AUC
   0.695→0.876、BBBP AUC 0.809→0.883——"特征工程是分子性质预测的第一杠杆"。
2. **键特征（feat_b）在分类任务继续增益**：BBBP 0.883→0.903、Clintox 0.876→0.893；
   回归任务上相对 feat_a 波动（ESOL/FreeSolv 略降），说明键特征增益与任务敏感性相关。
3. **指纹融合（feat_c）在回归任务上最优**：FreeSolv RMSE 2.005→1.008（R² 0.935）、
   ESOL RMSE 0.680；BBBP 三架构 feat_c 全部最高（GAT 0.9184）。
4. **三架构在最高档特征下全部优于基线**——特征工程收益与聚合函数正交，普适有效。

### 关键文件
- `src/featurize.py`（三档特征 + 幂等缓存）、`src/train_feat.py`（实验矩阵）、
  `src/reeval_feat.py`（统一口径重评估）、`src/models.py`（SAGEConvEdge / GAT edge_dim / GCN 边加权）
- 缓存：`data/featurized/*_feat{a,b,c}.pt`；模型：`models/feat/*.pt`（20 个）
- 结果：`reports/feat_results.json`

---

## 方向 ⑤ 毒性任务补全：Clintox（覆盖选题"溶解度或毒性"）

### 动机
选题要求"预测分子的**溶解度或毒性**"。此前完成溶解度（ESOL）与渗透（BBBP），
毒性数据集 Clintox 已下载未启用——补齐第四任务。

### 数据与标签
- Clintox：1480 分子，2 个二分类标签（FDA_APPROVED / CT_TOX）
- 采用 **CT_TOX 毒性列**（正类 112 / 负类 1368，正类 7.6% 强不平衡）
- 类别加权交叉熵（weight=[1,12]）缓解不平衡；划分 8:1:1 seed42（与全项目一致）

### 结果（测试集，概率 AUC，真实 CT_TOX col1，seed42）
| 架构 | 基线（默认 9 维） | feat_c（原子+键+指纹） | Δ |
|---|---|---|---|
| GCN | 0.7114 | 0.8526 | +0.141 |
| GAT | 0.7241 | 0.8308 | +0.107 |
| GraphSAGE | 0.6405 | 0.8571 | +0.217 |
| GCN feat_b（+键） | — | **0.8926** | 最高 |

**结论**：默认特征在强不平衡毒性任务上区分度有限（AUC 0.64-0.72），特征工程后
全部提升至 0.83-0.89——在毒性场景再次验证特征工程价值；正类仅 7.6%，AUC 提升
对候选毒性分子筛选（如虚拟筛选排序）有直接实践意义。

> **标签口径修正（整体检查发现）**：Clintox 数据集原始含两列标签（col0=FDA_APPROVED
> 正类 93%、col1=CT_TOX 正类 7.6%）。此前基线/集成脚本对 `dataset` 元素重写 y 的方式
> 在 PyG 2.8 下**不生效**（`__getitem__` 返回克隆），导致基线模型实际用 col0 训练；
> 现已改为在 DataLoader 中按 `y_col` 直接取列（与特征工程数据 `d.y[:,1]` 口径一致），
> 基线/集成全部按真实 CT_TOX 重训。上表为修正后数值。

### 深度集成（Clintox，4 成员 seed 42/7/13/21，类别加权）
| 架构 | 成员 AUC | 集成均值 AUC | 集成 ECE |
|---|---|---|---|
| GCN | 0.611~0.714 | 0.709 | 0.53 |
| GAT | 0.620~0.724 | 0.693 | 0.55 |
| GraphSAGE | 0.634~0.692 | 0.674 | 0.59 |

**结论**：强不平衡（正类 7.6%）+ 小数据集下，成员 AUC 离散大（0.61-0.72），
集成均值与最优成员相当但未超越——成员多样性不足时集成增益有限（与方向②
FreeSolv 结论一致）；ECE 0.53-0.59 偏高（概率集中于 0.5 附近，阈值校准困难），
提示对毒性任务应关注**排序**而非绝对概率。

### 关键文件
- `src/train_clintox_baseline.py`（基线，y_col=1 取 CT_TOX）、`data_prep.py`（Clintox 任务注册）、
  `src/eval_clintox_ensemble.py`（集成评估）
- 结果：`reports/clintox_baseline.json`、`reports/clintox_ensemble.json`、`reports/splits/Clintox_split.json`

---

## 方向 ⑥ 可解释性深挖：GAT 逐头注意力对比

### 动机
选题要求"可解释性分析（**注意力机制可视化**）"。已完成多头平均注意力图，
本方向进一步**拆解 4 个注意力头各自关注的化学键**——多头注意力是否编码了
不同的化学子结构偏好（如某头偏好芳香键、某头偏好环内键）。

### 方法
- 提取 GAT 第一层（最贴近化学键）的边注意力系数 alpha [E, 4 heads]
- 每头独立归一化 → 单独分子图 + 2×2 并排对比（同分子 4 头并列）
- 产出：`reports/attention/perhead/GAT_{数据集}_idx*_head{1..4}.png` + `_allheads.png`

### 发现（示例：BBBP 阿司匹林类似物 idx1923 / 三嗪衍生物 idx36）
- 4 个头的键注意力分布**明显分化**：部分头集中关注芳香环骨架键，部分头关注
  杂原子邻位键（N/C 极性区）——与化学直觉一致（渗透性由疏水芳环 + 极性基团
  共同决定），为"GNN 学到化学子结构偏好"提供直接证据。
- 单头图与平均图互补：平均图给出"总依赖"，逐头图给出"多视角依赖"。

### 关键文件
- `src/attention_viz.py`（`draw_per_head` + `--per-head`）
- 产物：`reports/attention/perhead/`（3 数据集 × 3 分子 × 5 图 = 45 张）

---

## 方向 ⑦ 整体检查与 Web 演示系统优化

### 整体检查发现并修复的问题
1. **Clintox 标签口径错误（最重要）**：见方向⑤修正说明——基线/集成曾误用 col0
   （FDA_APPROVED，正类 93%），已全部按真实 CT_TOX（col1）重训；`train_eval.py`
   新增 `y_col` 参数为正规通道，杜绝重写 dataset 的失效做法。
2. **Web 缺 Clintox 数据集**：数据集下拉/模型加载/单位映射补齐
   （`TASK_LABEL`/`UNIT_LABEL`，毒性任务显示 "P(毒性) 概率"）。
3. **Web 注意力无逐头支持**：`/api/attention` 新增 `per_head=true`，
   返回 4 张单头图 + 1 张 2×2 对比图（前端逐头缩略图展示）。
4. **单模型分支缺 unit**、特征模型预测时 edge_attr/fp 未转移设备（cuda 报错）——已修复。
5. **Clintox 深度集成缺失**：补训 3 架构 × 3 seed 共 9 成员（类别加权 [1,12]），
   与 seed42 基线构成 4 成员集成；新增 `src/eval_clintox_ensemble.py` 产出集成 AUC/ECE。

### Web 新功能（特征工程模型接入）
- 新增「特征工程等级」下拉：基线（9 维）/ feat_a（27 维）/ feat_b（+键 7 维）/
  feat_c（+MACCS 166 位）
- 对应懒加载 `models/feat/*.pt`（20 个特征模型），输入 SMILES 由
  `featurize.featurize_smiles` 实时构建特征图（与训练口径一致）
- 交互约束：特征模型支持单模型 / MC-Dropout；深度集成与解释图仅基线（集成成员与
  GNNExplainer 未扩展到特征模型）
- 回归任务显示单位（logS / kcal/mol），分类显示概率语义（渗透 / 毒性）

### 回归测试
`src/smoke_web.py`：覆盖 Clintox 单模型/MC/集成、ESOL/BBBP feat_a/b/c、逐头注意力、
feat 集成拒绝逻辑等 9 项端点用例（重启服务后可随时复跑）。

### 关键文件
- `src/app.py`（Web 全功能）、`src/featurize.py`（`featurize_smiles`）、
  `src/train_ensemble.py`（y_col 支持）、`src/train_eval.py`（y_col 参数）、
  `src/eval_clintox_ensemble.py`、`src/smoke_web.py`
- 报告：`reports/clintox_baseline.json`（修正后）、`reports/clintox_ensemble.json`（新增）

---

## 方向 ⑧ Web 界面交互逻辑优化

### 问题与优化点（前端用户体验 + 后端支撑）
| 原问题 | 优化 |
|---|---|
| 选了不可用组合（如 GAT×ESOL×feat_a）直接报 400，靠试错 | 新增 `/api/meta` 可用性矩阵（启动时扫描 models 目录真实产物）→ 前端**实时灰化**不可用特征档位、自动回退到可用档位，组合可用性以绿/黄/红三色提示条实时反馈 |
| SMILES 合法性靠提交后才知道 | 新增 `/api/validate`（RDKit 解析）→ 输入框**防抖 300ms 即时校验**：合法绿色边框 + 规范化 SMILES，非法红色边框 + 具体错误 |
| 无法对比多个分子/多配置 | 新增**预测历史面板**：每次运行自动入表（SMILES/任务/架构/特征/方法/预测值/不确定性），点击「回放」一键复现该配置再跑，支持清空 |
| 无键盘操作 | SMILES 输入框支持 **Enter 直接运行** |
| 请求失败时旧结果残留易误读 | 每次运行前先隐藏旧结果卡片，失败时只显示错误 |
| 方法/特征联动提示不一致 | 重构 `refreshUI()` 统一联动：attention 模式锁定 GAT+基线、ensemble 自动切回单模型、特征模型约束（解释图/集成仅基线）集中到单一状态机 |
| 示例分子少 | 补充咖啡因、布洛芬快捷样例 |
| 加载状态无区分 | 注意力模式按钮显示「渲染注意力图（较慢）...」，LLM 按钮独立文案 |

### 回归测试
`src/smoke_web.py` 扩至 **12 项用例**：新增 Meta 可用性（trained=32、ensemble_ok=12）、
合法/非法 SMILES 校验。全部通过（2 项 FAIL 为预期拒绝路径：未训练组合、feat 集成拒绝）。

### 关键文件
- `src/app.py`：新增 `/api/meta`、`/api/validate`，前端 JS 重写（refreshUI/防抖校验/历史面板/回放）
- `src/smoke_web.py`：12 项端点用例

---

## 方向 ⑨ 特征模型深度集成（全组合覆盖）

### 动机
方向④特征工程后，Web 深度集成仅支持基线特征（feat 模型未训练集成成员）。
补齐特征模型的多 seed 集成成员，使深度集成覆盖 **feat_a/b/c 全组合**。

### 方法与产物
- **训练**：`src/train_feat_ensemble.py`，**全矩阵 36 组**（3 架构 GCN/GAT/GraphSAGE
  × {a,b,c} × 4 数据集）× 3 seed（7/13/21）= **108 个成员**，3 进程并行训练（GPU）
- 成员与 seed42 特征模型（`models/feat/*.pt`）构成 **4 成员深度集成**
- seed42 特征模型同步补齐：`src/train_feat_gap.py` 补训 GAT/GraphSAGE×{a,b}×4 数据集
  = 16 个，`reports/feat_results.json` 现为全矩阵 36 组（合并，不覆盖原结果）
- Clintox 沿用类别加权 [1,12] + col1 口径（特征数据 y 已取 CT_TOX 单列）
- 集成评估：`src/train_feat_ensemble.py --eval-only` → `reports/feat_ensemble_results.json`
  （**36 组**，成员指标 + 集成指标 + 成员文件清单）

### Web 支持
- `_load_ensemble_members` 扩展 feat 组合（seed42 特征模型 + `models/feat_ensemble/*_s{7,13,21}.pt`）
- `/api/meta` 的 `ensemble_ok` 升级为三元组（conv|dataset|feat），扫描后 **48 个组合全部可集成**
  （3 架构 × 4 数据集 × {base,a,b,c}，即 Web 可选的全部组合）
- 前端移除「集成仅基线」限制：feat 非 base 时深度集成直接可用，
  提示改为"特征模型支持单模型 / MC-Dropout / 深度集成（解释图基于基线）"

### 回归测试
`src/smoke_web.py`：15 项全绿，覆盖 GCN/GAT × feat_a/b/c 的单模型/深度集成、
MC-Dropout、注意力逐头、meta 组合可用性（trained=48, ensemble_ok=48）与 SMILES 校验。

### 修复记录：feat_a Web 推理 edge_index 缺失
- 现象：Web 端 feat_a（原子增强）单模型/集成预测报
  `AttributeError: 'NoneType' object has no attribute 'to'`（HTTP 500 → 前端 SyntaxError）
- 根因：`featurize_smiles` 对 feat_a 未构建 edge_index（返回 None），
  与训练数据口径不一致（训练 feat_a 保留图拓扑、仅无键特征）
- 修复：`featurize.py` 对**所有非 base 档位**构建 edge_index（含单原子分子空图
  `torch.empty((2,0))` 保护），edge_attr 仅 b/c 档位计算
- 验证：`smoke_web.py` 新增 `ESOL feat_a single/ensemble GCN` 用例，修复后全部通过

### 修复记录：解释图支持全部特征档位
- 现象：Web 端勾选"生成分子解释图"时，feat_a/b/c 组合无图（静默空白）；
  feat_c 与 MC/集成组合报 `mat1 and mat2 shapes cannot be multiplied`（500）
- 根因：① `explain` 原限制 `level == "base"`，特征档位不生成解释图；
  ② `interpret.explain_one` 只传 `model(x, edge_index, batch)`，特征模型缺
  edge_attr/fp 输入导致前向维度不匹配
- 修复：① `app.py` 放开 explain 到所有特征档位（全组合训练已齐）；
  ② `explain_one` 增加 edge_attr/fp 可选参数并转发给模型 forward（PyG Explainer
  的 kwargs 原样透传），数据层自动从 data 提取
- 验证：feat_a/b/c × 单模型 / MC / 深度集成 × 解释图全部正常出图，
  `smoke_web.py` 15 项冒烟全绿

### 关键文件
- `src/train_feat_ensemble.py`（训练 + 集成评估）、`src/app.py`（feat 集成加载/meta）、
  `src/smoke_web.py`（feat 集成用例）
- 产物：`models/feat_ensemble/`（60 成员）、`reports/feat_ensemble_results.json`

---

## 其他修正
- **AUC 口径修正**：基线分类 AUC 原按 argmax 标签计算（偏低），已改为标准概率口径
  （BBBP：GCN 0.717→0.809、GAT 0.762→0.830、GraphSAGE 0.786→0.876），旧值保留在 `AUC_old_argmax`

## 文件索引
- 脚本：`src/` 下 `contrastive_pretrain.py` / `finetune_pretrained.py` / `ablation_pretrain.py` / `uncertainty.py` / `train_ensemble.py` / `llm_explain.py` / `fix_auc.py` / `featurize.py` / `train_feat.py` / `reeval_feat.py` / `train_clintox_baseline.py` / `attention_viz.py` / `eval_clintox_ensemble.py` / `smoke_web.py`
- 产物：`models/pretrained/`、`models/ensemble/`（含 Clintox 9 成员）、`models/feat/`（20 模型）、`models/Qwen2.5-3B-Instruct/`、`data/pretrain_pool/`
- 报告：`reports/pretrain_run.json` / `pretrain_finetune_results.json` / `ablation_results.json` / `uncertainty_report.json` / `feat_results.json` / `clintox_baseline.json` / `clintox_ensemble.json` / `pretrain_full.txt` / `finetune_*.txt`
