# 基于图神经网络的分子性质预测（CPU 可训的小型 GNN）

**工程实践课题 · 中国科学技术大学软件学院**

利用图神经网络（GCN / GAT / GraphSAGE）在 MoleculeNet 基准数据集（ESOL / FreeSolv / BBBP / Clintox）上预测分子性质（水溶解度、水合自由能、血脑屏障渗透、临床毒性），实现从数据准备、模型训练、多架构对比、特征工程、自监督预训练、不确定性量化、模型融合、可解释性分析到 Web 演示的完整闭环。

> **设备口径**：全部模型均可在 CPU 上训练与推理（参数量 < 100k）。历史 144 个主实验 checkpoint 实际在 GTX 1660 Ti（CUDA）上训练（`README_extensions.md`、`reports/comparison_results.json` 的 `device` 字段可查）；本轮新增的 81 个骨架多种子 checkpoint 与全部解释一致性/官能团实验在 CPU（torch 2.15.0+cpu）上完成。详见 `reports/项目报告.md` 第 1 节。

## 项目亮点

- **4 数据集 × 3 种 GNN 架构 × 3 档特征**：144 个主 checkpoint（CUDA 历史实验）+ 81 个骨架划分多种子 checkpoint（CPU 新增）
- **特征工程**：RDKit 27 维原子特征 + 7 维键特征 + 166 位 MACCS 指纹逐级增强，FreeSolv RMSE 降低 27.1%，Clintox AUC 提升 17.4 点
- **模型融合研究**（方法创新点）：从误差多样性诊断出发，验证"同构集成无效、特征档异构有效"的边界；4 种子复验给出均值 ± 标准差、95% bootstrap CI 与 Wilcoxon 检验
- **解释一致性**：每数据集 40 个代表性分子 × 9 配置 GNNExplainer，Top-5/Top-10 Jaccard + Spearman + 预测方差；ESOL/FreeSolv 判为"高"，BBBP 判为"中"
- **官能团级解释**：原子/键重要性聚合到羧基、羟基、芳香环、卤素等，输出 Top-3，严格区分 GNNExplainer 归因 / RDKit 识别 / LLM 叙述
- **GraphCL 自监督预训练**：15,289 无标签分子池，小样本任务（FreeSolv）增益最显著
- **不确定性量化**：MC-Dropout 置信区间 + 同构集成 + 异质集成（9 配置 × 4 种子）
- **端到端 Web 演示**：SMILES 预测 + 不确定性 + 结构着色 + LLM 解释 + 批量 CSV 预测/导出 + 离线规则模板降级

## 目录结构

```
GNN分子性质预测/
├── data/                      # MoleculeNet 数据集与特征化缓存（不入库）
├── models/
│   ├── feat/                  # 特征工程主模型（9 配置 × 4 数据集，seed42）
│   ├── feat_ensemble/         # 额外种子（7/13/21）权重，用于集成
│   ├── feat_scaffold/         # Murcko 骨架划分权重（seed42 + 新增 7/13/21）
│   ├── hybrid/                # HybridAgg 架构级融合（负结果，保留复现）
│   ├── distill/               # 蒸馏学生模型
│   └── ensemble/              # 深度集成成员
├── reports/
│   ├── splits/                # 随机划分 + *_scaffold_split.json 骨架划分
│   ├── diversity/             # 误差相关矩阵、融合预实验、热力图、预测缓存
│   ├── figures/               # 对比图表（含 scaffold_multiseed.png、explanation_consistency.png、feature_gain.png）
│   ├── interpret/             # GNNExplainer 解释结果（PNG + JSON）
│   ├── attention/             # GAT 逐头注意力热图（PNG + JSON）
│   ├── comparison_results.json        # 基线架构对比
│   ├── feat_results.json              # 特征工程结果
│   ├── fusion_scaffold.json           # 骨架划分单种子三方对比
│   ├── scaffold_multiseed_results.json  # 骨架 4 种子逐成员结果（含逐样本预测）
│   ├── scaffold_multiseed_analysis.json # 多种子均值/CI/显著性
│   ├── explanation_consistency.json   # 跨模型解释一致性
│   ├── functional_group_explanations.json # 官能团 Top-3
│   ├── model_manifest.json             # 模型版本清单
│   ├── hybrid_results.json            # HybridAgg（负结果）
│   ├── distill_results.json           # 蒸馏（负结果）
│   ├── fusion_summary.md              # 融合实验汇总
│   └── 项目报告.md                     # 工程实践结题报告（终稿）
├── 结题材料/                    # 结题报告、答辩 PPT、演示脚本、交付说明
├── src/
│   ├── data_prep.py           # 数据下载与预处理
│   ├── featurize.py           # 四档特征工程（BASE/A/B/C）
│   ├── models.py              # GCN / GAT / GraphSAGE / HybridAgg
│   ├── train_eval.py          # 基线训练与评估
│   ├── train_feat.py          # 特征工程实验（支持 --split scaffold）
│   ├── train_feat_ensemble.py # 深度集成训练
│   ├── train_scaffold_multiseed.py  # 骨架划分 9 配置 × 4 种子
│   ├── analyze_scaffold_multiseed.py # 多种子统计 + bootstrap + Wilcoxon
│   ├── analyze_error_diversity.py    # 误差多样性 + 融合预实验
│   ├── scaffold_split.py      # Murcko 骨架划分
│   ├── train_hybrid.py        # 架构级融合（负结果）
│   ├── distill_ensemble.py    # 异质集成蒸馏（负结果）
│   ├── fusion_scaffold.py     # 骨架划分单种子融合验证
│   ├── plot_feature_gain.py   # 特征工程收益图
│   ├── gen_figures.py         # 对比图表
│   ├── contrastive_pretrain.py # GraphCL 自监督预训练
│   ├── finetune_pretrained.py # 预训练模型微调
│   ├── uncertainty.py         # MC-Dropout 不确定性量化
│   ├── interpret.py           # GNNExplainer 可解释性
│   ├── explanation_consistency.py # 跨模型解释一致性
│   ├── functional_group_explain.py # 官能团聚合
│   ├── attention_viz.py       # 注意力可视化
│   ├── llm_explain.py         # LLM 化学解释（含规则模板降级）
│   ├── model_manifest.py      # 模型清单/版本
│   ├── app.py                 # Flask Web 服务与 API
│   ├── smoke_web.py           # Web 端点冒烟测试
│   └── task_config.py         # 目标列与类别权重统一配置
├── tests/                     # unittest 回归测试（35 项）
├── requirements.txt
└── README.md
```

## 环境与安装

- Python ≥ 3.10（本项目在 Python 3.13 / Windows 上验证）
- 依赖安装：

```bash
pip install -r requirements.txt
# 若需 CPU 版 torch（可选）：
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> 注：若遇到 PyG 2.8 + torch 2.14 的 `gcn_norm` in-place 类型提升报错，`src/compat_patch.py`
> 会在导入模型时自动应用兼容补丁，无需手动处理。

## 运行步骤

### 1. 数据准备

```bash
python src/data_prep.py                 # 下载 4 个 MoleculeNet 数据集
python src/featurize.py                 # 生成 BASE/A/B/C 四档特征缓存
```

### 2. 训练与多架构对比

```bash
python src/train_eval.py --convs GCN GAT GraphSAGE --names ESOL FreeSolv BBBP
python src/train_feat.py --names ESOL FreeSolv BBBP Clintox --levels a b c
```

### 3. 融合研究（按顺序复现）

```bash
python src/analyze_error_diversity.py            # 误差相关矩阵 + 融合预实验（零训练）
python src/scaffold_split.py                     # 生成 4 个 Murcko 骨架划分
python src/train_feat.py --split scaffold --names ESOL FreeSolv BBBP --levels a b c
python src/fusion_scaffold.py --names ESOL FreeSolv BBBP   # 单种子三方对比
python src/train_hybrid.py --names ESOL BBBP --levels c    # 架构级融合（负结果）
python src/distill_ensemble.py --names ESOL BBBP --epochs 300  # 蒸馏（负结果）
python src/gen_figures.py                        # 生成对比图表
```

### 4. 骨架划分多种子复验（CPU，新增）

```bash
python src/train_scaffold_multiseed.py    # 81 个新 checkpoint，7 进程
python src/analyze_scaffold_multiseed.py  # 均值±std、bootstrap CI、Wilcoxon
```

### 5. 解释一致性与官能团

```bash
python src/explanation_consistency.py --jobs 6 --num 40 --epochs 60
python src/functional_group_explain.py
python src/plot_feature_gain.py
```

### 6. 启动 Web 演示

```bash
python src/app.py
```

浏览器访问 **http://127.0.0.1:5000**，支持单模型 / 异质集成（9 配置 × 4 种子）/ MC-Dropout / 同构集成 / GAT 注意力五种分析方式，以及批量 CSV 预测与导出。

如需 LLM 化学解释，配置 DeepSeek API Key 即可（`deepseek_config.json` 或环境变量 `DEEPSEEK_API_KEY`），单次约 8 秒；无 Key 或断网时自动降级为规则模板解释。

## 实验结果汇总

训练配置：hidden=64，3 层图卷积，batch=32，Adam（lr=1e-3，wd=5e-4），early stopping（patience=40）。模型选择基于验证集，test 只评估一次；分类 AUC 为 softmax 概率口径。

### 基线架构对比（9 维特征）

| 数据集 | 指标 | GCN | GAT | GraphSAGE | 最优 |
|---|---|---|---|---|---|
| ESOL | RMSE ↓ | 0.795 | **0.748** | 0.787 | GAT |
| FreeSolv | RMSE ↓ | 2.005 | 2.205 | **1.384** | GraphSAGE |
| BBBP | AUC ↑ | 0.809 | 0.830 | **0.876** | GraphSAGE |
| Clintox | AUC ↑ | 0.711 | **0.724** | 0.641 | GAT |

### 融合效果（随机划分，test）

| 数据集 | 最强单模型 | 同构集成 | 异质等权 all9 | 增益 |
|---|---|---|---|---|
| ESOL | R² 0.9142 | 0.9108 | **0.9205** | +0.6 |
| FreeSolv | R² 0.9157 | 0.9191 | **0.9241** | +0.8 |
| BBBP | AUC 0.9082 | 0.9177 | **0.9292** | +2.1 |
| Clintox | AUC 0.9120 | 0.9027 | **0.9163** | +0.4 |

### 融合效果（骨架划分，4 种子 mean ± std）

| 数据集 | 最强单模型（val 选） | 同构（同架构×特征档） | 异质 all9 | Δ(all9−单模型) / 95% CI |
|---|---|---|---|---|
| ESOL | 0.7640 ± 0.0105 | 0.7730 ± 0.0028 | **0.7876 ± 0.0016** | +0.0296 [+0.0021, +0.0650] |
| FreeSolv | 0.5545 ± 0.0449 | 0.7620 ± 0.0021 | **0.7805 ± 0.0012** | +0.2189 [+0.0987, +0.4399] |
| BBBP | 0.7804 ± 0.0322 | **0.8243 ± 0.0043** | 0.8182 ± 0.0023 | +0.0789 [+0.0145, +0.1451] |

> 判定：三个数据集相对部署基线的 CI 均不含 0，可写"稳定提升"；但 ESOL/BBBP 逐种子配对中分别有 2/4、3/4 的 CI 含 0，幅度会随基线种子波动。BBBP 上 all9 相对同架构 12 成员为 −0.19 点（CI 含 0）→ **无提升**。详见 `reports/项目报告.md` 第 7 节。

### 解释一致性（40 分子 × 9 配置）

| 数据集 | Top-10 原子 Jaccard | Top-5 原子 Jaccard | 原子 Spearman | 预测 std | 等级 |
|---|---|---|---|---|---|
| ESOL | 0.758 | 0.530 | 0.358 | 0.411 | 高 |
| FreeSolv | 0.963 | 0.690 | 0.348 | 0.592 | 高 |
| BBBP | 0.489 | 0.316 | 0.365 | 0.116 | 中 |

![融合策略 × 数据划分](reports/figures/fusion_split_compare.png)
![骨架划分多种子](reports/figures/scaffold_multiseed.png)
![解释一致性](reports/figures/explanation_consistency.png)

### 结论

1. **没有单一架构全面胜出**：GAT 在 ESOL 上最优，GraphSAGE 在 FreeSolv / BBBP 上最优。架构选择应结合任务特性。
2. **特征工程收益大于换架构**：RDKit 原子特征贡献最大，MACCS 指纹对回归有效、对分类有限。
3. **融合的有效层级是"模型级"**：同构种子集成（误差相关 0.92–0.95）基本无增益；跨特征档 + 独立种子的异构集成才是主要增益来源。
4. **骨架划分下增益更大，但有效成分是"特征档多样性"**：同架构 3 特征档 × 4 种子（12 成员）已拿到 FreeSolv +20.1、BBBP +8.1 点；跨架构 all9 在 ESOL 再 +1.4 点（稳定）、FreeSolv +1.8 点（趋势性）、BBBP −0.2 点（无提升）。
5. **BBBP 上集成不提高上限**：事后最优单配置 GCN_feata（0.8339）高于所有集成；集成价值主要是降方差与免调模。
6. **解释一致性"集合高、排序中"**：Top-10 原子集合 Jaccard 0.49–0.96，但 Spearman 仅约 0.35；Web 应展示一致性等级，不能把单模型排序当唯一真值。
7. **负结果保留**：HybridAgg、NNLS 静态加权、知识蒸馏均失败，原因可归因（共享梯度摧毁多样性、小 val 过拟合、单模型容量不足）。详见 `reports/fusion_summary.md`。

## 测试记录（验证方式）

- **数据**：MoleculeNet 官方源，RDKit 解析 SMILES 构建分子图，BBBP 中 11 条无法解析的 SMILES 被跳过
- **划分**：随机 8:1:1（seed=42）+ Murcko 骨架划分，索引均落盘
- **训练**：固定种子，early stopping，测试集指标在独立划分上计算
- **多种子**：9 配置 × 4 种子 × 3 数据集 = 108 成员；bootstrap 2000 次 + Wilcoxon
- **可解释性**：GNNExplainer 对 12 个测试集分子生成案例解释；一致性实验覆盖 120 个分子 × 9 配置
- **回归测试**：`python -m unittest discover -s tests -v` → 35 项全部通过
- **Web 冒烟**：`python src/smoke_web.py` → 17 项端点用例全部通过
- **版本冻结**：commit `3c5268d` + annotated tag `v1.0-freeze-144ckpt`（144 个主 checkpoint 及报告/Web 版本）；后续实验在 `exp/scaffold-multiseed` 分支，避免覆盖

## 常见问题

- **数据集下载慢**：首次运行需从 deepchemdata S3 下载 CSV（每个约 1-5 MB），网络慢时请重试或代理。
- **PyG 安装失败**：确保 torch 已安装且版本匹配，再安装 torch_geometric；Windows 上推荐预编译 wheel。
- **Web 页面无图**：首次预测需运行 GNNExplainer（约 2-5 秒），属正常等待。
- **骨架划分指标偏低**：属正常现象，同一骨架不出现在训练集时泛化难度显著上升，这正是该划分的价值所在。
- **批量预测报错**：CSV 需含 `smiles` 列（可选 `id`），单文件 ≤200 行、单条 SMILES ≤300 字符；非法 SMILES 会逐行返回错误而不是中断整批。
- **无 DeepSeek Key**：解释接口自动降级为规则模板（`provider="rule_template"`），不影响预测与结构图。
