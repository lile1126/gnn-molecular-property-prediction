# 基于图神经网络的分子性质预测（CPU 可训的小型 GNN）

**工程实践课题 · 中国科学技术大学软件学院**

利用图神经网络（GCN / GAT / GraphSAGE）在 MoleculeNet 基准数据集（ESOL / FreeSolv / BBBP / Clintox）上预测分子性质（水溶解度、水合自由能、血脑屏障渗透、临床毒性），实现从数据准备、模型训练、多架构对比、特征工程、自监督预训练、不确定性量化、可解释性分析到 Web 演示的完整闭环。全部模型在 CPU/低配 GPU 上即可训练。

## 项目亮点

- **4 数据集 × 3 种 GNN 架构**共 12 个基线模型 + 36 组特征工程实验 + 4 成员深度集成全组合（48 组合全部可训练可推理）
- **GraphCL 自监督预训练**：15,289 无标签分子池，小样本任务（FreeSolv）增益最显著（RMSE 2.205 → 1.657）
- **不确定性量化**：MC-Dropout 置信区间 + 深度集成，输出可信预测范围
- **可解释链**：GNNExplainer 关键原子/键掩码 → GAT 逐头注意力可视化 → LLM 化学解释（本地 Qwen2.5-3B 或 DeepSeek API，一致性校验拦截幻觉）
- **端到端 Web 演示**：输入 SMILES 分子式 → 实时预测 + 不确定性 + 分子结构重要性着色 + LLM 解释
- **CPU 可训**：分子图规模小（几十节点），单模型参数量 < 100k，普通笔记本 CPU 完成全部训练

## 目录结构

```
GNN分子性质预测/
├── data/                      # MoleculeNet 数据集（自动下载缓存，不入库）
├── models/                    # 训练权重与本地 Qwen 模型（不入库）
├── reports/
│   ├── splits/                # train/val/test 划分索引
│   ├── interpret/             # GNNExplainer 解释结果（PNG + JSON）
│   ├── attention/             # GAT 逐头注意力热图（PNG + JSON）
│   ├── figures/               # 基线/特征工程对比可视化图表
│   ├── comparison_results.json# 基线对比实验结果
│   ├── feat_results.json      # 特征工程 36 组实验结果
│   ├── feat_ensemble_results.json  # 深度集成聚合结果
│   ├── pretrain_finetune_results.json  # GraphCL 预训练微调结果
│   └── uncertainty_report.json      # MC-Dropout/集成不确定性评估
├── src/
│   ├── data_prep.py           # 数据下载与预处理
│   ├── models.py              # GCN / GAT / GraphSAGE 模型定义
│   ├── train_eval.py          # 基线训练与评估
│   ├── train_feat.py          # 特征工程实验
│   ├── train_feat_ensemble.py # 深度集成训练
│   ├── contrastive_pretrain.py # GraphCL 自监督预训练
│   ├── finetune_pretrained.py # 预训练模型微调
│   ├── uncertainty.py         # MC-Dropout 不确定性量化
│   ├── interpret.py           # GNNExplainer 可解释性
│   ├── attention_viz.py       # 注意力可视化
│   ├── llm_explain.py         # LLM 化学解释（本地 Qwen / DeepSeek API）
│   ├── app.py                 # Flask Web 服务与 API
│   ├── task_config.py         # 数据集目标列与类别权重统一配置
│   ├── templates/index.html   # Web 页面模板
│   ├── static/css/app.css     # 响应式界面样式
│   ├── static/js/app.js       # 交互、校验、结果渲染与历史记录
│   └── compat_patch.py        # PyG/torch 版本兼容补丁
├── tests/                     # unittest 回归测试
├── requirements.txt
├── requirements-llm.txt       # 可选：本地 Qwen 解释依赖
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

### 1. 数据准备（自动下载 MoleculeNet 数据集）

```bash
python src/data_prep.py                 # 下载 ESOL / FreeSolv / BBBP 三个数据集
```

### 2. 训练与多架构对比

```bash
python src/train_eval.py --convs GCN GAT GraphSAGE --names ESOL FreeSolv BBBP
```

训练完成后：
- 9 个模型权重保存到 `models/`
- 对比结果保存到 `reports/comparison_results.json`
- 控制台输出每个模型的训练曲线摘要与测试集指标

### 3. 可解释性分析

```bash
python src/interpret.py --conv GCN --names ESOL FreeSolv BBBP --num 4
```

对每个数据集随机选取 4 个测试集分子运行 GNNExplainer，输出：
- `reports/interpret/{conv}_{dataset}_idx{i}.png`：分子结构按原子/键重要性着色
- `reports/interpret/{conv}_{dataset}_explain.json`：预测值、top 原子/键索引等

### 4. 启动 Web 演示

```bash
python src/app.py
```

浏览器访问 **http://127.0.0.1:5000**。界面采用左侧参数、右侧结果的科研工作台布局，支持：

- SMILES 即时校验与常用分子示例；
- 单模型、MC-Dropout、深度集成、GAT 注意力四种分析方式；
- 预测值、近似区间、结构解释图、逐头注意力与 LLM 化学解释；
- 最近分析历史、本地保存和一键回放；
- 桌面与移动端响应式布局。

如需本地 Qwen 化学解释，额外安装可选依赖：

```bash
pip install -r requirements-llm.txt
```

如需改用 DeepSeek API 生成解释（无需额外依赖），配置 API Key 后即可：

```bash
# 方式一：环境变量
set DEEPSEEK_API_KEY=sk-xxx        # Windows；macOS/Linux 用 export
# 方式二：项目根目录创建 deepseek_config.json
# {"api_key": "sk-xxx"}
```

Web 界面 LLM 按钮旁可选择生成方式（本地 Qwen / DeepSeek API）；命令行方式：

```bash
python src/llm_explain.py --smiles "CCCO" --dataset ESOL --provider deepseek
```

## 实验结果汇总

训练配置：hidden=64，3 层图卷积，batch=32，Adam（lr=1e-3，wd=5e-4），early stopping（patience=40），随机种子固定。

| 数据集 | 任务 | 指标 | GCN | GAT | GraphSAGE | 最优 |
|---|---|---|---|---|---|---|
| ESOL | 溶解度回归 | RMSE ↓ | 0.795 | **0.748** | 0.787 | GAT |
| ESOL | 溶解度回归 | R² ↑ | 0.849 | **0.867** | 0.852 | GAT |
| FreeSolv | 水合自由能回归 | RMSE ↓ | 2.005 | 2.205 | **1.384** | GraphSAGE |
| FreeSolv | 水合自由能回归 | R² ↑ | 0.742 | 0.688 | **0.877** | GraphSAGE |
| BBBP | 血脑屏障分类 | AUC ↑ | 0.717 | 0.762 | **0.786** | GraphSAGE |
| BBBP | 血脑屏障分类 | Accuracy ↑ | 0.820 | 0.834 | **0.854** | GraphSAGE |

### 结论

1. **没有单一架构全面胜出**：GAT 在 ESOL（小分子溶解度）上最优，GraphSAGE 在 FreeSolv 和 BBBP 上最优。架构选择应与任务/数据集特性结合。
2. GraphSAGE 在回归与分类任务上整体表现稳健，是最稳妥的默认选择。
3. 三种架构在 CPU 上均可在 3-5 分钟内完成训练，验证了小型 GNN 的资源友好性。
4. GNNExplainer 揭示的原子/键重要性具有化学合理性（如卤素取代基对溶解度的主导影响），说明模型学到了有意义的分子结构-性质关系。

## 测试记录（验证方式）

- **数据**：MoleculeNet 官方源下载，RDKit 解析 SMILES 构建分子图（9 维原子特征 + 3 维键特征），BBBP 中有 11 条无法解析的 SMILES 被 PyG 自动跳过
- **训练**：固定随机种子划分 train/val/test（8:1:1），early stopping 防过拟合，测试集指标在独立划分上计算
- **可解释性**：GNNExplainer 对 12 个测试集分子（3 数据集 × 4 个）生成解释，全部成功
- **Web**：Flask 服务加载 12 个基线模型，页面、静态资源、SMILES 校验、单模型、MC-Dropout、深度集成和注意力接口均通过实际请求验证
- **回归测试**：运行 `python -m unittest discover -s tests -v`，覆盖 Web 页面/静态资源、输入校验、MC 状态恢复、GraphSAGE 边特征前向和划分文件保护

## 常见问题

- **数据集下载慢**：首次运行需从 deepchemdata S3 下载 CSV（每个约 1-5 MB），网络慢时请重试或代理。
- **PyG 安装失败**：确保 torch 已安装且版本匹配，再安装 torch_geometric；Windows 上推荐用预编译 wheel。
- **Web 页面无图**：检查服务日志，首次预测需运行 GNNExplainer（约 2-5 秒），属正常等待。
