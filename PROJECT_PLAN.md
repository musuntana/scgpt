# PerturbScope-GPT 项目开发方案（本地开发版）

## 1. 项目定位

### 项目名称
PerturbScope-GPT：基于 Transformer 的单细胞扰动响应预测与靶点优先级排序系统

### 项目目标
构建一个面向 AI4Bio / AI4Science 求职展示的本地可运行 MVP，完成以下闭环：

1. 基于公开 Perturb-seq 单细胞数据构建稳定的数据处理 pipeline。
2. 训练一个本地机器可运行的扰动响应预测模型。
3. 用统一 baseline 和明确指标完成对比评估。
4. 输出可解释的靶点优先级排序结果。
5. 通过 Streamlit 提供交互式演示页面。

### 项目价值
- 展示单细胞数据工程能力：QC、归一化、HVG、样本配对、稀疏矩阵处理。
- 展示深度学习建模能力：Transformer、embedding 设计、训练与评估。
- 展示 AI4Bio 应用能力：扰动建模、DEG、靶点排序。
- 展示工程化能力：模块化代码、配置驱动、实验管理、可视化部署。

### 本地开发原则
- 单机可运行优先于大规模论文复现。
- 先做单数据集、单细胞系、单基因扰动的稳定 MVP。
- 先保证闭环，再扩展到更复杂的扰动或更大的数据规模。

## 2. 目标用户与使用场景

### 目标用户
- AI4Bio / 生信岗位面试官
- 对 perturbation modeling 感兴趣的研究人员
- 需要快速验证候选靶点影响的个人项目展示者

### 使用场景
- 输入 cell type 和 perturb gene，预测表达变化谱。
- 查看关键差异基因和 Top-k 候选基因排名。
- 对比 Transformer 与 MLP / XGBoost baseline 的效果。

## 3. 项目范围

### MVP 范围
- 一个锁定的数据集与固定预处理 pipeline
- 单细胞系、单基因扰动、control vs perturb 场景
- 一个 Transformer 主模型
- 两个 baseline：MLP、XGBoost
- 三个核心指标：Pearson correlation、MSE、Top-k DEG overlap
- 一个 Streamlit 演示页面

### 暂不纳入 MVP
- 多数据集联合训练
- 多基因组合扰动
- foundation model 级别预训练
- 复杂 batch correction 方案
- 分布式训练和完整 MLOps 平台
- 云部署与数据库服务

## 4. 数据与任务定义

### 4.1 锁定数据集

MVP 主数据集固定为：
- `scPerturb benchmark` 中的 `Norman2019` K562 数据，优先使用单基因扰动子集

选择理由：
- 社区认知度高，适合面试表达
- 结构相对清晰，便于从公开基准进入工程实现
- 可先使用处理后的 `AnnData` 版本，降低数据清洗复杂度
- 数据规模适中，适合本地机器做 MVP

纳入样本：
- 单基因扰动样本
- non-targeting 或 control 样本

排除样本：
- 多基因组合扰动
- metadata 缺失严重的样本
- 无法可靠匹配 control 的异常子集

### 4.2 数据 Pipeline

预处理主流程：

```text
raw / benchmark AnnData
  -> metadata sanity check
  -> filter low quality cells / genes
  -> normalize_total
  -> log1p
  -> highly_variable_genes
  -> keep sparse matrix through preprocessing
  -> build control mean by batch / cell type
  -> create delta expression targets
  -> seen / unseen split
  -> export processed tensors and metadata
```

核心产物：
- 原始 `AnnData`
- HVG 子集表达矩阵
- control 均值向量
- `gene_to_idx` 与 `perturb_to_idx`
- train / val / test split 索引
- 用于训练的张量文件或中间表

### 4.3 稀疏矩阵策略

scRNA-seq 数据天然稀疏，MVP 采用以下规则：

1. `AnnData.X` 在读取、QC、归一化、HVG 选择阶段保持 sparse。
2. 仅在 `HVG` 子集完成后，将当前样本切片转为 `float32 dense tensor`。
3. dense 转换仅发生在训练导出阶段，不在原始预处理中提前 densify。
4. 若内存不足，优先降低 `HVG` 数量和每个 perturbation 的采样上限，而不是直接扩模型。

### 4.4 Control / Perturb 配对策略

MVP 采用最稳定、最适合本地开发的配对方式：

- 只建模单基因扰动
- 对每个 batch 内的 control cells 计算平均表达向量
- 若 batch 信息不可用，则退化为同 cell type / cell line 下的全局 control 均值
- 每个 perturbed cell 的目标定义为：

```text
delta_expression = perturbed_expression - matched_control_mean
```

这样做的原因：
- 避免逐 cell 配对带来的噪声和组合爆炸
- 训练目标稳定
- 更适合本地机器做首版回归任务

## 5. 建模方案

### 5.1 任务定义

模型输入：
- control mean expression vector
- perturb gene id

模型输出：
- predicted delta expression

理由：
- 比直接预测 perturbed expression 更适合 perturbation response 建模
- 与 control / perturb 配对策略天然一致

### 5.2 Transformer 架构决策

MVP 采用本地优先架构：

```text
gene identity embedding
+ expression value projection
+ perturbation embedding
-> Transformer Encoder
-> MLP regression head
-> predicted delta expression
```

其中每个 gene token 的表示为：

```text
token_i =
  gene_embedding_i
+ value_projection(control_expression_i)
+ perturbation_embedding(p)
```

关键决策：
- 基因视为 token
- `perturbation embedding` 直接加到每个 gene token 上
- 不使用 special token
- 不使用 cross-attention
- 先不用更复杂的条件注入方式

这样做的原因：
- 结构最简单
- 参数量可控
- 训练更稳定
- 便于面试清楚解释

### 5.3 序列长度与显存控制

标准 self-attention 为 `O(n^2)`，因此本地开发必须先限制输入规模。

MVP 建议配置：
- 默认 `HVG = 512`
- 推荐工作区间 `512 ~ 800`
- 未做显存估算前，不超过 `1000`

推荐第一版模型参数：
- `d_model = 128`
- `n_heads = 4`
- `n_layers = 2`
- `ffn_dim = 256`
- `dropout = 0.1`
- `batch_size = 16`

扩容策略：
- 先放大数据量，再放大模型
- 若显存不足，优先减 `HVG`、`batch_size`、`n_layers`
- 不在 MVP 阶段引入 Performer / linear attention，除非标准 attention 已被证明确实不可运行

### 5.4 Positional Encoding 决策

基因没有天然序列顺序，因此：

- MVP 默认 `不使用位置编码`
- 固定基因顺序为保存下来的 `HVG index`
- 基因身份由 `gene embedding` 表达，而不是位置编码表达

若后续做扩展实验，可将可学习位置编码作为对照实验，而不是默认方案。

### 5.5 Loss

主损失：

```text
MSE Loss
```

可选正则：

```text
MSE Loss + lambda * L1
```

说明：
- `L1` 仅作轻量正则，不改变主任务定义
- 最优 checkpoint 以验证集主指标保存

## 6. Baseline 方案

### 6.1 MLP
- 输入：`control expression + perturbation embedding`
- 输出：predicted delta expression

### 6.2 XGBoost
- 作为传统机器学习 baseline
- 第一版以聚合任务或逐基因回归的轻量方式实现
- 目标是提供对照，而不是追求最优复杂实现

## 7. 评估方案

### 7.1 指标粒度

必须明确区分粒度。

主报告粒度：
- `per-perturbation`

定义：
- 对同一 perturbation condition 的预测结果求均值
- 将该均值向量与真实均值向量在所有基因上比较
- 计算 Pearson correlation 和 MSE

辅助粒度：
- `per-gene`
- `per-cell` 仅作补充分析，不作为主报告指标

### 7.2 核心指标
- Pearson correlation
- Mean Squared Error
- Top-k DEG overlap

### 7.3 DEG 定义

真实 DEG 统一定义为：
- 使用 `scanpy.tl.rank_genes_groups`
- 统计方法：`wilcoxon`
- 阈值：
  - `adjusted p-value < 0.05`
  - `abs(logfoldchange) > 0.25`

Top-k 真实 DEG 生成方式：
- 先按上述阈值筛选
- 再按统计 score 排序取前 `k`

Top-k 预测 DEG 生成方式：
- 按 `abs(predicted_delta_expression)` 排序取前 `k`

### 7.4 数据切分策略

MVP 明确保留两套评估协议：

1. `seen perturbation split`
   - 在每个 perturbation condition 内做分层切分
   - 用于快速验证模型是否学到基本映射

2. `unseen perturbation split`
   - 以 perturbation gene 为 group 切分
   - 某个 perturbation 只能出现在 train / val / test 其中一个 split
   - 用于评估对未见扰动的泛化

实施顺序：
- Phase 2 先跑 `seen split`
- Phase 3 再补 `unseen split`

## 8. Target Ranking 方案

### 8.1 Ranking 原则

attention 权重不作为 ranking 输入，只作为模型内部可视化参考。

MVP 的 `importance_score` 定义为：

```text
importance_score =
  0.5 * normalized_abs_predicted_delta
+ 0.5 * normalized_deg_significance
```

其中：

```text
deg_significance = -log10(adjusted_p_value + 1e-12)
```

并对两个分量分别做归一化后再相加。

### 8.2 设计理由
- 消除 attention 解释和 ranking 公式的矛盾
- score 含义更清晰
- 更适合求职项目中的可解释表达

### 8.3 后续扩展

如果后续需要调权重：
- 默认等权
- 可在验证集上做简单 grid search
- 任何权重调整必须写入配置和文档

## 9. 推荐工程结构

```text
PerturbScope-GPT/
├── AGENTS.md
├── PROJECT_PLAN.md
├── README.md
├── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_debug.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── io.py
│   │   ├── preprocess.py
│   │   ├── pairing.py
│   │   └── torch_dataset.py
│   ├── models/
│   │   ├── transformer.py
│   │   ├── mlp.py
│   │   └── xgboost_baseline.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── deg.py
│   ├── ranking/
│   │   └── target_ranking.py
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── seed.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   ├── preprocess_data.py
│   ├── train_transformer.py
│   ├── train_baselines.py
│   └── evaluate_model.py
└── tests/
    ├── test_dataset.py
    ├── test_metrics.py
    └── test_ranking.py
```

## 10. 分阶段开发计划

### Phase 0：项目初始化
目标：建立本地可持续迭代的工程骨架。

交付物：
- 目录结构
- `README.md`
- `requirements.txt`
- `configs/*.yaml`
- `AGENTS.md`

完成标准：
- 本地环境可创建并安装依赖
- 配置文件能表达数据、模型、训练三类参数

### Phase 1：数据 Pipeline
目标：从 `scPerturb/Norman2019` 稳定生成训练样本。

任务：
- 读取 benchmark 数据和 metadata
- 实现 QC、归一化、log1p、HVG
- 保持 sparse 直到导出阶段
- 建立 control mean 配对逻辑
- 导出 `delta expression` 训练数据

交付物：
- `src/data/preprocess.py`
- `src/data/pairing.py`
- `src/data/torch_dataset.py`
- `scripts/preprocess_data.py`

完成标准：
- 同一输入重复运行结果一致
- 样本维度、基因索引、扰动标签映射可追踪
- 输出能在本地被 DataLoader 直接读取

### Phase 2：模型与训练
目标：在本地机器上跑通 Transformer 与 baseline。

任务：
- 实现 `torch Dataset` 和 `DataLoader`
- 实现最小 Transformer
- 实现 MLP baseline
- 实现 XGBoost baseline
- 保存 checkpoint 与训练日志

交付物：
- `src/models/transformer.py`
- `src/models/mlp.py`
- `src/models/xgboost_baseline.py`
- `src/training/trainer.py`
- `scripts/train_transformer.py`

完成标准：
- `seen split` 至少成功训练一轮以上
- 产出验证集指标和最优 checkpoint
- 模型参数规模适合本地机器反复调试

### Phase 3：评估与 Ranking
目标：完成可解释、可比较的结果输出。

任务：
- 实现 per-perturbation Pearson 和 MSE
- 实现 Top-k DEG overlap
- 增加 `unseen perturbation split`
- 输出 gene importance ranking

交付物：
- `src/evaluation/metrics.py`
- `src/evaluation/deg.py`
- `src/ranking/target_ranking.py`
- `scripts/evaluate_model.py`

完成标准：
- 指标定义清晰且结果可复现
- ranking 字段含义明确
- 报告 seen / unseen 两类评估结果

### Phase 4：可视化与演示
目标：将研究型结果整理成求职可展示的 demo。

任务：
- 构建 Streamlit 应用
- 加载 saved model 和 ranking 结果
- 展示预测表达变化、DEG、target ranking
- 增加基础图表和错误处理

交付物：
- `app/streamlit_app.py`

完成标准：
- 用户可完成一次端到端推理
- 页面能清楚展示模型能力和局限性

## 11. 里程碑与验收

### M1：数据可用
- 能从目标数据集生成 processed dataset
- control mean、gene index、perturb index 均被持久化

### M2：模型可训
- Transformer 和至少一个 baseline 跑通
- 在 `seen split` 上输出可解释指标

### M3：泛化可评估
- 在 `unseen split` 上输出结果
- 明确 seen / unseen 的差异

### M4：项目可展示
- Streamlit demo 可运行
- 文档足以支持面试表达

## 12. 关键技术决策

### 决策 1：数据集固定
- 固定为 `scPerturb / Norman2019` 单基因扰动子集

### 决策 2：任务目标固定
- 预测 `delta expression`

### 决策 3：条件注入方式固定
- `perturbation embedding` 加到每个 gene token 上

### 决策 4：位置编码固定
- MVP 默认不用位置编码

### 决策 5：指标主粒度固定
- 主指标按 `per-perturbation`

### 决策 6：切分协议固定
- 先 `seen split`
- 后 `unseen perturbation split`

### 决策 7：ranking 公式固定
- 不使用 attention score
- 默认等权融合 `abs_delta` 与 `deg_significance`

## 13. 本地开发默认配置

建议以以下配置作为第一版起点：

```text
dataset: scPerturb / Norman2019 single-gene subset
HVG: 512
d_model: 128
n_heads: 4
n_layers: 2
batch_size: 16
epochs: 20
target: delta_expression
position_encoding: false
perturbation_injection: additive_to_all_gene_tokens
```

如果本地机器配置较弱：
- 先把 `HVG` 降到 `256`
- 将 `batch_size` 降到 `8`
- 先只用部分 perturbation 条件跑通流程

## 14. 风险与应对

### 风险 1：数据格式与 metadata 不统一
应对：
- 只锁定一个主数据集
- 将数据读取和预处理解耦

### 风险 2：本地算力不足
应对：
- 控制 `HVG` 上限
- 限制模型深度和 batch size
- 先跑 `seen split` 和小样本子集

### 风险 3：结果不稳定
应对：
- 固定随机种子
- 固定配置文件
- 记录 split 索引和预处理参数

### 风险 4：项目漂移成“大而全论文复现”
应对：
- 只做单数据集 MVP
- 先闭环数据、模型、评估、demo 四部分
- 任何新增复杂模块必须先更新 `PROJECT_PLAN.md`

## 15. Definition of Done

满足以下条件，项目第一版即完成：

1. 能从 `scPerturb / Norman2019` 生成 processed dataset。
2. Transformer 与至少一个 baseline 能训练并输出验证结果。
3. 能计算 per-perturbation Pearson、MSE、Top-k DEG overlap。
4. 能输出一份定义清晰的 target ranking 表。
5. Streamlit 页面可完成一次完整推理展示。
6. 所有关键决策均有文档记录，且适合本地机器复现。

## 16. 面试表达建议

建议按以下顺序介绍：

1. 问题定义：基于单细胞扰动数据预测基因表达响应。
2. 数据工程：Norman2019、HVG、稀疏矩阵处理、control mean 配对。
3. 建模选择：为何用 Transformer，为什么本地版先把 HVG 控制在 512 到 800。
4. 架构细节：扰动 embedding 如何注入、为何不使用位置编码。
5. 评估设计：per-perturbation 指标、seen / unseen 切分。
6. 可解释性：DEG 与 ranking，不把 attention 当因果证据。
7. 工程化：配置驱动、模块边界、Streamlit demo。

## 17. 下一步建议

优先执行顺序：

1. 搭建目录结构、`README.md`、`requirements.txt`、`configs/*.yaml`
2. 实现 `preprocess.py + pairing.py + torch_dataset.py`
3. 用 `HVG=512` 跑通第一个 Transformer baseline
4. 补齐评估与 ranking
5. 最后接 Streamlit demo
