# AutoRealize

AutoRealize 是一个面向真实工业数据场景的上游系统：输入“原始数据目录 + 简短任务描述”，输出可直接交给 ML-Master / AutoML 系统消费的标准化产物（`description.md`、`sample_submission.csv`、清洗后数据、过程报告）。

它重点解决三件事：

1. 数据认知：识别每个文件/字段的作用与风险。
2. 任务定义：产出无歧义的 Kaggle 风格任务说明。
3. 数据清洗：在可回滚、可检查的流程下做最小必要清洗。

## 1. 核心能力

- 分层智能体流程：`Orchestrator`（编排）→ `Architect`（方案/约束）→ `Ground Agents`（执行）。
- 自动模式配重调度：按数据与任务信号决定 P1/P2/P3 是否执行。
- 多格式解析（注册表模式）：`csv/xlsx/xls/json/txt/md/docx/pdf/toml/图片/压缩包`。
- 压缩包能力：支持 `zip/tar/tar.gz/rar(环境支持时)` 自动展开。
- JSON 兼容：可表格化时走表格分析；不可表格化时输出嵌套结构认知。
- 视觉认知：可调用 VLLM 对图片样本做语义摘要（目录级和文件级）。
- 强约束任务书生成：
  - 检查评估协议唯一性与无歧义性；
  - 检查文档引用文件必须真实存在（防幻觉文件名）。
- 清洗安全机制：
  - 渐进式执行（小样本到全量）；
  - 快照与回滚；
  - 契约检查 + 约束引擎 + checker 校验。
- 高可观测性：终端实时事件 + `realize_report/event_stream.jsonl` 结构化事件流（前端可直接消费）。

## 2. 安装

### 2.1 环境要求

- Python 3.10+（建议 3.11+）
- Windows / macOS / Linux

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

### 2.3 配置 API Key

DeepSeek（主 LLM）：

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-xxxx"

# Linux/macOS
export DEEPSEEK_API_KEY="sk-xxxx"
```

默认模型配置见 `autorealize/config.py`：

- `base_url = "https://api.deepseek.com"`
- `model_name = "deepseek-v4-pro"`

## 3. 快速开始

### 3.1 基本运行

```bash
python -m autorealize.cli \
  --input-root "path to data root" \
  --output-root "runs" \
  --task "预测下个月销量" \
  --run-name "task name"
```

### 3.2 常用命令

关闭清洗（只做认知+任务定义）：

```bash
python -m autorealize.cli \
  --input-root "path to data root" \
  --output-root "runs" \
  --task "预测下个月销量" \
  --run-name "task name"
  --no-cleaning
```

离线模式（不调用 LLM API）：

```bash
python -m autorealize.cli \
  --input-root "Sample/Sample1/raw2" \
  --output-root "runs" \
  --task "预测下个月销量" \
  --run-name "run_003_offline" \
  --offline
```

当数据无独立预测集时，自动生成 `predict_split`（默认关闭）：

```bash
python -m autorealize.cli \
  --input-root "path to data root" \
  --output-root "runs" \
  --task "预测下个月销量" \
  --run-name "task name"
  --auto-generate-predict-split
```

## 4. CLI 参数

- `--input-root`：原始数据目录（必填）
- `--output-root`：输出根目录（必填）
- `--task`：用户任务描述（必填）
- `--run-name`：本次运行名（必填，建议“编号+目的”）
- `--no-cleaning`：关闭数据清洗阶段
- `--offline`：离线模式，不调用 LLM API
- `--auto-generate-predict-split`：缺少预测集时自动生成预测切分
- `--parallel-cleaning`：显式开启文件级并行清洗（当前配置默认已开启）

## 5. 输出目录（给 ML-Master / AutoML）

运行后目录：`<output-root>/<run-name>/`

关键产物：

- `description.md`：面向下游建模系统的任务书（Kaggle 风格，含任务目标/评估协议/数据说明/提交格式）。
- `sample_submission.csv`：提交样例（优先复用原始样例；无样例时由系统生成）。
- `description_origin.md`：若输入里有原 `description.md`，会备份到这里。
- 数据文件：清洗后的数据按原结构平铺到运行根目录（非放在最终 `data/` 子目录）。
- `realize_report/`：过程文档与轨迹（供追踪、审计、前端可视化）。

`realize_report/` 典型内容：

- `data_description.md`：数据认知结果（文件级/字段级/关系级）
- `cleaning_report.md`：清洗目标、动作与结果
- `cleaning_scripts/`：执行过的清洗脚本归档
- `trajectory_events.jsonl`：阶段轨迹
- `event_stream.jsonl`：全量结构化事件流（推荐前端直接消费）
- `llm_traces.jsonl`：LLM 调用轨迹
- `run_summary.json`：运行摘要

## 6. 关键行为说明

### 6.1 `sample_submission.csv` 生成策略

1. 优先复用输入中已有样例（兼容 `sample_submission / sampleSubmission / sample-submission` 等命名）。
2. 若不存在，则结合任务语义、字段与表头生成。
3. 生成时会尽量遵循“业务键 + 目标列（或多概率列）”契约。

### 6.2 JSON 处理策略

- JSON 不一定是表格：
  - 可表格化：进入表格分析（列、统计、字段语义）。
  - 不可表格化：保留为结构语义（根类型、路径摘要）。
- 默认 `JSON 不参与清洗`（`enable_json_cleaning=False`），但仍参与认知与任务定义。

### 6.3 评估协议“无歧义”保障

系统会在生成 `description.md` 时执行质量门控，重点检查：

- 是否明确主指标与公式；
- 是否定义 `y_true` 来源；
- 是否固定切分规则与随机种子；
- 是否出现“推荐/可选/视情况”等歧义措辞；
- 是否引用了不存在文件名。

## 7. 并行与性能

默认已开启：

- P1 数据认知并行（逐文件）
- 关系发现并行
- 探查动作并行
- P3 文件级并行清洗

并行参数可在 `autorealize/config.py` 的 `ParallelConfig` 调整，例如：

- `cognition_max_workers`
- `relations_max_workers`
- `probe_max_workers`
- `cleaning_max_workers`

## 8. 配置入口

主配置文件：`autorealize/config.py`

建议重点关注：

- `RuntimeSwitches`：流程开关（认知/任务定义/清洗/契约/检查器）
- `DataConfig`：数据处理策略（JSON 清洗、压缩包、图片目录压缩展示等）
- `VLLMConfig`：视觉模型配置（base_url、model、key、失败降级）
- `PromptConfig`：提示词与质量门配置
- `ParallelConfig`：并行配置

## 9. 测试

```bash
pytest -q tests
```
