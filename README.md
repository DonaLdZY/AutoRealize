# AutoRealize

AutoRealize 是 AutoDecision 的数据认知与任务实现引擎。它把一个原始数据目录和自然语言需求转换为下游算法系统可以直接消费的任务包，包括任务书、精确数据说明、评估协议、输出合同和 AutoML 专用上下文。

AutoRealize 的目标不是简单概括文件，而是尽可能回答：数据实际有什么、字段准确叫什么、文件之间如何关联、任务应如何评价、方案必须输出什么，以及哪些约束不能违反。

## 输入与输出

输入：

- 一个包含数据、说明文档、图片或压缩包的目录。
- 可选的自然语言任务需求。
- 一份 YAML 配置。

核心输出：

- `description.md`：面向人和算法开发者的完整任务书。
- `sample_submission.csv`：官方样例复用或按输出合同生成的格式样例。
- `realize_report/data_description.md`：全局数据认知报告。
- `realize_report/automl_context.md`：明确供 MLEvolve 使用的结构化任务上下文。
- `realize_report/` 下的调查、合同、事件、LLM 用量和本地 artifact。

## 工作流程

```text
目录扫描
  -> 文件名模式分组与读取计划
  -> 文件解析、布局证据与表格画像
  -> 文件认知、Table Card、Relation Card
  -> QDI 问题驱动调查
  -> 权威需求与约束归并
  -> 任务范式、评估合同、输出合同
  -> description.md + automl_context.md + sample_submission.csv
```

### 1. 数据认知

- 扫描目录、建立文件清单和目录树。
- 解析 CSV、Excel、JSON、TXT/Markdown、DOCX、PDF、图片、YAML/TOML 和常见压缩包。
- 对表格生成 shape、精确列名、类型、空值、唯一值、类别分布、数值和日期统计。
- 对 Excel sheet 记录布局证据，辅助识别非首行表头、说明区、空白区、重复区和非规则文档式表格。
- 对文件名相似的文件建立分组；代表文件不足以证明同构时，可以扩展读取或拆分文件卡片。
- 推断表间一对一、一对多、多对多和共享属性关系。
- 将语义相近的实体字段标记为候选关系，并通过值交集、覆盖率等证据验证，不直接把名称相近当作等价。

### 2. QDI 调查

QDI（Question-Driven Investigation）根据现有证据提出尚未解决的问题，并在需要时生成受限的只读 Python 探查脚本。历史只保留短结论和问题账本，完整脚本输出落入本地 artifact，避免每轮把大段 stdout、metadata 和预览重新发送给 LLM。

典型问题包括：

- 某个字段是否真的是主键或外键。
- 两个候选实体字段的值域是否覆盖。
- 训练、预测、样例提交和说明文件分别扮演什么角色。
- 一个约束在数据中通过哪些字段表达。
- Excel 实际表头、sheet 结构和读取参数是什么。

### 3. 任务定义

- 保存完整 `original_requirements.txt` 作为权威需求来源。
- 判断回归、分类、时序、推荐、优化、决策或强化学习等问题范式。
- 明确任务对象、数据边界、字段来源、约束、泄漏风险和输出格式。
- 生成可复现的评估合同，包括指标、方向、公式、数据范围和验证协议。
- 对评估合同和 description 进行带缺陷理由的 LLM 修复，避免以程序化“未明确”文本代替最终内容。
- 生成供 MLEvolve 直接使用的 `automl_context.md` 和结构化 pack。

## 主要功能

- 多格式、混合目录和多 sheet 数据认知。
- 文件名模式分组、代表文件抽样与读取计划确认。
- 精确 schema、字段语义、公共字段与文件特有字段说明。
- Table Card、Relation Card、Filename Group Card 和约束记忆。
- QDI 问题队列、脚本探查、脚本修复与问题账本。
- 任务范式、数据访问协议、评估合同、输出合同和主任务协议。
- 原始 `description.md` 与用户需求的权威信息保留。
- `sample_submission.csv` 复用、生成和格式校验。
- 结构化事件流、当前状态、前端 manifest、LLM trace 与 token 汇总。
- 文本模型和可选视觉模型分别配置。

## 功能亮点

### 精确字段优先

AutoRealize 将数据中真实存在的列名作为 schema 权威来源。用户需求或 LLM 文本中的近似字段名可以作为语义线索，但不会无证据地覆盖真实列名。需要从多个字段推导的业务概念，应在任务书中写清来源与推导方式。

### Headroom 风格上下文编译

完整 preview、sheet profile、长文档、QDI stdout 和历史草稿保存在 `context_artifacts/`。LLM 默认只接收当前阶段需要的结构化证据包、短卡片和截断信息，从而降低未命中输入 token，同时保留本地追溯能力。

### 权威信息与压缩信息分层

原始需求、已有任务说明、当前错误、评估要求和关键约束不会被模糊摘要替代；可重复计算的统计和大对象才会被裁剪、卡片化或转为 artifact。

### 面向下游交付

`description.md` 面向人类理解，`automl_context.md` 面向 MLEvolve 编码与搜索。两者共享同一事实来源，但表达重点不同，避免把完整认知报告直接堆给下游模型。

## 环境要求

- Python 3.11 或 3.12，64 位版本
- 可访问 OpenAI-compatible 文本模型 API
- 读取 PDF、Office、Excel 和压缩包所需的系统权限
- 足够的磁盘空间保存输入副本、认知结果和 artifact

Windows、Linux 和 macOS 均可运行。视觉认知是可选能力；未配置视觉模型时，图片仍可保留基础元数据。

> 安全提示：QDI 会执行受限的只读探查代码，但这不是强隔离安全沙箱。只处理可信数据，并让服务保持监听在本机或受控内网。

## 安装

独立克隆：

```bash
git clone https://github.com/DonaLdZY/AutoRealize.git
cd AutoRealize
python -m venv .venv
```

Windows：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

在 AutoDecision 主仓库中使用时，也可以直接由根目录的 `requirements.txt` 统一安装。

## 配置

[`config/config.yaml`](config/config.yaml) 是唯一的正式默认配置，包含完整注释。CLI 未传 `--config` 时自动读取该文件；也可以通过 `--config` 指定任意其他位置的 YAML。

也可以由 CLI 输出当前默认配置：

```bash
python -m autorealize.cli --print-default-config
python -m autorealize.cli --write-default-config config.yaml
```

主要配置区：

| 配置区 | 作用 |
| --- | --- |
| `llm` | 文本模型、API、并发、重试、缓存、thinking、`reasoning_effort`、`max_tokens` |
| `switches` | 数据认知、任务定义、Headroom 路径、样例提交等功能开关 |
| `data` | 预览、表格画像、文件分组、布局和各格式读取参数 |
| `investigation` | QDI 问题、脚本、子问题和深度预算 |
| `vllm` | 可选视觉模型配置 |
| `prompt` | 输出语言、prompt 预算、description 与评估合同修复轮次 |
| `context` | Table/Relation/Group Card 与 artifact 摘要预算 |
| `parallel` | 文件认知、关系发现和探查动作并发 |
| `knowledge` | 本地知识存储与检索 |
| `telemetry`、`logging` | 事件流、状态、LLM trace 和日志 |
| `service` | API 服务任务、快照和停止行为 |

建议至少确认以下字段：

```yaml
llm:
  base_url: "https://api.deepseek.com"
  model_name: "deepseek-v4-pro"
  api_key: null
  max_tokens: null
  enable_thinking: null
  reasoning_effort: null

prompt:
  output_language: "zh"
```

`api_key` 非空时优先使用配置值；为空时读取 `DEEPSEEK_API_KEY`。`max_tokens` 为 `null` 或 `0` 时不主动发送限制，由 API 服务商采用默认值。不要提交带真实 Key 的配置。

## CLI 运行

通用命令：

```bash
python -m autorealize.cli \
  --input-root "/path/to/data" \
  --output-root "runs" \
  --task "请根据这些数据建立可复用的预测或决策方案" \
  --run-name "demo" \
  --config "config/config.yaml"
```

Windows PowerShell：

```powershell
python -m autorealize.cli `
  --input-root "D:\data\demo" `
  --output-root ".\runs" `
  --task "请分析数据并生成精确任务书" `
  --run-name "demo" `
  --config ".\config\config.yaml"
```

CLI 中还保留少量兼容性覆盖参数，例如 `--no-cognition`、`--no-task-definition`、`--llm-concurrency` 和 `--cognition-workers`。长期配置应优先写入 YAML。

## 服务模式

启动 FastAPI：

```bash
python -m uvicorn autorealize.service_api:app --host 127.0.0.1 --port 18101
```

常用接口：

- `GET /health`
- `POST /jobs/start`
- `GET /jobs/{job_id}`
- `POST /jobs/stop`
- `POST /snapshot`

启动后访问 `http://127.0.0.1:18101/docs` 查看请求结构。AutoDecision Gateway 默认通过此服务启动和轮询 AutoRealize 任务。

## 输出目录

```text
<output-root>/<run-name>/
|-- description.md
|-- sample_submission.csv
|-- description_origin.md              # 输入中已有 description.md 时的保留副本
`-- realize_report/
    |-- original_requirements.txt
    |-- data_description.md
    |-- automl_context.md
    |-- automl_context_pack.json
    |-- data_cognition_report.json
    |-- question_investigation_report.json
    |-- problem_paradigm_report.json
    |-- evaluation_contract_report.json
    |-- task_definition_report.json
    |-- submission_report.json
    |-- main_task_protocol.json
    |-- file_cognition/
    |-- context_artifacts/
    |-- event_stream.jsonl
    |-- current_state.json
    |-- frontend_manifest.json
    |-- llm_traces.jsonl
    `-- llm_usage_summary.json
```

部分文件受配置和输入条件控制，不保证每次运行都生成。例如关闭样例提交时不会要求 `sample_submission.csv`。

## 前端与下游集成

- `frontend_manifest.json`：模块、事件源和可展示产物索引。
- `current_state.json`：适合轮询的当前状态。
- `event_stream.jsonl`：按序追加的详细事件。
- `automl_context.md`：MLEvolve 的主要任务事实入口。
- `automl_context_pack.json`：供程序读取的结构化版本。
- `llm_usage_summary.json`：按阶段和 prompt part 统计 token、缓存与上下文形状。

下游不应仅凭模糊字段描述自行猜测 schema；应同时使用 `description.md`、`automl_context.md` 以及实际输入目录中的文件。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check autorealize tests --select E9,F63,F7,F82
```

测试覆盖配置解析、文件认知、JSON/Excel 流程、上下文编译、QDI、description 分章节生成、评估合同、样例提交、服务配置、事件流和 LLM 用量记录。默认单元测试不应依赖真实 API Key。

## 常见问题

### 为什么报告中没有完整 raw preview

完整对象通常保存在本地 `context_artifacts/`，prompt 只接收当前问题需要的证据。这是有意的 token 控制，不代表原始信息已被删除。

### 为什么字段名仍可能需要人工复核

AutoRealize 会校验真实 schema 并修正明显近似字段，但业务概念可能需要由多个字段推导。若数据本身存在歧义，应查看 QDI 结论、关系卡片和 `evaluation_contract_report.json`，而不是强制把两个名称相近的字段判为同一实体。

### 为什么运行时间较长

大目录、多 sheet、复杂关系和 QDI 会增加确定性画像与 LLM 调用。可在 YAML 中调整分组、预览、画像、QDI 和并发预算，但减少读取范围会降低事实覆盖率。

### 为什么输出语言混杂

将 `prompt.output_language` 设置为 `zh` 或 `en`。真实数据字段名、函数名和文件名会保留原文，不受自然语言设置影响。

## 使用边界

AutoRealize 生成的是可审计的任务定义和下游上下文，不是业务事实的最终法律或合规判定。正式交付前仍应由领域人员检查关键字段映射、评价公式、约束和数据授权。
