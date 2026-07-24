# AutoRealize

AutoRealize 是 AutoDecision 的数据认知与任务定义引擎。它把原始数据目录和自然语言需求转换为下游算法系统可以直接使用的任务包，包括精确数据说明、任务书、评估合同、输出合同和 AutoML 专用上下文。

AutoRealize 不只概括文件内容，还要尽量确认：数据实际包含什么、字段准确叫什么、文件和表如何关联、任务如何评价、方案必须输出什么，以及哪些事实和约束不能被下游模型误解。

## 输入与输出

输入：

- 原始数据目录。
- 可选的自然语言任务需求。
- 一份带注释的 YAML 配置。

核心输出：

- `description.md`：面向用户和算法开发者的完整任务书。
- `sample_submission.csv`：官方样例复用或根据输出合同生成的格式样例。
- `realize_report/data_description.md`：全局数据认知报告。
- `realize_report/automl_context.md`：明确供 MLEvolve 使用的结构化上下文。
- `realize_report/` 下的调查报告、合同、事件、LLM 用量和本地 artifact。

## 工作流程

```text
目录扫描
  -> 文件名模式与读取计划
  -> 文件解析、布局证据和表格画像
  -> 文件认知、Table Card、Relation Card
  -> QDI 问题驱动调查
  -> 权威需求与约束归并
  -> 任务范式、评估合同和输出合同
  -> description.md + automl_context.md + sample_submission.csv
```

## 主要能力

### 数据与文档认知

- 解析 CSV、Excel、JSON、TXT、Markdown、DOCX、PDF、图片、YAML、TOML 和常见压缩包。
- 为表格生成 shape、精确列名、类型、空值、唯一值、类别、数值和日期统计。
- 识别 Excel 的非首行表头、说明区、空白区、重复区和文档式布局，并给出读取建议。
- 对文件名相似的数据先比较表头和结构签名，再决定合并认知、扩展抽样或拆分文件卡片。
- 生成字段级关系证据，区分一对一、一对多、多对多和共享属性。
- 将语义相近的字段视为候选关系，通过值域交集和覆盖率验证，不因名称相似直接断言等价。

### QDI 调查

QDI（Question-Driven Investigation）针对尚未解决的问题生成受限的只读探查动作。模型会看到问题账本、近期动作摘要和当前相关证据；完整长输出保存在 artifact 中，需要时再按片段检索，避免反复发送全部历史。

QDI 适合确认：

- 主键、外键和跨表覆盖率。
- 候选实体字段是否指向同一业务对象。
- Excel 的实际表头、sheet 结构和读取参数。
- 训练集、预测集、样例提交和说明文档的角色。
- 约束、评估和输出要求对应的真实字段。

### 任务定义

- 保存完整 `original_requirements.txt` 作为权威需求来源。
- 判断预测、时序、推荐、优化、决策或强化学习等任务范式。
- 生成统一、可执行且可追溯的评估合同和输出合同。
- 先生成训练表、预测表、ID、目标列和任务类型候选，再仅对低置信度维度调用 LLM 消歧；LLM 只能选择真实候选，结果还会经过物理 schema 回验。
- 要求 `description.md` 中的数据字段与实际 schema 精确一致。
- 区分用户表达中的业务概念和数据中的物理字段；必要时说明派生规则，不伪造不存在的列。
- 最终对 `description.md`、评估/输出合同、sample submission、AutoML context 和主任务协议做结构化一致性审查；修复器只能替换被点名的二级章节。
- 生成给 MLEvolve 直接消费的 `automl_context.md`，补充精确 schema、读取方式、约束和数据事实。

### 上下文与成本控制

- 原始需求全文在任务定义核心调用中保持可见。
- 大型 metadata、预览、脚本输出和文档全文落到本地 artifact。
- prompt 使用稳定的结构化 evidence pack、artifact ref 和动态尾部。
- 任务定义阶段共享不可变的 provider-cache 前缀；近期阶段结果累积在尾部，接近预算时压缩为有损工作记忆。
- `prompt_cache_key_mode: auto` 只对 OpenAI 官方 API 发送基于稳定前缀摘要的路由 key；DeepSeek 等兼容 provider 默认不接收 OpenAI 专属字段，显式启用后若被拒绝也会自动降级重试。
- DeepSeek 使用自动磁盘上下文缓存，复用完全一致的消息前缀并读取 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`；官方地址统一归一化到 `/beta`，与下游 MLEvolve 的 Chat Prefix Completion 保持一致，同时仍兼容普通 Chat Completions。
- DeepSeek thinking 开启时不发送官方声明无效的采样参数；`reasoning_effort` 使用顶层字段，`thinking` 开关通过 SDK `extra_body` 发送。结构化 JSON 默认关闭 thinking，并按官方 JSON mode 建议附带紧凑合法示例。
- 压缩预算同时按字符和中英混合文本的估算 token 约束，并支持 `total` / `body_after_prefix` 两种计数范围。
- 被压缩的完整结果始终保存在本地 artifact；评估、输出和最终审查等关键阶段可按 `artifact_id + json_path` 规划最小范围回读。
- 摘要只用于导航，不能覆盖权威需求和程序验证事实；被省略且未回读的内容不得推断。
- QDI 历史保留问题、动作摘要、短结论和可检索引用，不重复携带全部 stdout。
- `llm_usage.jsonl` 和汇总文件记录 token、reasoning token、provider 缓存读写、后端 fingerprint、上下文形状和 artifact 信息；DeepSeek V4 成本按官方美元单价估算。

## 环境要求

- Conda、Miniconda 或 Miniforge
- Python 3.11 或 3.12，推荐 Python 3.12
- 可访问 OpenAI-compatible LLM API
- 对输入目录和输出目录的读写权限

PDF、Office 文档、Excel 和 QDI 会使用较完整的数据处理依赖。建议安装项目提供的全部 `requirements.txt`，不要只安装 Web 服务依赖。

## Conda 环境安装

### 在 AutoDecision 主仓库中使用

推荐直接复用主仓库的 `automl` 环境：

```bash
cd AutoDecision
conda env create -f environment.yml
conda activate automl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

环境已经存在时，只需激活并同步依赖：

```bash
conda activate automl
python -m pip install -r requirements.txt
```

### 独立安装 AutoRealize

```bash
git clone https://github.com/DonaLdZY/AutoRealize.git
cd AutoRealize
conda create -n autorealize python=3.12 pip -y
conda activate autorealize
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

确认解释器：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
```

## 配置

默认配置文件是 [`config/config.yaml`](config/config.yaml)。CLI 未指定 `--config` 时自动读取该文件，也可以使用其他位置的 YAML。

主要配置区：

| 配置区 | 作用 |
| --- | --- |
| `llm` | 模型、API、重试、并发、thinking、输出 token 和缓存 |
| `switches` | 数据认知、任务定义、低成本路由和样例提交等流程开关 |
| `data` | 文件解析、预览、画像、长文档切片和文件名模式抽样 |
| `investigation` | QDI 问题、动作、脚本、检索和上下文预算 |
| `prompt` | 控制/输出语言、下游语义消歧、源字段别名消歧和最终一致性审查 |
| `context` | 稳定前缀、动态记忆压缩、Headroom 比例和 artifact 回读阶段 |
| `knowledge` | 本地知识检索和候选集内语义重排 |
| `parallel` | 文件认知 worker 与并发策略 |
| `telemetry` / `logging` | 事件、详细日志、简略日志和 LLM usage |

API Key 建议通过环境变量提供。Linux / macOS：

```bash
export DEEPSEEK_API_KEY="your_api_key"
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY = "your_api_key"
```

如果 `llm.api_key` 在 YAML 中非空，则配置值优先；否则读取 `DEEPSEEK_API_KEY`。不要提交包含真实密钥的配置。

查看或导出默认配置：

```bash
python -m autorealize.cli --print-default-config
python -m autorealize.cli --write-default-config ./my-config.yaml
```

## CLI 运行

最小运行示例：

```bash
python -m autorealize.cli \
  --input-root /path/to/input \
  --output-root /path/to/output \
  --run-name demo \
  --task "预测下个月销量"
```

指定其他配置：

```bash
python -m autorealize.cli \
  --config /path/to/config.yaml \
  --input-root /path/to/input \
  --output-root /path/to/output \
  --run-name demo \
  --task "预测下个月销量"
```

Windows PowerShell：

```powershell
python -m autorealize.cli `
  --config ".\config\config.yaml" `
  --input-root "D:\data\demo" `
  --output-root "D:\runs" `
  --run-name "demo" `
  --task "预测下个月销量"
```

查看全部参数：

```bash
python -m autorealize.cli --help
```

## 服务模式

```bash
python -m uvicorn autorealize.service_api:app --host 127.0.0.1 --port 18101
```

常用接口：

- `GET /health`
- `POST /jobs/start`
- `GET /jobs/{job_id}`
- `POST /jobs/stop`
- `POST /snapshot`

访问 `http://127.0.0.1:18101/docs` 查看 OpenAPI 文档。AutoDecision Gateway 通过服务接口传入任务临时 YAML 和运行目录。

## 输出目录

一次运行通常生成：

```text
<output-root>/<run-name>/
|-- description.md
|-- original_requirements.txt
|-- sample_submission.csv
`-- realize_report/
    |-- data_description.md
    |-- automl_context.md
    |-- data_cognition_report.json
    |-- question_investigation_report.json
    |-- evaluation_contract_report.json
    |-- event_stream.jsonl
    |-- llm_usage.jsonl
    |-- llm_usage_summary.json
    `-- artifacts/
```

具体文件名和是否生成某类产物由配置控制。

## 日志与 token 统计

- 简略日志用于快速查看阶段、耗时、调用次数和结果。
- 详细日志保留解析、QDI、LLM 和异常诊断。
- `llm_usage.jsonl` 记录逐次调用的输入、缓存读取、缓存写入和输出 token。
- `llm_usage_summary.json` 按阶段、模型和 prompt 部分汇总。
- artifact 保留未进入 prompt 的完整证据，便于人工追溯。

## 测试

```bash
conda activate autorealize
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

在 AutoDecision 根环境中测试时改为：

```bash
conda activate automl
python -m pytest core/AutoRealize/tests -q
```

默认单元测试应使用 mock，不调用真实 LLM。使用真实模型验证长文档、QDI 或完整任务时会产生 API 费用。

## 常见问题

### 输出中的字段名与数据不一致

检查 `realize_report/automl_context.md` 和字段校验报告。物理字段必须来自实际 schema；业务概念如果需要由多个字段推导，应在任务书中写明推导逻辑，而不是生成一个不存在的列名。

### Excel 没有识别出正确表头

检查文件认知中的 layout evidence 和 reading note。非首行表头、合并单元格或文档式 Excel 可能需要 QDI 进一步探查；不要仅凭默认 `header=0` 读取。

### 相似文件被错误合并认知

提高文件名模式抽样数或启用结构签名检查。代表文件不一致时，系统应继续抽样、拆分分组，或将文件标记为异构而不是强行共享一张卡片。

### QDI 输出过长

完整输出会写入 artifact，prompt 只保留截断片段、动作摘要和引用。需要复查时应使用 artifact 检索或更精确的只读脚本，不应把全部历史输出重新拼回上下文。

### 服务提示缺少 API Key

确认 `config/config.yaml` 的 `llm.api_key` 非空，或在启动服务的同一 Conda 环境中设置 `DEEPSEEK_API_KEY`。

## 安全与许可证

- QDI 只应运行受限的只读探查代码；仍应避免对不可信输入目录授予不必要的系统权限。
- 不要提交真实 API Key、用户数据、运行 artifact、日志或生成的任务目录。
- 仓库加入明确许可证前，不应视为已经授权自由使用、修改或再分发。
