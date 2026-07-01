from __future__ import annotations

import json
import os
import os as _os
from dataclasses import fields, is_dataclass, dataclass, field
from pathlib import Path

from .utils.safe_json import write_json_safe


@dataclass
class LLMConfig:
    """大语言模型配置。"""

    # DeepSeek OpenAI 兼容 API 地址。
    # 例子: "https://api.deepseek.com"
    base_url: str = "https://api.deepseek.com"
    # 模型名称。需求指定为 deepseek-v4-pro。
    model_name: str = "deepseek-v4-pro"
    # API key，优先读取环境变量。
    # 例子: "sk-xxxx"
    api_key: str | None = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY"))
    # 失败重试次数（JSON 解析失败、网络抖动）。
    max_retries: int = 3
    # 温度，默认尽量稳定。
    temperature: float = 0.0
    # 单次输出最大 token。None/0 表示不传 max_tokens，由 API/provider 默认决定。
    max_tokens: int | None = None
    # 结构化 JSON 输出默认最大 token。None/0 表示不传 max_tokens，由 API/provider 默认决定。
    structured_max_tokens: int | None = None
    # 单次 LLM 请求超时时间（秒）。DeepSeek 偶尔会慢，设置超时可避免系统长时间看似卡死。
    # 例子: 180 表示 3 分钟仍未返回则失败并写入事件流。
    request_timeout_seconds: float = 180.0
    # 同一进程内允许并发请求 LLM 的最大数量。数据认知并行时会用它做限流。
    # 调大可更快，但也更容易触发 API 限流；调小更稳。
    max_concurrent_requests: int = 4
    # 是否在 LLM 输出非 JSON 时启用“重试+加强约束”。
    enforce_json_retry: bool = True
    # 是否启用 LLM 响应缓存。开启后相同 prompt/schema/model 会直接复用历史结果，便于调试与加速重复运行。
    enable_cache: bool = True
    # 命中缓存时是否也写入 llm_traces.jsonl，方便前端还原完整调用轨迹。
    trace_cache_hits: bool = True
    # DeepSeek thinking 开关；None 表示使用服务端默认。
    enable_thinking: bool | None = None
    # DeepSeek thinking effort，仅在 thinking 开启时有效。DeepSeek 会将 low/medium 映射到 high。
    reasoning_effort: str | None = None
    # 结构化 JSON 输出时是否关闭 thinking。分类/探查规划等小任务关闭 thinking 更稳定也更快。
    structured_disable_thinking: bool = True


@dataclass
class RuntimeSwitches:
    """Feature switches."""

    # Run the data cognition stage.
    run_data_cognition: bool = True
    # Run the task definition stage.
    run_task_definition: bool = True
    # Inject prompt few-shot examples where available.
    enable_fewshot: bool = False
    # Prefer a low-token AutoRealize route: skip legacy/redundant LLM stages and
    # trigger expensive investigation only when blocking evidence gaps exist.
    optimize_llm_cost: bool = True
    # Legacy intent planner. The newer paradigm/protocol/evaluation pipeline is
    # the source of truth, so this is disabled by default in low-token mode.
    run_architect_plan: bool = False
    # Legacy task classifier kept for compatibility; the paradigm classifier and
    # protocol bundle now carry the main task contract.
    run_legacy_task_classifier: bool = False
    # Final prose composer is expensive and the deterministic renderer rewrites
    # the final description afterwards, so keep it opt-in.
    run_description_final_composer: bool = False
    # Generate sample_submission.csv during task definition.
    generate_sample_submission: bool = True
    # When input data already contains description.md, keep it as the canonical
    # downstream task statement instead of letting generated prose rewrite it.
    prefer_original_description: bool = True
    # Frontend/backend switch: if true, AutoDecision may skip AutoRealize and
    # prepare an AutoML-ready folder directly from the human-provided description.
    direct_automl_from_description: bool = False
    # Auto mode or future human-in-the-loop mode.
    auto_mode: bool = True


@dataclass
class OrchestratorConfig:
    """Orchestrator routing policy."""

    # Enable weighted routing in auto mode.
    auto_enable_weighted_routing: bool = True
    # Data cognition phase weight.
    weight_data_cognition: float = 1.0
    # Task definition phase weight.
    weight_task_definition: float = 1.0
    # Minimum activation score in auto mode.
    # Example: 0.4 means a phase runs when score >= 0.4.
    base_min_activation_score: float = 0.4
    # Always run task definition so downstream AutoML gets a description.
    always_run_task_definition: bool = True


@dataclass
class SamplingConfig:
    """Data probing settings."""

    # Max LLM replanning retries after a probe script fails.
    # 0 disables retry; 2 means up to two retry rounds.
    probe_retry_max: int = 2


@dataclass
class DataConfig:
    """数据与文件处理配置。"""

    # 表格切片最大行数。默认只给下游展示 10 行，避免数据认知阶段线性膨胀上下文。
    preview_rows: int = 10
    # 切片单元格预算，用于控制上下文大小。
    preview_cell_budget: int = 2000
    # 表格统计默认最多读取多少行。用于避免 3GB 级 CSV 在数据认知阶段全量加载。
    # None 表示允许全量读取；推荐工业 demo 保持有限采样。
    table_profile_sample_rows: int | None = 20000
    # LLM 文件认知策略：
    # all = 所有已选中的表格/文档/结构化文件都交给 LLM 写文件认知；
    # none = 单文件阶段不调用 LLM；documents_only = 仅说明/文档类文件调用 LLM。
    # 单文件 LLM 不再调 probe 工具，只消费程序画像，避免逐文件多轮烧 token。
    llm_file_cognition_mode: str = "all"
    # 超过该字节数的 CSV 被视为大文件，只读取前 table_profile_sample_rows 行并记录 sampling 标记。
    large_table_threshold_bytes: int = 256 * 1024 * 1024
    # 类别列展示 top-k 值。若唯一值不足 top-k，画像会尽量完整列出。
    category_top_k: int = 10
    # 数值列异常值判定 z-score 阈值（稳健估计前置）。
    outlier_z_threshold: float = 4.0
    # 判定“几乎全空”的阈值。
    mostly_null_threshold: float = 0.98
    # 文件扫描时是否提取图片元数据。
    extract_image_metadata: bool = True
    # 是否在运行副本中自动解压压缩包（zip/tar/tar.gz/rar等，rar需环境支持）。
    enable_archive_extraction: bool = True
    # 自动解压时，每个压缩包最多展开的文件数，防止意外超大归档。
    archive_extract_file_limit: int = 50000
    # 自动解压后是否保留原压缩包文件（True=保留，False=删除）。
    keep_archive_after_extract: bool = True
    # 当某目录下图片文件数超过该阈值时，不再逐文件写入“文件级认知”，改为目录级汇总。
    image_dir_compact_threshold: int = 80
    # 图片目录压缩展示时，最多保留多少张样本图做文件级展示。
    image_dir_sample_file_count: int = 2
    # 同一目录下若文件名归一化后高度统一且数量达到该阈值，则只抽样读取。
    # 例子: sensor_001.log、sensor_002.log 会归一为 sensor_{num}.log。
    filename_pattern_min_group: int = 20
    # 同目录下相似表格文件达到多少个时启用抽样读取。
    # 例子: train/000d7d20__typewell.csv、train/00bbac68__typewell.csv 表头一致时只读代表文件。
    similar_table_min_files_to_sample: int = 3
    # 相似表格抽样时是否读取轻量表头签名，避免把同前缀但结构不同的表误合并。
    similar_table_use_header_signature: bool = True
    # 相似表格最多读取几个代表文件。
    similar_table_sample_file_count: int = 2
    # 是否允许 LLM 根据目录文件名提出分组正则；默认开启，程序验证后才用于抽样。
    enable_llm_filename_grouping: bool = True
    # 每个目录送给 LLM 判断分组模式的最多文件名数量。
    llm_filename_grouping_max_names: int = 80
    # 每个目录最多接受多少条 LLM 正则候选。
    llm_filename_grouping_max_patterns: int = 6
    # 命名模式抽样读取的最大文件数。
    pattern_sample_file_count: int = 3
    # 文件编码候选。
    text_encodings: tuple[str, ...] = ("utf-8", "utf-8-sig", "gb18030")
    # JSON 嵌套展开时的列名分隔符。
    # 例子: "__" -> user__profile__age；"." -> user.profile.age
    json_flatten_sep: str = "__"
    # JSON 嵌套展开最大深度。
    # None 表示不限深度；1 表示仅展开一层。
    json_flatten_max_level: int | None = None
    # 是否保留原始嵌套列内容（以 raw__<key> 列形式保存，内容为 JSON 字符串）。
    json_keep_raw_nested_columns: bool = False
    # 是否自动生成“预测/验证切分集”（默认关闭）。
    # False: 不生成任何虚拟 test 文件，description 只能引用真实存在文件；
    # True: 若原始数据缺少明确预测集，可生成一个 predict_split 文件供下游演练。
    auto_generate_predict_split: bool = False
    # 生成预测切分集时，样本占比（非时序任务）。
    generated_predict_split_ratio: float = 0.2
    # 时序任务生成预测切分集时，截取末尾窗口天数（按可解析日期列）。
    generated_predict_horizon_days: int = 30
    # 无官方 sample_submission 且无真实预测集时，自动生成的 sample_submission 最多保留多少行。
    # 注意它只是格式样例，不代表评测集行数。
    generated_sample_submission_max_rows: int = 20


@dataclass
class InvestigationConfig:
    """Question-driven cross-file investigation settings."""

    # Enable LLM-led cross-file question investigation after initial data cognition.
    enabled: bool = True
    # always = 每次都跑；on_demand = 只有读取/输出/评估/约束存在阻塞信号时跑；disabled = 跳过。
    trigger_mode: str = "always"
    # Maximum blocking questions the investigator may pursue in one run.
    max_questions: int = 5
    # Maximum plan/observe rounds before the investigator must summarize.
    max_rounds_per_run: int = 3
    # Allow the investigator to request sandboxed read-only Python scripts.
    allow_custom_readonly_python: bool = True
    # Per custom Python subprocess timeout.
    custom_python_timeout_seconds: float = 30.0
    # LLM-proposed read-only Python script repair attempts after execution/static validation fails.
    custom_python_max_retries: int = 3
    # Maximum stdout/stderr characters retained from custom Python.
    custom_python_max_stdout_chars: int = 12000
    # Maximum JSON result characters retained from any investigation script.
    max_result_chars: int = 20000
    # Legacy internal helper row limit. The investigator execution surface is
    # script-only, but these helpers remain available for tests/debugging.
    tool_sample_rows: int | None = 50000
    # Maximum rows returned in samples.
    max_sample_rows: int = 20


@dataclass
class VLLMConfig:
    """视觉模型（VLLM/OpenAI兼容）配置。"""

    # 是否启用视觉模型对图片样本进行语义描述（目录级用途推断）。
    enabled: bool = True
    # 视觉模型 OpenAI 兼容 base_url。
    # 示例: "https://open.bigmodel.cn/api/paas/v4/"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    # 视觉模型 API key。可改为环境变量注入。
    api_key: str = "0b88bd66cfaa4544ab2790b9f3366c84.SmTVkjhVl30IQzjk"
    # 视觉模型名称。
    model_name: str = "glm-4.6v-flashx"
    # 每个图片目录最多抽样几张图片发给视觉模型。
    max_images_per_dir: int = 2
    # 调用失败时是否静默降级到仅元数据模式。
    fail_silently: bool = True


@dataclass
class PromptConfig:
    """Prompt 上下文控制配置。"""

    # 模型最大上下文预算（粗略 token 预算）。
    prompt_token_budget: int = 12000
    # 超过预算的告警比例。
    prompt_budget_warn_ratio: float = 0.8
    # 宽表触发列投影的阈值。
    wide_table_column_threshold: int = 30
    # 执行时保留错误 traceback 的末尾字符数。
    traceback_tail_chars: int = 3000
    # description 质量门失败后的重试次数。
    description_quality_max_retries: int = 3
    # Evaluation contract LLM repair/finalization rounds. The last round is a
    # finalizer that must produce an executable human-facing contract.
    evaluation_contract_max_rounds: int = 3
    # Evaluation section reflection rounds after applying the contract.
    evaluation_reflection_max_rounds: int = 3
    # description 协议结构化输出的最大 token。None/0 表示不传 max_tokens，由 API/provider 默认决定。
    description_protocol_max_tokens: int | None = None
    # description 协议生成时注入的原始需求最大字符数。
    description_protocol_original_chars: int = 10000
    # description 协议生成时注入的数据认知摘要最大字符数。
    description_protocol_data_digest_chars: int = 8000
    # description 协议生成时传给 LLM 的关键文件数量上限。
    description_protocol_file_limit: int = 16
    # description 协议生成时每个文件展示的关键字段数量上限。
    description_protocol_fields_per_file: int = 12
    # Final description natural-language output language.
    # zh: Chinese prose; en: English prose; auto: do not add a global language constraint.
    output_language: str = "zh"
    # 是否启用“description 引用文件必须真实存在”的硬失败模式。
    # True: 若检测到不存在文件引用，触发重生成；重试耗尽仍失败则抛错终止该轮。
    # False: 仅做软修复（删除非法引用行）。
    enforce_description_real_file_refs: bool = True


@dataclass
class TelemetryConfig:
    """运行遥测与前端可视化接口配置。"""

    # 是否启用结构化事件流与当前状态快照。
    enabled: bool = True
    # 全量事件 JSONL 文件名，位于 realize_report 目录下。
    event_stream_filename: str = "event_stream.jsonl"
    # 当前运行状态 JSON 文件名，前端可轮询读取。
    current_state_filename: str = "current_state.json"
    # current_state.json 中保留最近事件数量。
    recent_events_limit: int = 200
    # 是否在每次 run 中写出 final_config.json。
    write_config_snapshot: bool = True
    # 是否写出 config_schema.json，便于前端自动生成配置面板。
    write_config_schema: bool = True


@dataclass
class KnowledgeConfig:
    """Knowledge store and research-context configuration."""

    # Enable local knowledge store. Current adapter is JSONL and can be replaced by vector DB / graph DB.
    enabled: bool = True
    # Knowledge store filename under realize_report.
    store_filename: str = "knowledge_store.jsonl"
    # Max retrieved entries injected into task definition.
    retrieval_top_k: int = 24
    # Max characters per knowledge entry.
    max_entry_chars: int = 2400
    # Write a manifest for future RAG/vector-store ingestion.
    write_rag_manifest: bool = True
    # Boost constraints, metrics and field glossary during lexical retrieval.
    boost_structured_knowledge: bool = True


@dataclass
class ParallelConfig:
    """并行执行配置。"""

    # 是否启用数据认知阶段并行（首轮逐文件读取/统计/摘要）。
    enable_parallel_cognition: bool = True
    # 数据认知并行 worker 数。
    cognition_max_workers: int = max(2, min(8, (_os.cpu_count() or 4)))
    # 是否启用跨文件关系发现并行。
    enable_parallel_relations: bool = True
    # 跨文件关系并行 worker 数。
    relations_max_workers: int = max(2, min(8, (_os.cpu_count() or 4)))
    # 是否启用探查脚本内部并行（preview/profile 等动作）。
    enable_parallel_probe_actions: bool = True
    # 探查脚本并行 worker 数。
    probe_max_workers: int = 4


@dataclass
class AutoRealizeConfig:
    """总配置。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    switches: RuntimeSwitches = field(default_factory=RuntimeSwitches)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    investigation: InvestigationConfig = field(default_factory=InvestigationConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    # 运行目录（会产生轨迹、快照、报告）。
    run_root: Path = Path("runs")

    @classmethod
    def from_env(cls) -> "AutoRealizeConfig":
        """从环境变量读取基础配置。"""
        cfg = cls()
        cfg.llm.api_key = os.environ.get("DEEPSEEK_API_KEY", cfg.llm.api_key)
        return cfg


_CONFIG_DESCRIPTIONS: dict[str, str] = {
    "base_url": "DeepSeek/OpenAI 兼容 API 地址。",
    "model_name": "文本大模型名称。",
    "api_key": "LLM API key，默认从 DEEPSEEK_API_KEY 读取。",
    "max_retries": "LLM 调用或结构化 JSON 解析失败后的最大重试次数。",
    "temperature": "LLM 温度，越低越稳定。",
    "max_tokens": "单次 LLM 输出最大 token 数；None/0 表示不传该参数，由 API/provider 默认决定。",
    "structured_max_tokens": "结构化 JSON 输出默认最大 token；None/0 表示不传该参数，由 API/provider 默认决定。",
    "request_timeout_seconds": "单次 LLM API 请求超时时间（秒）。",
    "max_concurrent_requests": "同一进程内 LLM API 最大并发请求数。",
    "enforce_json_retry": "结构化输出不合法时是否追加错误并要求重试。",
    "enable_cache": "是否缓存相同 prompt/schema/model 的 LLM 响应。",
    "trace_cache_hits": "缓存命中时是否仍写入 llm_traces.jsonl。",
    "enable_thinking": "DeepSeek thinking 开关；None 表示使用服务端默认。",
    "reasoning_effort": "DeepSeek thinking effort，可选 high/max。",
    "structured_disable_thinking": "结构化 JSON 输出时是否关闭 thinking，以提升稳定性与速度。",
    "run_data_cognition": "是否执行数据认知模块。",
    "run_task_definition": "是否执行任务定义模块并生成 description。",
    "optimize_llm_cost": "是否启用低 token 成本路径：跳过冗余 LLM 阶段并按需触发重型调查。",
    "run_architect_plan": "是否运行旧版意图规划 LLM；默认关闭，由范式分类、协议生成和评估合同接管。",
    "run_legacy_task_classifier": "是否运行旧版任务分类器；默认关闭以避免与范式分类器重复。",
    "run_description_final_composer": "是否运行最终全文润色 LLM；默认关闭，因为最终 description 由确定性 renderer 输出。",
    "generate_sample_submission": "是否在任务定义阶段生成 sample_submission.csv；关闭后仅生成 description.md 和任务定义报告。",
    "prefer_original_description": "若输入数据目录已有 description.md，则将其作为最高优先级任务说明并写入最终 description.md，避免自动生成内容改写提交格式。",
    "direct_automl_from_description": "若输入数据目录已有人工确认的 description.md，允许 AutoDecision 跳过 AutoRealize，直接准备 AutoML 输入目录。",
    "probe_retry_max": "探查脚本失败后，LLM 基于报错重规划并重试的最大次数。",
    "description_protocol_max_tokens": "description 协议结构化输出最大 token；None/0 表示不传该参数，由 API/provider 默认决定。",
    "description_protocol_original_chars": "description 协议生成时注入原始需求的最大字符数。",
    "description_protocol_data_digest_chars": "description 协议生成时注入数据认知摘要的最大字符数。",
    "description_protocol_file_limit": "description 协议生成时传给 LLM 的关键文件数量上限。",
    "description_protocol_fields_per_file": "description 协议生成时每个文件展示的关键字段数量上限。",
    "output_language": "最终 description 与 LLM 自然语言输出的语言约束：zh/en/auto。",
    "enable_fewshot": "是否向关键 prompt 注入 few-shot 示例。",
    "auto_mode": "是否使用自动模式；后续前端可切为人机确认模式。",
    "preview_rows": "表格预览最大行数。",
    "preview_cell_budget": "表格切片单元格预算，控制 prompt 大小。",
    "table_profile_sample_rows": "表格字段统计最多读取行数；None 表示全量读取。",
    "llm_file_cognition_mode": "文件级 LLM 认知策略：all/none/documents_only；默认 all，所有已选中的表格/文档/结构化文件都交给 LLM 写认知。",
    "large_table_threshold_bytes": "超过该体积的 CSV 视为大文件并启用采样统计。",
    "category_top_k": "类别值/枚举值统计 top-k。",
    "outlier_z_threshold": "数值异常检测 z-score 阈值。",
    "mostly_null_threshold": "判定列几乎全空的空值比例阈值。",
    "extract_image_metadata": "是否提取图片元数据。",
    "enable_archive_extraction": "是否自动解压 zip/tar/rar 等压缩包。",
    "archive_extract_file_limit": "单个压缩包最大展开文件数。",
    "keep_archive_after_extract": "自动解压后是否保留原压缩包。",
    "image_dir_compact_threshold": "图片目录超过该数量时改为目录级摘要。",
    "image_dir_sample_file_count": "图片目录级认知时抽样图片数。",
    "filename_pattern_min_group": "同命名模式文件超过该数量时启用抽样读取。",
    "similar_table_min_files_to_sample": "同目录、同命名模式、同表头结构的表格文件达到该数量时启用抽样读取。",
    "similar_table_use_header_signature": "相似表格抽样时是否读取轻量表头签名，防止误合并不同结构表。",
    "similar_table_sample_file_count": "相似表格最多读取几个代表文件。",
    "enable_llm_filename_grouping": "是否允许 LLM 根据目录文件名提出分组正则；程序验证后才用于抽样。",
    "llm_filename_grouping_max_names": "每个目录送给 LLM 判断分组模式的最多文件名数量。",
    "llm_filename_grouping_max_patterns": "每个目录最多接受多少条 LLM 正则候选。",
    "pattern_sample_file_count": "命名模式抽样读取的文件数。",
    "text_encodings": "文本文件编码候选列表。",
    "json_flatten_sep": "JSON 表格化时嵌套字段分隔符。",
    "json_flatten_max_level": "JSON 嵌套展开最大深度。",
    "json_keep_raw_nested_columns": "JSON 展开时是否保留原始嵌套列。",
    "auto_generate_predict_split": "缺少预测集时是否自动生成演练 predict_split。",
    "generated_predict_split_ratio": "非时序 predict_split 抽取比例。",
    "generated_predict_horizon_days": "时序 predict_split 截取末尾窗口天数。",
    "generated_sample_submission_max_rows": "无官方样例且无预测集时，生成 sample_submission 的最大样例行数。",
    "max_questions": "Question-Driven Investigator 每轮最多追问的关键跨文件问题数量。",
    "trigger_mode": "Question-Driven Investigator 触发策略：always/on_demand/disabled；默认 always。",
    "max_rounds_per_run": "Question-Driven Investigator 最多执行多少轮计划-观察循环。",
    "allow_custom_readonly_python": "是否允许 Question-Driven Investigator 请求沙盒只读 Python 脚本探查。",
    "custom_python_timeout_seconds": "自定义只读 Python 子进程超时时间（秒）。",
    "custom_python_max_retries": "Question-Driven Investigator 只读 Python 脚本失败后的最大修复重试次数。",
    "custom_python_max_stdout_chars": "自定义只读 Python stdout/stderr 最大保留字符数。",
    "max_result_chars": "调查脚本结果注入 LLM/报告前的最大字符数。",
    "evaluation_contract_max_rounds": "评估合同 LLM 审查/返修最大轮数。",
    "evaluation_reflection_max_rounds": "评估章节反思最大轮数。",
    "tool_sample_rows": "旧版内部调查辅助函数读取表格时最多采样行数；当前 LLM 调查入口只接受只读 Python 脚本。",
    "max_sample_rows": "调查工具返回样例行/键的最大数量。",
    "enabled": "是否启用该配置组功能。",
    "event_stream_filename": "结构化事件流 JSONL 文件名。",
    "current_state_filename": "当前状态快照 JSON 文件名。",
    "recent_events_limit": "状态快照保留最近事件数量。",
    "write_config_snapshot": "是否输出 final_config.json。",
    "write_config_schema": "是否输出 config_schema.json。",
    "store_filename": "本地知识库 JSONL 文件名。",
    "retrieval_top_k": "任务定义阶段检索注入的知识条数。",
    "max_entry_chars": "单条知识最大字符数。",
    "write_rag_manifest": "是否输出 RAG/知识库接入清单。",
    "boost_structured_knowledge": "检索时是否提升约束/指标/字段知识权重。",
    "enable_parallel_cognition": "是否并行执行逐文件数据认知。",
    "cognition_max_workers": "数据认知 worker 数。",
    "enable_parallel_relations": "是否并行执行跨文件关系发现。",
    "relations_max_workers": "关系发现 worker 数。",
    "enable_parallel_probe_actions": "是否并行执行探查动作。",
    "probe_max_workers": "探查动作 worker 数。",
    "run_root": "默认运行根目录。",
}


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, list):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _coerce_value(current, value):
    if isinstance(current, Path):
        return Path(value)
    if isinstance(current, tuple):
        return tuple(value)
    return value


def _deep_update_dataclass(obj, updates: dict) -> None:
    for key, value in updates.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _deep_update_dataclass(current, value)
        else:
            setattr(obj, key, _coerce_value(current, value))


def _schema_for_dataclass(obj) -> dict:
    descriptions = _CONFIG_DESCRIPTIONS
    props = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if is_dataclass(value):
            props[f.name] = _schema_for_dataclass(value)
        else:
            props[f.name] = {
                "type": type(value).__name__,
                "default": _jsonable(value),
                "description": descriptions.get(f.name, ""),
            }
    return {"type": "object", "properties": props}


# Attach helper methods after class creation; keeps dataclass definitions concise.
def _config_to_dict(self) -> dict:
    return _jsonable(self)


def _config_apply_dict(self, updates: dict) -> "AutoRealizeConfig":
    _deep_update_dataclass(self, updates)
    return self


def _config_from_file(cls, path: Path | str) -> "AutoRealizeConfig":
    cfg = cls.from_env()
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("配置文件必须是 JSON object。")
    cfg.apply_dict(data)
    return cfg


def _config_write_json(self, path: Path | str) -> None:
    write_json_safe(Path(path), self.to_dict(), indent=2)


def _config_schema(self) -> dict:
    return _schema_for_dataclass(self)


AutoRealizeConfig.to_dict = _config_to_dict
AutoRealizeConfig.apply_dict = _config_apply_dict
AutoRealizeConfig.from_file = classmethod(_config_from_file)
AutoRealizeConfig.write_json = _config_write_json
AutoRealizeConfig.schema_dict = _config_schema


