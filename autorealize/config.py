from __future__ import annotations

import os
import os as _os
from dataclasses import dataclass, field
from pathlib import Path


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
    # 单次输出最大 token。
    max_tokens: int = 4000
    # 是否在 LLM 输出非 JSON 时启用“重试+加强约束”。
    enforce_json_retry: bool = True


@dataclass
class RuntimeSwitches:
    """功能开关配置。"""

    # 是否执行数据认知流程。
    run_data_cognition: bool = True
    # 是否执行任务定义流程。
    run_task_definition: bool = True
    # 是否执行数据清洗流程。
    run_data_cleaning: bool = True
    # 是否启用模式契约静态检查。
    enable_contract_check: bool = True
    # ????????????value_constraints + post_conditions??
    enable_constraint_engine: bool = True
    # post_conditions ???????????????
    constraint_fail_on_unknown_rule: bool = False
    # 是否启用监控器规则检查（不调用 LLM）。
    enable_rule_monitor: bool = True
    # 是否启用渐进式采样。
    enable_progressive_sampling: bool = True
    # 是否启用置信度早停。
    enable_confidence_early_stop: bool = True
    # 是否启用“额外检查智能体”。
    enable_checker_agent: bool = True
    # 是否启用 prompt few-shot。
    enable_fewshot: bool = True
    # 是否允许 LLM 自主生成 Python 脚本。
    allow_llm_generated_script: bool = True
    # 自动模式或人机确认模式（当前实现默认自动）。
    auto_mode: bool = True


@dataclass
class OrchestratorConfig:
    """编排师调度策略配置。"""

    # 是否启用 auto 模式下的“配重调度”逻辑。
    auto_enable_weighted_routing: bool = True
    # 数据认知阶段配重。
    weight_data_cognition: float = 1.0
    # 任务定义阶段配重。
    weight_task_definition: float = 1.0
    # 数据清洗阶段配重。
    weight_data_cleaning: float = 1.0
    # auto 模式下阶段激活最小分值阈值。
    # 例子: 0.4 表示 score>=0.4 才执行该阶段。
    base_min_activation_score: float = 0.4
    # 是否强制始终执行任务定义阶段（推荐 True，确保输出给 ML-Master 的 description 始终可用）。
    always_run_task_definition: bool = True


@dataclass
class SamplingConfig:
    """渐进式采样与搜索配置。"""

    # 论文默认四级采样规模。
    xs_rows: int = 10
    s_rows: int = 100
    m_rows: int = 1000
    # 每一层最大 refine 次数。
    max_refine_per_level: int = 3
    # 失败后 phase 级回退阈值（>= 该值时升级 plan 级回退）。
    phase_backtrack_escalate_threshold: int = 2
    # 全局回退上限，防止死循环。
    global_backtrack_max: int = 8
    # 置信度早停阈值（论文示例 0.92）。
    confidence_commit_threshold: float = 0.92
    # MCTS 最大深度（用于限制蒙特卡洛搜索）。
    mcts_max_depth: int = 6
    # UCT 探索常数。
    uct_c: float = 1.4142


@dataclass
class DataConfig:
    """数据与文件处理配置。"""

    # 表格切片最大行数。
    preview_rows: int = 20
    # 切片单元格预算，用于控制上下文大小。
    preview_cell_budget: int = 2000
    # 类别列展示 top-k 值。
    category_top_k: int = 12
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
    # 文件编码候选。
    text_encodings: tuple[str, ...] = ("utf-8", "utf-8-sig", "gb18030")
    # JSON 文件是否参与数据清洗。
    # True: JSON 被识别为表格候选后，进入与 CSV/XLSX 相同的清洗流程；
    # False: JSON 仍参与数据认知/字段分析/文档总结，但在 P3 清洗阶段跳过。
    enable_json_cleaning: bool = False
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
    # 是否启用“description 引用文件必须真实存在”的硬失败模式。
    # True: 若检测到不存在文件引用，触发重生成；重试耗尽仍失败则抛错终止该轮。
    # False: 仅做软修复（删除非法引用行）。
    enforce_description_real_file_refs: bool = True


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
    # 是否启用数据清洗阶段按文件并行（默认开启）。
    enable_parallel_cleaning: bool = True
    # 数据清洗并行 worker 数（仅在 enable_parallel_cleaning=True 时生效）。
    cleaning_max_workers: int = max(2, min(6, (_os.cpu_count() or 4)))


@dataclass
class AutoRealizeConfig:
    """总配置。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    switches: RuntimeSwitches = field(default_factory=RuntimeSwitches)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    # 运行目录（会产生轨迹、快照、报告）。
    run_root: Path = Path("runs")

    @classmethod
    def from_env(cls) -> "AutoRealizeConfig":
        """从环境变量读取基础配置。"""
        cfg = cls()
        cfg.llm.api_key = os.environ.get("DEEPSEEK_API_KEY", cfg.llm.api_key)
        return cfg
