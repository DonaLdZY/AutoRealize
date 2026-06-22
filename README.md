# AutoRealize

AutoRealize is the upstream task-realization component for AutoDecision. It converts a raw data directory plus an optional natural-language requirement into an AutoML-ready task package:

- `description.md`
- `sample_submission.csv`
- process reports and telemetry under `realize_report/`

AutoRealize now contains two core stages:

1. Data cognition: inspect the directory tree, read files, profile tabular data, understand documents/images/archives/JSON, and discover cross-file relations.
2. Task definition: synthesize a Kaggle-style task description and submission format from the requirement and data cognition output.

## Capabilities

- Two-stage workflow: `DataCognitionModule` and `TaskDefinitionModule`.
- Parser registry for CSV, XLSX, JSON, TXT/MD, DOCX, PDF, TOML/YAML, images, and archives.
- Parallel data cognition for file reading and table profiling.
- Filename-aware reasoning for files such as `train`, `test`, `sampleSubmission`, `readme`, and requirement documents.
- JSON handling for both tabular JSON and nested/config-like JSON.
- Compact image-directory cognition with optional VLLM image summaries.
- Constraint memory extracted from documents, fields, metrics, time clues, and business rules.
- Kaggle-style `description.md` with task goal, data inventory, field descriptions, metric, validation protocol, submission format, constraints, and risks.
- `sample_submission.csv` generation or reuse, with format validation.
- Frontend-friendly observability through terminal logs, `event_stream.jsonl`, `current_state.json`, and `frontend_manifest.json`.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended. Windows, macOS, and Linux are supported.

## API Key

The text LLM uses a DeepSeek/OpenAI-compatible endpoint by default.

Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="sk-xxxx"
```

Linux/macOS:

```bash
export DEEPSEEK_API_KEY="sk-xxxx"
```

Default text model settings:

- `base_url = "https://api.deepseek.com"`
- `model_name = "deepseek-v4-pro"`

Vision model settings are in `VLLMConfig`.

## Quick Start

```powershell
python -m autorealize.cli `
  --input-root "Sample/Sample1/raw2" `
  --output-root "runs" `
  --task "forecast next month sales" `
  --run-name "run_001_sales_demo"
```

LLM configuration and network access are required. AutoRealize fails fast when the LLM API key or endpoint is unavailable.

Generate a demo `predict_split` when no independent prediction set exists:

```powershell
python -m autorealize.cli `
  --input-root "Sample/Sample1/raw2" `
  --output-root "runs" `
  --task "forecast next month sales" `
  --run-name "run_003_predict_split" `
  --auto-generate-predict-split
```

## Configuration

Print the default configuration:

```powershell
python -m autorealize.cli --print-default-config
```

Write the default configuration:

```powershell
python -m autorealize.cli --write-default-config config.json
```

Run with a configuration file:

```powershell
python -m autorealize.cli `
  --input-root "path/to/data" `
  --output-root "runs" `
  --task "business requirement" `
  --run-name "run_004_config" `
  --config "config.json"
```

Common fields:

- `switches.run_data_cognition`: run the data cognition stage.
- `switches.run_task_definition`: run task definition and submission generation.
- `data.auto_generate_predict_split`: generate a demo prediction split when no independent prediction set is detected.
- `data.table_profile_sample_rows`: maximum rows used for table profiling.
- `data.generated_sample_submission_max_rows`: maximum sample rows when a generated submission is only a format example.
- `parallel.enable_parallel_cognition`: enable parallel data cognition.
- `parallel.cognition_max_workers`: data cognition worker count.
- `telemetry.enabled`: write event stream and current state snapshots.
- `llm.enable_cache`: cache LLM responses.
- `llm.request_timeout_seconds`: per-request LLM timeout.
- `llm.max_concurrent_requests`: maximum concurrent LLM requests.
- `knowledge.enabled`: write the local knowledge store.

Each run writes `realize_report/config_schema.json`, which can be used by a frontend to build a configuration panel.

## Output Layout

Run directory: `<output-root>/<run-name>/`

Root-level artifacts for downstream AutoML:

- `description.md`: Kaggle-style task statement.
- `sample_submission.csv`: sample submission file.
- `description_origin.md`: backup when the input data already contains `description.md`.
- Data files: copied input data, flattened into the run root while preserving relative structure as much as possible.

Process reports under `realize_report/`:

- `data_description.md`: global data cognition document.
- `data_cognition_report.json`: structured cognition report.
- `task_definition_report.json`: structured task-definition report.
- `submission_report.json`: sample submission generation/reuse/validation report.
- `file_cognition/`: per-file cognition JSON/Markdown artifacts.
- `constraint_memory.json`: cross-stage constraint memory.
- `knowledge_base.json`: knowledge-base summary.
- `knowledge_store.jsonl`: local knowledge entries.
- `retrieved_knowledge.json`: knowledge retrieved for task definition.
- `rag_manifest.json`: manifest for future RAG/vector-store integration.
- `trajectory_events.jsonl`: legacy trajectory events.
- `trajectory.md`: trajectory index.
- `llm_traces.jsonl`: LLM request/response traces.
- `run_summary.json`: run summary.

## Frontend Integration

Recommended frontend entrypoints:

- `realize_report/frontend_manifest.json`: module/artifact/event-source manifest.
- `realize_report/event_stream.jsonl`: append-only structured event stream.
- `realize_report/current_state.json`: current state snapshot for polling.
- `realize_report/event_taxonomy.json`: event taxonomy and field descriptions.
- `realize_report/final_config.json`: resolved config used in the run.
- `realize_report/config_schema.json`: config schema and field descriptions.

Suggested frontend flow:

1. Load `frontend_manifest.json` to initialize module cards and artifact links.
2. Poll `current_state.json` for overall status, active component, and recent events.
3. Incrementally read `event_stream.jsonl` by `seq` for detailed timelines.
4. Use `classification.layer/scope` to group events by workflow lane.
5. Use `config_schema.json` to generate configuration forms.

## Workflow

### Data Cognition

1. Copy the input directory into the run workspace.
2. Write `directory_tree.txt`.
3. Apply filename-pattern sampling to avoid reading every file in massive homogeneous groups.
4. Read files in parallel:
   - Tables: columns, preview rows, numeric/categorical/datetime stats, nulls, abnormal tokens.
   - JSON: tabular expansion or nested-structure summary.
   - Documents: context, requirements, constraints, and data notes.
   - Images: metadata plus optional VLLM visual summary.
   - Archives: extraction or structure logging.
5. Write per-file cognition artifacts.
6. Discover cross-file relations, field alignment, time clues, and metric candidates.
7. Build `knowledge_base.json` and `knowledge_store.jsonl`.

### Task Definition

1. Read the natural-language requirement and data cognition output.
2. Retrieve relevant fields, constraints, metrics, and document notes from the knowledge store.
3. Classify the task type: regression, classification, time-series forecasting, recommendation, optimization, or decision modeling.
4. Infer the prediction/decision unit, target field, train/predict boundary, and leakage constraints.
5. Define a precise evaluation protocol: primary metric, formula, `y_true` source, split method, random seed, and reporting rules.
6. Write `description.md`.
7. Reuse or generate `sample_submission.csv` and validate the format.

## Tests

```powershell
pytest -q
```

Current test coverage includes:

- Output layout.
- JSON table handling and content preservation.
- Archive handling.
- Image cognition.
- Sample submission reuse/generation/validation.
- Frontend event stream and manifest.

## Current Positioning

AutoRealize is a research demo and engineering skeleton. It prioritizes a runnable two-stage workflow, AutoML-ready artifacts, observable execution, configurable behavior, and a replaceable local knowledge store.
