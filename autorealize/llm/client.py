from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ..config import AutoRealizeConfig
from ..models import LLMTrace

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("未找到 JSON 对象")
    return match.group(0)


class LLMClient:
    """OpenAI 兼容接口客户端（DeepSeek）。"""

    def __init__(self, config: AutoRealizeConfig, run_dir: Path) -> None:
        self.config = config
        if not self.config.llm.api_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY。")
        self.client = OpenAI(api_key=self.config.llm.api_key, base_url=self.config.llm.base_url)
        self.trace_path = run_dir / "llm_traces.jsonl"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_trace(self, trace: LLMTrace) -> None:
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace.model_dump(), ensure_ascii=False) + "\n")

    def ask_text(self, system_prompt: str, user_prompt: str, prompt_name: str) -> str:
        logger.info("[LLM] 正在生成文本: prompt=%s model=%s", prompt_name, self.config.llm.model_name)
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.config.llm.model_name,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        text = response.choices[0].message.content or ""
        dt = time.perf_counter() - t0
        logger.info("[LLM] 生成完成: prompt=%s | %.2fs", prompt_name, dt)
        self._log_trace(
            LLMTrace(
                prompt_name=prompt_name,
                request=user_prompt[:10000],
                response=text[:12000],
                parsed_ok=True,
            )
        )
        return text

    def ask_structured(
        self,
        model_cls: type[T],
        system_prompt: str,
        user_prompt: str,
        prompt_name: str,
        fewshot: str = "",
    ) -> T:
        schema_text = json.dumps(model_cls.model_json_schema(), ensure_ascii=False, indent=2)
        base_user = (
            f"{user_prompt}\n\n"
            f"你必须严格输出一个 JSON 对象，且必须满足以下 JSON Schema：\n{schema_text}\n"
            "禁止输出解释性文字、Markdown、代码块。"
        )
        if fewshot:
            base_user = f"参考few-shot示例：\n{fewshot}\n\n{base_user}"

        last_error = ""
        for attempt in range(1, self.config.llm.max_retries + 1):
            logger.info(
                "[LLM] 正在生成结构化结果: prompt=%s attempt=%s/%s model=%s",
                prompt_name,
                attempt,
                self.config.llm.max_retries,
                self.config.llm.model_name,
            )
            t0 = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": base_user},
                    *(
                        [
                            {
                                "role": "user",
                                "content": (
                                    "上次输出不合格，错误如下：\n"
                                    f"{last_error}\n"
                                    "现在只输出合法 JSON 对象。"
                                ),
                            }
                        ]
                        if last_error and self.config.llm.enforce_json_retry
                        else []
                    ),
                ],
                stream=False,
            )
            text = response.choices[0].message.content or ""
            try:
                json_obj = json.loads(_extract_json_object(text))
                parsed = model_cls.model_validate(json_obj)
                dt = time.perf_counter() - t0
                logger.info("[LLM] 生成完成: prompt=%s attempt=%s | %.2fs", prompt_name, attempt, dt)
                self._log_trace(
                    LLMTrace(
                        prompt_name=prompt_name,
                        request=base_user[:12000],
                        response=text[:12000],
                        parsed_ok=True,
                    )
                )
                return parsed
            except (ValueError, json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                self._log_trace(
                    LLMTrace(
                        prompt_name=prompt_name,
                        request=base_user[:12000],
                        response=text[:12000],
                        parsed_ok=False,
                        error=last_error[:2000],
                    )
                )
                logger.warning("结构化输出失败，attempt=%s, error=%s", attempt, last_error)
        raise RuntimeError(f"LLM 结构化输出失败: {last_error}")
