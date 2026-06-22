from __future__ import annotations

from ..config import AutoRealizeConfig
from .archive_parser import ArchiveParser
from .docx_parser import DocxParser
from .image_parser import ImageParser
from .json_toml_parser import JsonParser, TomlParser
from .pdf_parser import PdfParser
from .registry import ParserRegistry
from .table_parser import TableParser
from .text_parser import TextParser


def build_registry(config: AutoRealizeConfig) -> ParserRegistry:
    registry = ParserRegistry()
    registry.register(ArchiveParser())
    registry.register(TextParser(encodings=config.data.text_encodings))
    registry.register(DocxParser())
    registry.register(PdfParser())
    registry.register(
        TableParser(
            preview_rows=config.data.preview_rows,
            sample_rows=config.data.table_profile_sample_rows,
        )
    )
    registry.register(
        JsonParser(
            flatten_sep=config.data.json_flatten_sep,
            flatten_max_level=config.data.json_flatten_max_level,
            keep_raw_nested_columns=config.data.json_keep_raw_nested_columns,
            preview_rows=config.data.preview_rows,
        )
    )
    registry.register(TomlParser())
    if config.data.extract_image_metadata:
        registry.register(ImageParser())
    return registry
