from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import BaseParser, ParsedFile


class TableParser(BaseParser):
    supported_suffixes = (".csv", ".xlsx", ".xls")
    kind = "table"

    def __init__(self, preview_rows: int) -> None:
        self.preview_rows = preview_rows

    def _read(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                return pd.read_csv(path)
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="gb18030")
        return pd.read_excel(path)

    def parse(self, path: Path) -> ParsedFile:
        df = self._read(path)
        preview = df.head(self.preview_rows).to_dict(orient="records")
        return ParsedFile(
            path=path,
            kind=self.kind,
            text_summary=f"表格行列: {df.shape[0]} x {df.shape[1]}",
            preview=preview,
            columns=[str(c) for c in df.columns.tolist()],
            metadata={
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            },
        )
