"""Non-destructive loading and cleaning helpers for tabular research data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
import re
from typing import BinaryIO

import numpy as np
import pandas as pd


MISSING_SENTINELS = {
    "",
    ".",
    "-",
    "--",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "missing",
    "欠損",
    "不明",
}


@dataclass
class CleaningReport:
    """Audit trail describing every automatic cleaning operation."""

    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    renamed_columns: dict[str, str] = field(default_factory=dict)
    dropped_empty_rows: int = 0
    dropped_empty_columns: list[str] = field(default_factory=list)
    dropped_duplicate_rows: int = 0
    numeric_conversions: list[str] = field(default_factory=list)
    datetime_conversions: list[str] = field(default_factory=list)
    missing_values_standardized: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rewind(source: str | Path | BinaryIO) -> None:
    if hasattr(source, "seek"):
        source.seek(0)


def load_dataset(source: str | Path | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    """Load CSV, TSV, XLSX, or XLS data from a path or file-like object.

    CSV decoding tries UTF-8 (including BOM) and Japanese CP932. Delimiters are
    inferred, so comma-, tab-, and semicolon-separated files are accepted.
    """

    name = filename or getattr(source, "name", None) or str(source)
    suffix = Path(name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        _rewind(source)
        return pd.read_excel(source)
    if suffix not in {".csv", ".tsv", ".txt", ""}:
        raise ValueError(f"Unsupported file type: {suffix}. Use CSV, TSV, XLSX, or XLS.")

    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            _rewind(source)
            return pd.read_csv(source, sep=None, engine="python", encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not read the delimited file. " + " | ".join(errors))


def _normalise_column_name(value: object, position: int) -> str:
    text = str(value).strip()
    if not text or text.lower().startswith("unnamed"):
        text = f"column_{position + 1}"
    text = re.sub(r"[\s\-/\\:;,.()\[\]{}]+", "_", text)
    text = re.sub(r"[^\w]+", "", text, flags=re.UNICODE).strip("_")
    if not text:
        text = f"column_{position + 1}"
    if text[0].isdigit():
        text = "col_" + text
    return text


def _unique_names(columns: list[object]) -> tuple[list[str], dict[str, str]]:
    used: dict[str, int] = {}
    names: list[str] = []
    renamed: dict[str, str] = {}
    for position, original in enumerate(columns):
        base = _normalise_column_name(original, position)
        count = used.get(base, 0)
        used[base] = count + 1
        new = base if count == 0 else f"{base}_{count + 1}"
        names.append(new)
        if str(original) != new:
            renamed[str(original)] = new
    return names, renamed


def _standardise_strings(series: pd.Series) -> tuple[pd.Series, int]:
    before_missing = int(series.isna().sum())

    def convert(value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return np.nan if stripped.lower() in MISSING_SENTINELS else stripped

    result = series.map(convert)
    return result, int(result.isna().sum()) - before_missing


def _numeric_candidate(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    is_percent = text.str.endswith("%", na=False)
    cleaned = (
        text.str.replace(",", "", regex=False)
        .str.replace(r"^[¥￥$€£]", "", regex=True)
        .str.replace("%", "", regex=False)
    )
    # Force a NumPy float dtype so mixed percentage/non-percentage values can
    # be divided safely on pandas versions that infer nullable integer dtype.
    numeric = pd.to_numeric(cleaned, errors="coerce").astype(float)
    if is_percent.any():
        numeric = numeric.where(~is_percent, numeric / 100.0)
    return numeric


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a cleaned copy and a complete cleaning audit report.

    The original DataFrame is never modified. Automatic type conversion is
    deliberately conservative: an object column becomes numeric only when at
    least 85% of its non-missing values parse successfully.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    result = df.copy(deep=True)
    original_shape = result.shape

    names, renamed = _unique_names(list(result.columns))
    result.columns = names

    standardised = 0
    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column]) or isinstance(
            result[column].dtype, pd.StringDtype
        ):
            result[column], changed = _standardise_strings(result[column])
            standardised += changed

    empty_columns = [str(c) for c in result.columns[result.isna().all()]]
    if empty_columns:
        result = result.drop(columns=empty_columns)
    before_rows = len(result)
    result = result.dropna(how="all")
    dropped_empty_rows = before_rows - len(result)

    before_dedup = len(result)
    result = result.drop_duplicates()
    dropped_duplicates = before_dedup - len(result)

    numeric_conversions: list[str] = []
    datetime_conversions: list[str] = []
    for column in list(result.columns):
        series = result[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.StringDtype)
        ):
            continue
        non_missing = int(series.notna().sum())
        if non_missing == 0:
            continue
        numeric = _numeric_candidate(series)
        ratio = float(numeric.notna().sum()) / non_missing
        if ratio >= 0.85:
            result[column] = numeric
            numeric_conversions.append(str(column))
            continue
        lowered = str(column).lower()
        if any(token in lowered for token in ("date", "datetime", "日付", "年月日")):
            converted = pd.to_datetime(series, errors="coerce")
            if float(converted.notna().sum()) / non_missing >= 0.85:
                result[column] = converted
                datetime_conversions.append(str(column))

    result = result.reset_index(drop=True)
    report = CleaningReport(
        original_shape=original_shape,
        cleaned_shape=result.shape,
        renamed_columns=renamed,
        dropped_empty_rows=dropped_empty_rows,
        dropped_empty_columns=empty_columns,
        dropped_duplicate_rows=dropped_duplicates,
        numeric_conversions=numeric_conversions,
        datetime_conversions=datetime_conversions,
        missing_values_standardized=standardised,
    )
    if result.empty:
        report.notes.append("No rows remain after cleaning.")
    return result, report


def dataframe_to_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to UTF-8 CSV bytes for downloads."""

    return df.to_csv(index=False).encode("utf-8-sig")
