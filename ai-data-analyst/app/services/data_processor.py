import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Encodings to try in order (covers UTF-8, Western European, and Thai)
_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "tis-620", "cp874"]
# Common CSV delimiters to probe
_DELIMITERS = [",", ";", "\t", "|"]


class DataProcessor:
    """Handles loading, inspection, cleaning, and statistical analysis of tabular data."""

    # ------------------------------------------------------------------ #
    # 1. load_file                                                         #
    # ------------------------------------------------------------------ #

    def load_file(self, filepath: str) -> pd.DataFrame:
        """Load CSV / XLSX / JSON into a DataFrame.

        Auto-detects encoding (utf-8, latin-1, tis-620) and CSV delimiter.
        Raises FileNotFoundError / ValueError on bad input.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = path.suffix.lower()
        logger.info("Loading file: %s (type=%s)", filepath, ext)

        try:
            if ext == ".csv":
                return self._load_csv(path)
            elif ext in (".xlsx", ".xls"):
                return self._load_excel(path)
            elif ext == ".json":
                return self._load_json(path)
            else:
                raise ValueError(f"Unsupported file type: '{ext}'. Allowed: csv, xlsx, xls, json")
        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            logger.error("Failed to load %s: %s", filepath, exc)
            raise RuntimeError(f"Could not load file '{filepath}': {exc}") from exc

    def _load_csv(self, path: Path) -> pd.DataFrame:
        # Try each encoding
        for encoding in _ENCODINGS:
            for delimiter in _DELIMITERS:
                try:
                    df = pd.read_csv(
                        path,
                        encoding=encoding,
                        sep=delimiter,
                        engine="python",
                    )
                    # A successful parse with the right delimiter produces > 1 column
                    # (unless the file genuinely has one column — accept it anyway)
                    if df.shape[1] >= 1:
                        logger.debug("CSV loaded with encoding=%s, delimiter=%r", encoding, delimiter)
                        return df
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
        raise RuntimeError(f"Could not decode CSV file '{path}' with any known encoding/delimiter.")

    def _load_excel(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_excel(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read Excel file: {exc}") from exc

    def _load_json(self, path: Path) -> pd.DataFrame:
        for encoding in _ENCODINGS:
            try:
                df = pd.read_json(path, encoding=encoding)
                logger.debug("JSON loaded with encoding=%s", encoding)
                return df
            except (UnicodeDecodeError, ValueError):
                continue
        raise RuntimeError(f"Could not decode JSON file '{path}' with any known encoding.")

    # ------------------------------------------------------------------ #
    # 2. get_overview                                                      #
    # ------------------------------------------------------------------ #

    def get_overview(self, df: pd.DataFrame) -> dict:
        """Return a comprehensive overview dict for a DataFrame."""
        try:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

            missing_counts = df.isnull().sum()
            missing_pct = (df.isnull().mean() * 100).round(2)

            missing_values = {
                col: {
                    "count": int(missing_counts[col]),
                    "percent": float(missing_pct[col]),
                }
                for col in df.columns
            }

            # Human-readable dtype mapping
            dtype_map = {
                col: self._friendly_dtype(df[col])
                for col in df.columns
            }

            preview = df.head(5).replace({np.nan: None}).to_dict(orient="records")

            overview = {
                "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "column_names": df.columns.tolist(),
                "dtypes": dtype_map,
                "missing_values": missing_values,
                "duplicates": int(df.duplicated().sum()),
                "memory_usage": self._memory_usage(df),
                "preview": preview,
                "numeric_columns": num_cols,
                "categorical_columns": cat_cols,
                "datetime_columns": dt_cols,
            }

            logger.info("Overview generated for DataFrame %s", df.shape)
            return overview

        except Exception as exc:
            logger.error("get_overview failed: %s", exc)
            raise RuntimeError(f"Failed to generate overview: {exc}") from exc

    def _friendly_dtype(self, series: pd.Series) -> str:
        dtype = series.dtype
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if pd.api.types.is_float_dtype(dtype):
            return "float"
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        return "text"

    def _memory_usage(self, df: pd.DataFrame) -> dict:
        total_bytes = df.memory_usage(deep=True).sum()
        return {
            "total_bytes": int(total_bytes),
            "total_mb": round(total_bytes / (1024 ** 2), 4),
        }

    # ------------------------------------------------------------------ #
    # 3. auto_clean                                                        #
    # ------------------------------------------------------------------ #

    def auto_clean(
        self, df: pd.DataFrame, options: dict = {}
    ) -> tuple[pd.DataFrame, dict]:
        """Clean a DataFrame and return (cleaned_df, cleaning_report).

        Steps:
          1. Standardize column names
          2. Drop columns with > 50 % missing (configurable)
          3. Impute numeric → median, categorical → mode
          4. Remove duplicate rows
          5. Auto-parse date columns
        """
        try:
            report: dict[str, Any] = {
                "before": {
                    "shape": list(df.shape),
                    "missing_total": int(df.isnull().sum().sum()),
                    "duplicates": int(df.duplicated().sum()),
                },
                "actions": [],
            }

            df = df.copy()

            # 1. Standardize column names
            original_cols = df.columns.tolist()
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(r"[\s\-]+", "_", regex=True)
                .str.replace(r"[^\w]", "", regex=True)
            )
            renamed = {o: n for o, n in zip(original_cols, df.columns) if o != n}
            if renamed:
                report["actions"].append({"step": "rename_columns", "renamed": renamed})

            # 2. Drop columns with too many missing values
            threshold = options.get("drop_missing_threshold", 0.5)
            missing_ratio = df.isnull().mean()
            cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                report["actions"].append({
                    "step": "drop_high_missing_columns",
                    "dropped": cols_to_drop,
                    "threshold": threshold,
                })

            # 3. Impute missing values
            imputed: dict[str, str] = {}
            for col in df.columns:
                if df[col].isnull().sum() == 0:
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_val = df[col].median()
                    df[col] = df[col].fillna(fill_val)
                    imputed[col] = f"median ({fill_val:.4g})"
                else:
                    mode_vals = df[col].mode()
                    if not mode_vals.empty:
                        df[col] = df[col].fillna(mode_vals[0])
                        imputed[col] = f"mode ({mode_vals[0]})"
            if imputed:
                report["actions"].append({"step": "impute_missing", "columns": imputed})

            # 4. Remove duplicates
            before_dedup = len(df)
            df = df.drop_duplicates()
            removed = before_dedup - len(df)
            if removed:
                report["actions"].append({"step": "remove_duplicates", "removed": removed})

            # 5. Auto-parse date columns
            parsed_dates: list[str] = []
            for col in df.select_dtypes(include="object").columns:
                if any(kw in col for kw in ("date", "time", "dt", "timestamp")):
                    try:
                        df[col] = pd.to_datetime(df[col])
                        parsed_dates.append(col)
                    except Exception:
                        pass
            if parsed_dates:
                report["actions"].append({"step": "parse_dates", "columns": parsed_dates})

            report["after"] = {
                "shape": list(df.shape),
                "missing_total": int(df.isnull().sum().sum()),
                "duplicates": int(df.duplicated().sum()),
            }

            logger.info(
                "auto_clean done: %s → %s, actions=%d",
                report["before"]["shape"],
                report["after"]["shape"],
                len(report["actions"]),
            )
            return df, report

        except Exception as exc:
            logger.error("auto_clean failed: %s", exc)
            raise RuntimeError(f"Cleaning failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # 4. get_statistics                                                    #
    # ------------------------------------------------------------------ #

    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Return descriptive statistics, value counts, correlations, skew/kurtosis."""
        try:
            num_df = df.select_dtypes(include="number")
            cat_df = df.select_dtypes(include="object")

            # Numeric describe
            numeric_stats: dict[str, Any] = {}
            if not num_df.empty:
                desc = num_df.describe().round(4)
                desc.loc["skewness"] = num_df.skew().round(4)
                desc.loc["kurtosis"] = num_df.kurtosis().round(4)
                numeric_stats = desc.to_dict()

            # Categorical value counts (top 10)
            categorical_stats: dict[str, Any] = {}
            for col in cat_df.columns:
                vc = df[col].value_counts().head(10)
                categorical_stats[col] = {
                    "unique_count": int(df[col].nunique()),
                    "top_values": vc.to_dict(),
                }

            # Correlation matrix
            correlations: dict[str, Any] = {}
            if num_df.shape[1] >= 2:
                correlations = num_df.corr().round(4).to_dict()

            # Per-column skew & kurtosis
            skew_kurt: dict[str, Any] = {}
            for col in num_df.columns:
                skew_kurt[col] = {
                    "skewness": round(float(num_df[col].skew()), 4),
                    "kurtosis": round(float(num_df[col].kurtosis()), 4),
                }

            logger.info("Statistics computed for %d columns", df.shape[1])
            return {
                "numeric_stats": numeric_stats,
                "categorical_stats": categorical_stats,
                "correlations": correlations,
                "skewness_kurtosis": skew_kurt,
            }

        except Exception as exc:
            logger.error("get_statistics failed: %s", exc)
            raise RuntimeError(f"Failed to compute statistics: {exc}") from exc

    # ------------------------------------------------------------------ #
    # 5. detect_column_types                                               #
    # ------------------------------------------------------------------ #

    def detect_column_types(self, df: pd.DataFrame) -> dict[str, str]:
        """Classify each column into a semantic type.

        Types:
            "id"                 – unique > 90 % of rows
            "boolean"            – exactly 2 unique non-null values
            "datetime"           – datetime dtype or parseable date strings
            "numeric_continuous" – float dtype
            "numeric_discrete"   – int dtype with < 20 unique values
            "categorical"        – object dtype or int with < 20 unique values
            "text"               – object dtype with avg string length > 50
        """
        try:
            n_rows = len(df)
            result: dict[str, str] = {}

            for col in df.columns:
                series = df[col].dropna()
                n_unique = series.nunique()
                dtype = df[col].dtype

                # Datetime dtype
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    result[col] = "datetime"
                    continue

                # Boolean dtype or 2-value columns
                if pd.api.types.is_bool_dtype(dtype) or n_unique == 2:
                    result[col] = "boolean"
                    continue

                # ID-like: nearly all values are unique
                if n_rows > 0 and (n_unique / n_rows) > 0.9:
                    result[col] = "id"
                    continue

                # Object / string columns
                if pd.api.types.is_object_dtype(dtype):
                    # Try to detect hidden datetime strings
                    if self._looks_like_datetime(series):
                        result[col] = "datetime"
                        continue
                    # Long free text
                    avg_len = series.astype(str).str.len().mean()
                    if avg_len > 50:
                        result[col] = "text"
                        continue
                    result[col] = "categorical"
                    continue

                # Numeric
                if pd.api.types.is_float_dtype(dtype):
                    result[col] = "numeric_continuous"
                    continue

                if pd.api.types.is_integer_dtype(dtype):
                    if n_unique < 20:
                        result[col] = "numeric_discrete"
                    else:
                        result[col] = "numeric_continuous"
                    continue

                # Fallback
                result[col] = "categorical"

            logger.info("Column types detected for %d columns", len(result))
            return result

        except Exception as exc:
            logger.error("detect_column_types failed: %s", exc)
            raise RuntimeError(f"Failed to detect column types: {exc}") from exc

    def _looks_like_datetime(self, series: pd.Series, sample_size: int = 50) -> bool:
        """Heuristic: try parsing a sample of string values as datetimes."""
        sample = series.dropna().head(sample_size).astype(str)
        if sample.empty:
            return False
        parsed = 0
        for val in sample:
            try:
                pd.to_datetime(val)
                parsed += 1
            except Exception:
                pass
        return (parsed / len(sample)) > 0.8
