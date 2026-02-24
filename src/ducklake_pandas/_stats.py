"""Statistics extraction for column stats computation."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from ducklake_pandas._schema import duckdb_type_to_pandas

if TYPE_CHECKING:
    from ducklake_pandas._catalog import ColumnInfo, ColumnStats, FileInfo


# Types that support min/max statistics
_STAT_TYPES = {
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64",
    "boolean", "object",
}


def _parse_stat_value(value: str | None, pandas_type: str) -> object:
    """Parse a DuckLake stat value string into a Python value."""
    if value is None or value == "NULL":
        return None

    # Strip surrounding quotes if present
    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        value = value[1:-1].replace("''", "'")

    try:
        if pandas_type in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"):
            return int(value)
        if pandas_type in ("Float32", "Float64"):
            return float(value)
        if pandas_type == "boolean":
            low = value.lower()
            if low in ("true", "1"):
                return True
            if low in ("false", "0"):
                return False
            return None
        if pandas_type == "object":
            return value
    except (ValueError, TypeError, ArithmeticError):
        return None

    return None


def _stat_value_to_str(value: object, col_type: str) -> str | None:
    """Serialize a Python value to a DuckLake stat string."""
    if value is None:
        return None
    t = col_type.lower()
    if t == "boolean":
        return "true" if value else "false"
    return str(value)


def is_stat_supported_type(col_type: str) -> bool:
    """Check if a DuckDB column type supports statistics."""
    try:
        pandas_type = duckdb_type_to_pandas(col_type)
        return pandas_type in _STAT_TYPES or "datetime64" in pandas_type
    except ValueError:
        return False
