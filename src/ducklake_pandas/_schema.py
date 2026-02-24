"""DuckDB SQL type to pandas/numpy type mapping."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


# DuckDB type string → pandas/numpy dtype string
_SIMPLE_TYPE_MAP: dict[str, str] = {
    # SQL standard names (uppercase)
    "BOOLEAN": "boolean",
    "BOOL": "boolean",
    "TINYINT": "Int8",
    "SMALLINT": "Int16",
    "INTEGER": "Int32",
    "INT": "Int32",
    "BIGINT": "Int64",
    "HUGEINT": "Int64",  # pandas has no Int128; use Int64
    "UTINYINT": "UInt8",
    "USMALLINT": "UInt16",
    "UINTEGER": "UInt32",
    "UINT": "UInt32",
    "UBIGINT": "UInt64",
    "UHUGEINT": "UInt64",  # pandas has no UInt128; use UInt64
    "FLOAT": "Float32",
    "REAL": "Float32",
    "DOUBLE": "Float64",
    "VARCHAR": "object",
    "TEXT": "object",
    "STRING": "object",
    "BLOB": "object",
    "BYTEA": "object",
    "DATE": "object",  # store as Python date objects
    "TIME": "object",  # store as Python time objects
    "TIMESTAMP": "datetime64[us]",
    "TIMESTAMP_S": "datetime64[s]",
    "TIMESTAMP_MS": "datetime64[ms]",
    "TIMESTAMP_NS": "datetime64[ns]",
    "TIMESTAMP WITH TIME ZONE": "datetime64[us]",
    "TIMESTAMPTZ": "datetime64[us]",
    "TIMESTAMP_TZ": "datetime64[us]",
    "INTERVAL": "object",
    "UUID": "object",
    "JSON": "object",
    "BIT": "object",
    "TIME_NS": "object",
    "TIMESTAMP_US": "datetime64[us]",
    "TIMETZ": "object",
    "TIME_TZ": "object",
    "TIME WITH TIME ZONE": "object",
    "GEOMETRY": "object",
    "VARIANT": "object",
    "UNKNOWN": "object",
    # DuckDB internal type names (lowercase, as stored in ducklake_column.column_type)
    "INT8": "Int8",
    "INT16": "Int16",
    "INT32": "Int32",
    "INT64": "Int64",
    "INT128": "Int64",
    "UINT8": "UInt8",
    "UINT16": "UInt16",
    "UINT32": "UInt32",
    "UINT64": "UInt64",
    "UINT128": "UInt64",
    "FLOAT32": "Float32",
    "FLOAT16": "Float32",
    "FLOAT64": "Float64",
}


def duckdb_type_to_pandas(type_str: str) -> str:
    """
    Convert a DuckDB SQL type string to a pandas dtype string.

    Parameters
    ----------
    type_str
        DuckDB type string as stored in ducklake_column.column_type.
        Examples: "BIGINT", "VARCHAR", "STRUCT(a INTEGER, b VARCHAR)",
                  "INTEGER[]", "DECIMAL(18,3)"

    Returns
    -------
    str
        A pandas-compatible dtype string.
    """
    t = type_str.strip().upper()

    if t in _SIMPLE_TYPE_MAP:
        return _SIMPLE_TYPE_MAP[t]

    # DECIMAL(precision, scale) / NUMERIC(precision, scale)
    if re.match(r"(?:DECIMAL|NUMERIC)\s*\(\s*\d+\s*,\s*\d+\s*\)", t):
        return "Float64"  # pandas doesn't have native Decimal dtype

    # VARCHAR(N) / CHAR(N) / BPCHAR(N)
    if re.match(r"(?:VARCHAR|CHAR|BPCHAR|CHARACTER VARYING|CHARACTER)\s*\(\s*\d+\s*\)", t):
        return "object"

    # Array types, LIST, STRUCT, MAP → store as object (Python objects)
    if t.endswith("[]") or t.startswith("LIST") or t.startswith("STRUCT") or t.startswith("MAP"):
        return "object"

    # Fallback
    msg = f"Unsupported DuckDB type: {type_str}"
    raise ValueError(msg)


# Reverse mapping: pandas dtype → DuckDB internal type string
_PANDAS_TO_DUCKDB: dict[str, str] = {
    "bool": "boolean",
    "boolean": "boolean",
    "int8": "int8",
    "Int8": "int8",
    "int16": "int16",
    "Int16": "int16",
    "int32": "int32",
    "Int32": "int32",
    "int64": "int64",
    "Int64": "int64",
    "uint8": "uint8",
    "UInt8": "uint8",
    "uint16": "uint16",
    "UInt16": "uint16",
    "uint32": "uint32",
    "UInt32": "uint32",
    "uint64": "uint64",
    "UInt64": "uint64",
    "float32": "float32",
    "Float32": "float32",
    "float64": "float64",
    "Float64": "float64",
    "object": "varchar",
    "string": "varchar",
    "str": "varchar",
}


def pandas_dtype_to_duckdb(dtype: Any) -> str:
    """
    Convert a pandas dtype to a DuckDB internal type string.

    Returns lowercase type strings as stored in ``ducklake_column.column_type``.

    Parameters
    ----------
    dtype
        A pandas dtype, numpy dtype, or string representation.
    """
    dtype_str = str(dtype)

    # Check direct mapping
    result = _PANDAS_TO_DUCKDB.get(dtype_str)
    if result is not None:
        return result

    # Handle numpy dtypes
    if hasattr(dtype, 'kind'):
        kind = dtype.kind
        if kind == 'b':
            return "boolean"
        if kind == 'i':
            size = dtype.itemsize
            return {1: "int8", 2: "int16", 4: "int32", 8: "int64"}.get(size, "int64")
        if kind == 'u':
            size = dtype.itemsize
            return {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}.get(size, "uint64")
        if kind == 'f':
            size = dtype.itemsize
            return {2: "float32", 4: "float32", 8: "float64"}.get(size, "float64")
        if kind == 'M':
            return "timestamp"
        if kind == 'U' or kind == 'O':
            return "varchar"
        if kind == 'S':
            return "blob"

    # Handle pandas extension types by string
    if "datetime64" in dtype_str:
        if "ns" in dtype_str:
            return "timestamp_ns"
        if "ms" in dtype_str:
            return "timestamp_ms"
        return "timestamp"
    if "timedelta" in dtype_str:
        return "interval"

    # Fallback
    return "varchar"


def resolve_column_type(
    column_id: int,
    column_type: str,
    all_columns: list,
) -> str:
    """
    Resolve a column's pandas dtype string using the DuckLake column hierarchy.

    DuckLake stores compound types (list, struct, map) as a parent-child
    hierarchy in the ducklake_column table. For pandas, compound types
    are stored as Python objects.

    Parameters
    ----------
    column_id
        The column_id of this column.
    column_type
        The type string (e.g. "list", "struct", "map", "int32", "varchar").
    all_columns
        All columns for this table (including nested children).

    Returns
    -------
    str
        A pandas-compatible dtype string.
    """
    t = column_type.strip().upper()

    if t in ("LIST", "STRUCT", "MAP"):
        return "object"

    return duckdb_type_to_pandas(column_type)
