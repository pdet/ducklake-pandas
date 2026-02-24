"""Unit tests for the DuckDB type -> pandas type mapping."""

from __future__ import annotations

import pytest

from ducklake_pandas._schema import duckdb_type_to_pandas


class TestSimpleTypes:
    """Test mapping of simple scalar types."""

    @pytest.mark.parametrize(
        "duckdb_type,expected",
        [
            # SQL standard names
            ("BOOLEAN", "boolean"),
            ("TINYINT", "Int8"),
            ("SMALLINT", "Int16"),
            ("INTEGER", "Int32"),
            ("BIGINT", "Int64"),
            ("HUGEINT", "Int64"),
            ("UTINYINT", "UInt8"),
            ("USMALLINT", "UInt16"),
            ("UINTEGER", "UInt32"),
            ("UBIGINT", "UInt64"),
            ("UHUGEINT", "UInt64"),
            ("FLOAT", "Float32"),
            ("DOUBLE", "Float64"),
            ("VARCHAR", "object"),
            ("BLOB", "object"),
            ("DATE", "object"),
            ("TIME", "object"),
            ("TIMESTAMP", "datetime64[us]"),
            ("TIMESTAMP_S", "datetime64[s]"),
            ("TIMESTAMP_MS", "datetime64[ms]"),
            ("TIMESTAMP_NS", "datetime64[ns]"),
            ("TIMESTAMP WITH TIME ZONE", "datetime64[us]"),
            ("TIMESTAMPTZ", "datetime64[us]"),
            ("INTERVAL", "object"),
            ("UUID", "object"),
            ("JSON", "object"),
            # DuckLake internal type names (lowercase)
            ("boolean", "boolean"),  # maps through BOOLEAN
            ("int8", "Int8"),
            ("int16", "Int16"),
            ("int32", "Int32"),
            ("int64", "Int64"),
            ("int128", "Int64"),
            ("uint8", "UInt8"),
            ("uint16", "UInt16"),
            ("uint32", "UInt32"),
            ("uint64", "UInt64"),
            ("uint128", "UInt64"),
            ("float32", "Float32"),
            ("float64", "Float64"),
            ("varchar", "object"),
            ("blob", "object"),
            ("date", "object"),
            ("time", "object"),
            ("time_ns", "object"),
            ("timestamp", "datetime64[us]"),
            ("timestamp_us", "datetime64[us]"),
            ("timestamp_s", "datetime64[s]"),
            ("timestamp_ms", "datetime64[ms]"),
            ("timestamp_ns", "datetime64[ns]"),
            ("timestamptz", "datetime64[us]"),
            ("timetz", "object"),
            ("interval", "object"),
            ("uuid", "object"),
            ("json", "object"),
            ("geometry", "object"),
            ("variant", "object"),
            ("unknown", "object"),
        ],
    )
    def test_simple_type_mapping(self, duckdb_type, expected):
        result = duckdb_type_to_pandas(duckdb_type)
        assert result == expected


class TestDecimalType:
    """Test DECIMAL type mapping."""

    def test_decimal(self):
        result = duckdb_type_to_pandas("DECIMAL(18, 3)")
        assert result == "Float64"

    def test_decimal_no_spaces(self):
        result = duckdb_type_to_pandas("DECIMAL(10,2)")
        assert result == "Float64"


class TestComplexTypes:
    """Test complex/nested type mapping."""

    def test_list_bracket_syntax(self):
        result = duckdb_type_to_pandas("INTEGER[]")
        assert result == "object"

    def test_list_function_syntax(self):
        result = duckdb_type_to_pandas("LIST(VARCHAR)")
        assert result == "object"

    def test_struct(self):
        result = duckdb_type_to_pandas("STRUCT(a INTEGER, b VARCHAR)")
        assert result == "object"

    def test_map(self):
        result = duckdb_type_to_pandas("MAP(VARCHAR, INTEGER)")
        assert result == "object"

    def test_nested_list_of_structs(self):
        result = duckdb_type_to_pandas("STRUCT(x INTEGER, y VARCHAR)[]")
        assert result == "object"

    def test_nested_struct_with_list(self):
        result = duckdb_type_to_pandas("STRUCT(a INTEGER, b INTEGER[])")
        assert result == "object"

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported"):
            duckdb_type_to_pandas("COMPLETELY_FAKE_TYPE")

    def test_decimal_lowercase(self):
        """DuckLake stores decimal as lowercase 'decimal(w,s)'."""
        result = duckdb_type_to_pandas("decimal(18,3)")
        assert result == "Float64"

    def test_numeric_type(self):
        result = duckdb_type_to_pandas("NUMERIC(10,2)")
        assert result == "Float64"


class TestParameterizedTypes:
    """Test types with length/precision parameters."""

    def test_varchar_with_length(self):
        result = duckdb_type_to_pandas("VARCHAR(255)")
        assert result == "object"

    def test_char_with_length(self):
        result = duckdb_type_to_pandas("CHAR(10)")
        assert result == "object"

    def test_bpchar_with_length(self):
        result = duckdb_type_to_pandas("BPCHAR(5)")
        assert result == "object"
