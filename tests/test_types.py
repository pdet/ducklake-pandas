"""Type mapping tests for ducklake-pandas."""

from __future__ import annotations

import math

import pandas as pd
import numpy as np
import pytest

from ducklake_pandas import read_ducklake


class TestScalarTypes:
    """Test reading tables with various scalar DuckDB types."""

    def test_integer_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a TINYINT,
                b SMALLINT,
                c INTEGER,
                d BIGINT
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (1, 100, 10000, 1000000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert list(result.iloc[0]) == [1, 100, 10000, 1000000]

    def test_unsigned_integer_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a UTINYINT,
                b USMALLINT,
                c UINTEGER,
                d UBIGINT
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (1, 100, 10000, 1000000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert len(result) == 1
        assert list(result.columns) == ["a", "b", "c", "d"]

    def test_float_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT, b DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (3.14, 2.718281828)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert abs(result["a"].iloc[0] - 3.14) < 0.01
        assert abs(result["b"].iloc[0] - 2.718281828) < 1e-6

    def test_boolean(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a BOOLEAN)")
        cat.execute("INSERT INTO ducklake.test VALUES (true), (false), (NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        values = result["a"].tolist()
        assert values[0] is True or values[0] == True
        assert values[1] is False or values[1] == False

    def test_varchar(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES ('hello'), ('world'), ('')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].tolist() == ["hello", "world", ""]

    def test_date(self, ducklake_catalog):
        import datetime

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DATE)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15'), ('2024-12-31')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)

    def test_timestamp(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)

    def test_decimal(self, ducklake_catalog):
        from decimal import Decimal

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DECIMAL(10, 2))")
        cat.execute("INSERT INTO ducklake.test VALUES (123.45), (678.90)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)


class TestComplexTypes:
    """Test reading tables with complex/nested DuckDB types."""

    def test_list_type(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER[])")
        cat.execute("INSERT INTO ducklake.test VALUES ([1, 2, 3]), ([4, 5])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        values = result["a"].tolist()
        assert values[0] == [1, 2, 3]
        assert values[1] == [4, 5]

    def test_struct_type(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(x INTEGER, y VARCHAR))")
        cat.execute("INSERT INTO ducklake.test VALUES ({'x': 1, 'y': 'hello'})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        val = result["a"].iloc[0]
        assert val == {"x": 1, "y": "hello"}


class TestMixedColumns:
    """Test tables with multiple column types."""

    def test_mixed_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                id INTEGER,
                name VARCHAR,
                score DOUBLE,
                active BOOLEAN,
                created DATE
            )
        """)
        cat.execute("""
            INSERT INTO ducklake.test VALUES
                (1, 'Alice', 95.5, true, '2024-01-01'),
                (2, 'Bob', 87.3, false, '2024-01-02'),
                (3, 'Charlie', NULL, true, NULL)
        """)
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 5)
        assert result["id"].tolist() == [1, 2, 3]
        assert result["name"].tolist() == ["Alice", "Bob", "Charlie"]


class TestFloatEdgeCases:
    """Test float edge cases: NaN, infinity."""

    def test_float_nan(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT, b DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES ('NaN'::FLOAT, 'NaN'::DOUBLE)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 2)
        assert math.isnan(result["a"].iloc[0])
        assert math.isnan(result["b"].iloc[0])

    def test_float_infinity(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT, b DOUBLE)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES ('inf'::FLOAT, '-inf'::DOUBLE)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 2)
        assert math.isinf(result["a"].iloc[0]) and result["a"].iloc[0] > 0
        assert math.isinf(result["b"].iloc[0]) and result["b"].iloc[0] < 0


class TestStringEdgeCases:
    """Test string edge cases."""

    def test_varchar_empty_and_null(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (''), (NULL), ('hello')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 1)
        values = result["a"].tolist()
        assert "" in values
        assert "hello" in values
        # One value should be None/NaN
        null_count = sum(1 for v in values if v is None or (isinstance(v, float) and math.isnan(v)))
        assert null_count == 1
