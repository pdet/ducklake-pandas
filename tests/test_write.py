"""Tests for ducklake-pandas write path."""

from __future__ import annotations

import os

import duckdb
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ducklake_pandas import (
    create_ducklake_table,
    read_ducklake,
    write_ducklake,
)
from ducklake_pandas._schema import pandas_dtype_to_duckdb


# ---------------------------------------------------------------------------
# Reverse type mapping
# ---------------------------------------------------------------------------


class TestPandasTypeToDuckdb:
    """Test pandas_dtype_to_duckdb reverse mapping."""

    def test_integer_types(self):
        assert pandas_dtype_to_duckdb("Int8") == "int8"
        assert pandas_dtype_to_duckdb("Int16") == "int16"
        assert pandas_dtype_to_duckdb("Int32") == "int32"
        assert pandas_dtype_to_duckdb("Int64") == "int64"

    def test_unsigned_integer_types(self):
        assert pandas_dtype_to_duckdb("UInt8") == "uint8"
        assert pandas_dtype_to_duckdb("UInt16") == "uint16"
        assert pandas_dtype_to_duckdb("UInt32") == "uint32"
        assert pandas_dtype_to_duckdb("UInt64") == "uint64"

    def test_float_types(self):
        assert pandas_dtype_to_duckdb("Float32") == "float32"
        assert pandas_dtype_to_duckdb("Float64") == "float64"

    def test_string(self):
        assert pandas_dtype_to_duckdb("object") == "varchar"

    def test_boolean(self):
        assert pandas_dtype_to_duckdb("boolean") == "boolean"
        assert pandas_dtype_to_duckdb("bool") == "boolean"


# ---------------------------------------------------------------------------
# CREATE TABLE
# ---------------------------------------------------------------------------


class TestCreateTable:
    """Test create_ducklake_table."""

    def test_create_simple_table(self, make_write_catalog):
        cat = make_write_catalog()
        schema_dict = {"a": "int32", "b": "varchar", "c": "float64"}

        create_ducklake_table(cat.metadata_path, "test", schema_dict)

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (0, 3)
        assert list(result.columns) == ["a", "b", "c"]

    def test_create_table_already_exists(self, make_write_catalog):
        cat = make_write_catalog()
        schema_dict = {"a": "int32"}

        create_ducklake_table(cat.metadata_path, "test", schema_dict)

        with pytest.raises(ValueError, match="already exists"):
            create_ducklake_table(cat.metadata_path, "test", schema_dict)

    def test_create_table_metadata_correct(self, make_write_catalog):
        cat = make_write_catalog()
        schema_dict = {"a": "int32", "b": "varchar"}

        create_ducklake_table(cat.metadata_path, "test", schema_dict)

        row = cat.query_one(
            "SELECT table_name, path FROM ducklake_table WHERE table_name = 'test'"
        )
        assert row is not None
        assert row[0] == "test"
        assert row[1] == "test/"

        rows = cat.query_all(
            "SELECT column_name, column_type FROM ducklake_column "
            "WHERE table_id = (SELECT table_id FROM ducklake_table WHERE table_name = 'test') "
            "ORDER BY column_order"
        )
        assert rows == [("a", "int32"), ("b", "varchar")]

        row = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row is not None
        assert "created_table" in row[0]
        assert "test" in row[0]

    def test_create_table_duckdb_interop(self, make_write_catalog):
        """DuckDB can read a table created by ducklake-pandas."""
        cat = make_write_catalog()
        schema_dict = {"a": "int32", "b": "varchar"}

        create_ducklake_table(cat.metadata_path, "test", schema_dict)

        df = pd.DataFrame({"a": [1, 2], "b": ["hello", "world"]})
        write_ducklake(df, cat.metadata_path, "test", mode="append")

        pdf = cat.read_with_duckdb("test")
        assert len(pdf) == 2
        assert sorted(pdf["a"].tolist()) == [1, 2]

    def test_create_multiple_tables(self, make_write_catalog):
        cat = make_write_catalog()

        create_ducklake_table(cat.metadata_path, "t1", {"a": "int32"})
        create_ducklake_table(cat.metadata_path, "t2", {"x": "varchar", "y": "float64"})

        r1 = read_ducklake(cat.metadata_path, "t1")
        r2 = read_ducklake(cat.metadata_path, "t2")
        assert r1.shape == (0, 1)
        assert r2.shape == (0, 2)


# ---------------------------------------------------------------------------
# INSERT (write_ducklake)
# ---------------------------------------------------------------------------


class TestWriteDucklake:
    """Test write_ducklake with various modes."""

    def test_write_mode_error_new_table(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["x", "y", "z"]

    def test_write_mode_error_existing_table_raises(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="already exists"):
            write_ducklake(df, cat.metadata_path, "test", mode="error")

    def test_write_mode_append_new_table(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2]})

        write_ducklake(df, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].tolist()) == [1, 2]

    def test_write_mode_append_existing(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df2 = pd.DataFrame({"a": [3, 4], "b": ["z", "w"]})

        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 4
        assert sorted(result["a"].tolist()) == [1, 2, 3, 4]

    def test_write_mode_overwrite_replaces(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pd.DataFrame({"a": [10, 20], "b": ["new1", "new2"]})

        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="overwrite")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 2
        assert sorted(result["a"].tolist()) == [10, 20]

    def test_write_invalid_mode_raises(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})

        with pytest.raises(ValueError, match="Invalid write mode"):
            write_ducklake(df, cat.metadata_path, "test", mode="invalid")


# ---------------------------------------------------------------------------
# DuckDB interop
# ---------------------------------------------------------------------------


class TestDuckDBInterop:
    """Verify catalogs written by ducklake-pandas are readable by DuckDB."""

    def test_basic_interop(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["hello", "world", "test"]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        pdf = cat.read_with_duckdb("test")
        assert len(pdf) == 3
        assert sorted(pdf["a"].tolist()) == [1, 2, 3]

    def test_interop_multiple_inserts(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pd.DataFrame({"x": [10, 20]})
        df2 = pd.DataFrame({"x": [30]})

        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        pdf = cat.read_with_duckdb("test")
        assert sorted(pdf["x"].tolist()) == [10, 20, 30]

    def test_interop_after_overwrite(self, make_write_catalog):
        cat = make_write_catalog()
        df1 = pd.DataFrame({"v": [1, 2, 3]})
        df2 = pd.DataFrame({"v": [99]})

        write_ducklake(df1, cat.metadata_path, "test", mode="append")
        write_ducklake(df2, cat.metadata_path, "test", mode="overwrite")

        pdf = cat.read_with_duckdb("test")
        assert pdf["v"].tolist() == [99]

    def test_interop_null_values(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({
            "a": pd.array([1, None, 3], dtype="Int32"),
            "b": ["hello", None, "world"],
        })

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        pdf = cat.read_with_duckdb("test")
        assert len(pdf) == 3


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Write and read with ducklake-pandas."""

    def test_roundtrip_basic(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "value": [1.1, 2.2, 3.3],
        })

        write_ducklake(df, cat.metadata_path, "test", mode="error")
        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("id").reset_index(drop=True)
        df = df.sort_values("id").reset_index(drop=True)
        assert result["id"].tolist() == df["id"].tolist()
        assert result["name"].tolist() == df["name"].tolist()

    def test_roundtrip_append_then_read(self, make_write_catalog):
        cat = make_write_catalog()

        for i in range(5):
            df = pd.DataFrame({"batch": [i], "val": [i * 10]})
            write_ducklake(df, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 5
        assert sorted(result["batch"].tolist()) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Column statistics
# ---------------------------------------------------------------------------


class TestColumnStats:
    """Verify column statistics are correctly computed."""

    def test_stats_registered(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({
            "a": pd.array([1, 5, 3], dtype="Int32"),
            "b": ["x", "z", "y"],
        })

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        stats = cat.query_all(
            "SELECT column_id, null_count, min_value, max_value "
            "FROM ducklake_file_column_stats "
            "ORDER BY column_id"
        )
        assert len(stats) >= 2

        a_stats = [s for s in stats if s[0] == 1]
        assert len(a_stats) == 1
        assert a_stats[0][1] == 0  # null_count
        assert a_stats[0][2] == "1"  # min
        assert a_stats[0][3] == "5"  # max

    def test_stats_with_nulls(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({
            "a": pd.array([1, None, 3], dtype="Int32"),
        })

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        stats = cat.query_one(
            "SELECT null_count, min_value, max_value "
            "FROM ducklake_file_column_stats WHERE column_id = 1"
        )
        assert stats is not None
        assert stats[0] == 1  # null_count
        assert stats[1] == "1"  # min
        assert stats[2] == "3"  # max


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestWriteEdgeCases:
    """Test edge cases in the write path."""

    def test_write_single_row(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [42]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].tolist() == [42]

    def test_write_many_columns(self, make_write_catalog):
        cat = make_write_catalog()
        data = {f"col_{i}": [i] for i in range(20)}
        df = pd.DataFrame(data)

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 20)

    def test_snapshot_progression(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")
        snap_count_1 = cat.query_one("SELECT COUNT(*) FROM ducklake_snapshot")[0]

        write_ducklake(pd.DataFrame({"a": [2]}), cat.metadata_path, "test", mode="append")
        snap_count_2 = cat.query_one("SELECT COUNT(*) FROM ducklake_snapshot")[0]

        assert snap_count_2 > snap_count_1

    def test_parquet_file_naming(self, make_write_catalog):
        cat = make_write_catalog()
        df = pd.DataFrame({"a": [1]})

        write_ducklake(df, cat.metadata_path, "test", mode="error")

        row = cat.query_one("SELECT path FROM ducklake_data_file LIMIT 1")
        assert row is not None
        assert row[0].startswith("ducklake-")
        assert row[0].endswith(".parquet")

    def test_time_travel_after_write(self, make_write_catalog):
        cat = make_write_catalog()

        df1 = pd.DataFrame({"a": [1, 2]})
        write_ducklake(df1, cat.metadata_path, "test", mode="error")

        snap_after_create = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        df2 = pd.DataFrame({"a": [3, 4]})
        write_ducklake(df2, cat.metadata_path, "test", mode="append")

        result_latest = read_ducklake(cat.metadata_path, "test")
        assert result_latest.shape[0] == 4

        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_after_create
        )
        assert result_old.shape[0] == 2
        assert sorted(result_old["a"].tolist()) == [1, 2]


# ---------------------------------------------------------------------------
# DuckDB writes, ducklake-pandas reads (regression)
# ---------------------------------------------------------------------------


class TestDuckDBWritePandasRead:
    """Verify read path works with DuckDB-created catalogs."""

    def test_duckdb_write_pandas_read(self, tmp_path):
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        con.close()

        result = read_ducklake(metadata_path, "test")
        assert result.shape == (2, 2)
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == ["hello", "world"]
