# ducklake-pandas

> **This project is a proof of concept. It was 100% written by [Claude Code](https://docs.anthropic.com/en/docs/build-with-claude/claude-code/overview) (Anthropic's AI coding agent). It is not intended for production use.**

A pure-Python [Pandas](https://pandas.pydata.org/) integration for [DuckLake](https://ducklake.select/) catalogs — both read and write.

Reads and writes DuckLake metadata directly from SQLite or PostgreSQL and scans the underlying Parquet data files through Pandas' native Parquet reader. **No DuckDB runtime dependency.**

## Installation

```bash
pip install ducklake-pandas

# With PostgreSQL catalog support
pip install ducklake-pandas[postgres]
```

Runtime dependencies: `pandas >= 2.0` and `pyarrow >= 12.0`. SQLite catalogs use Python's built-in `sqlite3`. PostgreSQL catalogs require the `postgres` extra (adds `psycopg2`).

## Quick start

### Reading data

```python
from ducklake_pandas import read_ducklake

# Read a table into a DataFrame
df = read_ducklake("catalog.ducklake", "my_table")

# Select specific columns
df = read_ducklake("catalog.ducklake", "my_table", columns=["x", "y"])

# Time travel
df = read_ducklake("catalog.ducklake", "my_table", snapshot_version=3)
df = read_ducklake("catalog.ducklake", "my_table", snapshot_time="2025-01-15T10:30:00")

# PostgreSQL-backed catalog
df = read_ducklake("postgresql://user:pass@localhost/mydb", "my_table")
```

### Writing data

```python
import pandas as pd
from ducklake_pandas import write_ducklake

df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})

# Create and populate a new table
write_ducklake(df, "catalog.ducklake", "users", mode="error")

# Append rows
write_ducklake(new_rows, "catalog.ducklake", "users", mode="append")

# Overwrite all data
write_ducklake(df, "catalog.ducklake", "users", mode="overwrite")
```

### DDL operations

```python
from ducklake_pandas import (
    create_ducklake_table,
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    alter_ducklake_set_partitioned_by,
    drop_ducklake_table,
    rename_ducklake_table,
    create_ducklake_schema,
    drop_ducklake_schema,
    create_ducklake_view,
    drop_ducklake_view,
)

# Schema management
create_ducklake_schema("catalog.ducklake", "analytics")
drop_ducklake_schema("catalog.ducklake", "analytics", cascade=True)

# Table management — schema_dict uses DuckDB type strings
create_ducklake_table("catalog.ducklake", "events", {"ts": "timestamp", "value": "double"})
rename_ducklake_table("catalog.ducklake", "events", "event_log")
drop_ducklake_table("catalog.ducklake", "event_log")

# Column management — dtype is a DuckDB type string
alter_ducklake_add_column("catalog.ducklake", "users", "email", "varchar")
alter_ducklake_rename_column("catalog.ducklake", "users", "email", "contact_email")
alter_ducklake_drop_column("catalog.ducklake", "users", "contact_email")

# Partitioning
alter_ducklake_set_partitioned_by("catalog.ducklake", "events", ["region", "date"])

# Views
create_ducklake_view("catalog.ducklake", "active_users", "SELECT * FROM users WHERE active = true")
drop_ducklake_view("catalog.ducklake", "active_users")
```

### DML operations

```python
from ducklake_pandas import delete_ducklake, update_ducklake, merge_ducklake

# Delete rows matching a predicate (callable: DataFrame -> Series[bool])
deleted = delete_ducklake("catalog.ducklake", "users", lambda df: df["active"] == False)

# Update rows
updated = update_ducklake(
    "catalog.ducklake", "users",
    updates={"status": "inactive"},
    predicate=lambda df: df["last_login"] < "2024-01-01",
)

# Merge (upsert)
rows_updated, rows_inserted = merge_ducklake(
    "catalog.ducklake", "users", source_df, on="id",
    when_matched_update=True,
    when_not_matched_insert=True,
)
```

### Catalog inspection

```python
from ducklake_pandas import DuckLakeCatalog

catalog = DuckLakeCatalog("catalog.ducklake")

catalog.snapshots()           # All snapshots
catalog.current_snapshot()    # Latest snapshot ID
catalog.list_schemas()        # All schemas
catalog.list_tables()         # Tables in a schema
catalog.table_info()          # Per-table storage metadata
catalog.list_files("users")   # Data files and delete files
catalog.options()             # Catalog key-value metadata
catalog.settings()            # Backend type and data path

# Change data feed
catalog.table_insertions("users", start_version=1, end_version=5)
catalog.table_deletions("users", start_version=1, end_version=5)
catalog.table_changes("users", start_version=1, end_version=5)
```

### Maintenance

```python
from ducklake_pandas import expire_snapshots, vacuum_ducklake

# Expire old snapshots (metadata cleanup)
expired = expire_snapshots("catalog.ducklake", keep_last_n=10)

# Delete orphaned Parquet files (disk cleanup)
deleted = vacuum_ducklake("catalog.ducklake")
```

## Features

### Read path
- **Eager reads** via `read_ducklake()`
- **Column projection** via the `columns` parameter
- **File pruning** via column-level min/max statistics and partition values
- **Time travel** by snapshot version or timestamp
- **Delete file handling** via Iceberg-compatible positional deletes
- **Schema evolution** — ADD COLUMN, DROP COLUMN, RENAME COLUMN all handled transparently
- **Inlined data** — small tables stored directly in catalog metadata
- **Partition pruning** for identity-transform partitions
- **Column renames** — old Parquet files with old names seamlessly reconciled

### Write path
- **INSERT** — append, overwrite, or error-on-exists modes
- **DELETE** — predicate-based row deletion with position-delete files
- **UPDATE** — atomic delete + insert in a single snapshot
- **MERGE** — upsert with configurable matched/unmatched behavior
- **CREATE TABLE AS** — single-snapshot table creation with data
- **Data inlining** — small inserts stored as rows in catalog metadata
- **Partitioned writes** — Hive-style directory layout per partition key

### DDL
- **CREATE/DROP TABLE** with full snapshot versioning
- **ADD/DROP/RENAME COLUMN** with schema evolution tracking
- **CREATE/DROP SCHEMA** with cascade support
- **RENAME TABLE** preserving table identity
- **SET PARTITIONED BY** for identity-transform partitioning
- **CREATE/DROP VIEW** with `OR REPLACE` support

### Catalog inspection
- Snapshot history and time travel metadata
- Per-table storage statistics (file counts, sizes)
- Data file and delete file listing
- Schema and table enumeration
- Key-value catalog options
- **Change data feed** — insertions, deletions, and update detection

### Maintenance
- **expire_snapshots** — remove old snapshot metadata
- **vacuum** — delete orphaned Parquet files

### Backend support
- **SQLite** — via Python stdlib `sqlite3` (zero-dependency)
- **PostgreSQL** — via `psycopg2` (optional extra)
- Full interoperability with DuckDB's DuckLake extension

## DuckDB interoperability

ducklake-pandas produces catalogs that are fully interoperable with DuckDB's DuckLake extension. You can:

- Create catalogs with DuckDB, read/write with ducklake-pandas
- Create catalogs with ducklake-pandas, read/query with DuckDB
- Mix operations freely — both tools read the same metadata format

```python
# Create catalog with DuckDB
import duckdb
con = duckdb.connect()
con.execute("INSTALL ducklake; LOAD ducklake")
con.execute("ATTACH 'ducklake:sqlite:catalog.ducklake' AS lake (DATA_PATH 'data/')")
con.execute("CREATE TABLE lake.users (id INTEGER, name VARCHAR)")
con.execute("INSERT INTO lake.users VALUES (1, 'Alice'), (2, 'Bob')")
con.close()

# Read with ducklake-pandas
from ducklake_pandas import read_ducklake
df = read_ducklake("catalog.ducklake", "users")
```

See the [DuckDB Interop Guide](https://github.com/pdet/ducklake-pandas/wiki/DuckDB-Interop) for detailed interop patterns.

## Supported data types

| DuckLake / DuckDB type | Pandas type | Notes |
|---|---|---|
| `TINYINT` / `int8` | `Int8` | |
| `SMALLINT` / `int16` | `Int16` | |
| `INTEGER` / `int32` | `Int32` | |
| `BIGINT` / `int64` | `Int64` | |
| `UTINYINT` / `uint8` | `UInt8` | |
| `USMALLINT` / `uint16` | `UInt16` | |
| `UINTEGER` / `uint32` | `UInt32` | |
| `UBIGINT` / `uint64` | `UInt64` | |
| `FLOAT` / `float32` | `Float32` | |
| `DOUBLE` / `float64` | `Float64` | |
| `BOOLEAN` | `bool` | |
| `VARCHAR` | `object` (str) | |
| `BLOB` | `object` (bytes) | |
| `DATE` | `object` (date) | |
| `TIME` / `timetz` | `object` (time) | |
| `TIMESTAMP` | `datetime64[us]` | |
| `TIMESTAMP_MS` | `datetime64[ms]` | |
| `TIMESTAMP_NS` | `datetime64[ns]` | |
| `TIMESTAMP_S` | `datetime64[s]` | |
| `TIMESTAMPTZ` | `datetime64[us]` | |
| `DECIMAL(p, s)` | `object` (Decimal) | |
| `UUID` | `object` | Binary in Parquet |
| `JSON` | `object` | Binary in Parquet |
| `HUGEINT` | `Int64` | Limited: DuckDB writes as Float64 in Parquet |
| `INTERVAL` | `object` | Limited: Pandas Parquet reader limitation |
| `LIST(T)` | `object` (list) | Recursive nesting supported |
| `STRUCT(...)` | `object` (dict) | Recursive nesting supported |
| `MAP(K, V)` | `object` (list of dicts) | Limited: Pandas Parquet reader issue |

## Architecture

```
src/ducklake_pandas/
    __init__.py       Public API (all functions and DuckLakeCatalog)
    _backend.py       Backend adapters (SQLite, PostgreSQL)
    _catalog.py       Metadata reader (snapshots, tables, columns, files, stats)
    _catalog_api.py   DuckLakeCatalog inspection class
    _dataset.py       Pandas dataset reader
    _schema.py        DuckLake type -> Pandas type mapping
    _stats.py         Column statistics for file pruning
    _writer.py        Catalog writer (tables, data, DDL, views, maintenance)
```

See the [Architecture Overview](https://github.com/pdet/ducklake-pandas/wiki/Architecture) for a detailed deep-dive.

## Development

```bash
git clone https://github.com/pdet/ducklake-pandas.git
cd ducklake-pandas
pip install -e ".[dev]"
```

### Running tests

```bash
pytest                    # Full suite (SQLite backend)
pytest -n auto            # Parallel execution
pytest -k "test_views"    # Specific pattern

# With PostgreSQL backend
DUCKLAKE_PG_DSN="postgresql://user:pass@localhost/testdb" pytest
```

Test suite: **553 tests** (4 xfailed for known DuckDB/Pandas limitations). Tests are parametrized over backends — SQLite always runs; PostgreSQL runs when `DUCKLAKE_PG_DSN` is set.

## Documentation

- [Architecture Overview](https://github.com/pdet/ducklake-pandas/wiki/Architecture)
- [API Reference](https://github.com/pdet/ducklake-pandas/wiki/API-Reference)
- [Configuration Guide](https://github.com/pdet/ducklake-pandas/wiki/Configuration)
- [Feature Examples](https://github.com/pdet/ducklake-pandas/wiki/Examples)
- [DuckDB Interop Guide](https://github.com/pdet/ducklake-pandas/wiki/DuckDB-Interop)

## License

MIT
