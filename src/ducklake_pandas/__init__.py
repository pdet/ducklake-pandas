"""ducklake-pandas: Read and write DuckLake catalogs with pandas DataFrames."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from ducklake_pandas._catalog_api import DuckLakeCatalog
from ducklake_pandas._dataset import DuckLakeDataset
from ducklake_pandas._writer import DuckLakeCatalogWriter, Predicate

__all__ = [
    "read_ducklake",
    "write_ducklake",
    "overwrite_ducklake",
    "delete_ducklake",
    "update_ducklake",
    "merge_ducklake",
    "create_ducklake_table",
    "create_table_as_ducklake",
    "alter_ducklake_table",
    "alter_ducklake_add_column",
    "alter_ducklake_drop_column",
    "alter_ducklake_rename_column",
    "alter_ducklake_set_partitioned_by",
    "drop_ducklake_table",
    "create_ducklake_schema",
    "drop_ducklake_schema",
    "rename_ducklake_table",
    "expire_snapshots",
    "vacuum_ducklake",
    "create_ducklake_view",
    "drop_ducklake_view",
    "DuckLakeCatalog",
    "DuckLakeDataset",
    "DuckLakeCatalogWriter",
]


def read_ducklake(
    metadata_path: str,
    table: str,
    *,
    schema: str = "main",
    columns: list[str] | None = None,
    snapshot_version: int | None = None,
    snapshot_time: str | None = None,
    data_path: str | None = None,
) -> pd.DataFrame:
    """
    Read a DuckLake table into a pandas DataFrame.

    Parameters
    ----------
    metadata_path
        Path to the DuckLake catalog file or PostgreSQL connection string.
    table
        Table name.
    schema
        Schema name (default ``"main"``).
    snapshot_version
        Read at a specific snapshot version.
    snapshot_time
        Read at or before a specific timestamp.
    data_path
        Override the data path stored in the catalog.

    Returns
    -------
    pd.DataFrame
        The table data.

    Examples
    --------
    >>> import ducklake_pandas as dlp
    >>> df = dlp.read_ducklake("catalog.ducklake", "my_table")
    """
    ds = DuckLakeDataset(
        metadata_path=metadata_path,
        table_name=table,
        schema_name=schema,
        snapshot_version=snapshot_version,
        snapshot_time=snapshot_time,
        data_path_override=data_path,
    )
    result = ds.read()
    if columns is not None:
        result = result[[c for c in columns if c in result.columns]]
    return result


def write_ducklake(
    df: pd.DataFrame,
    metadata_path: str,
    table: str,
    *,
    schema: str = "main",
    mode: str = "error",
    data_path: str | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Write a DataFrame to a DuckLake table.

    Parameters
    ----------
    mode
        Write mode:
        - ``"error"`` (default): Fail if the table already exists.
        - ``"append"``: Append data to an existing table. Creates the
          table if it does not exist.
        - ``"overwrite"``: Replace all data in the table. Creates the
          table if it does not exist.

    Examples
    --------
    >>> import ducklake_pandas as dlp
    >>> dlp.write_ducklake(df, "catalog.ducklake", "my_table", mode="append")
    """
    valid_modes = ("error", "append", "overwrite")
    if mode not in valid_modes:
        msg = f"Invalid write mode '{mode}'. Must be one of {valid_modes}"
        raise ValueError(msg)

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        if mode == "error":
            writer.create_table_with_data(table, df, schema_name=schema)
        elif mode == "append":
            # Check if table exists; if not, create it
            snap_id, _, _, _ = writer._get_latest_snapshot()
            if writer._table_exists(table, schema, snap_id) is None:
                writer.create_table_with_data(table, df, schema_name=schema)
            else:
                if len(df) > 0:
                    writer.insert_data(df, table, schema_name=schema)
        elif mode == "overwrite":
            snap_id, _, _, _ = writer._get_latest_snapshot()
            if writer._table_exists(table, schema, snap_id) is None:
                writer.create_table_with_data(table, df, schema_name=schema)
            else:
                writer.overwrite_data(df, table, schema_name=schema)


def overwrite_ducklake(
    df: pd.DataFrame,
    metadata_path: str,
    table: str,
    *,
    schema: str = "main",
    data_path: str | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """
    Overwrite all data in a DuckLake table.

    Returns the new snapshot ID.
    """
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.overwrite_data(df, table, schema_name=schema)


def delete_ducklake(
    predicate: Predicate,
    metadata_path: str,
    table: str,
    *,
    schema: str = "main",
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """
    Delete rows matching a predicate from a DuckLake table.

    The predicate is a callable ``(DataFrame) -> Series[bool]``.
    Returns the number of deleted rows.

    Examples
    --------
    >>> import ducklake_pandas as dlp
    >>> dlp.delete_ducklake(
    ...     lambda df: df["age"] > 30,
    ...     "catalog.ducklake",
    ...     "my_table",
    ... )
    """
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.delete_data(predicate, table, schema_name=schema)


def update_ducklake(
    updates: dict[str, Any],
    predicate: Predicate,
    metadata_path: str,
    table: str,
    *,
    schema: str = "main",
    data_path: str | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """
    Update rows matching a predicate in a DuckLake table.

    Parameters
    ----------
    updates
        Dict mapping column names to new values. Values can be
        literals or ``Callable[[pd.DataFrame], pd.Series]`` for
        computed updates.
    predicate
        Callable ``(DataFrame) -> Series[bool]``.

    Returns the number of rows updated.

    Examples
    --------
    >>> import ducklake_pandas as dlp
    >>> dlp.update_ducklake(
    ...     {"status": "active"},
    ...     lambda df: df["id"] == 1,
    ...     "catalog.ducklake",
    ...     "my_table",
    ... )
    """
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.update_data(updates, predicate, table, schema_name=schema)


def merge_ducklake(
    source_df: pd.DataFrame,
    metadata_path: str,
    table: str,
    on: str | list[str],
    *,
    when_matched_update: dict[str, Any] | bool | None = None,
    when_not_matched_insert: bool = True,
    schema: str = "main",
    data_path: str | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> tuple[int, int]:
    """
    MERGE source_df into an existing DuckLake table.

    Returns ``(rows_updated, rows_inserted)``.

    Examples
    --------
    >>> import ducklake_pandas as dlp
    >>> dlp.merge_ducklake(
    ...     new_data,
    ...     "catalog.ducklake",
    ...     "my_table",
    ...     on="id",
    ...     when_matched_update=True,
    ... )
    """
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.merge_data(
            source_df, table, on,
            when_matched_update=when_matched_update,
            when_not_matched_insert=when_not_matched_insert,
            schema_name=schema,
        )


def create_ducklake_table(
    metadata_path: str,
    table: str,
    schema_dict: dict[str, str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    schema: str = "main",
    data_path: str | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """
    Create a new DuckLake table.

    Either provide ``schema_dict`` (mapping column names to DuckDB type strings)
    for an empty table, or ``df`` to create a table with data inferred from
    the DataFrame.

    Returns the table_id (for empty tables) or snapshot_id (if data was provided).

    Examples
    --------
    >>> import ducklake_pandas as dlp
    >>> dlp.create_ducklake_table(
    ...     "catalog.ducklake", "my_table",
    ...     {"id": "int64", "name": "varchar"},
    ... )
    """
    if schema_dict is None and df is None:
        msg = "Either schema_dict or df must be provided"
        raise ValueError(msg)

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        if df is not None:
            return writer.create_table_with_data(table, df, schema_name=schema)
        assert schema_dict is not None
        return writer.create_table(table, schema_dict, schema_name=schema)


def alter_ducklake_table(
    metadata_path: str,
    table: str,
    *,
    add_column: tuple[str, str] | None = None,
    add_column_default: Any = None,
    rename_column: tuple[str, str] | None = None,
    drop_column: str | None = None,
    set_partitioned_by: list[str] | None = None,
    schema: str = "main",
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Alter an existing DuckLake table.

    Exactly one of the keyword arguments should be provided.

    Examples
    --------
    >>> import ducklake_pandas as dlp
    >>> dlp.alter_ducklake_table(
    ...     "catalog.ducklake", "my_table",
    ...     add_column=("email", "varchar"),
    ... )
    """
    ops = sum([
        add_column is not None,
        rename_column is not None,
        drop_column is not None,
        set_partitioned_by is not None,
    ])
    if ops != 1:
        msg = "Exactly one alter operation must be specified"
        raise ValueError(msg)

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        if add_column is not None:
            col_name, dtype = add_column
            writer.add_column(
                table, col_name, dtype,
                default=add_column_default, schema_name=schema,
            )
        elif rename_column is not None:
            old_name, new_name = rename_column
            writer.rename_column(table, old_name, new_name, schema_name=schema)
        elif drop_column is not None:
            writer.drop_column(table, drop_column, schema_name=schema)
        elif set_partitioned_by is not None:
            writer.set_partitioned_by(table, set_partitioned_by, schema_name=schema)


def drop_ducklake_table(
    metadata_path: str,
    table: str,
    *,
    schema: str = "main",
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Drop a table from the DuckLake catalog."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.drop_table(table, schema_name=schema)


def create_ducklake_schema(
    metadata_path: str,
    schema_name: str,
    *,
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """Create a new schema in the DuckLake catalog. Returns the schema_id."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.create_schema(schema_name)


def drop_ducklake_schema(
    metadata_path: str,
    schema_name: str,
    *,
    cascade: bool = False,
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Drop a schema from the DuckLake catalog."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.drop_schema(schema_name, cascade=cascade)


def rename_ducklake_table(
    metadata_path: str,
    old_table: str,
    new_table: str,
    *,
    schema: str = "main",
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Rename a table in the DuckLake catalog."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.rename_table(old_table, new_table, schema_name=schema)


# ---------------------------------------------------------------------------
# Individual ALTER functions (convenience wrappers)
# ---------------------------------------------------------------------------


def alter_ducklake_add_column(
    metadata_path: str,
    table: str,
    column_name: str,
    dtype: str,
    *,
    default: Any = None,
    schema: str = "main",
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Add a column to an existing DuckLake table."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.add_column(table, column_name, dtype, default=default, schema_name=schema)


def alter_ducklake_drop_column(
    metadata_path: str,
    table: str,
    column_name: str,
    *,
    schema: str = "main",
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Drop a column from an existing DuckLake table."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.drop_column(table, column_name, schema_name=schema)


def alter_ducklake_rename_column(
    metadata_path: str,
    table: str,
    old_column_name: str,
    new_column_name: str,
    *,
    schema: str = "main",
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Rename a column in an existing DuckLake table."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.rename_column(table, old_column_name, new_column_name, schema_name=schema)


def alter_ducklake_set_partitioned_by(
    metadata_path: str,
    table: str,
    column_names: list[str],
    *,
    schema: str = "main",
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Set partitioning on an existing DuckLake table."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.set_partitioned_by(table, column_names, schema_name=schema)


def create_table_as_ducklake(
    metadata_path: str,
    table: str,
    df: pd.DataFrame,
    *,
    schema: str = "main",
    data_path: str | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """Create a new table from a DataFrame (CTAS). Returns the new snapshot ID."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.create_table_with_data(table, df, schema_name=schema)


def expire_snapshots(
    metadata_path: str,
    *,
    older_than_snapshot: int | None = None,
    keep_last_n: int | None = None,
    data_path: str | None = None,
) -> int:
    """Expire old snapshots. Returns the number of snapshots expired."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
    ) as writer:
        return writer.expire_snapshots(
            older_than_snapshot=older_than_snapshot,
            keep_last_n=keep_last_n,
        )


def vacuum_ducklake(
    metadata_path: str,
    *,
    data_path: str | None = None,
) -> int:
    """Delete orphaned Parquet files. Returns the number of files deleted."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
    ) as writer:
        return writer.vacuum()


def create_ducklake_view(
    metadata_path: str,
    view_name: str,
    sql: str,
    *,
    schema: str = "main",
    or_replace: bool = False,
    column_aliases: str = "",
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """Create a view in the DuckLake catalog. Returns the view_id."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.create_view(
            view_name, sql,
            schema_name=schema,
            or_replace=or_replace,
            column_aliases=column_aliases,
        )


def drop_ducklake_view(
    metadata_path: str,
    view_name: str,
    *,
    schema: str = "main",
    data_path: str | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Drop a view from the DuckLake catalog."""
    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=data_path,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.drop_view(view_name, schema_name=schema)
