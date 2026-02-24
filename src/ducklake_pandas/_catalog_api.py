"""DuckLake catalog utility functions — Python equivalents of DuckLake's DuckDB extension functions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd

from ducklake_pandas._catalog import DuckLakeCatalogReader
from ducklake_pandas._schema import duckdb_type_to_pandas

if TYPE_CHECKING:
    from pathlib import Path


class DuckLakeCatalog:
    """
    High-level interface for inspecting DuckLake catalog metadata.

    Provides Python equivalents of DuckLake's DuckDB extension utility
    functions (``ducklake_snapshots``, ``ducklake_table_info``, etc.).
    All methods return ``pd.DataFrame``.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (``.ducklake`` or ``.db``),
        or a PostgreSQL connection string.
    data_path
        Override the data path stored in the catalog.

    Examples
    --------
    >>> catalog = DuckLakeCatalog("catalog.ducklake")
    >>> catalog.snapshots()
    >>> catalog.current_snapshot()
    >>> catalog.table_info()
    >>> catalog.list_files("my_table")
    """

    def __init__(self, path: str | Path, *, data_path: str | Path | None = None) -> None:
        self._metadata_path = os.fspath(path)
        self._data_path_override = os.fspath(data_path) if data_path is not None else None

    def _reader(self) -> DuckLakeCatalogReader:
        return DuckLakeCatalogReader(self._metadata_path, data_path_override=self._data_path_override)

    def __enter__(self) -> DuckLakeCatalog:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    # ------------------------------------------------------------------
    # Snapshot functions
    # ------------------------------------------------------------------

    def snapshots(self) -> pd.DataFrame:
        """
        List all snapshots in the catalog.

        Returns a DataFrame with columns:
        - ``snapshot_id`` (int)
        - ``snapshot_time`` (str)
        - ``schema_version`` (int)
        """
        with self._reader() as reader:
            rows = reader.get_all_snapshots()
        if not rows:
            return pd.DataFrame(columns=["snapshot_id", "snapshot_time", "schema_version"])
        return pd.DataFrame({
            "snapshot_id": [r[0] for r in rows],
            "snapshot_time": [str(r[2]) for r in rows],
            "schema_version": [r[1] for r in rows],
        })

    def current_snapshot(self) -> int:
        """Get the current (latest) snapshot ID."""
        with self._reader() as reader:
            snap = reader.get_current_snapshot()
        return snap.snapshot_id

    # ------------------------------------------------------------------
    # Table metadata
    # ------------------------------------------------------------------

    def table_info(self, *, schema: str = "main") -> pd.DataFrame:
        """
        Get per-table storage metadata.

        Returns a DataFrame with columns:
        - ``table_name`` (str)
        - ``table_id`` (int)
        - ``file_count`` (int)
        - ``file_size_bytes`` (int)
        - ``delete_file_count`` (int)
        - ``delete_row_count`` (int)
        """
        with self._reader() as reader:
            snap = reader.get_current_snapshot()
            schemas = reader.get_all_schemas(snap.snapshot_id)
            schema_row = None
            for s in schemas:
                if s[1] == schema:
                    schema_row = s
                    break
            if schema_row is None:
                return pd.DataFrame(columns=[
                    "table_name", "table_id", "file_count",
                    "file_size_bytes", "delete_file_count", "delete_row_count",
                ])

            tables = reader.get_all_tables(schema_row[0], snap.snapshot_id)
            result_rows = []
            for table_id, table_name in tables:
                data_files = reader.get_data_files(table_id, snap.snapshot_id)
                delete_files = reader.get_delete_files(table_id, snap.snapshot_id)
                file_count = len(data_files)
                file_size = sum(f.file_size_bytes for f in data_files)
                del_count = len(delete_files)
                del_size = sum(f.delete_count for f in delete_files)
                result_rows.append((table_name, table_id, file_count, file_size, del_count, del_size))

        if not result_rows:
            return pd.DataFrame(columns=[
                "table_name", "table_id", "file_count",
                "file_size_bytes", "delete_file_count", "delete_row_count",
            ])

        return pd.DataFrame({
            "table_name": [r[0] for r in result_rows],
            "table_id": [r[1] for r in result_rows],
            "file_count": [r[2] for r in result_rows],
            "file_size_bytes": [r[3] for r in result_rows],
            "delete_file_count": [r[4] for r in result_rows],
            "delete_row_count": [r[5] for r in result_rows],
        })

    def list_files(
        self,
        table: str,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pd.DataFrame:
        """
        List data files and delete files for a table.

        Returns a DataFrame with columns:
        - ``data_file`` (str)
        - ``data_file_size_bytes`` (int)
        - ``delete_file`` (str or None)
        - ``delete_row_count`` (int or None)
        """
        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()

            table_info = reader.get_table(table, schema, snap.snapshot_id)
            data_files = reader.get_data_files(table_info.table_id, snap.snapshot_id)
            delete_files = reader.get_delete_files(table_info.table_id, snap.snapshot_id)

            del_map: dict[int, list[tuple[str, int]]] = {}
            for df in delete_files:
                path = reader.resolve_data_file_path(df.path, df.path_is_relative, table_info)
                del_map.setdefault(df.data_file_id, []).append((path, df.delete_count))

            rows: list[tuple[str, int, str | None, int | None]] = []
            for f in data_files:
                data_path = reader.resolve_data_file_path(f.path, f.path_is_relative, table_info)
                del_entries = del_map.get(f.data_file_id, [])
                if del_entries:
                    for del_path, del_count in del_entries:
                        rows.append((data_path, f.file_size_bytes, del_path, del_count))
                else:
                    rows.append((data_path, f.file_size_bytes, None, None))

        if not rows:
            return pd.DataFrame(columns=[
                "data_file", "data_file_size_bytes", "delete_file", "delete_row_count",
            ])

        return pd.DataFrame({
            "data_file": [r[0] for r in rows],
            "data_file_size_bytes": [r[1] for r in rows],
            "delete_file": [r[2] for r in rows],
            "delete_row_count": [r[3] for r in rows],
        })

    # ------------------------------------------------------------------
    # Schema / table listing
    # ------------------------------------------------------------------

    def list_schemas(self, *, snapshot_version: int | None = None) -> pd.DataFrame:
        """
        List all schemas in the catalog.

        Returns a DataFrame with columns:
        - ``schema_id`` (int)
        - ``schema_name`` (str)
        """
        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()
            rows = reader.get_all_schemas(snap.snapshot_id)

        if not rows:
            return pd.DataFrame(columns=["schema_id", "schema_name"])

        return pd.DataFrame({
            "schema_id": [r[0] for r in rows],
            "schema_name": [r[1] for r in rows],
        })

    def list_tables(
        self,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pd.DataFrame:
        """
        List all tables in a schema.

        Returns a DataFrame with columns:
        - ``table_id`` (int)
        - ``table_name`` (str)
        """
        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()

            schemas = reader.get_all_schemas(snap.snapshot_id)
            schema_row = None
            for s in schemas:
                if s[1] == schema:
                    schema_row = s
                    break
            if schema_row is None:
                return pd.DataFrame(columns=["table_id", "table_name"])

            tables = reader.get_all_tables(schema_row[0], snap.snapshot_id)

        if not tables:
            return pd.DataFrame(columns=["table_id", "table_name"])

        return pd.DataFrame({
            "table_id": [r[0] for r in tables],
            "table_name": [r[1] for r in tables],
        })

    # ------------------------------------------------------------------
    # Catalog options
    # ------------------------------------------------------------------

    def options(self) -> pd.DataFrame:
        """
        Get catalog key-value options from ducklake_metadata.

        Returns a DataFrame with columns:
        - ``key`` (str)
        - ``value`` (str)
        """
        with self._reader() as reader:
            rows = reader.get_all_metadata()

        if not rows:
            return pd.DataFrame(columns=["key", "value"])

        return pd.DataFrame({
            "key": [r[0] for r in rows],
            "value": [r[1] for r in rows],
        })

    def settings(self) -> pd.DataFrame:
        """
        Get catalog type and data path.

        Returns a DataFrame with columns:
        - ``catalog_type`` (str) — ``"sqlite"`` or ``"postgresql"``
        - ``data_path`` (str)
        """
        with self._reader() as reader:
            catalog_type = reader._backend.__class__.__name__
            if "sqlite" in catalog_type.lower():
                cat_type = "sqlite"
            else:
                cat_type = "postgresql"
            data_path = reader.data_path

        return pd.DataFrame({
            "catalog_type": [cat_type],
            "data_path": [data_path],
        })

    # ------------------------------------------------------------------
    # Change data feed
    # ------------------------------------------------------------------

    def table_insertions(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pd.DataFrame:
        """
        Get rows inserted between two snapshots.

        Reads the Parquet data files added in ``(start_version, end_version]``.

        Returns a DataFrame with ``snapshot_id`` column plus all table columns.
        """
        with self._reader() as reader:
            snap = reader.get_snapshot_at_version(end_version)
            table_info = reader.get_table(table, schema, snap.snapshot_id)
            all_columns = reader.get_all_columns(table_info.table_id, snap.snapshot_id)
            column_names = [c.column_name for c in all_columns if c.parent_column is None]

            files_with_snap = reader.get_data_files_in_range_with_snapshot(
                table_info.table_id, start_version, end_version
            )

            if not files_with_snap:
                return pd.DataFrame(columns=["snapshot_id"] + column_names)

            frames: list[pd.DataFrame] = []
            for file_info, begin_snap in files_with_snap:
                path = reader.resolve_data_file_path(
                    file_info.path, file_info.path_is_relative, table_info
                )
                df = pd.read_parquet(path)
                available = [c for c in column_names if c in df.columns]
                df = df[available]
                df = df.copy()
                df.insert(0, "snapshot_id", begin_snap)
                frames.append(df)

        result = pd.concat(frames, ignore_index=True)
        cols = ["snapshot_id"] + [c for c in result.columns if c != "snapshot_id"]
        return result[cols]

    def table_deletions(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pd.DataFrame:
        """
        Get rows deleted between two snapshots.

        Returns a DataFrame with ``snapshot_id`` column plus all table columns.
        """
        with self._reader() as reader:
            snap = reader.get_snapshot_at_version(end_version)
            table_info = reader.get_table(table, schema, snap.snapshot_id)
            all_columns = reader.get_all_columns(table_info.table_id, snap.snapshot_id)
            column_names = [c.column_name for c in all_columns if c.parent_column is None]

            del_files = reader.get_delete_files_in_range(
                table_info.table_id, start_version, end_version
            )

            if not del_files:
                return pd.DataFrame(columns=["snapshot_id"] + column_names)

            frames: list[pd.DataFrame] = []
            for del_info, begin_snap in del_files:
                del_path = reader.resolve_data_file_path(
                    del_info.path, del_info.path_is_relative, table_info
                )
                del_df = pd.read_parquet(del_path)

                data_file = reader.get_data_file_by_id(del_info.data_file_id)
                if data_file is None:
                    continue

                data_path = reader.resolve_data_file_path(
                    data_file.path, data_file.path_is_relative, table_info
                )
                data_df = pd.read_parquet(data_path)

                if "pos" in del_df.columns:
                    positions = del_df["pos"].tolist()
                    row_id_start = data_file.row_id_start
                    local_positions = [p - row_id_start for p in positions]
                    local_positions = [p for p in local_positions if 0 <= p < len(data_df)]
                    if local_positions:
                        deleted_rows = data_df.iloc[local_positions].copy()
                        available = [c for c in column_names if c in deleted_rows.columns]
                        deleted_rows = deleted_rows[available].copy()
                        deleted_rows.insert(0, "snapshot_id", begin_snap)
                        frames.append(deleted_rows)

        if not frames:
            return pd.DataFrame(columns=["snapshot_id"] + column_names)

        result = pd.concat(frames, ignore_index=True)
        cols = ["snapshot_id"] + [c for c in result.columns if c != "snapshot_id"]
        return result[cols]

    def table_changes(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pd.DataFrame:
        """
        Get combined change data feed between two snapshots.

        Returns a DataFrame with columns:
        - ``snapshot_id`` (int)
        - ``change_type`` (str) — ``'insert'``, ``'delete'``,
          ``'update_preimage'``, ``'update_postimage'``
        - All table columns
        """
        insertions = self.table_insertions(table, start_version, end_version, schema=schema)
        deletions = self.table_deletions(table, start_version, end_version, schema=schema)

        has_ins = len(insertions) > 0
        has_del = len(deletions) > 0

        if not has_ins and not has_del:
            source = insertions if len(insertions.columns) >= len(deletions.columns) else deletions
            cols = ["snapshot_id", "change_type"] + [
                c for c in source.columns if c != "snapshot_id"
            ]
            return pd.DataFrame(columns=cols)

        ins_snapshots = set(insertions["snapshot_id"].tolist()) if has_ins else set()
        del_snapshots = set(deletions["snapshot_id"].tolist()) if has_del else set()
        update_snapshots = ins_snapshots & del_snapshots

        frames: list[pd.DataFrame] = []

        if has_ins:
            if update_snapshots:
                pure_ins = insertions[~insertions["snapshot_id"].isin(update_snapshots)].copy()
                update_post = insertions[insertions["snapshot_id"].isin(update_snapshots)].copy()
                if len(pure_ins) > 0:
                    pure_ins["change_type"] = "insert"
                    frames.append(pure_ins)
                if len(update_post) > 0:
                    update_post["change_type"] = "update_postimage"
                    frames.append(update_post)
            else:
                ins = insertions.copy()
                ins["change_type"] = "insert"
                frames.append(ins)

        if has_del:
            if update_snapshots:
                pure_del = deletions[~deletions["snapshot_id"].isin(update_snapshots)].copy()
                update_pre = deletions[deletions["snapshot_id"].isin(update_snapshots)].copy()
                if len(pure_del) > 0:
                    pure_del["change_type"] = "delete"
                    frames.append(pure_del)
                if len(update_pre) > 0:
                    update_pre["change_type"] = "update_preimage"
                    frames.append(update_pre)
            else:
                dels = deletions.copy()
                dels["change_type"] = "delete"
                frames.append(dels)

        if not frames:
            return pd.DataFrame(columns=["snapshot_id", "change_type"])

        result = pd.concat(frames, ignore_index=True)
        other_cols = [c for c in result.columns if c not in ("snapshot_id", "change_type")]
        return result[["snapshot_id", "change_type"] + other_cols]
