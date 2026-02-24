"""DuckLake dataset reader for pandas."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from ducklake_pandas._catalog import DuckLakeCatalogReader
from ducklake_pandas._schema import duckdb_type_to_pandas

if TYPE_CHECKING:
    from datetime import datetime

    from ducklake_pandas._catalog import (
        ColumnHistoryEntry,
        ColumnInfo,
        DeleteFileInfo,
        FileInfo,
        FilePartitionValue,
        PartitionColumnDef,
        TableInfo,
    )


def _cast_inlined_to_schema(
    df: pd.DataFrame, schema: dict[str, str]
) -> pd.DataFrame:
    """Cast inlined data from SQLite types to the catalog schema.

    SQLite stores all integers as BIGINT (Int64), booleans as 0/1 integers,
    dates/timestamps as strings, and decimals as strings.  This function
    casts each column to the expected pandas type so the data matches
    the dataset schema.
    """
    for col_name in df.columns:
        if col_name not in schema:
            continue
        target = schema[col_name]
        if target == "boolean":
            df[col_name] = df[col_name].astype("boolean")
        elif "datetime64" in target:
            df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
        elif target in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"):
            try:
                df[col_name] = df[col_name].astype(target)
            except (ValueError, TypeError):
                pass
        elif target in ("Float32", "Float64"):
            try:
                df[col_name] = df[col_name].astype(target)
            except (ValueError, TypeError):
                pass
    return df


def _is_active_at(entry: ColumnHistoryEntry, snapshot: int) -> bool:
    """Check if a column history entry was active at a given snapshot."""
    return entry.begin_snapshot <= snapshot and (
        entry.end_snapshot is None or entry.end_snapshot > snapshot
    )


def _top_level_history(history: list[ColumnHistoryEntry]) -> list[ColumnHistoryEntry]:
    """Filter history to top-level columns only."""
    return [e for e in history if e.parent_column is None]


def _has_renames(
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> bool:
    """Check if any column has been renamed or has a drop+re-add conflict."""
    top_history = _top_level_history(history)
    top_current = [c for c in current_columns if c.parent_column is None]
    current_names = {c.column_id: c.column_name for c in top_current}
    current_name_set = set(current_names.values())

    for entry in top_history:
        if entry.column_id in current_names:
            if entry.column_name != current_names[entry.column_id]:
                return True
        elif entry.column_name in current_name_set:
            return True

    return False


def _get_physical_name(
    column_id: int,
    file_begin_snapshot: int,
    history: list[ColumnHistoryEntry],
) -> str | None:
    """Get the physical column name that was active when the file was written."""
    for entry in history:
        if entry.column_id != column_id:
            continue
        if _is_active_at(entry, file_begin_snapshot):
            return entry.column_name
    return None


def _get_rename_map(
    file_begin_snapshot: int,
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> dict[str, str]:
    """Get {physical_name -> target_name} for top-level columns that differ."""
    top_history = _top_level_history(history)
    top_current = [c for c in current_columns if c.parent_column is None]
    rename_map: dict[str, str] = {}
    current_ids = {c.column_id for c in top_current}
    current_name_set = {c.column_name for c in top_current}

    for col in top_current:
        physical = _get_physical_name(col.column_id, file_begin_snapshot, top_history)
        if physical is not None and physical != col.column_name:
            rename_map[physical] = col.column_name

    for entry in top_history:
        if entry.column_id in current_ids:
            continue
        if _is_active_at(entry, file_begin_snapshot):
            if entry.column_name in current_name_set and entry.column_name not in rename_map:
                rename_map[entry.column_name] = f"__ducklake_dropped_{entry.column_id}__"

    return rename_map


def _group_files_by_rename_map(
    files: list[FileInfo],
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> list[tuple[dict[str, str], list[FileInfo]]]:
    """Group files by their rename map.

    Returns list of (rename_map, files_list) tuples.
    """
    groups: dict[tuple, list[FileInfo]] = defaultdict(list)
    rename_maps: dict[tuple, dict[str, str]] = {}

    for f in files:
        rmap = _get_rename_map(f.begin_snapshot, history, current_columns)
        rmap_key = frozenset(rmap.items())

        groups[rmap_key].append(f)
        if rmap_key not in rename_maps:
            rename_maps[rmap_key] = rmap

    return [
        (rename_maps[k], group)
        for k, group in groups.items()
    ]


@dataclass
class DuckLakeDataset:
    """
    Dataset reader for DuckLake tables.

    Reads DuckLake catalog metadata and Parquet data files to produce
    pandas DataFrames.
    """

    metadata_path: str
    table_name: str
    schema_name: str
    snapshot_version: int | None = None
    snapshot_time: datetime | str | None = None
    data_path_override: str | None = None

    def __post_init__(self) -> None:
        if self.snapshot_version is not None and self.snapshot_time is not None:
            msg = "Cannot specify both snapshot_version and snapshot_time"
            raise ValueError(msg)

    def _get_reader(self) -> DuckLakeCatalogReader:
        return DuckLakeCatalogReader(
            self.metadata_path,
            data_path_override=self.data_path_override,
        )

    def _resolve_snapshot(self, reader: DuckLakeCatalogReader) -> Any:
        if self.snapshot_version is not None:
            return reader.get_snapshot_at_version(self.snapshot_version)
        if self.snapshot_time is not None:
            return reader.get_snapshot_at_time(self.snapshot_time)
        return reader.get_current_snapshot()

    @staticmethod
    def _build_schema_from_columns(all_columns: list[ColumnInfo]) -> dict[str, str]:
        """Build the pandas schema dict from the column hierarchy."""
        top_level = [c for c in all_columns if c.parent_column is None]
        schema_dict: dict[str, str] = {}
        for col in top_level:
            schema_dict[col.column_name] = duckdb_type_to_pandas(col.column_type)
        return schema_dict

    def schema(self) -> dict[str, str]:
        """Return the table schema as a dict of {column_name: pandas_dtype}."""
        with self._get_reader() as reader:
            snapshot = self._resolve_snapshot(reader)
            table = reader.get_table(self.table_name, self.schema_name, snapshot.snapshot_id)
            all_columns = reader.get_all_columns(table.table_id, snapshot.snapshot_id)
            return self._build_schema_from_columns(all_columns)

    def read(self) -> pd.DataFrame:
        """
        Read the DuckLake table and return a pandas DataFrame.

        Resolves metadata, reads Parquet data files, handles column renames,
        deletion files, and inlined data.
        """
        with self._get_reader() as reader:
            snapshot = self._resolve_snapshot(reader)

            table = reader.get_table(
                self.table_name, self.schema_name, snapshot.snapshot_id
            )
            all_columns = reader.get_all_columns(table.table_id, snapshot.snapshot_id)
            columns = [c for c in all_columns if c.parent_column is None]
            column_names = [c.column_name for c in columns]
            schema_dict = self._build_schema_from_columns(all_columns)

            # Get data files
            data_files = reader.get_data_files(table.table_id, snapshot.snapshot_id)

            if not data_files:
                # Check for inlined data
                inlined = reader.read_inlined_data(
                    table.table_id,
                    snapshot.snapshot_id,
                    column_names,
                )
                if inlined is not None and len(inlined) > 0:
                    return _cast_inlined_to_schema(inlined, schema_dict)
                # Empty table
                return pd.DataFrame(columns=column_names)

            # Get delete files
            delete_files = reader.get_delete_files(table.table_id, snapshot.snapshot_id)

            # Build delete positions lookup: data_file_id -> set of positions
            delete_positions: dict[int, set[int]] = {}
            if delete_files:
                del_by_file: dict[int, list[str]] = defaultdict(list)
                for df_info in delete_files:
                    path = reader.resolve_data_file_path(
                        df_info.path, df_info.path_is_relative, table
                    )
                    del_by_file[df_info.data_file_id].append(path)

                for data_file_id, del_paths in del_by_file.items():
                    positions: set[int] = set()
                    for del_path in del_paths:
                        try:
                            del_df = pd.read_parquet(del_path)
                            positions.update(del_df["pos"].tolist())
                        except Exception:
                            pass
                    if positions:
                        delete_positions[data_file_id] = positions

            # Detect column renames
            history = reader.get_column_history(table.table_id)
            has_rename = _has_renames(history, all_columns)

            frames: list[pd.DataFrame] = []

            if not has_rename:
                # Fast path: no renames
                for f in data_files:
                    source = reader.resolve_data_file_path(
                        f.path, f.path_is_relative, table
                    )
                    df = pd.read_parquet(source)

                    # Apply deletions
                    deleted = delete_positions.get(f.data_file_id)
                    if deleted:
                        df = df.drop(index=[i for i in deleted if i < len(df)]).reset_index(drop=True)

                    # Keep only current schema columns
                    available = [c for c in column_names if c in df.columns]
                    df = df[available]
                    frames.append(df)
            else:
                # Rename path: group files by rename map
                groups = _group_files_by_rename_map(data_files, history, all_columns)

                for rename_map, group_files_list in groups:
                    for f in group_files_list:
                        source = reader.resolve_data_file_path(
                            f.path, f.path_is_relative, table
                        )
                        df = pd.read_parquet(source)

                        # Apply deletions
                        deleted = delete_positions.get(f.data_file_id)
                        if deleted:
                            df = df.drop(index=[i for i in deleted if i < len(df)]).reset_index(drop=True)

                        # Apply column renames
                        if rename_map:
                            applicable = {k: v for k, v in rename_map.items() if k in df.columns}
                            if applicable:
                                df = df.rename(columns=applicable)

                        # Drop columns not in current schema
                        keep = [n for n in column_names if n in df.columns]
                        df = df[keep]
                        frames.append(df)

            # Combine all data files
            if not frames:
                result = pd.DataFrame(columns=column_names)
            elif len(frames) == 1:
                result = frames[0]
            else:
                result = pd.concat(frames, ignore_index=True)

            # Add inlined data if present
            inlined = reader.read_inlined_data(
                table.table_id,
                snapshot.snapshot_id,
                column_names,
            )
            if inlined is not None and len(inlined) > 0:
                inlined = _cast_inlined_to_schema(inlined, schema_dict)
                result = pd.concat([result, inlined], ignore_index=True)

            # Add missing columns
            for col_name in column_names:
                if col_name not in result.columns:
                    result[col_name] = None

            # Reorder columns to match schema
            result = result[[c for c in column_names if c in result.columns]]

            return result
