"""Exports: CSV, Parquet, DuckDB, and ClickHouse DDL + dictionaries + example queries."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .clickhouse import (
    write_clickhouse_ddl,
    write_example_queries,
    write_load_script,
    write_serving_views,
)

__all__ = [
    "write_dataset",
    "write_clickhouse_ddl",
    "write_example_queries",
    "write_serving_views",
    "write_load_script",
]


def _sanitise(frame: pl.DataFrame) -> pl.DataFrame:
    """Flatten anything CSV cannot represent (lists, structs) to strings."""
    casts = []
    for name, dtype in zip(frame.columns, frame.dtypes):
        if dtype in (pl.List, pl.Struct) or isinstance(dtype, (pl.List, pl.Struct)):
            casts.append(pl.col(name).cast(pl.Utf8).alias(name))
    return frame.with_columns(casts) if casts else frame


def write_dataset(dataset, output_dir: Path, formats: list[str] | None = None) -> dict[str, list[str]]:
    """Write every table in the requested formats. Returns the files written."""
    output_dir = Path(output_dir)
    formats = formats or list(dataset.config.get("outputs"))
    written: dict[str, list[str]] = {fmt: [] for fmt in formats}

    if "csv" in formats:
        target = output_dir / "csv"
        target.mkdir(parents=True, exist_ok=True)
        for name, frame in sorted(dataset.tables.items()):
            path = target / f"{name}.csv"
            _sanitise(frame).write_csv(path)
            written["csv"].append(str(path))

    if "parquet" in formats:
        target = output_dir / "parquet"
        target.mkdir(parents=True, exist_ok=True)
        for name, frame in sorted(dataset.tables.items()):
            path = target / f"{name}.parquet"
            frame.write_parquet(path, compression="zstd")
            written["parquet"].append(str(path))

    if "duckdb" in formats:
        import duckdb

        target = output_dir / "duckdb"
        target.mkdir(parents=True, exist_ok=True)
        path = target / "volteo.duckdb"
        if path.exists():
            path.unlink()
        connection = duckdb.connect(str(path))
        try:
            for name, frame in sorted(dataset.tables.items()):
                arrow_table = _sanitise(frame).to_arrow()  # noqa: F841 - referenced by SQL below
                connection.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM arrow_table')
        finally:
            connection.close()
        written["duckdb"].append(str(path))

    if "clickhouse_ddl" in formats:
        target = output_dir / "clickhouse"
        target.mkdir(parents=True, exist_ok=True)
        written["clickhouse_ddl"].append(str(write_clickhouse_ddl(dataset, target)))
        written["clickhouse_ddl"].append(str(write_example_queries(dataset, target)))
        written["clickhouse_ddl"].append(str(write_serving_views(dataset, target)))
        written["clickhouse_ddl"].append(str(write_load_script(dataset, target)))

    return written
