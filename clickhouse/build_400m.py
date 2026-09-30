"""Build a 400M-row fact_gl in ClickHouse from the REAL generator.

A single generator run tops out near 1.3M GL rows: fact_gl is trial-balance grain, so
its size is bounded by families x markets x customers x months, and inflating that with
fake dimensions is exactly what spec 6 forbids. Holding 400M rows in one polars frame
would also need far more RAM than the machine has.

So the table is built as an ENSEMBLE: N independent generator runs, each with its own
seed, each individually calibrated and each individually reconciling. They are unioned
under a `run_id` column. Every row is real generated output — nothing is fabricated with
rand() — and any single `run_id` still ties out on its own.

    python clickhouse/build_400m.py --target 400000000
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from volteogen.pipeline import build_dataset  # noqa: E402

CH = ["clickhouse", "client", "--host", "127.0.0.1"]

DDL = """
CREATE TABLE IF NOT EXISTS volteo.fact_gl_400m
(
    run_id        UInt16,
    fiscal_year   UInt16,
    period_no     UInt8,
    account_class LowCardinality(String),
    account_id    UInt32,
    entity        LowCardinality(String),
    cc_id         UInt32,
    product_id    UInt32,
    market_id     UInt32,
    amount        Float64
)
ENGINE = MergeTree
PARTITION BY intDiv(run_id, 32)
ORDER BY (account_id, cc_id, product_id, market_id, run_id);
"""

# The projection carries every dimension the dashboard can drill, INCLUDING the account.
# That is what makes the key space realistic: with account, entity and period in the
# grain there are ~1.4M distinct positions rather than the 1,540 you get from
# (class x cc x product x market) alone.
PROJECTION = """
ALTER TABLE volteo.fact_gl_400m ADD PROJECTION IF NOT EXISTS p_dims
(
    SELECT account_class, account_id, entity, fiscal_year, period_no,
           cc_id, product_id, market_id, sum(amount)
    GROUP BY account_class, account_id, entity, fiscal_year, period_no,
             cc_id, product_id, market_id
);
"""


def ch(query: str, stdin=None) -> str:
    result = subprocess.run(CH + ["--query", query], stdin=stdin,
                            capture_output=stdin is None, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr if result.stderr else "clickhouse error")
    return (result.stdout or "").strip()


def key_maps() -> dict[str, dict]:
    """node_id -> surrogate id, read back from the hierarchy tables already in ClickHouse."""
    maps: dict[str, dict] = {}
    for dim, table in (("cc", "hier_cc"), ("product", "hier_product"), ("market", "hier_market")):
        rows = ch(f"SELECT node_id, id FROM volteo.{table} FORMAT TSV").splitlines()
        maps[dim] = {r.split("\t")[0]: int(r.split("\t")[1]) for r in rows if "\t" in r}
    # Accounts are keyed by (entity, local code): the four charts reuse code numbers.
    rows = ch("SELECT entity, code, id FROM volteo.map_account_key FORMAT TSV").splitlines()
    maps["account"] = {}
    for row in rows:
        parts = row.split("\t")
        if len(parts) == 3:
            maps["account"][(parts[0], parts[1])] = int(parts[2])
    return maps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=400_000_000)
    parser.add_argument("--months", type=int, default=60)
    parser.add_argument("--customers", type=int, default=400)
    parser.add_argument("--products", type=int, default=900)
    parser.add_argument("--markets", type=int, default=40)
    args = parser.parse_args()

    ch(DDL)
    ch(PROJECTION)
    maps = key_maps()
    if not maps["cc"]:
        print("hierarchy tables are empty — run clickhouse/drilldown_setup.sql first")
        return 1

    already = int(ch("SELECT count() FROM volteo.fact_gl_400m") or 0)
    run_id = int(ch("SELECT ifNull(max(run_id), 0) FROM volteo.fact_gl_400m") or 0)
    print(f"starting at {already:,} rows, run_id {run_id}", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="volteo400m_"))
    started = time.time()
    try:
        while already < args.target:
            run_id += 1
            overrides = {
                "seed": 42 + run_id,
                "period.months": args.months,
                "cardinality.customers.named_per_entity": args.customers,
                "cardinality.products.target_leaves": args.products,
                "cardinality.markets.target_leaves": args.markets,
                "volume.target_gl_rows": 0,
            }
            dataset = build_dataset(preset="M", overrides=overrides, verbose=False)
            gl = dataset["fact_gl"]

            account_ids = [
                maps["account"].get((entity, code), 0)
                for entity, code in zip(gl["entity"].to_list(), gl["account_code"].to_list())
            ]
            frame = gl.with_columns(
                pl.Series("account_id", account_ids, dtype=pl.UInt32)
            ).select(
                pl.lit(run_id, pl.UInt16).alias("run_id"),
                pl.col("fiscal_year").cast(pl.UInt16),
                pl.col("period_no").cast(pl.UInt8),
                pl.col("account_class"),
                pl.col("account_id"),
                pl.col("entity"),
                pl.col("cost_center_node_id").replace_strict(maps["cc"], default=0)
                    .cast(pl.UInt32).alias("cc_id"),
                pl.col("product_node_id").replace_strict(maps["product"], default=0)
                    .cast(pl.UInt32).alias("product_id"),
                pl.col("market_node_id").replace_strict(maps["market"], default=0)
                    .cast(pl.UInt32).alias("market_id"),
                pl.col("amount_gc_actual_rates").fill_null(0.0).cast(pl.Float64).alias("amount"),
            )

            path = tmp / f"run_{run_id}.parquet"
            frame.write_parquet(path, compression="zstd")
            with path.open("rb") as handle:
                subprocess.run(
                    CH + ["--query", "INSERT INTO volteo.fact_gl_400m FORMAT Parquet",
                          "--input_format_null_as_default=1"],
                    stdin=handle, check=True,
                )
            path.unlink()

            already += frame.height
            rate = already / max(1e-9, time.time() - started)
            eta = (args.target - already) / rate / 60 if rate else 0
            print(f"  run {run_id:>4}: +{frame.height:>9,} -> {already:>12,} "
                  f"({already/args.target*100:5.1f}%)  {rate:>9,.0f} rows/s  ETA {eta:5.1f} min",
                  flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"DONE: {already:,} rows across {run_id} generator runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
