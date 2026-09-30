"""Build a 400M-row JOURNAL-LINE table for ONE company from the real generator.

The 400M ensemble table stacked 295 companies, so its figures were meaningless at group
level and its key space was degenerate enough that a projection collapsed it 4,868x.
This is the opposite: a single company, one continuous history, at document-line grain.

    doc_no x line_no x posting  is unique per row

so the table CANNOT be pre-aggregated — a projection over the drillable dimensions
would be as large as the table itself. Queries have to do real work, which is the point.

How one company gets to 400M rows without inflating the ledger's grain:

* the GL stays trial-balance-monthly, exactly as spec 6 requires;
* volume comes from DOCUMENTS, spec 2's sub-ledger density lever — a trade business
  books hundreds of invoice lines per SKU/market/month, not one;
* each invoice line posts twice (revenue and cost of sales), and the monthly GL rows
  for payroll, facilities and the rest are carried through as their own journal lines.

Chunking: every chunk runs the generator with the SAME seed and a shifted period
window, so products, markets, customers and cost centres are identical across chunks
and the months are contiguous. The union is one company's history, not an ensemble.

    python clickhouse/build_journal.py --target 400000000
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
CREATE TABLE IF NOT EXISTS volteo.fact_journal
(
    fiscal_year   UInt16,
    period_no     UInt8,
    entity        LowCardinality(String),
    account_id    UInt32,
    account_class LowCardinality(String),
    cc_id         UInt32,
    product_id    UInt32,
    market_id     UInt32,
    customer_id   LowCardinality(String),
    doc_no        String,
    line_no       UInt16,
    posting       UInt8,
    amount        Float64
)
ENGINE = MergeTree
PARTITION BY fiscal_year
ORDER BY (account_id, cc_id, product_id, market_id, doc_no, line_no, posting);
"""


def ch(query: str, stdin=None) -> str:
    result = subprocess.run(CH + ["--query", query], stdin=stdin,
                            capture_output=stdin is None, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "clickhouse error")
    return (result.stdout or "").strip()


def account_map() -> dict[tuple[str, str], int]:
    rows = ch("SELECT entity, code, id FROM volteo.map_account_key FORMAT TSV").splitlines()
    out = {}
    for row in rows:
        parts = row.split("\t")
        if len(parts) == 3:
            out[(parts[0], parts[1])] = int(parts[2])
    return out


def node_map(table: str) -> dict[str, int]:
    rows = ch(f"SELECT node_id, id FROM volteo.{table} FORMAT TSV").splitlines()
    return {r.split("\t")[0]: int(r.split("\t")[1]) for r in rows if "\t" in r}


def journal_lines(dataset, accounts, ccs, products, markets) -> pl.DataFrame:
    """Invoice lines post twice; monthly GL rows carry through as their own lines."""
    sales = dataset["fact_sales_lines"]
    router = dataset.router
    market_nodes = dataset["dim_market_node"].filter(pl.col("is_leaf"))
    country = dict(zip(market_nodes["node_id"].to_list(), market_nodes["country"].to_list()))

    entities = sorted(set(sales["entity"].unique().to_list()))
    categories = sorted(set(sales["category_name"].drop_nulls().to_list()))
    countries = sorted({c for c in country.values() if c}) + [None]
    routes = pl.DataFrame([
        {"entity": e, "country": c, "category_name": k,
         "rev_code": router.revenue_account(e, c, k)}
        for e in entities for c in countries for k in categories
    ])
    cogs_code = {e: router.account_for(e, "cogs_standard") for e in entities}
    country_frame = pl.DataFrame(
        [{"market_node_id": k, "country": v} for k, v in country.items()])

    base = (
        sales.join(country_frame, on="market_node_id", how="left")
        .join(routes, on=["entity", "country", "category_name"], how="left")
        .with_columns(
            pl.col("entity").replace_strict(cogs_code, default="").alias("cogs_code"),
            pl.col("product_node_id").replace_strict(products, default=0)
                .cast(pl.UInt32).alias("product_id"),
            pl.col("market_node_id").replace_strict(markets, default=0)
                .cast(pl.UInt32).alias("market_id"),
        )
    )

    def account_ids(entity_col, code_col):
        return [accounts.get((e, c), 0) for e, c in zip(entity_col, code_col)]

    common = [
        pl.col("fiscal_year").cast(pl.UInt16), pl.col("period_no").cast(pl.UInt8),
        pl.col("entity"), pl.lit(0, pl.UInt32).alias("cc_id"),
        pl.col("product_id"), pl.col("market_id"),
        pl.col("customer_id").cast(pl.Utf8), pl.col("doc_no"),
        pl.col("line_no").cast(pl.UInt16),
    ]
    revenue = base.with_columns(
        pl.Series("account_id", account_ids(base["entity"].to_list(), base["rev_code"].to_list()),
                  dtype=pl.UInt32)
    ).select(*common, pl.col("account_id"), pl.lit("revenue_trade").alias("account_class"),
             pl.lit(1, pl.UInt8).alias("posting"),
             (-pl.col("invoiced_amount")).cast(pl.Float64).alias("amount"))
    cogs = base.with_columns(
        pl.Series("account_id", account_ids(base["entity"].to_list(), base["cogs_code"].to_list()),
                  dtype=pl.UInt32)
    ).select(*common, pl.col("account_id"), pl.lit("cogs_standard").alias("account_class"),
             pl.lit(2, pl.UInt8).alias("posting"),
             pl.col("std_cost_amount").cast(pl.Float64).alias("amount"))

    # Everything that never had a document: monthly payroll, facilities, depreciation…
    gl = dataset["fact_gl"].filter(
        (pl.col("source") != "subledger")
        & (pl.col("ledger") != "TAX")
        & (pl.col("account_class") != "balance_sheet")
        & (~pl.col("entity").is_in(["ELIM", "GRP"]))
    )
    gl_ids = [accounts.get((e, c), 0)
              for e, c in zip(gl["entity"].to_list(), gl["account_code"].to_list())]
    other = gl.with_columns(pl.Series("account_id", gl_ids, dtype=pl.UInt32)).select(
        pl.col("fiscal_year").cast(pl.UInt16), pl.col("period_no").cast(pl.UInt8),
        pl.col("entity"),
        pl.col("cost_center_node_id").replace_strict(ccs, default=0).cast(pl.UInt32).alias("cc_id"),
        pl.col("product_node_id").replace_strict(products, default=0).cast(pl.UInt32).alias("product_id"),
        pl.col("market_node_id").replace_strict(markets, default=0).cast(pl.UInt32).alias("market_id"),
        pl.lit("").alias("customer_id"),
        (pl.lit("GL-") + pl.col("period_key") + pl.lit("-") + pl.col("account_code")).alias("doc_no"),
        pl.lit(0, pl.UInt16).alias("line_no"),
        pl.col("account_id"), pl.col("account_class"),
        pl.lit(3, pl.UInt8).alias("posting"),
        pl.col("amount_gc_actual_rates").fill_null(0.0).cast(pl.Float64).alias("amount"),
    )

    order = ["fiscal_year", "period_no", "entity", "account_id", "account_class",
             "cc_id", "product_id", "market_id", "customer_id", "doc_no", "line_no",
             "posting", "amount"]
    return pl.concat([revenue.select(order), cogs.select(order), other.select(order)],
                     how="vertical")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=400_000_000)
    parser.add_argument("--chunk-months", type=int, default=6)
    parser.add_argument("--density-min", type=int, default=180)
    parser.add_argument("--density-max", type=int, default=420)
    args = parser.parse_args()

    ch(DDL)
    accounts = account_map()
    ccs, products, markets = (node_map("hier_cc"), node_map("hier_product"),
                              node_map("hier_market"))
    if not accounts:
        print("run clickhouse/account_hierarchy.sql first")
        return 1

    total = int(ch("SELECT count() FROM volteo.fact_journal") or 0)
    tmp = Path(tempfile.mkdtemp(prefix="volteo_journal_"))
    started = time.time()
    chunk = 0
    year, month = 2015, 7          # one long, contiguous history

    try:
        while total < args.target:
            chunk += 1
            dataset = build_dataset(preset="M", overrides={
                "seed": 42,                       # SAME seed: identical dimensions
                "period.start": f"{year:04d}-{month:02d}",
                "period.months": args.chunk_months,
                "volume.target_gl_rows": 0,
                "cardinality.subledger.lines_per_driver_min": args.density_min,
                "cardinality.subledger.lines_per_driver_max": args.density_max,
                "cardinality.products.target_leaves": 900,
                "cardinality.markets.target_leaves": 40,
                "cardinality.customers.named_per_entity": 60,
            }, verbose=False)

            frame = journal_lines(dataset, accounts, ccs, products, markets)
            path = tmp / f"chunk_{chunk}.parquet"
            frame.write_parquet(path, compression="zstd")
            with path.open("rb") as handle:
                subprocess.run(
                    CH + ["--query", "INSERT INTO volteo.fact_journal FORMAT Parquet",
                          "--input_format_null_as_default=1"], stdin=handle, check=True)
            path.unlink()

            total += frame.height
            month += args.chunk_months
            while month > 12:
                month -= 12
                year += 1
            rate = total / max(1e-9, time.time() - started)
            print(f"  chunk {chunk:>3} ({year}-{month:02d}): +{frame.height:>10,} -> "
                  f"{total:>12,} ({total/args.target*100:5.1f}%)  {rate:>9,.0f} rows/s  "
                  f"ETA {(args.target-total)/rate/60:5.1f} min", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"DONE: {total:,} journal lines, {chunk} chunks, "
          f"{chunk * args.chunk_months / 12:.1f} years of history", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
