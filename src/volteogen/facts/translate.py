"""Currency translation: local amounts -> group currency at the right rate type.

Every amount is posted in LOCAL currency (spec 6.6); group-currency measures are
derived here so the identity ``amount_gc = amount_lc x rate`` is checkable (V6).

Two group measures, because they answer different questions:

``amount_gc_actual_rates``  translated at the month's average rate — what happened.
``amount_gc_budget_rates``  translated at the fiscal year's fixed budget rate — the
                            constant-currency view, which strips FX out of growth.
"""

from __future__ import annotations

import polars as pl


def translate_gl(gl: pl.DataFrame, fx: pl.DataFrame, group_currency: str = "USD") -> pl.DataFrame:
    rates = fx.select(
        pl.col("from_currency").alias("currency"),
        "fiscal_year",
        pl.col("period_no").alias("_rate_period"),
        pl.col("rate_avg").alias("fx_rate_avg"),
        pl.col("rate_closing").alias("fx_rate_closing"),
        pl.col("rate_budget").alias("fx_rate_budget"),
    ).unique(subset=["currency", "fiscal_year", "_rate_period"], keep="first")

    # P13 has no month of its own, so it shares its fiscal year's closing month (P12)
    # rate. Group/ELIM rows are already in group currency and translate at 1.0.
    out = gl.with_columns(
        pl.col("period_no").clip(upper_bound=12).alias("_rate_period")
    ).join(rates, on=["currency", "fiscal_year", "_rate_period"], how="left").drop("_rate_period")

    out = out.with_columns(
        pl.col("fx_rate_avg").fill_null(1.0),
        pl.col("fx_rate_closing").fill_null(1.0),
        pl.col("fx_rate_budget").fill_null(1.0),
    )

    return out.with_columns(
        (pl.col("amount_lc") * pl.col("fx_rate_avg")).round(2).alias("amount_gc_actual_rates"),
        (pl.col("amount_lc") * pl.col("fx_rate_budget")).round(2).alias("amount_gc_budget_rates"),
        pl.lit(group_currency).alias("group_currency"),
    )
