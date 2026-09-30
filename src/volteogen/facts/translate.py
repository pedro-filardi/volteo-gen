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
        "period_key",
        pl.col("rate_avg").alias("fx_rate_avg"),
        pl.col("rate_closing").alias("fx_rate_closing"),
        pl.col("rate_budget").alias("fx_rate_budget"),
    ).unique(subset=["currency", "period_key"], keep="first")

    out = gl.join(rates, on=["currency", "period_key"], how="left")

    # P13 shares its fiscal year's closing month rate; group/ELIM rows are already in
    # group currency and translate at 1.0.
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
