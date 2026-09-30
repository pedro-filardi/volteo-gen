"""FX rates: monthly average and closing, plus a fixed budget rate per fiscal year.

Three rate types exist because reporting needs all three (spec 4.7):

``avg``      translates P&L flows
``closing``  translates balances
``budget``   fixed per FY at the prior September spot, which is what gives
             constant-currency ("at budget rates") analysis something to compare

Rates follow a random walk around a drifting mean rather than i.i.d. draws, so
month-on-month FX moves are autocorrelated the way real rates are.
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from ..rng import SeedBank

GROUP_CCY_RATE = 1.0


def build_fx_rates(cfg: Config, calendar: pl.DataFrame, seeds: SeedBank) -> pl.DataFrame:
    rng = seeds.rng("dims", "fx")
    pairs = list(cfg.get("fx.pairs"))
    group_ccy = str(cfg.get("fx.group_currency"))
    vol = float(cfg.get("fx.monthly_vol"))
    start_rates = dict(cfg.get("fx.start_rates"))
    drift = dict(cfg.get("fx.drift", {}))

    months = calendar.filter(~pl.col("is_period13")).sort("month_index")
    rows: list[dict] = []

    for pair in pairs:
        base_ccy = pair[:3]
        rate = float(start_rates[pair])
        mean = rate
        for row in months.iter_rows(named=True):
            mean *= 1.0 + float(drift.get(pair, 0.0))
            # Mean-reverting random walk: shock plus a light pull toward the drifting mean.
            shock = rng.normal(0.0, vol)
            rate = rate * (1.0 + shock) + 0.08 * (mean - rate)
            closing = rate * (1.0 + rng.normal(0.0, vol * 0.4))
            rows.append(
                {
                    "pair": pair,
                    "from_currency": base_ccy,
                    "to_currency": group_ccy,
                    "period_key": row["period_key"],
                    "fiscal_year": row["fiscal_year"],
                    "period_no": row["period_no"],
                    "month_index": row["month_index"],
                    "rate_avg": round(rate, 6),
                    "rate_closing": round(closing, 6),
                }
            )

    frame = pl.DataFrame(rows)

    # Budget rate: the prior FY's September spot, held flat for the whole budget year.
    # September is period 3 of a July-start fiscal year.
    budget_rows = []
    for pair in pairs:
        pair_rates = frame.filter(pl.col("pair") == pair).sort("month_index")
        for fy in sorted(pair_rates["fiscal_year"].unique().to_list()):
            prior = pair_rates.filter(
                (pl.col("fiscal_year") == fy - 1) & (pl.col("period_no") == 3)
            )
            source = prior if prior.height else pair_rates.filter(pl.col("fiscal_year") == fy)
            if not source.height:
                continue
            budget_rows.append(
                {
                    "pair": pair,
                    "fiscal_year": fy,
                    "rate_budget": round(float(source["rate_avg"][0]), 6),
                    "basis": "prior FY September spot" if prior.height else "first available spot",
                }
            )
    budget = pl.DataFrame(budget_rows)

    frame = frame.join(budget.select("pair", "fiscal_year", "rate_budget"), on=["pair", "fiscal_year"], how="left")

    # The group currency translates to itself at 1.0 in every rate type.
    self_rows = []
    for row in months.iter_rows(named=True):
        self_rows.append(
            {
                "pair": f"{group_ccy}{group_ccy}",
                "from_currency": group_ccy,
                "to_currency": group_ccy,
                "period_key": row["period_key"],
                "fiscal_year": row["fiscal_year"],
                "period_no": row["period_no"],
                "month_index": row["month_index"],
                "rate_avg": GROUP_CCY_RATE,
                "rate_closing": GROUP_CCY_RATE,
                "rate_budget": GROUP_CCY_RATE,
            }
        )
    return pl.concat([frame, pl.DataFrame(self_rows)], how="diagonal").sort(
        ["pair", "month_index"]
    )


def rate_lookup(fx: pl.DataFrame, rate_type: str = "avg") -> dict[tuple[str, str], float]:
    """``(from_currency, period_key) -> rate`` for the requested rate type."""
    column = {"avg": "rate_avg", "closing": "rate_closing", "budget": "rate_budget"}[rate_type]
    return {
        (row["from_currency"], row["period_key"]): float(row[column])
        for row in fx.iter_rows(named=True)
        if row[column] is not None
    }
