"""The P&L cube: actuals, budget and forecast in one fact (``fact_pnl``).

This is the table an FP&A team actually reports from. ``fact_gl`` is the books;
``fact_budget`` / ``fact_forecast`` are planning submissions at a coarser grain and in
local currency only. None of them can be compared with the others directly, so this
module brings all three onto one grain and one set of conventions:

**One grain.** entity x scenario x period x group account x product family — the
coarsest grain the plan exists at. Actuals are lifted onto it through
``map_local_to_group``; the local chart is irrelevant to group reporting. A local
account with no valid mapping lands on ``UNMAPPED`` (a suspense bucket) rather than
disappearing, so group totals still tie to the books and the gap stays visible.

**One sign.** Ledger sign throughout (debit positive, revenue negative), as in fact_gl.
The report bridge flips signs for presentation, so there is one rule, not two.

**Two group-currency measures**, following the usual group reporting policy:

``amount_gc``     reported: actuals at the month's average rate; budget at the fixed
                  budget rate it was planned at; forecast closed months at actual rates
                  and open months at the budget rate.
``amount_gc_cc``  constant currency: every scenario at the fiscal year's budget rate,
                  so ACT vs BUD in this measure carries no FX effect.

**Planned eliminations.** Entities plan their intercompany sales like any other sale,
so a consolidated plan that simply sums entities overstates group revenue. The plan
therefore carries its own ELIM rows: budget rolls forward the prior year's actual
eliminations with the same volume uplift as the budget itself; forecast copies actual
eliminations for closed months and takes the board budget's for open ones.

**Non-controlling interest.** The minority share of each partly-owned entity's result is
posted on GRP as account class ``nci`` in every scenario, so the statement can show
profit attributable to owners of the parent.
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from .scenarios import BUDGET_VOLUME_UPLIFT

UNMAPPED = "UNMAPPED"
NCI_CLASS = "nci"
NCI_ACCOUNT = "G6900"
NON_PNL_CLASSES = ("balance_sheet",)
CONSOLIDATION_ENTITIES = ("ELIM", "GRP")

KEYS = [
    "scenario_key", "entity", "fiscal_year", "period_no",
    "group_account", "account_class", "product_node_id",
]


def _actuals(gl: pl.DataFrame, local_to_group: pl.DataFrame, entities: pl.DataFrame) -> pl.DataFrame:
    """ACT lifted to group accounts, keeping the translated measures from fact_gl."""
    frame = gl.filter(
        (pl.col("scenario_key") == "ACT")
        & (pl.col("ledger") != "TAX")
        & (~pl.col("account_class").is_in(NON_PNL_CLASSES))
    ).join(entities.select("entity", "coa_id"), on="entity", how="left")

    mapping = local_to_group.select(
        "coa_id",
        pl.col("local_account").alias("account_code"),
        pl.col("group_account").alias("_mapped"),
        "valid_from", "valid_to", "mapping_gap_from", "mapping_gap_to",
    ).unique(subset=["coa_id", "account_code"], keep="first")

    period = pl.col("period_key")
    in_gap = (
        pl.col("mapping_gap_from").is_not_null()
        & (period >= pl.col("mapping_gap_from"))
        & (period <= pl.col("mapping_gap_to"))
    )
    out_of_validity = (
        (pl.col("valid_from").is_not_null() & (period < pl.col("valid_from")))
        | (pl.col("valid_to").is_not_null() & (period > pl.col("valid_to")))
    )
    frame = frame.join(mapping, on=["coa_id", "account_code"], how="left").with_columns(
        pl.when(pl.col("entity").is_in(CONSOLIDATION_ENTITIES))
        .then(pl.col("account_code"))
        .when(pl.col("_mapped").is_null() | in_gap | out_of_validity)
        .then(pl.lit(UNMAPPED))
        .otherwise(pl.col("_mapped"))
        .alias("group_account")
    )

    return (
        frame.group_by(KEYS + ["currency"])
        .agg(
            pl.col("amount_lc").sum(),
            pl.col("amount_gc_actual_rates").sum().alias("amount_gc"),
            pl.col("amount_gc_budget_rates").sum().alias("amount_gc_cc"),
            pl.col("fx_rate_avg").first().alias("fx_rate"),
            pl.col("fx_rate_budget").first(),
        )
        .with_columns(pl.lit(True).alias("is_closed_month"))
    )


def _rates(fx: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(currency, fy, period) -> average rate, and (currency, fy) -> budget rate."""
    average = fx.select(
        pl.col("from_currency").alias("currency"), "fiscal_year", "period_no",
        pl.col("rate_avg").alias("_rate_avg"),
    ).unique(subset=["currency", "fiscal_year", "period_no"], keep="first")
    budget = fx.select(
        pl.col("from_currency").alias("currency"), "fiscal_year",
        pl.col("rate_budget").alias("fx_rate_budget"),
    ).drop_nulls().unique(subset=["currency", "fiscal_year"], keep="first")
    return average, budget


def _plan(
    budget: pl.DataFrame,
    forecast: pl.DataFrame,
    entities: pl.DataFrame,
    fx: pl.DataFrame,
) -> pl.DataFrame:
    """Budget and forecast submissions translated to group currency."""
    frames = []
    if budget.height:
        frames.append(budget.with_columns(pl.lit(False).alias("is_closed_month")))
    if forecast.height:
        frames.append(forecast)
    if not frames:
        return pl.DataFrame()

    plan = (
        pl.concat(frames, how="diagonal")
        .filter(~pl.col("account_class").is_in(NON_PNL_CLASSES))
        .rename({"amount": "amount_lc"})
        .join(entities.select("entity", "currency"), on="entity", how="left")
        .group_by(KEYS + ["currency", "is_closed_month"])
        .agg(pl.col("amount_lc").sum())
    )

    average, budget_rate = _rates(fx)
    plan = (
        plan.join(average, on=["currency", "fiscal_year", "period_no"], how="left")
        .join(budget_rate, on=["currency", "fiscal_year"], how="left")
        .with_columns(pl.col("fx_rate_budget").fill_null(1.0), pl.col("_rate_avg").fill_null(1.0))
        .with_columns(
            # Closed forecast months are actuals, so they carry actual rates.
            pl.when(pl.col("is_closed_month"))
            .then(pl.col("_rate_avg"))
            .otherwise(pl.col("fx_rate_budget"))
            .alias("fx_rate")
        )
        .with_columns(
            (pl.col("amount_lc") * pl.col("fx_rate")).round(2).alias("amount_gc"),
            (pl.col("amount_lc") * pl.col("fx_rate_budget")).round(2).alias("amount_gc_cc"),
        )
        .drop("_rate_avg")
    )
    return plan


def _group_rows(frame: pl.DataFrame, group_currency: str) -> pl.DataFrame:
    """Rows booked at group level: already in group currency, so every rate is 1.0."""
    return frame.with_columns(
        pl.lit(group_currency).alias("currency"),
        pl.col("amount_gc").alias("amount_lc"),
        pl.lit(1.0).alias("fx_rate"),
        pl.lit(1.0).alias("fx_rate_budget"),
    )


def _planned_eliminations(
    actual: pl.DataFrame, scenarios: pl.DataFrame, board_version: str, group_currency: str
) -> pl.DataFrame:
    """ELIM rows for every plan scenario (see module docstring)."""
    elim = actual.filter(pl.col("entity") == "ELIM")
    if elim.height == 0:
        return pl.DataFrame()
    value_cols = ["amount_gc", "amount_gc_cc"]
    base = elim.filter(~pl.col("is_period13")) if "is_period13" in elim.columns else elim
    base = base.select(
        "entity", "fiscal_year", "period_no", "group_account", "account_class",
        "product_node_id", *value_cols,
    )

    parts = []
    budgets = scenarios.filter(pl.col("scenario_type") == "BUD")
    for scenario in budgets.to_dicts():
        rolled = base.filter(pl.col("fiscal_year") == int(scenario["fiscal_year"]) - 1).with_columns(
            pl.lit(scenario["scenario_key"]).alias("scenario_key"),
            pl.lit(int(scenario["fiscal_year"]), dtype=pl.Int64).alias("fiscal_year"),
            *[(pl.col(c) * (1.0 + BUDGET_VOLUME_UPLIFT)).round(2).alias(c) for c in value_cols],
            pl.lit(False).alias("is_closed_month"),
        )
        parts.append(rolled)
    planned = pl.concat(parts, how="diagonal") if parts else pl.DataFrame()

    forecasts = scenarios.filter(pl.col("scenario_type") == "FC")
    for scenario in forecasts.to_dicts():
        fy, closed = int(scenario["fiscal_year"]), int(scenario["closed_months"])
        copied = base.filter(
            (pl.col("fiscal_year") == fy) & (pl.col("period_no") <= closed)
        ).with_columns(
            pl.lit(scenario["scenario_key"]).alias("scenario_key"),
            pl.lit(True).alias("is_closed_month"),
        )
        parts.append(copied)
        if planned.height:
            board = f"{board_version}_FY{fy}"
            open_months = planned.filter(
                (pl.col("scenario_key") == board) & (pl.col("period_no") > closed)
            ).with_columns(pl.lit(scenario["scenario_key"]).alias("scenario_key"))
            parts.append(open_months)

    if not parts:
        return pl.DataFrame()
    return _group_rows(pl.concat(parts, how="diagonal"), group_currency)


def _nci(frame: pl.DataFrame, entities: pl.DataFrame, group_currency: str) -> pl.DataFrame:
    """Minority share of each partly-owned entity's result, booked on GRP."""
    owned = {
        row["entity"]: float(row["nci_pct"])
        for row in entities.to_dicts()
        if float(row.get("nci_pct") or 0.0) > 0
    }
    if not owned:
        return pl.DataFrame()
    pct = pl.col("entity").replace_strict(owned, default=0.0, return_dtype=pl.Float64)
    # Ledger sign: a profit is a negative sum, and the minority's share of it is a debit
    # (a positive charge against the owners' result).
    result = (
        frame.filter(pl.col("entity").is_in(list(owned)))
        .group_by(["scenario_key", "entity", "fiscal_year", "period_no", "is_closed_month"])
        .agg(pl.col("amount_gc").sum(), pl.col("amount_gc_cc").sum())
        .with_columns(
            (-pct * pl.col("amount_gc")).round(2).alias("amount_gc"),
            (-pct * pl.col("amount_gc_cc")).round(2).alias("amount_gc_cc"),
        )
        .group_by(["scenario_key", "fiscal_year", "period_no", "is_closed_month"])
        .agg(pl.col("amount_gc").sum(), pl.col("amount_gc_cc").sum())
        .with_columns(
            pl.lit("GRP").alias("entity"),
            pl.lit(NCI_ACCOUNT).alias("group_account"),
            pl.lit(NCI_CLASS).alias("account_class"),
            pl.lit("~NA~").alias("product_node_id"),
        )
    )
    return _group_rows(result, group_currency)


def build_pnl(
    cfg: Config,
    gl: pl.DataFrame,
    budget: pl.DataFrame,
    forecast: pl.DataFrame,
    local_to_group: pl.DataFrame,
    entities: pl.DataFrame,
    scenarios: pl.DataFrame,
    calendar: pl.DataFrame,
    fx: pl.DataFrame,
) -> pl.DataFrame:
    group_currency = str(cfg.get("fx.group_currency"))
    board_version = list(cfg.get("scenarios.budget_versions"))[-1]
    periods = calendar.select("fiscal_year", "period_no", "period_key", "quarter", "is_period13")

    actual = _actuals(gl, local_to_group, entities).join(
        periods, on=["fiscal_year", "period_no"], how="left"
    )
    parts = [actual]
    plan = _plan(budget, forecast, entities, fx)
    if plan.height:
        parts.append(plan)
    elim = _planned_eliminations(actual, scenarios, board_version, group_currency)
    if elim.height:
        parts.append(elim)

    frame = pl.concat(
        [p.drop("period_key", "quarter", "is_period13", strict=False) for p in parts],
        how="diagonal",
    )
    nci = _nci(frame, entities, group_currency)
    if nci.height:
        frame = pl.concat([frame, nci], how="diagonal")

    meta = scenarios.select("scenario_key", "scenario_type", "version")
    return (
        frame.join(meta, on="scenario_key", how="left")
        .join(periods, on=["fiscal_year", "period_no"], how="left")
        .with_columns(
            pl.col("amount_lc").round(2),
            pl.col("amount_gc").round(2),
            pl.col("amount_gc_cc").round(2),
            pl.lit(group_currency).alias("group_currency"),
        )
        .select(
            "scenario_key", "scenario_type", "version", "entity",
            "fiscal_year", "period_no", "period_key", "quarter", "is_period13",
            "is_closed_month", "group_account", "account_class", "product_node_id",
            "currency", "amount_lc", "fx_rate", "fx_rate_budget",
            "group_currency", "amount_gc", "amount_gc_cc",
        )
        .sort(["scenario_key", "entity", "fiscal_year", "period_no", "group_account"])
    )


def build_income_statement(
    pnl: pl.DataFrame, lines: pl.DataFrame, bridge: pl.DataFrame
) -> pl.DataFrame:
    """``rpt_income_statement``: every report line for every entity/scenario/period.

    Values are in presentation sign (revenue positive, costs negative) because the
    bridge weights already carry it; a column of lines sums to the subtotal below it.
    """
    if lines.height == 0 or pnl.height == 0:
        return pl.DataFrame()
    return (
        pnl.join(bridge, on="account_class", how="inner")
        .group_by(["report_id", "line_id", "scenario_key", "entity", "fiscal_year", "period_no"])
        .agg(
            (pl.col("amount_gc") * pl.col("weight")).sum().round(2).alias("amount_gc"),
            (pl.col("amount_gc_cc") * pl.col("weight")).sum().round(2).alias("amount_gc_cc"),
            (pl.col("amount_lc") * pl.col("weight")).sum().round(2).alias("amount_lc"),
        )
        .join(
            lines.select("report_id", "line_id", "label", "sort_order", "indent", "is_subtotal"),
            on=["report_id", "line_id"],
            how="left",
        )
        .join(
            pnl.select("scenario_key", "scenario_type", "version").unique(),
            on="scenario_key",
            how="left",
        )
        .select(
            "report_id", "scenario_key", "scenario_type", "version", "entity",
            "fiscal_year", "period_no", "sort_order", "line_id", "label", "indent",
            "is_subtotal", "amount_gc", "amount_gc_cc", "amount_lc",
        )
        .sort(["report_id", "scenario_key", "entity", "fiscal_year", "period_no", "sort_order"])
    )
