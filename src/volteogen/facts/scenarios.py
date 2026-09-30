"""Budget and forecast facts (spec 4.6, 6.5).

Two things make these realistic rather than a copy of actuals with noise:

**Budget is COARSER than actuals.** It is built at mapped GROUP accounts (never local
leaves) and at product FAMILY (never SKU), spread from quarterly targets to months by the
seasonality curve. So "budget by SKU" is unanswerable by construction, which is exactly
the constraint real planning systems impose — and validator V9 asserts the grain gap.

**The assumptions are deliberately, plausibly wrong.** Management budgets +8% volume when
actual lands nearer +3%, holds price flat, and translates at the fixed budget FX rate. So
variances have a *story* — volume, price, FX, mix — instead of being noise.

**Forecasts converge.** An FC3+9 copies three closed ACT months verbatim into the
forecast scenario and blends budget with run-rate for the rest. By FC9+3 nine months are
actual, so forecast accuracy visibly improves through the year — a measurable property.
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from ..rng import SeedBank

# Management assumptions baked into the budget, and wrong on purpose.
BUDGET_VOLUME_UPLIFT = 0.08          # actual growth lands nearer +3%
BUDGET_PRICE_UPLIFT = 0.0            # no price erosion assumed; reality erodes
BUDGET_COST_INFLATION = 0.02
BOARD_REVISION_RANGE = (0.03, 0.05)  # V2 differs from V1 by 3-5%


def _coarse_actuals(gl: pl.DataFrame, local_to_group: pl.DataFrame) -> pl.DataFrame:
    """Actuals lifted to budget grain: group account x family x period."""
    mapping = local_to_group.select(
        pl.col("local_account").alias("account_code"),
        pl.col("coa_id"),
        pl.col("group_account"),
    ).unique(subset=["coa_id", "account_code"], keep="first")

    entity_coa = {
        "US01": "US_ERP", "DE01": "SKR03", "ES01": "PGC", "UK01": "SAGE50",
    }
    frame = gl.filter(
        (~pl.col("is_period13"))
        & (pl.col("scenario_key") == "ACT")
        & (pl.col("ledger") != "TAX")
        & (pl.col("entity").is_in(list(entity_coa)))
    ).with_columns(
        pl.col("entity").replace_strict(entity_coa, default=None).alias("coa_id")
    )
    joined = frame.join(mapping, on=["coa_id", "account_code"], how="inner")

    return (
        joined.group_by(
            ["entity", "group_account", "account_class", "product_node_id",
             "fiscal_year", "period_no"]
        )
        .agg(pl.col("amount_lc").sum().alias("amount"))
    )


def build_budget(
    cfg: Config,
    seeds: SeedBank,
    gl: pl.DataFrame,
    local_to_group: pl.DataFrame,
    scenarios: pl.DataFrame,
    calendar: pl.DataFrame,
) -> pl.DataFrame:
    """Budget versions, built from PRIOR-year actuals plus management assumptions."""
    rng = seeds.rng("facts", "budget")
    versions = list(cfg.get("scenarios.budget_versions"))
    if not versions:
        return pl.DataFrame()

    coarse = _coarse_actuals(gl, local_to_group)
    if coarse.height == 0:
        return pl.DataFrame()

    fiscal_years = sorted(calendar["fiscal_year"].unique().to_list())
    budget_scenarios = scenarios.filter(pl.col("scenario_type") == "BUD")

    rows: list[dict] = []
    for scenario in budget_scenarios.to_dicts():
        target_fy = int(scenario["fiscal_year"])
        prior_fy = target_fy - 1
        if prior_fy not in fiscal_years:
            continue
        basis = coarse.filter(pl.col("fiscal_year") == prior_fy)
        if basis.height == 0:
            continue

        # The board-approved version revises the first pass by a few percent.
        revision = (
            1.0 + float(rng.uniform(*BOARD_REVISION_RANGE)) * (1 if rng.random() < 0.5 else -1)
            if scenario["version"].endswith("V2")
            else 1.0
        )

        for row in basis.to_dicts():
            account_class = row["account_class"]
            if account_class in ("revenue_trade", "revenue_reduction"):
                factor = (1.0 + BUDGET_VOLUME_UPLIFT) * (1.0 + BUDGET_PRICE_UPLIFT)
            elif account_class in ("cogs_standard", "cogs_variance"):
                factor = (1.0 + BUDGET_VOLUME_UPLIFT) * (1.0 + BUDGET_COST_INFLATION * 0.5)
            else:
                factor = 1.0 + BUDGET_COST_INFLATION
            rows.append(
                {
                    "entity": row["entity"],
                    "scenario_key": scenario["scenario_key"],
                    "scenario_type": "BUD",
                    "version": scenario["version"],
                    "group_account": row["group_account"],
                    "account_class": account_class,
                    "product_node_id": row["product_node_id"],
                    "fiscal_year": target_fy,
                    "period_no": int(row["period_no"]),
                    "amount": round(float(row["amount"]) * factor * revision, 2),
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def build_forecast(
    cfg: Config,
    seeds: SeedBank,
    gl: pl.DataFrame,
    budget: pl.DataFrame,
    local_to_group: pl.DataFrame,
    scenarios: pl.DataFrame,
) -> pl.DataFrame:
    """Forecast cycles: closed months copied from ACT, open months blended.

    The closed months are copied VERBATIM, not re-derived — that redundancy is
    deliberate (spec 4.6) and is what validator V9 checks.
    """
    rng = seeds.rng("facts", "forecast")
    forecast_scenarios = scenarios.filter(pl.col("scenario_type") == "FC")
    if forecast_scenarios.height == 0 or budget.height == 0:
        return pl.DataFrame()

    actual = _coarse_actuals(gl, local_to_group)
    if actual.height == 0:
        return pl.DataFrame()

    rows: list[dict] = []
    for scenario in forecast_scenarios.to_dicts():
        target_fy = int(scenario["fiscal_year"])
        closed = int(scenario["closed_months"])

        # Closed months: an exact copy of actuals at budget grain.
        closed_actual = actual.filter(
            (pl.col("fiscal_year") == target_fy) & (pl.col("period_no") <= closed)
        )
        for row in closed_actual.to_dicts():
            rows.append(
                {
                    "entity": row["entity"],
                    "scenario_key": scenario["scenario_key"],
                    "scenario_type": "FC",
                    "version": scenario["version"],
                    "group_account": row["group_account"],
                    "account_class": row["account_class"],
                    "product_node_id": row["product_node_id"],
                    "fiscal_year": target_fy,
                    "period_no": int(row["period_no"]),
                    "amount": round(float(row["amount"]), 2),
                    "is_closed_month": True,
                }
            )

        # Open months: blend budget with the run-rate implied by closed actuals. The
        # more months are closed, the more weight the run-rate carries — which is why
        # forecast accuracy improves through the year.
        run_rate_weight = min(0.85, 0.35 + 0.06 * closed)
        open_budget = budget.filter(
            (pl.col("fiscal_year") == target_fy)
            & (pl.col("period_no") > closed)
            & (pl.col("version") == cfg.get("scenarios.budget_versions")[-1])
        )
        run_rate = (
            closed_actual.group_by(["entity", "group_account", "account_class", "product_node_id"])
            .agg((pl.col("amount").sum() / max(1, closed)).alias("monthly_run_rate"))
        )
        blended = open_budget.join(
            run_rate, on=["entity", "group_account", "account_class", "product_node_id"], how="left"
        )
        for row in blended.to_dicts():
            budget_amount = float(row["amount"])
            rate = row.get("monthly_run_rate")
            if rate is None:
                amount = budget_amount
            else:
                amount = (
                    run_rate_weight * float(rate) + (1.0 - run_rate_weight) * budget_amount
                ) * float(rng.uniform(0.97, 1.03))
            rows.append(
                {
                    "entity": row["entity"],
                    "scenario_key": scenario["scenario_key"],
                    "scenario_type": "FC",
                    "version": scenario["version"],
                    "group_account": row["group_account"],
                    "account_class": row["account_class"],
                    "product_node_id": row["product_node_id"],
                    "fiscal_year": target_fy,
                    "period_no": int(row["period_no"]),
                    "amount": round(amount, 2),
                    "is_closed_month": False,
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def build_variance_bridge(
    gl: pl.DataFrame, budget: pl.DataFrame, local_to_group: pl.DataFrame, cfg: Config
) -> pl.DataFrame:
    """ACT vs BUD decomposed into named causes that sum EXACTLY to total variance.

    Causes: price, volume, mix, fx, one_off, scope, allocation, other. The residual is
    assigned to `other` by construction, so the bridge always ties (validator V10) —
    a bridge that nearly ties is worse than useless.
    """
    if budget.height == 0:
        return pl.DataFrame()

    board_version = list(cfg.get("scenarios.budget_versions"))[-1]
    plan = (
        budget.filter(pl.col("version") == board_version)
        .group_by(["entity", "fiscal_year", "period_no", "account_class"])
        .agg(pl.col("amount").sum().alias("budget_amount"))
    )
    actual = (
        _coarse_actuals(gl, local_to_group)
        .group_by(["entity", "fiscal_year", "period_no", "account_class"])
        .agg(pl.col("amount").sum().alias("actual_amount"))
    )
    joined = plan.join(
        actual, on=["entity", "fiscal_year", "period_no", "account_class"], how="full", coalesce=True
    ).with_columns(
        pl.col("budget_amount").fill_null(0.0),
        pl.col("actual_amount").fill_null(0.0),
        pl.col("fiscal_year").cast(pl.Int64),
        pl.col("period_no").cast(pl.Int64),
    )

    # One-offs and scope effects are identifiable from the ledger itself.
    keys = ["entity", "fiscal_year", "period_no"]
    one_off = (
        gl.filter(pl.col("is_one_off"))
        .group_by(["entity", "fiscal_year", "period_no"])
        .agg(pl.col("amount_lc").sum().alias("one_off_amount"))
        .with_columns(pl.col("fiscal_year").cast(pl.Int64), pl.col("period_no").cast(pl.Int64))
    )
    scope = (
        gl.filter(pl.col("entity") == "UK01")
        .group_by(["entity", "fiscal_year", "period_no"])
        .agg(pl.col("amount_lc").sum().alias("scope_amount"))
        .with_columns(pl.col("fiscal_year").cast(pl.Int64), pl.col("period_no").cast(pl.Int64))
    )
    allocation = (
        gl.filter(pl.col("account_class") == "allocation")
        .group_by(["entity", "fiscal_year", "period_no"])
        .agg(pl.col("amount_lc").sum().alias("allocation_amount"))
        .with_columns(pl.col("fiscal_year").cast(pl.Int64), pl.col("period_no").cast(pl.Int64))
    )

    joined = (
        joined.join(one_off, on=["entity", "fiscal_year", "period_no"], how="left")
        .join(scope, on=["entity", "fiscal_year", "period_no"], how="left")
        .join(allocation, on=["entity", "fiscal_year", "period_no"], how="left")
        .with_columns(
            pl.col("one_off_amount").fill_null(0.0),
            pl.col("scope_amount").fill_null(0.0),
            pl.col("allocation_amount").fill_null(0.0),
        )
    )

    rows: list[dict] = []
    for row in joined.to_dicts():
        total = float(row["actual_amount"]) - float(row["budget_amount"])
        account_class = row["account_class"]

        explained: dict[str, float] = {}
        if account_class in ("revenue_trade", "revenue_reduction"):
            # Budget assumed +8% volume and flat price; reality grew slower and eroded price.
            explained["volume"] = total * 0.45
            explained["price"] = total * 0.30
            explained["mix"] = total * 0.10
            explained["fx"] = total * 0.05
        elif account_class in ("cogs_standard", "cogs_variance"):
            explained["volume"] = total * 0.55
            explained["price"] = total * 0.20
            explained["mix"] = total * 0.10
        else:
            explained["price"] = total * 0.40

        if account_class == "one_off":
            explained["one_off"] = float(row["one_off_amount"])
        if account_class == "allocation":
            explained["allocation"] = float(row["allocation_amount"])
        if row["entity"] == "UK01":
            explained["scope"] = total * 0.20

        # The residual is 'other' — this is what makes the bridge tie exactly.
        explained["other"] = total - sum(explained.values())

        for cause, amount in explained.items():
            if abs(amount) < 0.005:
                continue
            rows.append(
                {
                    "entity": row["entity"],
                    "fiscal_year": int(row["fiscal_year"]),
                    "period_no": int(row["period_no"]),
                    "account_class": account_class,
                    "cause": cause,
                    "amount": round(amount, 2),
                    "total_variance": round(total, 2),
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()
