"""GL assembler: fact_gl at monthly trial-balance grain.

Grain is ``entity x ledger x account x applicable-dims x period x scenario`` — and
"applicable-dims" is enforced from the matrix, not chosen ad hoc. Revenue and COGS rows
are DERIVED by aggregating the sub-ledger so the two tie exactly (V7); everything else
is derived from the physical drivers.

The three-state dimension rule (spec 5) is applied here and only here, via
:func:`volteogen.dims.applicability.resolve_dimension_value`:

``~NA~``       structurally not applicable
``UNASSIGNED`` applicable but untagged (a defect)
real member    applicable and tagged
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from ..dims.applicability import NA, UNASSIGNED, Applicability, resolve_dimension_value
from ..rng import SeedBank
from .account_router import AccountRouter

GL_SCHEMA_ORDER = [
    "entity", "ledger", "scenario_key", "account_code", "account_class",
    "period_key", "fiscal_year", "period_no", "is_period13",
    "product_node_id", "market_node_id", "customer_id", "cost_center_node_id",
    "ic_partner", "profit_center_node_id",
    "amount_lc", "currency", "quantity", "source", "is_topside", "is_one_off",
]


def _row(
    applicability: Applicability,
    entity: str,
    ledger: str,
    account_code: str,
    account_class: str,
    period: dict,
    amount: float,
    currency: str,
    source: str,
    product=None,
    market=None,
    customer=None,
    cost_center=None,
    ic_partner=None,
    profit_center=None,
    quantity: float | None = None,
    scenario_key: str = "ACT",
    is_one_off: bool = False,
    is_topside: bool = False,
) -> dict:
    """Build one GL row, resolving every dimension through the applicability matrix."""
    return {
        "entity": entity,
        "ledger": ledger,
        "scenario_key": scenario_key,
        "account_code": account_code,
        "account_class": account_class,
        "period_key": period["period_key"],
        "fiscal_year": int(period["fiscal_year"]),
        "period_no": int(period["period_no"]),
        "is_period13": bool(period.get("is_period13", False)),
        "product_node_id": resolve_dimension_value(
            applicability, account_class, "product", product, condition_met=product is not None
        ),
        "market_node_id": resolve_dimension_value(
            applicability, account_class, "market", market, condition_met=market is not None
        ),
        "customer_id": resolve_dimension_value(
            applicability, account_class, "customer", customer, condition_met=customer is not None
        ),
        "cost_center_node_id": resolve_dimension_value(
            applicability, account_class, "cost_center", cost_center,
            condition_met=cost_center is not None,
        ),
        "ic_partner": resolve_dimension_value(
            applicability, account_class, "ic_partner", ic_partner,
            condition_met=ic_partner is not None,
        ),
        "profit_center_node_id": profit_center or NA,
        "amount_lc": round(float(amount), 2),
        "currency": currency,
        "quantity": quantity,
        "source": source,
        "is_topside": is_topside,
        "is_one_off": is_one_off,
    }



# ---------------------------------------------------------------------------
# Vectorised block construction.
#
# `_row()` resolves five dimensions through the applicability matrix PER ROW, which
# is fine for the small event blocks but dominates the build once revenue and COGS
# reach millions of rows. The rule is CONSTANT for a given account class, so the whole
# per-row decision collapses to one polars expression per column, evaluated once for
# the entire block.
# ---------------------------------------------------------------------------
_DIM_COLUMNS = (
    ("product", "product_node_id"),
    ("market", "market_node_id"),
    ("customer", "customer_id"),
    ("cost_center", "cost_center_node_id"),
    ("ic_partner", "ic_partner"),
)


def _dim_exprs(applicability: Applicability, account_class: str, sources: dict) -> list:
    """One expression per dimension column, resolved once for the whole block.

    Mirrors :func:`resolve_dimension_value` exactly: `~NA~` where the matrix says the
    dimension does not apply, `UNASSIGNED` where it is required but absent, the member
    otherwise. Conditional rules fall to `~NA~` when the condition is not met, which for
    a vectorised block means a null source.
    """
    out = []
    for dimension, column in _DIM_COLUMNS:
        rule = applicability.rule(account_class, dimension)
        expr = sources.get(dimension)
        if rule == "na":
            out.append(pl.lit(NA).alias(column))
        elif rule.startswith("conditional"):
            out.append(
                pl.lit(NA).alias(column) if expr is None
                else expr.fill_null(pl.lit(NA)).alias(column)
            )
        elif rule in ("applicable", "mandatory"):
            out.append(
                pl.lit(UNASSIGNED).alias(column) if expr is None
                else expr.fill_null(pl.lit(UNASSIGNED)).alias(column)
            )
        else:
            out.append(
                pl.lit(NA).alias(column) if expr is None
                else expr.fill_null(pl.lit(NA)).alias(column)
            )
    return out


def _block(
    frame: pl.DataFrame,
    applicability: Applicability,
    *,
    account_class: str,
    account_code,
    amount,
    source,
    ledger,
    currency,
    quantity=None,
    dims: dict | None = None,
) -> pl.DataFrame:
    """Build a full GL block from a source frame, entirely in polars."""
    return frame.select(
        pl.col("entity"),
        ledger.alias("ledger"),
        pl.lit("ACT").alias("scenario_key"),
        account_code.alias("account_code"),
        pl.lit(account_class).alias("account_class"),
        pl.col("period_key"),
        pl.col("fiscal_year").cast(pl.Int64),
        pl.col("period_no").cast(pl.Int64),
        pl.lit(False).alias("is_period13"),
        *_dim_exprs(applicability, account_class, dims or {}),
        pl.lit(NA).alias("profit_center_node_id"),
        amount.round(2).alias("amount_lc"),
        currency.alias("currency"),
        (pl.lit(None, pl.Float64) if quantity is None else quantity.cast(pl.Float64)).alias("quantity"),
        source.alias("source"),
        pl.lit(False).alias("is_topside"),
        pl.lit(False).alias("is_one_off"),
    )


def assemble_gl(
    cfg: Config,
    pack: dict,
    seeds: SeedBank,
    applicability: Applicability,
    router: AccountRouter,
    entities: pl.DataFrame,
    calendar: pl.DataFrame,
    sales: pl.DataFrame,
    reductions: pl.DataFrame,
    headcount: pl.DataFrame,
    energy: pl.DataFrame,
    machine_hours: pl.DataFrame,
    cost_centers: pl.DataFrame,
    markets: pl.DataFrame,
) -> pl.DataFrame:
    rng = seeds.rng("facts", "gl")
    economics = pack["economics"]

    # The pack's cost_structure states the shape of the cost base. Payroll is already
    # industry-shaped (headcount x the pack's salary bands), so the discretionary
    # baselines are scaled RELATIVE to payroll to match it. Without this, a services
    # firm would carry a hardware firm's marketing budget and the calibrator would have
    # to distort it back by an order of magnitude.
    structure = economics.get("cost_structure") or {}
    payroll_share = float(structure.get("payroll", 0.19)) or 0.19
    _REFERENCE = {"marketing": 0.3158, "other_opex": 0.2105}  # consumer-electronics ratios
    opex_shape = {
        key: (float(structure.get(key, ref * payroll_share)) / payroll_share) / ref
        for key, ref in _REFERENCE.items()
    }

    periods = {r["period_key"]: r for r in calendar.to_dicts()}
    currency = {r["entity"]: r["currency"] for r in entities.to_dicts()}
    primary_ledger = {
        r["entity"]: ("IFRS" if r["parallel_ledgers"] else "LOCAL")
        for r in entities.to_dicts()
    }

    # UK01 was acquired mid-history and must not post anything before that month —
    # otherwise the M&A scope effect in group growth disappears.
    consolidated_from = {
        r["entity"]: int(r["consolidated_from_month"]) for r in entities.to_dicts()
    }
    month_of_period = {
        r["period_key"]: int(r["month_index"]) for r in calendar.to_dicts()
    }

    def in_scope(entity: str, period_key: str) -> bool:
        return month_of_period.get(period_key, 0) >= consolidated_from.get(entity, 0)

    market_country = {
        r["node_id"]: r.get("country") for r in markets.filter(pl.col("is_leaf")).to_dicts()
    }
    cc_by_entity_function: dict[tuple[str, str], list[str]] = {}
    for row in cost_centers.filter(pl.col("is_leaf")).to_dicts():
        key = (row["entity"], row.get("function") or "GA")
        cc_by_entity_function.setdefault(key, []).append(row["node_id"])

    rows: list[dict] = []
    blocks: list[pl.DataFrame] = []

    # -- Revenue and COGS: aggregated FROM the sub-ledger so V7 ties by construction --
    revenue_grain = (
        sales.group_by(
            ["entity", "period_key", "fiscal_year", "period_no", "family_name",
             "category_name", "market_node_id", "customer_id", "product_node_id"]
        )
        .agg(
            pl.col("invoiced_amount").sum().alias("revenue"),
            pl.col("std_cost_amount").sum().alias("cogs"),
            pl.col("qty").sum().alias("qty"),
        )
        .sort(
            ["entity", "period_key", "family_name", "market_node_id", "customer_id",
             "product_node_id"]
        )
    )

    # GL carries product at FAMILY grain even though the sub-ledger is per SKU.
    family_node = _family_node_lookup(sales)

    # Family-grain product tag, joined rather than looked up per row.
    family_frame = pl.DataFrame(
        [{"entity": e, "family_name": f, "family_node": n} for (e, f), n in family_node.items()]
    ) if family_node else pl.DataFrame(
        {"entity": [], "family_name": [], "family_node": []},
        schema={"entity": pl.Utf8, "family_name": pl.Utf8, "family_node": pl.Utf8},
    )
    country_frame = pl.DataFrame(
        [{"market_node_id": k, "country": v} for k, v in market_country.items()]
    ) if market_country else pl.DataFrame(
        {"market_node_id": [], "country": []},
        schema={"market_node_id": pl.Utf8, "country": pl.Utf8},
    )

    # Revenue account routing depends on (entity, country, category) only — a few
    # hundred combinations — so it resolves to a join instead of a call per row.
    categories = sorted(set(revenue_grain["category_name"].drop_nulls().to_list()))
    countries = sorted({c for c in market_country.values() if c}) + [None]
    route_rows = []
    posting_entities = sorted(set(revenue_grain["entity"].unique().to_list()))
    for entity in posting_entities:
        for country in countries:
            for category in categories:
                route_rows.append({
                    "entity": entity, "country": country, "category_name": category,
                    "revenue_account": router.revenue_account(entity, country, category),
                })
    routes = pl.DataFrame(route_rows) if route_rows else pl.DataFrame()

    cogs_account = {e: router.account_for(e, "cogs_standard") for e in posting_entities}

    grain = (
        revenue_grain
        .join(country_frame, on="market_node_id", how="left")
        .join(family_frame, on=["entity", "family_name"], how="left")
    )
    if routes.height:
        grain = grain.join(routes, on=["entity", "country", "category_name"], how="left")
    else:
        grain = grain.with_columns(pl.lit(None, pl.Utf8).alias("revenue_account"))

    product_tag = (
        pl.when(pl.col("product_node_id") == UNASSIGNED)
        .then(pl.lit(UNASSIGNED))
        .otherwise(pl.coalesce([pl.col("family_node"), pl.col("family_name")]))
    )
    ledger_expr = pl.col("entity").replace_strict(primary_ledger, default="LOCAL")
    currency_expr = pl.col("entity").replace_strict(currency, default="USD")

    blocks.append(_block(
        grain, applicability, account_class="revenue_trade",
        account_code=pl.col("revenue_account"), amount=-pl.col("revenue"),
        source=pl.lit("subledger"), ledger=ledger_expr, currency=currency_expr,
        quantity=pl.col("qty"),
        dims={"product": product_tag, "market": pl.col("market_node_id"),
              "customer": pl.col("customer_id")},
    ))
    blocks.append(_block(
        grain, applicability, account_class="cogs_standard",
        account_code=pl.col("entity").replace_strict(cogs_account, default=""),
        amount=pl.col("cogs"), source=pl.lit("subledger"),
        ledger=ledger_expr, currency=currency_expr, quantity=pl.col("qty"),
        dims={"product": product_tag},
    ))

    # -- Revenue reductions: family grain, country grain, named customers only --------
    kinds = sorted(set(reductions["reduction_kind"].unique().to_list())) if reductions.height else []
    reduction_account = {
        (e, k): router.revenue_reduction_account(e, k) for e in posting_entities for k in kinds
    }
    if reductions.height:
        red = (
            reductions.join(family_frame, on=["entity", "family_name"], how="left")
            .with_columns(
                pl.struct(["entity", "reduction_kind"]).map_elements(
                    lambda s: reduction_account.get((s["entity"], s["reduction_kind"]), ""),
                    return_dtype=pl.Utf8,
                ).alias("reduction_account")
            )
        )
        blocks.append(_block(
            red, applicability, account_class="revenue_reduction",
            account_code=pl.col("reduction_account"), amount=-pl.col("amount"),
            source=pl.lit("gross_to_net:") + pl.col("reduction_kind"),
            ledger=ledger_expr, currency=currency_expr,
            dims={"product": pl.coalesce([pl.col("family_node"), pl.col("family_name")]),
                  "market": pl.col("market_node_id"), "customer": pl.col("customer_id")},
        ))

    # -- Payroll: mandatory cost centre, country-specific employer charges ------------
    salary_bands = economics["payroll"]["salary_band_monthly"]
    charge_rates = economics["payroll"]["employer_charge_rate"]
    merit = economics["payroll"]["merit_raise"]
    merit_period = int(str(merit["period"]).replace("P", ""))
    merit_pct = float(merit["pct"])
    country_of = {r["entity"]: r["country"] for r in entities.to_dicts()}

    for row in headcount.to_dicts():
        entity = row["entity"]
        if not in_scope(entity, row["period_key"]):
            continue
        period = periods[row["period_key"]]
        band = salary_bands.get(row["function"], {"min": 3500, "max": 6500})
        avg_salary = float(rng.uniform(band["min"], band["max"]))
        if int(row["period_no"]) >= merit_period:
            avg_salary *= 1.0 + merit_pct
        gross = avg_salary * int(row["headcount"])
        # REALISM: same gross salary, very different total cost by country.
        charge = gross * float(charge_rates.get(country_of.get(entity, "US"), 0.15))
        rows.append(
            _row(
                applicability, entity, primary_ledger[entity],
                router.account_for(entity, "payroll"), "payroll", period,
                gross + charge, currency[entity], "payroll",
                cost_center=row["cost_center_node_id"], quantity=float(row["headcount"]),
            )
        )

    # -- Facilities: energy billed two months in arrears -----------------------------
    lag_index = {
        (r["entity"], int(r["month_index"])): r for r in energy.to_dicts()
    }
    for row in energy.to_dicts():
        entity = row["entity"]
        if not in_scope(entity, row["period_key"]):
            continue
        lag = int(row["invoice_lag_months"])
        source_row = lag_index.get((entity, int(row["month_index"]) - lag))
        usage = float((source_row or row)["energy_kwh"])
        period = periods[row["period_key"]]
        site_cc = _site_cost_center(cc_by_entity_function, entity)
        rows.append(
            _row(
                applicability, entity, primary_ledger[entity],
                router.account_for(entity, "facilities"), "facilities", period,
                usage * float(rng.uniform(0.16, 0.24)) + float(rng.uniform(9000, 26000)),
                currency[entity], "facilities", cost_center=site_cc, quantity=usage,
            )
        )

    # -- COGS variances: PLANT level, unallocatable to product (the spec's key case) --
    # An industry pack can disable a whole class (professional services has no plant,
    # so no production variance exists). Honour that before generating anything.
    variance_cfg = economics["cogs_variance"]
    if applicability.is_disabled("cogs_variance"):
        machine_hours = machine_hours.head(0)
    absorption_rate = float(variance_cfg["absorption_rate_per_unit"])
    plant_cc = _plant_cost_center(cost_centers)
    for row in machine_hours.to_dicts():
        entity = row["entity"]
        if entity not in currency:
            continue
        period = periods[row["period_key"]]
        utilisation = float(row["utilisation"])
        produced = float(row["produced_units"])
        # Absorption swings sign: under-absorbed below capacity, over-absorbed above.
        absorption = (utilisation - 1.0) * produced * absorption_rate * -1
        price_var = produced * float(
            rng.normal(variance_cfg["price_variance_pct"]["mu"], variance_cfg["price_variance_pct"]["sigma"])
        )
        usage_var = produced * float(
            rng.normal(variance_cfg["usage_variance_pct"]["mu"], variance_cfg["usage_variance_pct"]["sigma"])
        )
        for label, amount in (
            ("absorption", absorption), ("price", price_var), ("usage", usage_var)
        ):
            rows.append(
                _row(
                    applicability, entity, primary_ledger[entity],
                    router.account_for(entity, "cogs_variance"), "cogs_variance", period,
                    amount, currency[entity], f"variance:{label}", cost_center=plant_cc,
                )
            )

    # -- Marketing: 60% product campaigns (family-tagged), 40% brand (~NA~) -----------
    product_share = float(economics["opex"]["marketing_product_campaign_share"])
    families = sorted({f for (_, f) in family_node})
    for entity in currency:
        if entity not in {r["entity"] for r in headcount.to_dicts()}:
            continue
        mkt_ccs = cc_by_entity_function.get((entity, "MKT")) or cc_by_entity_function.get((entity, "GA"), [])
        if not mkt_ccs:
            continue
        for period_key, period in periods.items():
            if period.get("is_period13") or not in_scope(entity, period_key):
                continue
            base = float(rng.uniform(40000, 180000)) * _entity_size(entity) * opex_shape["marketing"]
            # Lumpy: marketing arrives in bursts, not a flat monthly line.
            if rng.random() < 0.18:
                base *= float(rng.uniform(1.8, 3.2))
            is_product_campaign = rng.random() < product_share
            family_tag = (
                family_node.get((entity, str(rng.choice(families))))
                if is_product_campaign and families
                else None
            )
            rows.append(
                _row(
                    applicability, entity, primary_ledger[entity],
                    router.account_for(entity, "marketing"), "marketing", period,
                    base, currency[entity], "marketing",
                    product=family_tag,
                    market=None,
                    cost_center=str(rng.choice(mkt_ccs)),
                )
            )

    # -- Depreciation: flat with step-changes at capex -------------------------------
    for entity in currency:
        ccs = [c for key, values in cc_by_entity_function.items() if key[0] == entity for c in values]
        if not ccs:
            continue
        level = float(rng.uniform(30000, 120000)) * _entity_size(entity)
        for period_key, period in sorted(periods.items()):
            if period.get("is_period13") or not in_scope(entity, period_key):
                continue
            if rng.random() < 0.06:      # a capex event steps the run-rate up
                level *= float(rng.uniform(1.05, 1.22))
            rows.append(
                _row(
                    applicability, entity, primary_ledger[entity],
                    router.account_for(entity, "depreciation"), "depreciation", period,
                    level, currency[entity], "depreciation", cost_center=str(rng.choice(ccs)),
                )
            )

    # -- Other opex: per-CC baselines x inflation x noise, plus lumpy items -----------
    inflation = float(economics["opex"]["inflation_annual"])
    lumpy_share = float(economics["opex"]["lumpy_share"])
    for row in headcount.to_dicts():
        entity = row["entity"]
        if not in_scope(entity, row["period_key"]):
            continue
        period = periods[row["period_key"]]
        years = int(row["month_index"]) / 12.0
        base = float(rng.uniform(1800, 5200)) * int(row["headcount"]) * (1.0 + inflation) ** years
        if rng.random() < lumpy_share * 0.25:
            base *= float(rng.uniform(1.5, 2.6))
        rows.append(
            _row(
                applicability, entity, primary_ledger[entity],
                router.account_for(entity, "other_opex"), "other_opex", period,
                base * 0.35 * opex_shape["other_opex"], currency[entity], "opex",
                cost_center=row["cost_center_node_id"],
            )
        )

    frame = pl.DataFrame(rows)
    frame = frame.select([c for c in GL_SCHEMA_ORDER if c in frame.columns])
    if blocks:
        ordered = [b.select([c for c in GL_SCHEMA_ORDER if c in b.columns]) for b in blocks]
        frame = pl.concat([frame, *ordered], how="diagonal")
    return _shape_cost_base(frame, structure)


# Classes whose level is discretionary and therefore shapeable.
_SHAPEABLE = ("payroll", "facilities", "marketing", "other_opex", "depreciation")


def _shape_cost_base(frame: pl.DataFrame, structure: dict) -> pl.DataFrame:
    """Scale discretionary opex to the industry pack's cost_structure shares.

    Payroll and opex are generated from headcount and per-cost-centre baselines, which
    do NOT scale with revenue. So raising SKU or market cardinality multiplies revenue
    while leaving the cost base flat, and the margin calibrator is then asked for
    40x adjustments it should never have to make.

    Sizing the cost base as a share of revenue first — exactly what cost_structure
    states — keeps the calibrator's levers near 1.0 at any cardinality, and leaves it
    doing what it is for: hitting the narrative, not repairing the scale.
    """
    shares = {c: float(structure.get(c, 0.0)) for c in _SHAPEABLE}
    if not any(shares.values()):
        return frame

    revenue = (
        frame.filter(pl.col("account_class").is_in(["revenue_trade", "revenue_reduction"]))
        .group_by("entity").agg((-pl.col("amount_lc").sum()).alias("revenue"))
    )
    actual = (
        frame.filter(pl.col("account_class").is_in(_SHAPEABLE))
        .group_by(["entity", "account_class"]).agg(pl.col("amount_lc").sum().alias("actual"))
    )
    if revenue.height == 0 or actual.height == 0:
        return frame

    # Cost base as a share of revenue, leaving room for a mid-single-digit net margin.
    cost_base_ratio = 0.92
    targets = actual.join(revenue, on="entity", how="inner").with_columns(
        pl.col("account_class").replace_strict(shares, default=0.0).alias("share")
    ).with_columns(
        (pl.col("revenue") * cost_base_ratio * pl.col("share")).alias("target")
    ).with_columns(
        pl.when((pl.col("actual").abs() > 1.0) & (pl.col("share") > 0))
        .then(pl.col("target") / pl.col("actual"))
        .otherwise(1.0)
        .clip(0.05, 40.0)
        .alias("shape_factor")
    ).select("entity", "account_class", "shape_factor")

    return (
        frame.join(targets, on=["entity", "account_class"], how="left")
        .with_columns(
            pl.when(pl.col("shape_factor").is_not_null())
            .then((pl.col("amount_lc") * pl.col("shape_factor")).round(2))
            .otherwise(pl.col("amount_lc"))
            .alias("amount_lc")
        )
        .drop("shape_factor")
    )


def _family_node_lookup(sales: pl.DataFrame) -> dict[tuple[str, str], str]:
    """Map (entity, family_name) to a stable family node id for GL product tagging."""
    out: dict[tuple[str, str], str] = {}
    for row in sales.select("entity", "family_name", "product_node_id_true").unique().to_dicts():
        key = (row["entity"], row["family_name"])
        if key not in out:
            # The family node is the SKU node's ancestor prefix; derive it structurally.
            parts = str(row["product_node_id_true"]).split(":")
            out[key] = ":".join(parts[:4]) if len(parts) >= 4 else row["product_node_id_true"]
    return out


def _site_cost_center(cc_map: dict[tuple[str, str], list[str]], entity: str) -> str | None:
    for function in ("GA", "LOG", "PRD", "SLS"):
        found = cc_map.get((entity, function))
        if found:
            return found[0]
    return None


def _plant_cost_center(cost_centers: pl.DataFrame) -> str | None:
    plant = cost_centers.filter(
        (pl.col("entity") == "DE01") & (pl.col("function") == "PRD") & pl.col("is_leaf")
    )
    return plant["node_id"][0] if plant.height else None


def _entity_size(entity: str) -> float:
    return {"US01": 1.0, "DE01": 0.85, "ES01": 0.38, "UK01": 0.30}.get(entity, 0.5)
