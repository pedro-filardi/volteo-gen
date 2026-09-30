"""Physical drivers. No P&L number is ever drawn directly (spec 6).

Everything downstream is a consequence of these quantities:

``units``         by SKU x market x month = base demand (family-level, ABC-weighted)
                  x seasonality x trend x launch curve x promo x lognormal noise
``headcount``     by cost centre — integer, sticky, with hiring steps
``energy``        by site — seasonal and INVERSE to sales (heating in winter)
``machine_hours`` at the plant — drives absorption, which is why COGS variances swing
                  sign with volume

The launch curve is the electronics-specific bit: a new model ramps over three months
and cannibalises its predecessor by 40%, so a family's total is smooth while its
members are not.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import Config
from ..rng import SeedBank

SITE_BY_ENTITY = {"US01": "US-HQ", "DE01": "DE-PLANT", "ES01": "ES-OFFICE", "UK01": "UK-OFFICE"}


def _seasonality_lookup(pack: dict) -> dict[str, list[float]]:
    curves = dict(pack["seasonality"]["curves"])
    return {k: [float(x) for x in v] for k, v in curves.items()}


def build_unit_drivers(
    cfg: Config,
    pack: dict,
    seeds: SeedBank,
    products: pl.DataFrame,
    markets: pl.DataFrame,
    coverage: pl.DataFrame,
    calendar: pl.DataFrame,
    entities: pl.DataFrame,
) -> pl.DataFrame:
    """fact_drivers: units by entity x SKU x market x month."""
    rng = seeds.rng("facts", "drivers", "units")
    curves = _seasonality_lookup(pack)
    notes = pack["seasonality"].get("notes") or {}
    noise_sigma = float(pack["seasonality"]["demand_noise"]["sigma"])
    lifecycle = pack["product_taxonomy"]["lifecycle"]
    ramp_months = int(lifecycle["ramp_months"])
    cannibalisation = float(lifecycle["cannibalisation_of_predecessor"])

    months = calendar.filter(~pl.col("is_period13")).sort("month_index")
    month_rows = months.to_dicts()

    leaves = products.filter(pl.col("is_leaf") & pl.col("sku_code").is_not_null())
    leaf_rows = leaves.to_dicts()

    market_leaves = markets.filter(pl.col("is_leaf"))
    market_by_id = {r["node_id"]: r for r in market_leaves.to_dicts()}

    entity_scale = _entity_scale(entities)

    promo = notes.get("promo_event") or {}
    promo_period = str(promo.get("period", "")).replace("P", "")
    promo_period_no = int(promo_period) if promo_period.isdigit() else None
    promo_uplift = float(promo.get("uplift", 0.0))
    promo_divisions = set(promo.get("divisions") or [])

    bts = notes.get("back_to_school") or {}
    bts_periods = {int(str(p).replace("P", "")) for p in (bts.get("periods") or [])}
    bts_uplift = float(bts.get("uplift", 0.0))

    trough = notes.get("post_holiday_trough") or {}
    trough_periods = {int(str(p).replace("P", "")) for p in (trough.get("periods") or [])}
    trough_factor = float(trough.get("factor", 0.0))

    # Per-entity market lists, so DE01 really does invoice Austria and Poland.
    entity_markets: dict[str, list[str]] = {}
    for row in coverage.to_dicts():
        entity_markets.setdefault(row["entity"], []).append(row["market_node_id"])

    records: list[dict] = []
    for entity, market_ids in entity_markets.items():
        scale = entity_scale.get(entity, 1.0)
        consolidated_from = _consolidated_from(entities, entity)
        for product in leaf_rows:
            division = product["division_name"]
            curve = curves.get(division) or curves.get("Audio")
            trend_annual = float(product["demand_trend_annual"] or 0.0)
            abc_weight = float(product["abc_weight"] or 1.0)
            launch = int(product["launch_period"] or 0)
            eol = product["eol_period"]

            # Base demand is per family, spread by ABC class — the level, not the shape.
            base = abc_weight * float(rng.uniform(60.0, 220.0)) * scale

            for market_id in market_ids:
                market = market_by_id.get(market_id)
                if market is None:
                    continue
                market_weight = float(rng.uniform(0.35, 1.6))

                for month in month_rows:
                    index = int(month["month_index"])
                    if index < consolidated_from:
                        continue
                    if index < launch:
                        continue
                    if eol is not None and index >= int(eol):
                        continue

                    period_no = int(month["period_no"])
                    factor = curve[period_no - 1]

                    if period_no in bts_periods:
                        factor *= 1.0 + bts_uplift
                    if period_no in trough_periods:
                        factor *= 1.0 + trough_factor
                    if promo_period_no == period_no and division in promo_divisions:
                        factor *= 1.0 + promo_uplift

                    age = index - launch
                    if age < ramp_months:
                        factor *= (age + 1) / (ramp_months + 1)

                    years = index / 12.0
                    factor *= (1.0 + trend_annual) ** years

                    # Late-life decay standing in for the successor's cannibalisation.
                    if eol is not None and index >= int(eol) - ramp_months:
                        factor *= 1.0 + cannibalisation * 0.5

                    units = base * market_weight * factor * float(
                        rng.lognormal(0.0, noise_sigma)
                    )
                    units = int(max(0, round(units)))
                    if units <= 0:
                        continue

                    records.append(
                        {
                            "entity": entity,
                            "product_node_id": product["node_id"],
                            "market_node_id": market_id,
                            "period_key": month["period_key"],
                            "month_index": index,
                            "fiscal_year": int(month["fiscal_year"]),
                            "period_no": period_no,
                            "units": units,
                            "driver_type": "units",
                        }
                    )

    if not records:
        raise ValueError("unit driver generation produced no rows — check lifecycle windows")
    return pl.DataFrame(records)


def _entity_scale(entities: pl.DataFrame) -> dict[str, float]:
    """Relative size of each entity's demand base."""
    return {"US01": 1.0, "DE01": 0.85, "ES01": 0.38, "UK01": 0.30}


def _consolidated_from(entities: pl.DataFrame, entity: str) -> int:
    row = entities.filter(pl.col("entity") == entity)
    return int(row["consolidated_from_month"][0]) if row.height else 0


def build_headcount_drivers(
    cfg: Config, pack: dict, seeds: SeedBank, cost_centers: pl.DataFrame, calendar: pl.DataFrame
) -> pl.DataFrame:
    """Headcount by cost centre: integer, sticky, stepping up at hiring months."""
    rng = seeds.rng("facts", "drivers", "headcount")
    hiring_steps = set(int(m) for m in pack["economics"]["payroll"]["hiring_step_months"])
    months = calendar.filter(~pl.col("is_period13")).sort("month_index").to_dicts()
    leaves = cost_centers.filter(pl.col("is_leaf")).to_dicts()

    records = []
    for cc in leaves:
        function = cc.get("function") or "GA"
        headcount = int(rng.integers(2, 14 if function in ("PRD", "SLS") else 8))
        for month in months:
            index = int(month["month_index"])
            if index in hiring_steps and rng.random() < 0.45:
                headcount += int(rng.integers(1, 3))
            # Rare attrition so the series is sticky but not monotonic.
            elif rng.random() < 0.03 and headcount > 2:
                headcount -= 1
            records.append(
                {
                    "cost_center_node_id": cc["node_id"],
                    "entity": cc["entity"],
                    "function": function,
                    "period_key": month["period_key"],
                    "month_index": index,
                    "fiscal_year": int(month["fiscal_year"]),
                    "period_no": int(month["period_no"]),
                    "headcount": headcount,
                    "driver_type": "headcount",
                }
            )
    return pl.DataFrame(records)


def build_site_drivers(
    cfg: Config, pack: dict, seeds: SeedBank, calendar: pl.DataFrame, entities: list[str]
) -> pl.DataFrame:
    """Energy usage by site — seasonal and inverse to sales, with an invoice lag."""
    rng = seeds.rng("facts", "drivers", "energy")
    curve = [float(x) for x in pack["seasonality"]["energy_curve"]]
    lag = int(pack["seasonality"]["energy_invoice_lag_months"])
    months = calendar.filter(~pl.col("is_period13")).sort("month_index").to_dicts()

    records = []
    for entity in entities:
        site = SITE_BY_ENTITY.get(entity, f"{entity}-SITE")
        base = float(rng.uniform(18000, 52000))
        for month in months:
            period_no = int(month["period_no"])
            usage = base * curve[period_no - 1] * float(rng.lognormal(0.0, 0.06))
            records.append(
                {
                    "entity": entity,
                    "site": site,
                    "period_key": month["period_key"],
                    "month_index": int(month["month_index"]),
                    "fiscal_year": int(month["fiscal_year"]),
                    "period_no": period_no,
                    "energy_kwh": round(usage, 1),
                    "invoice_lag_months": lag,
                    "driver_type": "energy",
                }
            )
    return pl.DataFrame(records)


def build_machine_hours(
    cfg: Config, pack: dict, seeds: SeedBank, units: pl.DataFrame, calendar: pl.DataFrame
) -> pl.DataFrame:
    """Machine hours at the plant, derived from the units it actually produces.

    Capacity utilisation from these hours is what makes absorption variance swing sign.
    """
    rng = seeds.rng("facts", "drivers", "machine_hours")
    capacity = float(pack["economics"]["cogs_variance"]["plant_capacity_units_per_month"])

    produced = (
        units.group_by(["period_key", "month_index", "fiscal_year", "period_no"])
        .agg(pl.col("units").sum().alias("group_units"))
        .sort("month_index")
    )

    records = []
    for row in produced.to_dicts():
        hours = row["group_units"] * float(rng.uniform(0.0028, 0.0042))
        records.append(
            {
                "entity": "DE01",
                "plant": "DE-PLANT",
                "period_key": row["period_key"],
                "month_index": row["month_index"],
                "fiscal_year": row["fiscal_year"],
                "period_no": row["period_no"],
                "machine_hours": round(hours, 1),
                "produced_units": int(row["group_units"]),
                "capacity_units": capacity,
                "utilisation": round(row["group_units"] / capacity, 4),
                "driver_type": "machine_hours",
            }
        )
    return pl.DataFrame(records)
