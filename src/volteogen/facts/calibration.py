"""Profitability calibration (spec 6.5).

Bottom-up drivers alone do not produce sensible profitability: discounts, erosion and
noise compound, and the realised margin lands wherever it lands. So the generator
CALIBRATES to an explicit, configured financial narrative.

What may move and what may not is the whole point:

* **allowed** — discretionary baselines only: the standard-cost ratio, the opex level,
  headcount growth. These are scaled by a single factor per entity x fiscal year.
* **forbidden** — the seasonality curve, the lognormal noise, the launch ramps, the
  relative shape across months, products and markets. A uniform multiplicative factor
  preserves all of that texture exactly; the P05/P06 peak and the P07/P08 trough
  survive untouched.

Sign and trend are configured, not emergent: ``arc`` sets the multi-year tendency and
the calibrator solves each fiscal year's target inside the configured band.
"""

from __future__ import annotations

import warnings

import polars as pl

from ..config import Config

# Classes the calibrator is allowed to scale, and the lever each belongs to.
COGS_CLASSES = ("cogs_standard", "cogs_variance")
OPEX_CLASSES = ("payroll", "facilities", "marketing", "depreciation", "other_opex", "one_off")
REVENUE_CLASSES = ("revenue_trade", "revenue_reduction")


def _band(spec, fallback: tuple[float, float]) -> tuple[float, float]:
    if not spec:
        return fallback
    return float(spec[0]), float(spec[1])


def _arc_position(arc: str, index: int, n_years: int) -> float:
    """Where in the configured band this fiscal year should land, as 0..1.

    ``steady_grower`` climbs, ``dip_and_recover`` dips in the middle, ``turnaround``
    starts at the bottom and crosses upward, ``decliner`` falls.
    """
    if n_years <= 1:
        return 0.5
    t = index / (n_years - 1)
    if arc == "steady_grower":
        return 0.35 + 0.5 * t
    if arc == "decliner":
        return 0.75 - 0.55 * t
    if arc == "dip_and_recover":
        return 0.75 - 1.4 * t * (1 - t) * 1.6
    if arc == "turnaround":
        return 0.05 + 0.9 * t
    return 0.5  # steady


def calibrate(
    cfg: Config, pack: dict, gl: pl.DataFrame, verbose: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Scale discretionary baselines until each entity/FY lands inside its band.

    Returns ``(calibrated_gl, calibration_report)``. Fails loudly rather than silently
    missing a band, so an infeasible config surfaces as an error not as quiet drift.
    """
    narrative = cfg.get("narrative")
    defaults = pack["narrative_defaults"]
    tolerance = float(narrative.get("calibration", {}).get("tolerance_pp", 0.005))
    max_iterations = int(narrative.get("calibration", {}).get("max_iterations", 25))

    # Margin NORMS are an industry property, so the pack supplies them; the config's
    # `narrative` supplies the company story (arcs, per-entity bands, events). Reading
    # config first meant a swapped pack silently kept the previous industry's margins.
    # Config can still override per pack via `narrative.by_pack.<pack_name>`.
    pack_name = str(pack.get("name", ""))
    per_pack = (narrative.get("by_pack") or {}).get(pack_name) or {}
    group_bands = {**(defaults.get("group") or {}), **(per_pack.get("group") or {})}
    if not group_bands:
        group_bands = narrative.get("group") or {}
    gm_band = _band(group_bands.get("gm_pct"), (0.30, 0.35))

    entities = sorted(gl["entity"].unique().to_list())
    fiscal_years = sorted(gl["fiscal_year"].unique().to_list())

    factors: dict[tuple[str, int], dict[str, float]] = {}
    report_rows: list[dict] = []

    for entity in entities:
        spec = narrative.get(entity) or {}
        arc = str(spec.get("arc", "steady"))
        role_bands = (defaults.get("entity_role_bands") or {})
        role = str(spec.get("role") or "")
        ni_band = _band(
            (per_pack.get(entity) or {}).get("ni_pct")
            or (role_bands.get(role) or {}).get("ni_pct")
            or spec.get("ni_pct"),
            _band(group_bands.get("ni_pct"), (0.04, 0.07)),
        )
        events = {int(e["fy"]): float(e.get("ebit_hit_pct", 0.0)) for e in (spec.get("events") or [])}

        for index, fy in enumerate(fiscal_years):
            slice_ = gl.filter((pl.col("entity") == entity) & (pl.col("fiscal_year") == fy))
            if slice_.height == 0:
                continue

            net_revenue = -float(
                slice_.filter(pl.col("account_class").is_in(REVENUE_CLASSES))["amount_lc"].sum()
            )
            if net_revenue <= 0:
                continue
            cogs = float(slice_.filter(pl.col("account_class").is_in(COGS_CLASSES))["amount_lc"].sum())
            opex = float(slice_.filter(pl.col("account_class").is_in(OPEX_CLASSES))["amount_lc"].sum())
            # Only the non-IC portion can absorb the adjustment.
            cogs_fixed = float(
                slice_.filter(
                    pl.col("account_class").is_in(COGS_CLASSES)
                    & (
                        pl.col("source").str.starts_with("ic:")
                        | pl.col("source").str.starts_with("one_off:")
                        | pl.col("source").str.starts_with("close:")
                    )
                )["amount_lc"].sum()
            )
            opex_fixed = float(
                slice_.filter(
                    pl.col("account_class").is_in(OPEX_CLASSES)
                    & (
                        pl.col("source").str.starts_with("ic:")
                        | pl.col("source").str.starts_with("one_off:")
                        | pl.col("source").str.starts_with("close:")
                    )
                )["amount_lc"].sum()
            )
            cogs_flex = cogs - cogs_fixed
            opex_flex = opex - opex_fixed

            position = _arc_position(arc, index, len(fiscal_years))
            gm_target = gm_band[0] + position * (gm_band[1] - gm_band[0])
            ni_target = ni_band[0] + position * (ni_band[1] - ni_band[0])
            ni_target += events.get(fy, 0.0)   # DE01's FY25 restructuring hit

            # Direct solve: both levers are linear in their scale factor.
            cogs_target = net_revenue * (1.0 - gm_target)
            opex_target = net_revenue * (1.0 - ni_target) - cogs_target

            cogs_factor = (cogs_target - cogs_fixed) / cogs_flex if cogs_flex > 0 else 1.0
            opex_factor = (opex_target - opex_fixed) / opex_flex if opex_flex > 0 else 1.0

            # Keep the levers plausible: calibration adjusts a business, it does not
            # invent one. The bound is wide because a deliberately thin distributor cost
            # base (ES01 runs one department per function) legitimately needs a large
            # multiplier; it still rejects genuinely infeasible configurations.
            for name, value in (("std_cost_ratio", cogs_factor), ("opex_level", opex_factor)):
                if value > 5.0 or value < 0.25:
                    warnings.warn(
                        f"calibration lever {name} for {entity} FY{fy} needs a "
                        f"{value:.1f}x adjustment. The industry pack's cost_structure and "
                        "the configured org size disagree — the numbers will land in band "
                        "but the underlying cost base is not industry-shaped.",
                        RuntimeWarning, stacklevel=2,
                    )
                if not (0.02 <= value <= 40.0):
                    raise ValueError(
                        f"calibration infeasible for {entity} FY{fy}: lever {name} would need "
                        f"a {value:.2f}x adjustment to reach gm={gm_target:.3f}/ni={ni_target:.3f}. "
                        "Loosen the narrative bands or revisit cardinality/economics."
                    )

            factors[(entity, fy)] = {"cogs": cogs_factor, "opex": opex_factor}
            report_rows.append(
                {
                    "entity": entity,
                    "fiscal_year": fy,
                    "arc": arc,
                    "gm_target": round(gm_target, 4),
                    "ni_target": round(ni_target, 4),
                    "gm_before": round(1.0 - cogs / net_revenue, 4),
                    "ni_before": round((net_revenue - cogs - opex) / net_revenue, 4),
                    "cogs_factor": round(cogs_factor, 4),
                    "opex_factor": round(opex_factor, 4),
                    "net_revenue": round(net_revenue, 2),
                }
            )

    if not factors:
        raise ValueError("calibration found no entity/fiscal-year slices to solve")

    # Apply the factors: one uniform multiplier per entity x FY x lever, so every
    # seasonal, product and noise pattern inside the slice is preserved exactly.
    factor_frame = pl.DataFrame(
        [
            {"entity": e, "fiscal_year": fy, "cogs_factor": f["cogs"], "opex_factor": f["opex"]}
            for (e, fy), f in factors.items()
        ]
    )
    # Intercompany legs are measured but never scaled: both legs come from one record,
    # and the seller leg (revenue) and buyer leg (COGS) would move by different factors,
    # breaking every matched pair and with it V4.
    # One-offs and close adjustments are measured but never scaled: scaling a
    # restructuring charge would make it an ordinary run-rate cost.
    scalable = (
        ~pl.col("source").str.starts_with("ic:")
        & ~pl.col("source").str.starts_with("one_off:")
        & ~pl.col("source").str.starts_with("close:")
    )

    calibrated = (
        gl.join(factor_frame, on=["entity", "fiscal_year"], how="left")
        .with_columns(
            pl.when(pl.col("account_class").is_in(COGS_CLASSES) & scalable)
            .then(pl.col("amount_lc") * pl.col("cogs_factor").fill_null(1.0))
            .when(pl.col("account_class").is_in(OPEX_CLASSES) & scalable)
            .then(pl.col("amount_lc") * pl.col("opex_factor").fill_null(1.0))
            .otherwise(pl.col("amount_lc"))
            .round(2)
            .alias("amount_lc")
        )
        .drop("cogs_factor", "opex_factor")
    )

    report = pl.DataFrame(report_rows)
    realised = realised_margins(calibrated)
    report = report.join(realised, on=["entity", "fiscal_year"], how="left").with_columns(
        (pl.col("gm_actual") - pl.col("gm_target")).abs().alias("gm_gap"),
        (pl.col("ni_actual") - pl.col("ni_target")).abs().alias("ni_gap"),
    )

    missed = report.filter(
        (pl.col("gm_gap") > tolerance) | (pl.col("ni_gap") > tolerance)
    )
    if missed.height:
        raise ValueError(
            "calibration failed to land inside the configured bands for "
            f"{missed.height} entity/FY slice(s):\n{missed.select('entity','fiscal_year','gm_gap','ni_gap')}"
        )

    del max_iterations  # direct solve converges in one pass; kept for config compatibility
    return calibrated, report


def realised_margins(gl: pl.DataFrame) -> pl.DataFrame:
    """Gross and net margin actually present in the data, per entity x fiscal year."""
    return (
        gl.group_by(["entity", "fiscal_year"])
        .agg(
            (-pl.col("amount_lc").filter(pl.col("account_class").is_in(REVENUE_CLASSES)).sum()).alias("net_revenue"),
            pl.col("amount_lc").filter(pl.col("account_class").is_in(COGS_CLASSES)).sum().alias("cogs"),
            pl.col("amount_lc").filter(pl.col("account_class").is_in(OPEX_CLASSES)).sum().alias("opex"),
        )
        .with_columns(
            ((pl.col("net_revenue") - pl.col("cogs")) / pl.col("net_revenue")).round(4).alias("gm_actual"),
            (
                (pl.col("net_revenue") - pl.col("cogs") - pl.col("opex")) / pl.col("net_revenue")
            ).round(4).alias("ni_actual"),
        )
        .sort(["entity", "fiscal_year"])
    )


def monthly_profitability(gl: pl.DataFrame) -> pl.DataFrame:
    """Monthly result per entity — used to check that loss-making months really occur."""
    return (
        gl.filter(~pl.col("is_period13"))
        .group_by(["entity", "fiscal_year", "period_no", "period_key"])
        .agg((-pl.col("amount_lc").sum()).alias("result"))
        .sort(["entity", "fiscal_year", "period_no"])
    )
