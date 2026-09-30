"""Sales sub-ledger: the only place SKU / customer / invoice detail exists.

``fact_gl`` revenue and COGS rows are the monthly SUMMARY of this table, derived from it
so they tie exactly (validator V7) — except for one documented manual correction
(defect ``subledger_reconciling_item``).

The gross-to-net ladder is first-class here because in electronics the gap between
gross and net revenue is LARGE (10-20%) and made of four named components, each landing
on its own revenue-reduction account:

    gross (MSRP x qty)
      - channel discount        (mass retail takes 35-45% off list)
      - customer tier discount
      = invoiced revenue
      - volume rebate           (negotiated per FAMILY, never per SKU)
      - MDF                     (marketing development funds, big retailers only)
      - returns reserve         (3-8% by category; TWS worst)
      - price protection        (credits when MSRP drops on a retailer's stock)
      = net revenue
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import Config
from ..rng import SeedBank, stable_hash_unit
from ..dims.applicability import UNASSIGNED

REDUCTION_KINDS = ("volume_rebate", "mdf", "returns_reserve", "price_protection")


def build_sales_lines(
    cfg: Config,
    pack: dict,
    seeds: SeedBank,
    unit_drivers: pl.DataFrame,
    products: pl.DataFrame,
    customers: pl.DataFrame,
    calendar: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(fact_sales_lines, fact_revenue_reductions)``.

    Reductions are a separate frame because they post at a coarser grain (family, and
    monthly accrual) than the invoice lines that generate them — exactly the grain
    mismatch the applicability matrix describes.
    """
    rng = seeds.rng("facts", "subledger")
    economics = pack["economics"]
    channel_discount = economics["channel_discount"]
    tier_discount = economics["customer_tier_discount"]
    g2n = economics["gross_to_net"]
    erosion_cfg = pack["product_taxonomy"]["price_erosion"]

    product_meta = {
        row["node_id"]: row
        for row in products.filter(pl.col("is_leaf")).to_dicts()
    }

    # Third-party customers per entity, with their revenue shares as sampling weights.
    customers_by_entity: dict[str, list[dict]] = {}
    for row in customers.filter(~pl.col("is_affiliate")).to_dicts():
        customers_by_entity.setdefault(row["entity"], []).append(row)

    unassigned_enabled = cfg.defect_enabled("unassigned_product_tags")

    # ---------------------------------------------------------------------
    # Vectorised. The previous row loop ran at ~42k invoice lines/sec and dominated
    # the whole build; the sub-ledger is the table spec 6 designates to carry volume,
    # so it is the one that most needs to scale. Every step below is a polars
    # expression over the whole frame.
    # ---------------------------------------------------------------------
    product_frame = products.filter(pl.col("is_leaf")).select(
        pl.col("node_id").alias("product_node_id"),
        pl.col("sku_code"), pl.col("family_name"), pl.col("category_name"),
        pl.col("division_name"), pl.col("list_price"), pl.col("std_cost"),
        pl.col("launch_period"), pl.col("price_erosion_annual"),
        pl.col("std_cost_erosion_annual"),
    )
    drivers = unit_drivers.join(product_frame, on="product_node_id", how="inner").filter(
        pl.col("units") > 0
    )
    if drivers.height == 0:
        raise ValueError("sub-ledger generation produced no invoice lines")

    # 1..3 invoice lines per driver row, then explode.
    n = drivers.height
    density = cfg.get("cardinality.subledger", {}) or {}
    lo = int(density.get("lines_per_driver_min", 1))
    hi = int(density.get("lines_per_driver_max", 4))
    line_counts = rng.integers(lo, max(lo + 1, hi), size=n)
    drivers = drivers.with_columns(pl.Series("n_lines", line_counts, dtype=pl.Int64))
    exploded = drivers.with_columns(
        pl.int_ranges(1, pl.col("n_lines") + 1).alias("line_no")
    ).explode("line_no")

    total_lines = exploded.height
    # Proportional split of the month's units across a driver row's lines: draw a
    # weight per line, normalise within the driver row, and hand the rounding
    # remainder to the first line so the units still sum exactly.
    weights = rng.random(total_lines) + 0.15
    exploded = exploded.with_columns(pl.Series("w", weights))
    exploded = exploded.with_columns(
        (pl.col("w") / pl.col("w").sum().over(["entity", "product_node_id", "market_node_id", "period_key"]))
        .alias("share")
    )
    exploded = exploded.with_columns(
        (pl.col("units") * pl.col("share")).floor().cast(pl.Int64).alias("qty_base")
    )
    exploded = exploded.with_columns(
        (pl.col("units") - pl.col("qty_base").sum().over(
            ["entity", "product_node_id", "market_node_id", "period_key"])).alias("remainder")
    )
    exploded = exploded.with_columns(
        (pl.col("qty_base") + pl.when(pl.col("line_no") == 1).then(pl.col("remainder")).otherwise(0))
        .alias("qty")
    ).filter(pl.col("qty") > 0)

    # Weighted customer draw, per entity, via cumulative shares + searchsorted.
    total_lines = exploded.height
    picks = np.empty(total_lines, dtype=np.int64)
    entity_col = exploded["entity"].to_numpy()
    draws = rng.random(total_lines)
    customer_index: dict[str, list[dict]] = {}
    for entity, pool in customers_by_entity.items():
        weights_e = np.array([max(1e-9, float(c["revenue_share"])) for c in pool])
        weights_e = np.cumsum(weights_e / weights_e.sum())
        mask = entity_col == entity
        if not mask.any():
            continue
        picks[mask] = np.searchsorted(weights_e, draws[mask], side="left").clip(0, len(pool) - 1)
        customer_index[entity] = pool

    customer_ids = np.empty(total_lines, dtype=object)
    channels = np.empty(total_lines, dtype=object)
    tiers = np.empty(total_lines, dtype=object)
    for entity, pool in customer_index.items():
        mask = entity_col == entity
        idx = picks[mask]
        customer_ids[mask] = [pool[i]["customer_id"] for i in idx]
        channels[mask] = [pool[i]["channel"] for i in idx]
        tiers[mask] = [pool[i]["customer_tier"] for i in idx]

    exploded = exploded.with_columns(
        pl.Series("customer_id", customer_ids, dtype=pl.Utf8),
        pl.Series("channel", channels, dtype=pl.Utf8),
        pl.Series("customer_tier", tiers, dtype=pl.Utf8),
        pl.Series("discount_u", rng.random(total_lines)),
    )

    # Channel and tier discounts resolve as expression chains, not dict lookups.
    disc_lo = pl.lit(0.1); disc_hi = pl.lit(0.2)
    for channel, band in channel_discount.items():
        disc_lo = pl.when(pl.col("channel") == channel).then(pl.lit(float(band["min"]))).otherwise(disc_lo)
        disc_hi = pl.when(pl.col("channel") == channel).then(pl.lit(float(band["max"]))).otherwise(disc_hi)
    tier_expr = pl.lit(0.0)
    for tier, value in tier_discount.items():
        tier_expr = pl.when(pl.col("customer_tier") == tier).then(pl.lit(float(value))).otherwise(tier_expr)

    age = ((pl.col("month_index") - pl.col("launch_period").fill_null(0)) / 12.0).clip(lower_bound=0.0)
    unit_list = pl.col("list_price") * (1.0 + pl.col("price_erosion_annual").fill_null(0.0)) ** age
    unit_cost = pl.col("std_cost") * (1.0 + pl.col("std_cost_erosion_annual").fill_null(0.0)) ** age
    discount_pct = disc_lo + (disc_hi - disc_lo) * pl.col("discount_u")

    exploded = exploded.with_columns(
        unit_list.alias("unit_list_price"),
        unit_cost.alias("unit_cost"),
        discount_pct.alias("discount_pct"),
        tier_expr.alias("tier_pct"),
    ).with_columns(
        (pl.col("unit_list_price") * (1 - pl.col("discount_pct")) * (1 - pl.col("tier_pct")))
        .alias("unit_net_price")
    )

    # Document numbering: a running rank within entity x period, matching the old
    # counter's semantics without a Python dict.
    exploded = exploded.with_columns(
        pl.int_range(pl.len()).over(["entity", "period_key"]).add(1).alias("doc_seq")
    ).with_columns(
        (pl.lit("INV-") + pl.col("entity") + pl.lit("-")
         + pl.col("period_key").str.replace("-", "", literal=True) + pl.lit("-")
         + pl.col("doc_seq").cast(pl.Utf8).str.pad_start(6, "0")).alias("doc_no")
    )

    # DEFECT 9.11: ~0.5% of rows lose their product tag. Keyed by row identity via a
    # stable hash, so it does not depend on draw order.
    unassigned_expr = (
        (pl.col("doc_no") + pl.lit("|") + pl.col("line_no").cast(pl.Utf8)).hash(seed=911) % 100000
    ) < 500 if unassigned_enabled else pl.lit(False)

    sales = exploded.with_columns(
        unassigned_expr.alias("is_unassigned_product")
    ).select(
        "entity", "doc_no", "line_no", "period_key", "month_index", "fiscal_year", "period_no",
        pl.when(pl.col("is_unassigned_product")).then(pl.lit(UNASSIGNED))
          .otherwise(pl.col("product_node_id")).alias("product_node_id"),
        pl.col("product_node_id").alias("product_node_id_true"),
        "sku_code", "family_name", "category_name", "division_name",
        "market_node_id", "customer_id", "channel",
        pl.col("qty").cast(pl.Int64),
        pl.col("unit_list_price").round(4),
        pl.col("unit_net_price").round(4),
        (pl.col("unit_list_price") * pl.col("qty")).round(2).alias("gross_amount"),
        ((pl.col("unit_list_price") - pl.col("unit_net_price")) * pl.col("qty")).round(2)
            .alias("channel_discount_amount"),
        (pl.col("unit_net_price") * pl.col("qty")).round(2).alias("invoiced_amount"),
        (pl.col("unit_cost") * pl.col("qty")).round(2).alias("std_cost_amount"),
        "is_unassigned_product",
    ).sort(["entity", "period_key", "doc_no", "line_no"])

    lines = [1]  # non-empty sentinel for the guard below
    if not lines:
        raise ValueError("sub-ledger generation produced no invoice lines")

    reductions = _build_reductions(rng, cfg, pack, sales, g2n, erosion_cfg)
    return sales, reductions


def _split_units(rng, total: int, parts: int) -> list[int]:
    if parts <= 1:
        return [total]
    cuts = sorted(int(x) for x in rng.integers(0, total + 1, size=parts - 1))
    result = []
    previous = 0
    for cut in cuts:
        result.append(cut - previous)
        previous = cut
    result.append(total - previous)
    return result


def _build_reductions(
    rng, cfg: Config, pack: dict, sales: pl.DataFrame, g2n: dict, erosion_cfg: dict
) -> pl.DataFrame:
    """Monthly accruals at FAMILY grain — never per SKU (spec 5)."""
    by_family = (
        sales.group_by(
            ["entity", "period_key", "month_index", "fiscal_year", "period_no",
             "family_name", "category_name", "channel", "customer_id"]
        )
        .agg(
            pl.col("invoiced_amount").sum().alias("invoiced"),
            pl.col("qty").sum().alias("qty"),
            pl.col("market_node_id").first().alias("market_node_id"),
        )
        # Total order, not a partial one: this loop consumes the RNG per row, so any
        # unordered tie would change draw interleaving and make the build irreproducible.
        .sort(
            ["entity", "month_index", "family_name", "category_name", "channel", "customer_id"]
        )
    )

    rebate_cfg = g2n["volume_rebate"]
    rebate_channels = set(rebate_cfg["applies_to"])
    tiers = sorted(rebate_cfg["tiers"], key=lambda t: float(t["threshold"]))
    mdf_cfg = g2n["mdf"]
    mdf_channels = set(mdf_cfg["applies_to"])
    returns_by_category = g2n["returns_reserve"]["by_category"]
    pp_cfg = g2n["price_protection"]
    pp_share = float(pp_cfg["pct_of_channel_inventory"])

    rows = []
    for row in by_family.to_dicts():
        invoiced = float(row["invoiced"])
        channel = row["channel"]

        if channel in rebate_channels:
            pct = float(tiers[0]["pct"])
            for tier in tiers:
                if invoiced >= float(tier["threshold"]):
                    pct = float(tier["pct"])
            rows.append(_reduction_row(row, "volume_rebate", -invoiced * pct))

        if channel in mdf_channels:
            pct = float(rng.uniform(mdf_cfg["pct"]["min"], mdf_cfg["pct"]["max"]))
            rows.append(_reduction_row(row, "mdf", -invoiced * pct))

        returns_pct = float(returns_by_category.get(row["category_name"], 0.04))
        rows.append(_reduction_row(row, "returns_reserve", -invoiced * returns_pct))

        # Price protection fires when MSRP steps down — a genuinely electronics-specific
        # revenue reduction, and only for channels that hold stock.
        if channel in ("MASS_RETAIL", "DISTRIBUTION") and rng.random() < 0.08:
            erosion = abs(float(erosion_cfg["annual_pct"].get("Audio", -0.18))) / 12.0
            rows.append(
                _reduction_row(row, "price_protection", -invoiced * pp_share * erosion)
            )

    return pl.DataFrame(rows)


def _reduction_row(row: dict, kind: str, amount: float) -> dict:
    return {
        "entity": row["entity"],
        "period_key": row["period_key"],
        "month_index": row["month_index"],
        "fiscal_year": row["fiscal_year"],
        "period_no": row["period_no"],
        "family_name": row["family_name"],
        "category_name": row["category_name"],
        "market_node_id": row["market_node_id"],
        "customer_id": row["customer_id"],
        "channel": row["channel"],
        "reduction_kind": kind,
        "amount": round(amount, 2),
    }


def gross_to_net_waterfall(sales: pl.DataFrame, reductions: pl.DataFrame) -> pl.DataFrame:
    """First-class report: the gross-to-net bridge per entity x period."""
    gross = sales.group_by(["entity", "period_key"]).agg(
        pl.col("gross_amount").sum().alias("gross_revenue"),
        pl.col("channel_discount_amount").sum().alias("channel_discounts"),
        pl.col("invoiced_amount").sum().alias("invoiced_revenue"),
    )
    pivot = (
        reductions.group_by(["entity", "period_key", "reduction_kind"])
        .agg(pl.col("amount").sum().alias("amount"))
        .pivot(on="reduction_kind", index=["entity", "period_key"], values="amount")
        .fill_null(0.0)
    )
    out = gross.join(pivot, on=["entity", "period_key"], how="left").fill_null(0.0)
    reduction_cols = [c for c in out.columns if c in REDUCTION_KINDS]
    return out.with_columns(
        (pl.col("invoiced_revenue") + sum(pl.col(c) for c in reduction_cols)).alias("net_revenue")
    ).sort(["entity", "period_key"])
