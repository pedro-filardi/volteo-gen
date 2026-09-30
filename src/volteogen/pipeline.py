"""Build orchestration: config + seeds -> a complete, reconciling dataset.

Deterministic end to end: the same ``seed`` and config reproduce byte-identical output,
and each module draws from its own child stream so regenerating one does not shift the
others.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .config import Config, load_config, load_pack, CONFIG_DIR, SEEDS_DIR
from .dims.applicability import Applicability, classify_accounts, load_applicability
from .dims.calendar import build_calendar, build_scenarios
from .dims.customers import build_customers
from .dims.entities import ENTITY_PROFILES, build_entities, build_ledgers
from .dims.fx import build_fx_rates
from .dims.hierarchy import assert_irregularity, to_bridge, to_levels
from .dims.markets import build_entity_market_coverage, build_markets
from .dims.orgdims import (
    build_cost_center_versions,
    build_cost_centers,
    build_profit_centers,
)
from .dims.products import build_products
from .facts.account_router import AccountRouter
from .facts.drivers import (
    build_headcount_drivers,
    build_machine_hours,
    build_site_drivers,
    build_unit_drivers,
)
from .facts.calibration import calibrate, monthly_profitability, realised_margins
from .facts.closing import (
    build_balance_sheet,
    build_one_offs,
    build_period13_adjustments,
    build_tax_ledger,
)
from .facts.gl import assemble_gl
from .facts.ic import (
    build_allocations,
    build_eliminations,
    build_ic_transactions,
    build_nci,
    build_topside,
    ic_gl_rows,
)
from .maps.reports import build_report_tables
from .maps import (
    build_group_coa_hierarchy,
    build_management_hierarchy,
    build_group_coa,
    build_group_to_gaap,
    build_local_to_group,
    build_management_map,
    build_statutory_tags,
)
from .facts.translate import translate_gl
from .defects import (
    DefectLog,
    inject_dormant_accounts,
    inject_mapping_drift_uk,
    inject_sku_relaunch,
    inject_subledger_reconciling_item,
    inject_unmapped_accounts,
)
from .facts.pnl import build_income_statement, build_pnl
from .facts.scenarios import build_budget, build_forecast, build_variance_bridge
from .facts.subledger import build_sales_lines, gross_to_net_waterfall
from .rng import SeedBank
from .seedload import ENTITY_COA, SeedBundle, load_seeds


def _solve_volume(cfg: Config, pack: dict, verbose: bool = False) -> None:
    """Back-solve grain density toward ``volume.target_gl_rows`` (spec 2).

    Measured behaviour, not assumption: fact_gl revenue grain is
    entity x period x FAMILY x market x customer, and customers SATURATE — raising them
    from 12 to 120 moves fact_gl only 243k -> 262k, because the invoice lines that feed
    it are fixed by (sku x market x month) and simply spread thinner.

    So the honest levers for fact_gl are months, market leaves and the pack's family
    count. SKU count drives fact_sales_lines, not fact_gl — which is spec 6's point that
    GL stays trial-balance grain and volume comes from sub-ledgers. This adjusts what it
    can and leaves the residual to be reported rather than silently missed.
    """
    target = int(cfg.get("volume.target_gl_rows", 0) or 0)
    if target <= 0:
        return

    families = sum(
        len(category.get("families") or category.get("leaves") or [])
        for division in pack["product_taxonomy"]["divisions"]
        for category in division["categories"]
    )
    from .dims.markets import ENTITY_COVERAGE  # local import avoids a module-load cycle

    markets = sum(max(1, len(ENTITY_COVERAGE.get(e, []))) for e in cfg.entities)
    # Rows per named customer, fitted against measured builds. Beyond ~25 customers the
    # combination space is saturated and extra customers buy almost nothing.
    per_customer = max(1, int(families * markets * cfg.months * 0.28))
    saturation_cap = 25

    current = int(cfg.get("cardinality.customers.named_per_entity"))
    solved = max(2, min(saturation_cap, round(target / max(1, per_customer))))
    if solved != current:
        cfg.set("cardinality.customers.named_per_entity", int(solved))
        if verbose:
            print(f"  volume: named customers/entity {current} -> {solved}", flush=True)


@dataclass
class Dataset:
    """Every table the generator produces, plus build metadata."""

    tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    config: Config | None = None
    pack: dict | None = None
    seeds: SeedBundle | None = None
    applicability: Applicability | None = None
    router: AccountRouter | None = None
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __getitem__(self, name: str) -> pl.DataFrame:
        return self.tables[name]

    def add(self, name: str, frame: pl.DataFrame) -> pl.DataFrame:
        self.tables[name] = frame
        return frame

    def summary(self) -> pl.DataFrame:
        return pl.DataFrame(
            [{"table": k, "rows": v.height, "columns": v.width} for k, v in sorted(self.tables.items())]
        ).sort("rows", descending=True)


def build_dataset(
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: Path | None = None,
    seeds_dir: Path | None = None,
    verbose: bool = True,
) -> Dataset:
    cfg = load_config(preset=preset, overrides=overrides, config_dir=config_dir)
    pack = load_pack(cfg)
    seeds = SeedBank(cfg.seed)
    dataset = Dataset(config=cfg, pack=pack)
    defect_log = DefectLog()

    def step(name: str):
        start = time.time()

        class _Timer:
            def __enter__(self_inner):
                if verbose:
                    print(f"  {name} ...", flush=True)
                return self_inner

            def __exit__(self_inner, *exc):
                dataset.timings[name] = time.time() - start

        return _Timer()

    _solve_volume(cfg, pack, verbose=verbose)

    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    sdir = Path(seeds_dir) if seeds_dir else SEEDS_DIR

    # -- Seeds and the applicability contract -----------------------------------
    with step("seeds"):
        bundle = load_seeds(sdir)
        dataset.seeds = bundle
        applicability = load_applicability(cdir, pack)
        dataset.applicability = applicability
        accounts = classify_accounts(bundle.accounts, applicability)
        dataset.add("dim_account_node", accounts)
        dataset.add("meta_account_dimensionality", applicability.matrix)
        dataset.add("dim_gaap_node", bundle.gaap_nodes)
        dataset.add("map_gaap_calc_arcs", bundle.gaap_arcs)
        dataset.add("bridge_usgaap", bundle.gaap_bridge)

    # -- Dimensions --------------------------------------------------------------
    with step("dimensions"):
        calendar = dataset.add("dim_calendar", build_calendar(cfg))
        dataset.add("dim_scenario", build_scenarios(cfg, calendar))
        entities = dataset.add("dim_entity", build_entities(cfg))
        dataset.add("dim_ledger", build_ledgers(cfg, entities))
        dataset.add("dim_fx_rate", build_fx_rates(cfg, calendar, seeds))

        products = build_products(cfg, pack, seeds, cfg.months)
        products = inject_sku_relaunch(cfg, seeds, products, defect_log)
        dataset.add("dim_product_node", products)
        markets = dataset.add("dim_market_node", build_markets(cfg, seeds))
        coverage = dataset.add(
            "dim_entity_market_coverage", build_entity_market_coverage(markets, cfg.entities)
        )
        cost_centers = dataset.add("dim_cost_center_node", build_cost_centers(cfg, seeds, cfg.entities))
        dataset.add(
            "dim_cost_center_version", build_cost_center_versions(cost_centers, cfg, calendar)
        )
        profit_centers = dataset.add("dim_profit_center_node", build_profit_centers(cfg, pack))
        entity_country = {k: v["country"] for k, v in ENTITY_PROFILES.items()}
        customers = dataset.add(
            "dim_customer", build_customers(cfg, seeds, cfg.entities, entity_country)
        )

    # -- V11: fail the build if a synthetic hierarchy came out regular -----------
    with step("irregularity checks (V11)"):
        checks = []
        for name, frame in (
            ("product", products),
            ("market", markets),
            ("cost_center", cost_centers),
        ):
            checks.append(assert_irregularity(frame, name))
        dataset.add("meta_hierarchy_irregularity", pl.DataFrame(checks))

    # -- Derived hierarchy artifacts (never the source of truth) -----------------
    with step("derived hierarchies"):
        for name, frame in (
            ("product", products),
            ("market", markets),
            ("cost_center", cost_centers),
            ("profit_center", profit_centers),
        ):
            dataset.add(f"bridge_{name}", to_bridge(frame, name))
            dataset.add(f"dim_{name}_levels", to_levels(frame, name))

        # Spec 8 also requires bridges over every ACCOUNT tree, not just the synthetic
        # dims: the four local charts have wildly different depth (Sage 2, SKR03/PGC 4,
        # Business Central 6), so a depth-agnostic bridge is the only way one query can
        # roll up all of them.
        for coa_id in sorted(accounts["coa_id"].unique().to_list()):
            sub = accounts.filter(pl.col("coa_id") == coa_id).rename({"code": "account_code"})
            key = coa_id.lower()
            dataset.add(f"bridge_{key}", to_bridge(sub, key))
            dataset.add(f"dim_{key}_levels", to_levels(sub, key))

    # -- Drivers ------------------------------------------------------------------
    with step("drivers"):
        units = dataset.add(
            "fact_drivers_units",
            build_unit_drivers(cfg, pack, seeds, products, markets, coverage, calendar, entities),
        )
        headcount = dataset.add(
            "fact_drivers_headcount", build_headcount_drivers(cfg, pack, seeds, cost_centers, calendar)
        )
        energy = dataset.add(
            "fact_drivers_energy", build_site_drivers(cfg, pack, seeds, calendar, cfg.entities)
        )
        machine_hours = dataset.add(
            "fact_drivers_machine_hours", build_machine_hours(cfg, pack, seeds, units, calendar)
        )

    # -- Sub-ledger ---------------------------------------------------------------
    with step("sales sub-ledger"):
        sales, reductions = build_sales_lines(cfg, pack, seeds, units, products, customers, calendar)
        dataset.add("fact_sales_lines", sales)
        dataset.add("fact_revenue_reductions", reductions)
        dataset.add("rpt_gross_to_net", gross_to_net_waterfall(sales, reductions))

    # -- GL ------------------------------------------------------------------------
    with step("GL assembly"):
        router = AccountRouter(accounts, ENTITY_COA)
        dataset.router = router
        gl = assemble_gl(
            cfg, pack, seeds, applicability, router, entities, calendar,
            sales, reductions, headcount, energy, machine_hours, cost_centers, markets,
        )
        dataset.add("fact_gl", gl)

    # -- Group mechanics: intercompany and allocations (spec 7) -------------------
    with step("intercompany + allocations"):
        transactions = dataset.add(
            "fact_ic_transaction",
            build_ic_transactions(
                cfg, pack, seeds, gl, calendar, dataset["dim_fx_rate"], entities
            ),
        )
        ic_gl = ic_gl_rows(
            cfg, applicability, router, transactions, calendar, entities, dataset["dim_fx_rate"]
        )
        alloc_gl, alloc_rules = build_allocations(
            cfg, applicability, router, seeds, gl, cost_centers, profit_centers, calendar, entities
        )
        dataset.add("allocation_rules", alloc_rules)
        parts = [gl, ic_gl] + ([alloc_gl] if alloc_gl.height else [])
        gl = pl.concat(parts, how="diagonal")

    # -- Close mechanics: P13 and one-offs, before calibration so they land INSIDE
    #    the configured margin bands rather than on top of them (spec 6.5).
    with step("close: P13 + one-offs"):
        closing_parts = [gl]
        for frame in (
            build_one_offs(cfg, applicability, router, seeds, gl, calendar, entities, cost_centers),
            build_period13_adjustments(
                cfg, applicability, router, seeds, gl, calendar, entities, cost_centers
            ),
        ):
            if frame.height:
                closing_parts.append(frame)
        gl = pl.concat(closing_parts, how="diagonal") if len(closing_parts) > 1 else gl

    # -- Calibration to the configured financial narrative (spec 6.5) -------------
    with step("margin calibration"):
        calibrated, report = calibrate(cfg, pack, gl, verbose=verbose)
        dataset.add("fact_gl", calibrated)
        dataset.add("meta_calibration", report)
        dataset.add("rpt_margins_by_fy", realised_margins(calibrated))
        dataset.add("rpt_monthly_result", monthly_profitability(calibrated))
        gl = calibrated

    # -- Balance sheet and the TAX parallel ledger -------------------------------
    with step("balance sheet + TAX ledger"):
        extra = [gl]
        bs = build_balance_sheet(cfg, applicability, router, seeds, gl, calendar, entities)
        if bs.height:
            extra.append(bs)
        tax = build_tax_ledger(cfg, applicability, router, seeds, gl, entities, calendar)
        if tax.height:
            extra.append(tax)
        if len(extra) > 1:
            gl = pl.concat(extra, how="diagonal")
        dataset.add("fact_gl", gl)

    # -- Currency translation (spec 4.7 / 6.6) -----------------------------------
    with step("fx translation"):
        gl = translate_gl(gl, dataset["dim_fx_rate"], str(cfg.get("fx.group_currency")))
        dataset.add("fact_gl", gl)

    # -- Mappings, eliminations, NCI and topside ---------------------------------
    with step("mappings + consolidation"):
        group_coa = dataset.add("dim_group_account", build_group_coa())
        dataset.add(
            "map_local_to_group",
            inject_unmapped_accounts(
                cfg, seeds, build_local_to_group(accounts, group_coa), defect_log
            ),
        )
        dataset.add("map_group_to_gaap", build_group_to_gaap(group_coa, bundle))
        dataset.add(
            "map_management",
            inject_mapping_drift_uk(cfg, build_management_map(group_coa), defect_log),
        )
        dataset.add("map_statutory_tags", build_statutory_tags(accounts))

        group_tree = dataset.add("dim_group_coa_node", build_group_coa_hierarchy(group_coa))
        dataset.add("bridge_group_coa", to_bridge(group_tree, "group_coa"))
        dataset.add("dim_group_coa_levels", to_levels(group_tree, "group_coa"))
        mgmt_tree = dataset.add(
            "dim_management_node", build_management_hierarchy(dataset["map_management"])
        )
        dataset.add("bridge_management", to_bridge(mgmt_tree, "management"))
        dataset.add("dim_management_levels", to_levels(mgmt_tree, "management"))

        # Report layouts as data, so a BI tool needs no statement logic of its own.
        report_lines, report_bridge = build_report_tables(cdir / "reports")
        if report_lines.height:
            dataset.add("dim_report_line", report_lines)
            dataset.add("bridge_report_line", report_bridge)

        eliminations = dataset.add(
            "fact_elimination",
            build_eliminations(cfg, transactions, ic_gl, dataset["map_local_to_group"], pack, calendar),
        )
        dataset.add("fact_topside", build_topside(cfg, calendar))
        dataset.add("fact_nci", build_nci(gl, entities, eliminations))

    # -- Ledger-level defect injection (spec 9) ----------------------------------
    with step("defect injection"):
        gl, accounts = inject_dormant_accounts(cfg, seeds, gl, accounts, defect_log)
        gl = inject_subledger_reconciling_item(cfg, seeds, gl, defect_log)
        dataset.add("fact_gl", gl)
        dataset.add("dim_account_node", accounts)
        dataset.add("meta_defects_injected", defect_log.frame())

    # -- Budget and forecast scenarios (spec 4.6) --------------------------------
    with step("budget + forecast"):
        budget = build_budget(
            cfg, seeds, gl, dataset["map_local_to_group"], dataset["dim_scenario"], calendar
        )
        dataset.add("fact_budget", budget)
        dataset.add(
            "fact_forecast",
            build_forecast(cfg, seeds, gl, budget, dataset["map_local_to_group"], dataset["dim_scenario"]),
        )
        dataset.add(
            "fact_bridge_variance",
            build_variance_bridge(gl, budget, dataset["map_local_to_group"], cfg),
        )
        # The first fiscal year has no prior-year basis, so no budget (and therefore no
        # forecast) exists for it. Keep only scenarios that actually carry data.
        planned = set()
        for name in ("fact_budget", "fact_forecast"):
            if dataset[name].height:
                planned |= set(dataset[name]["scenario_key"].unique().to_list())
        dataset.add(
            "dim_scenario",
            dataset["dim_scenario"].filter(
                (pl.col("scenario_type") == "ACT") | pl.col("scenario_key").is_in(list(planned))
            ),
        )


        # Eliminations and topside journals are appended to fact_gl as their own
        # entities (ELIM / GRP) rather than living only in side tables. Consolidation
        # then falls out of an ordinary GROUP BY, so a BI tool can produce a
        # consolidated statement with no bespoke logic: the group total is simply every
        # entity summed. Both are already in group currency, so the FX columns are 1:1.
        adjustment_frames = []
        for frame, flag in ((eliminations, False), (dataset["fact_topside"], True)):
            if frame is None or frame.height == 0:
                continue
            adjustment_frames.append(
                frame.with_columns(
                    pl.col("amount_lc").alias("amount_gc_actual_rates"),
                    pl.col("amount_lc").alias("amount_gc_budget_rates"),
                    pl.lit(1.0).alias("fx_rate_avg"),
                    pl.lit(1.0).alias("fx_rate_closing"),
                    pl.lit(1.0).alias("fx_rate_budget"),
                    pl.lit("~NA~").alias("product_node_id"),
                    pl.lit("~NA~").alias("market_node_id"),
                    pl.lit("~NA~").alias("customer_id"),
                    pl.lit("~NA~").alias("cost_center_node_id"),
                    pl.lit("~NA~").alias("ic_partner"),
                    pl.lit("~NA~").alias("profit_center_node_id"),
                    pl.lit("consolidation" if not flag else "topside").alias("source"),
                    pl.lit(flag).alias("is_topside"),
                    pl.lit(False).alias("is_one_off"),
                    pl.lit(str(cfg.get("fx.group_currency"))).alias("group_currency"),
                )
            )
        if adjustment_frames:
            gl = pl.concat([gl] + adjustment_frames, how="diagonal")
            dataset.add("fact_gl", gl)

        if router.substitutions:
            for (entity, account_class), fallback in sorted(router.substitutions.items()):
                dataset.notes.append(
                    f"account fallback: {entity} has no {account_class} account; "
                    f"postings routed to {fallback}"
                )

    # -- P&L cube: ACT, BUD and FC on one grain and one currency policy ----------
    with step("p&l cube"):
        pnl = dataset.add(
            "fact_pnl",
            build_pnl(
                cfg, dataset["fact_gl"], dataset["fact_budget"], dataset["fact_forecast"],
                dataset["map_local_to_group"], dataset["dim_entity"], dataset["dim_scenario"],
                calendar, dataset["dim_fx_rate"],
            ),
        )
        if "dim_report_line" in dataset.tables:
            dataset.add(
                "rpt_income_statement",
                build_income_statement(
                    pnl, dataset["dim_report_line"], dataset["bridge_report_line"]
                ),
            )

    # Volume reporting. NOTE: the spec's back-solver (config `volume.target_gl_rows`)
    # is NOT implemented — the generator does not yet adjust grain density to hit the
    # target. Report the gap explicitly rather than let it pass unnoticed.
    target = int(cfg.get("volume.target_gl_rows"))
    tolerance = float(cfg.get("volume.tolerance", 0.2))
    actual = dataset["fact_gl"].height
    if target and abs(actual - target) > target * tolerance:
        dataset.notes.append(
            f"volume: fact_gl has {actual:,} rows vs target {target:,} "
            f"(+/-{tolerance:.0%}). The back-solver saturated the customer lever; "
            "fact_gl is trial-balance grain, so its size is bounded by "
            "families x markets x months. To grow it further raise period.months or add "
            "market leaves (dims/markets.py GEOGRAPHY) or families (industry pack). SKU "
            "count grows fact_sales_lines, not fact_gl -- which is spec 6's point that "
            "volume belongs in the sub-ledgers."
        )

    return dataset
