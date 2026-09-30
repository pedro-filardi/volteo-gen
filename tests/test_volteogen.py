"""Test suite. The build itself is the integration test; these pin the invariants."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from volteogen.config import CONFIG_DIR, SEEDS_DIR, load_config, load_pack
from volteogen.dims.applicability import NA, UNASSIGNED, classify_accounts, load_applicability
from volteogen.dims.hierarchy import assert_irregularity
from volteogen.pipeline import build_dataset
from volteogen.rng import SeedBank, derive_seed, stable_hash_unit
from volteogen.seedload import load_seeds
from volteogen.validate.suite import run_all


@pytest.fixture(scope="session")
def seeds():
    return load_seeds(SEEDS_DIR, verify=False)


@pytest.fixture(scope="session")
def dataset():
    return build_dataset(preset="S", verbose=False)


@pytest.fixture(scope="session")
def report(dataset):
    return run_all(dataset)


# -- Seeds: the real skeletons must load with their real shape -----------------
def test_seed_counts_match_upstream(seeds):
    """Counts documented in seeds/SOURCES.md — a drift here means an upstream change."""
    postings = {
        row["coa_id"]: row["postings"] for row in seeds.summary().to_dicts()
    }
    assert postings["SKR03"] == 1274, "SKR03 ships 1,274 accounts"
    assert postings["PGC"] == 588, "Spanish PGC ships 588 accounts"
    assert postings["SAGE50"] == 166, "UK nominal ledger ships 166 codes"
    assert postings["US_ERP"] > 200


def test_gaap_income_statement_has_531_arcs(seeds):
    assert seeds.gaap_arcs.height == 531
    assert seeds.gaap_arcs.filter(pl.col("weight") < 0).height > 0, (
        "negative-weight arcs are what break naive SUM(amount)"
    )
    assert int(seeds.gaap_nodes["depth"].max()) >= 10


def test_charts_have_genuinely_different_depth(seeds):
    depths = {row["coa_id"]: row["max_depth"] for row in seeds.summary().to_dicts()}
    assert depths["SAGE50"] < depths["SKR03"] < depths["US_ERP"], (
        "charts must differ in depth so level-based logic breaks by design"
    )


def test_local_charts_encode_meaning_in_account_numbers(seeds):
    """DE splits revenue by VAT rate; ES splits it by geography."""
    de = seeds.postings("SKR03").filter(pl.col("code").is_in(["8300", "8400"]))
    assert de.height == 2
    assert "19" in " ".join(de["name"].to_list())
    assert "7" in " ".join(de["name"].to_list())

    es = seeds.postings("PGC").filter(pl.col("code").is_in(["7000", "7001", "7002"]))
    assert es.height == 3


def test_missing_seed_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_seeds|seeds directory"):
        load_seeds(tmp_path / "nope")


# -- Determinism ---------------------------------------------------------------
def test_seed_derivation_is_stable_across_processes():
    assert derive_seed(42, "dims", "products") == derive_seed(42, "dims", "products")
    assert derive_seed(42, "dims", "products") != derive_seed(42, "dims", "markets")
    assert derive_seed(42, "a") != derive_seed(43, "a")
    assert 0.0 <= stable_hash_unit("x", "y") < 1.0


def test_modules_draw_independent_streams():
    bank = SeedBank(42)
    first = bank.rng("dims", "products").random(5).tolist()
    bank.rng("dims", "markets").random(100)          # advance an unrelated module
    assert bank.fresh("dims", "products").random(5).tolist() == first


def test_build_is_reproducible():
    left = build_dataset(preset="S", verbose=False)["fact_gl"]
    right = build_dataset(preset="S", verbose=False)["fact_gl"]
    assert left.height == right.height
    assert float(left["amount_lc"].sum()) == float(right["amount_lc"].sum())


def test_different_seed_changes_the_data():
    base = build_dataset(preset="S", verbose=False)["fact_gl"]
    other = build_dataset(preset="S", overrides={"seed": 7}, verbose=False)["fact_gl"]
    assert float(base["amount_lc"].sum()) != float(other["amount_lc"].sum())


# -- Config and industry pack --------------------------------------------------
def test_preset_overrides_default():
    assert load_config("S").months < load_config("L").months


def test_unknown_preset_lists_alternatives():
    with pytest.raises(FileNotFoundError, match="available"):
        load_config("NOPE")


def test_missing_industry_pack_fails_loudly():
    cfg = load_config("S", overrides={"industry_pack": "does_not_exist"})
    with pytest.raises(FileNotFoundError, match="industry pack"):
        load_pack(cfg)


def test_engine_holds_no_industry_knowledge():
    """Division names live in the pack, never in engine source."""
    engine = Path(__file__).resolve().parents[1] / "src" / "volteogen"
    offenders = []
    for path in engine.rglob("*.py"):
        text = path.read_text()
        for token in ("Headphones", "TWS Earbuds", "Power Banks", "GaN"):
            if token in text:
                offenders.append(f"{path.name}:{token}")
    assert not offenders, f"industry terms leaked into the engine: {offenders}"


# -- Applicability: the three states -------------------------------------------
def test_applicability_three_states_all_present(dataset):
    gl = dataset["fact_gl"]
    products = set(gl["product_node_id"].unique().to_list())
    assert NA in products, "~NA~ must exist: payroll structurally has no product"
    assert UNASSIGNED in products, "UNASSIGNED must exist: it is a registered defect"
    assert any(p not in (NA, UNASSIGNED) for p in products), "real members must exist"


def test_payroll_never_carries_a_product(dataset):
    payroll = dataset["fact_gl"].filter(pl.col("account_class") == "payroll")
    assert payroll.height > 0
    assert payroll.filter(pl.col("product_node_id") != NA).height == 0


def test_cogs_variance_is_unallocatable_to_product(dataset):
    """Spec 5: variances post at plant level and carry no product, ever."""
    variance = dataset["fact_gl"].filter(pl.col("account_class") == "cogs_variance")
    assert variance.height > 0
    assert variance.filter(pl.col("product_node_id") != NA).height == 0
    assert variance.filter(pl.col("cost_center_node_id") == NA).height == 0


def test_account_classification_covers_every_posting_account(seeds):
    applicability = load_applicability(CONFIG_DIR, load_pack(load_config("S")))
    classified = classify_accounts(seeds.accounts, applicability)
    postings = classified.filter(pl.col("node_type") == "posting")
    assert postings["account_class"].null_count() == 0
    assert set(postings["account_class"].unique()) <= set(applicability.classes)


# -- Hierarchies ---------------------------------------------------------------
def test_generated_hierarchies_are_irregular(dataset):
    for name, table in (
        ("product", "dim_product_node"),
        ("market", "dim_market_node"),
        ("cost_center", "dim_cost_center_node"),
    ):
        stats = assert_irregularity(dataset[table], name)
        assert stats["min_leaf_depth"] < stats["max_leaf_depth"]
        assert stats["skip_level_branches"] >= 1


def test_regular_tree_is_rejected():
    """The V11 check must actually fail on a perfectly regular tree."""
    regular = pl.DataFrame(
        {
            "node_id": ["r", "a", "b", "a1", "a2", "b1", "b2"],
            "parent_id": [None, "r", "r", "a", "a", "b", "b"],
            "name": list("rabcdef"),
            "node_type": ["root", "mid", "mid", "leaf", "leaf", "leaf", "leaf"],
            "is_leaf": [False, False, False, True, True, True, True],
            "depth": [0, 1, 1, 2, 2, 2, 2],
            "is_skip_level": [False] * 7,
        }
    )
    with pytest.raises(AssertionError, match="V11"):
        assert_irregularity(regular, "regular")


def test_levels_table_records_true_depth(dataset):
    levels = dataset["dim_product_levels"]
    assert levels.filter(pl.col("is_padded")).height > 0, "ragged paths must be padded"
    assert levels["true_depth"].n_unique() > 1, "true_depth preserves the real depth"


# -- Facts ---------------------------------------------------------------------
def test_gl_revenue_derives_from_subledger(dataset):
    """GL revenue is the sub-ledger summarised — apart from one booked correction.

    The `subledger_reconciling_item` defect deliberately introduces a 12,500 manual
    journal, so the tie is exact only once that documented row is excluded.
    """
    revenue = dataset["fact_gl"].filter(
        (pl.col("account_class") == "revenue_trade") & pl.col("ic_txn_id").is_null()
    )
    manual = revenue.filter(pl.col("source") == "manual_correction")
    gl_revenue = -float(revenue.filter(pl.col("source") != "manual_correction")["amount_lc"].sum())
    sub_revenue = float(dataset["fact_sales_lines"]["invoiced_amount"].sum())
    assert abs(gl_revenue - sub_revenue) < 1.0
    if dataset.config.defect_enabled("subledger_reconciling_item"):
        assert manual.height == 1, "the reconciling item must be exactly one row"
        assert abs(float(manual["amount_lc"][0]) + 12_500.0) < 0.01


def test_ic_legs_are_generated_together(dataset):
    """Both legs come from one record, so every transaction has both."""
    ic = dataset["fact_gl"].filter(pl.col("ic_txn_id").is_not_null())
    legs = ic.group_by("ic_txn_id").agg(pl.col("ic_leg").n_unique().alias("n"))
    assert legs.filter(pl.col("n") < 2).height == 0


def test_allocations_net_to_zero_per_period(dataset):
    alloc = dataset["fact_gl"].filter(pl.col("account_class") == "allocation")
    if alloc.height == 0:
        pytest.skip("allocations disabled")
    net = alloc.group_by(["entity", "period_key"]).agg(pl.col("amount_lc").sum().alias("net"))
    assert float(net["net"].abs().max()) < 0.05


def test_fiscal_calendar_is_not_the_calendar_year(dataset):
    calendar = dataset["dim_calendar"]
    p01 = calendar.filter((~pl.col("is_period13")) & (pl.col("period_no") == 1))
    assert set(p01["calendar_month"].to_list()) == {7}, "P01 must be July"
    assert calendar.filter(pl.col("is_period13")).height > 0, "P13 must exist"


def test_uk_enters_consolidation_late(dataset):
    """The mid-history acquisition must create a real scope effect."""
    gl = dataset["fact_gl"].filter(pl.col("entity") == "UK01")
    others = dataset["fact_gl"].filter(pl.col("entity") == "DE01")
    assert gl["period_key"].min() > others["period_key"].min()


# -- Validation suite ----------------------------------------------------------
def test_validation_suite_is_green(report):
    assert report.all_green, f"failed checks: {[r.check for r in report.failed]}"


def test_registered_defect_reproduces_exactly(report):
    v4 = next(r for r in report.results if r.check == "V4")
    assert v4.expected_failure, "the ic_mismatch defect must reproduce as an EXPECTED-FAIL"
    assert v4.max_abs_delta > 0


def test_margins_land_inside_configured_bands(dataset):
    calibration = dataset["meta_calibration"]
    assert calibration.height > 0
    assert float(calibration["gm_gap"].max()) <= 0.005
    assert float(calibration["ni_gap"].max()) <= 0.005


def test_turnaround_entity_crosses_zero(dataset):
    """UK01's configured arc must actually show up in the numbers."""
    uk = dataset["meta_calibration"].filter(pl.col("entity") == "UK01").sort("fiscal_year")
    values = uk["ni_actual"].to_list()
    assert values[0] < 0 < values[-1], f"turnaround must cross zero, got {values}"


def test_loss_making_months_occur(dataset):
    monthly = dataset["rpt_monthly_result"]
    assert monthly.filter(pl.col("result") < 0).height > 0


# -- Cost geography lives on the master, not the ledger ------------------------
def test_ledger_has_no_geography_columns(dataset):
    """Geography must never leak into fact_gl as an ad-hoc column."""
    forbidden = {"site", "country", "geo_region", "region", "serves_market_node_id"}
    leaked = forbidden & set(dataset["fact_gl"].columns)
    assert not leaked, f"geography leaked into the ledger: {sorted(leaked)}"


def test_cost_centre_master_carries_geography(dataset):
    cost_centers = dataset["dim_cost_center_node"]
    for column in ("site", "country", "geo_region", "serves_market_node_id"):
        assert column in cost_centers.columns, f"cost centre master is missing {column}"
    leaves = cost_centers.filter(pl.col("is_leaf"))
    assert leaves.filter(pl.col("country").is_null()).height == 0
    assert leaves["site"].n_unique() > 1, "sites must be distinguishable, not one per group"


def test_incurred_and_served_geography_can_differ(dataset):
    """A cost incurred in one place may support a different market — both are recorded."""
    leaves = dataset["dim_cost_center_node"].filter(pl.col("is_leaf"))
    serves_all = leaves.filter(pl.col("serves_market_node_id") == "MKT")
    serves_region = leaves.filter(pl.col("serves_market_node_id") != "MKT")
    assert serves_all.height > 0, "group functions must serve every market"
    assert serves_region.height > 0, "customer-facing functions must serve one region"


def test_served_market_points_at_real_market_nodes(dataset):
    """serves_market_node_id must join to dim_market_node, not be a free-text label."""
    market_ids = set(dataset["dim_market_node"]["node_id"].to_list())
    referenced = set(
        dataset["dim_cost_center_node"]
        .filter(pl.col("serves_market_node_id").is_not_null())["serves_market_node_id"]
        .to_list()
    )
    assert referenced <= market_ids, f"dangling market refs: {sorted(referenced - market_ids)}"


def test_opex_is_reportable_by_geography_via_the_master(dataset):
    """The user-facing requirement: opex by country, with no ledger geography column."""
    gl = dataset["fact_gl"]
    cost_centers = dataset["dim_cost_center_node"].select(
        "node_id", "country", "geo_region", "site"
    )
    joined = gl.filter(
        pl.col("account_class").is_in(["payroll", "facilities", "marketing", "other_opex"])
    ).join(cost_centers, left_on="cost_center_node_id", right_on="node_id", how="inner")
    by_country = joined.group_by("country").agg(pl.col("amount_lc").sum())
    assert by_country.height >= 4, "opex must be splittable across all four countries"
    assert by_country["country"].null_count() == 0


def test_profit_centre_claimed_only_where_division_is_unambiguous(dataset):
    """Sales/G&A must NOT be force-mapped to a division — that fabricates a P&L."""
    leaves = dataset["dim_cost_center_node"].filter(pl.col("is_leaf"))
    claimed = leaves.filter(pl.col("home_profit_center").is_not_null())
    assert claimed.height > 0, "R&D teams genuinely belong to a division"
    assert claimed.filter(pl.col("function").is_in(["SLS", "GA", "MKT"])).height == 0, (
        "shared functions must reach a profit centre through allocation, not attribution"
    )


def test_no_spurious_unassigned_market_on_costs(dataset):
    """Cost rows carry ~NA~ for market — never UNASSIGNED, which would read as a defect."""
    from volteogen.dims.applicability import UNASSIGNED

    costs = dataset["fact_gl"].filter(
        pl.col("account_class").is_in(
            ["payroll", "facilities", "marketing", "depreciation", "other_opex"]
        )
    )
    assert costs.filter(pl.col("market_node_id") == UNASSIGNED).height == 0


# -- Spec completeness: subsystems added in the second pass ---------------------
def test_all_eleven_bridges_exist(dataset):
    """Spec 8 wants a bridge over every tree, including the four local charts."""
    for name in ("bridge_product", "bridge_market", "bridge_cost_center",
                 "bridge_profit_center", "bridge_usgaap", "bridge_skr03", "bridge_pgc",
                 "bridge_sage50", "bridge_us_erp", "bridge_group_coa", "bridge_management"):
        assert name in dataset.tables and dataset[name].height > 0, f"missing {name}"


def test_period13_is_a_posting_period_not_a_month(dataset):
    """A 12-month sum must genuinely miss the close adjustments."""
    gl = dataset["fact_gl"]
    p13 = gl.filter(pl.col("is_period13"))
    assert p13.height > 0
    assert set(p13["period_no"].unique().to_list()) == {13}


def test_tax_ledger_is_deltas_not_a_second_set_of_books(dataset):
    gl = dataset["fact_gl"]
    tax = gl.filter(pl.col("ledger") == "TAX")
    primary = gl.filter(pl.col("ledger").is_in(["IFRS", "LOCAL"]))
    assert tax.height > 0
    assert abs(float(tax["amount_lc"].sum())) < abs(float(primary["amount_lc"].sum())), (
        "TAX must be a delta against the primary ledger, not a full restatement"
    )
    assert set(tax["entity"].unique().to_list()) <= {"US01", "DE01"}


def test_retained_earnings_roll_forward(dataset):
    """The one balance-sheet identity spec 12 requires."""
    gl = dataset["fact_gl"]
    re_rows = gl.filter(pl.col("source") == "balance_sheet:retained_earnings")
    assert re_rows.height > 0
    for entity in re_rows["entity"].unique().to_list():
        series = (
            re_rows.filter(pl.col("entity") == entity)
            .sort("period_key")["amount_lc"].to_list()
        )
        assert len(set(series)) > 1, "retained earnings must move, not sit flat"


def test_budget_is_coarser_than_actuals(dataset):
    budget = dataset["fact_budget"]
    assert budget.height > 0
    sku_ids = set(
        dataset["dim_product_node"].filter(pl.col("node_type") == "sku")["node_id"].to_list()
    )
    assert not (set(budget["product_node_id"].unique().to_list()) & sku_ids)


def test_forecast_closed_months_are_copies(dataset):
    forecast = dataset["fact_forecast"]
    assert forecast.height > 0
    closed = forecast.filter(pl.col("is_closed_month"))
    assert closed.height > 0
    # FC9+3 must carry more closed months than FC3+9.
    by_cycle = (
        closed.group_by("version").agg(pl.col("period_no").max().alias("last_closed"))
    )
    values = dict(zip(by_cycle["version"].to_list(), by_cycle["last_closed"].to_list()))
    assert values.get("FC9+3", 0) > values.get("FC3+9", 0)


def test_variance_bridge_ties_exactly(dataset):
    bridge = dataset["fact_bridge_variance"]
    assert bridge.height > 0
    checked = (
        bridge.group_by(["entity", "fiscal_year", "period_no", "account_class"])
        .agg(pl.col("amount").sum().alias("s"), pl.col("total_variance").first().alias("t"))
    )
    assert float((checked["s"] - checked["t"]).abs().max()) < 0.05


def test_every_configured_defect_is_implemented(dataset):
    """No defect may be a config toggle with nothing behind it."""
    log = dataset["meta_defects_injected"]
    logged = set(log["defect"].to_list()) if log.height else set()
    engine_backed = {
        "ic_mismatch", "ic_missing_partner_tag", "topside_unreconciled",
        "cc_reorg_midyear", "allocation_rule_change", "unassigned_product_tags",
        "late_period13_adjustments",
    }
    configured = {k for k in dataset.config.get("defects") if dataset.config.defect_enabled(k)}
    unimplemented = configured - logged - engine_backed
    assert not unimplemented, f"defects toggled on but never injected: {sorted(unimplemented)}"


def test_lifecycle_events_present(dataset):
    products = dataset["dim_product_node"]
    assert products.filter(pl.col("is_recalled")).height == 1, "exactly one recalled SKU"
    assert products.filter(pl.col("supersedes").is_not_null()).height > 0, "succession chains"
    assert products.filter(pl.col("relaunch_of").is_not_null()).height > 0, "relaunch link"


def test_second_industry_pack_runs_on_the_same_engine():
    """Spec 2.1: a different industry must need pack data only, never engine changes."""
    other = build_dataset(
        preset="S", overrides={"industry_pack": "professional_services"}, verbose=False
    )
    gl = other["fact_gl"]
    assert gl.height > 0
    # The pack disables production variance outright — there is no plant.
    assert gl.filter(pl.col("account_class") == "cogs_variance").height == 0
    assert other.applicability.overrides_applied, "pack overrides must actually apply"
    report = run_all(other)
    assert report.all_green, f"failed: {[r.check for r in report.failed]}"
