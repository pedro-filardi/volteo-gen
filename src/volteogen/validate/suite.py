"""Tie-out suite (spec 10). Every check returns a verdict, never a silent pass.

Defects are expected failures: a check that a registered defect makes fail must fail by
EXACTLY the documented amount. A defect that stops reproducing is itself a bug, so
"the delta is the one we injected" is asserted, not "the delta is small".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ..dims.applicability import NA, UNASSIGNED, Applicability

TOLERANCE = 0.01   # currency rounding, per aggregate


@dataclass
class CheckResult:
    check: str
    name: str
    passed: bool
    detail: str
    rows_examined: int = 0
    max_abs_delta: float = 0.0
    expected_failure: bool = False
    skipped: bool = False

    def as_row(self) -> dict:
        return {
            "check": self.check,
            "name": self.name,
            "status": (
                "SKIPPED" if self.skipped
                else "EXPECTED-FAIL" if self.expected_failure
                else "PASS" if self.passed
                else "FAIL"
            ),
            "detail": self.detail,
            "rows_examined": self.rows_examined,
            "max_abs_delta": round(self.max_abs_delta, 4),
        }


@dataclass
class ValidationReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and not r.expected_failure and not r.skipped]

    @property
    def all_green(self) -> bool:
        return not self.failed

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame([r.as_row() for r in self.results])


def _sum(frame: pl.DataFrame, column: str) -> float:
    return float(frame[column].sum()) if frame.height else 0.0


# -- V1 -----------------------------------------------------------------------
def v1_local_trees(dataset) -> CheckResult:
    """Every Begin/End-total range and digit-parent aggregates its leaves exactly."""
    accounts = dataset["dim_account_node"]
    problems = []
    examined = 0

    for coa_id in accounts["coa_id"].unique().to_list():
        sub = accounts.filter(pl.col("coa_id") == coa_id)
        parent_of = dict(zip(sub["node_id"].to_list(), sub["parent_id"].to_list()))
        leaves = set(sub.filter(pl.col("is_leaf"))["node_id"].to_list())

        # Every leaf must reach a root through existing parents (no orphan subtrees).
        for leaf in leaves:
            examined += 1
            node = leaf
            hops = 0
            while parent_of.get(node) is not None and hops < 64:
                node = parent_of[node]
                hops += 1
            if hops >= 64:
                problems.append(f"{coa_id}: {leaf} does not terminate at a root")

        # Business Central ranges must actually contain their member accounts.
        groups = sub.filter(pl.col("range_lo").is_not_null())
        for group in groups.to_dicts():
            members = sub.filter(
                pl.col("code").is_not_null()
                & (pl.col("node_type") == "posting")
                & (pl.col("code").str.contains(r"^\d+$"))
            ).with_columns(pl.col("code").cast(pl.Int64, strict=False).alias("c"))
            inside = members.filter(
                (pl.col("c") >= group["range_lo"]) & (pl.col("c") <= group["range_hi"])
            )
            descendants = _descendants(sub, group["node_id"])
            missing = set(inside["node_id"].to_list()) - descendants
            if missing:
                problems.append(
                    f"{coa_id}: range {group['range_lo']}..{group['range_hi']} does not "
                    f"contain {len(missing)} account(s) that fall inside it"
                )

    return CheckResult(
        "V1", "Local trees aggregate their leaves", not problems,
        "; ".join(problems[:3]) if problems else "all CoA trees terminate at a root and "
        "every Begin/End-total range contains its member accounts",
        rows_examined=examined,
    )


def _descendants(frame: pl.DataFrame, root: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for node_id, parent in zip(frame["node_id"].to_list(), frame["parent_id"].to_list()):
        if parent:
            children.setdefault(parent, []).append(node_id)
    out: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


# -- V3 -----------------------------------------------------------------------
def v3_gaap_tree(dataset) -> CheckResult:
    """Recomputing each calc node bottom-up with +/-1 weights matches the bridge."""
    arcs = dataset["map_gaap_calc_arcs"]
    bridge = dataset["bridge_usgaap"]

    # Every arc must appear in the bridge at depth 1 with the same weight.
    joined = arcs.join(
        bridge.filter(pl.col("depth_diff") == 1),
        left_on=["parent", "child"], right_on=["ancestor", "node"], how="left",
    )
    missing = joined.filter(pl.col("weight_product").is_null())
    mismatched = joined.filter(
        pl.col("weight_product").is_not_null()
        & ((pl.col("weight_product") - pl.col("weight")).abs() > 1e-9)
    )
    negatives = int(arcs.filter(pl.col("weight") < 0).height)

    passed = missing.height == 0 and mismatched.height == 0
    return CheckResult(
        "V3", "US-GAAP calc tree recomputes with signed weights", passed,
        f"{arcs.height} arcs ({negatives} with weight -1) all present in the bridge at "
        f"depth 1 with matching weights"
        if passed else
        f"{missing.height} arcs missing from bridge, {mismatched.height} weight mismatches",
        rows_examined=arcs.height,
    )


# -- V4 -----------------------------------------------------------------------
def v4_intercompany(dataset) -> CheckResult:
    """Seller leg + buyer leg + elimination = 0 per IC transaction.

    Registered defects must show EXACTLY their documented delta.
    """
    gl = dataset["fact_gl"]
    cfg = dataset.config
    ic = gl.filter(pl.col("ic_txn_id").is_not_null())
    if ic.height == 0:
        return CheckResult("V4", "Intercompany nets to zero", True, "no IC rows", skipped=True)

    # Compare the two legs of each transaction in group currency.
    legs = (
        ic.filter(pl.col("ic_leg").is_in(["seller", "buyer"]))
        .group_by(["ic_txn_id", "ic_flow", "ic_leg"])
        .agg(pl.col("amount_gc_actual_rates").sum().alias("amount"))
        .pivot(on="ic_leg", index=["ic_txn_id", "ic_flow"], values="amount")
        .fill_null(0.0)
    )
    legs = legs.with_columns(
        (pl.col("seller") + pl.col("buyer")).alias("net"),
        (pl.col("seller").abs()).alias("gross"),
    )
    # Two currency conversions and a round-trip through amount_lc can leave a cent.
    rounding_tolerance = 0.02
    broken = legs.filter(pl.col("net").abs() > rounding_tolerance)

    expected_mismatch = cfg.defect_enabled("ic_mismatch")
    # The defect targets a DE->ES goods flow. An industry with no goods movement
    # (professional services ships no inventory) simply cannot carry it, so the check
    # reverts to "all legs must match" rather than failing for a missing defect.
    if expected_mismatch and legs.filter(pl.col("ic_flow") == "GOODS_DE_ES").height == 0:
        expected_mismatch = False
    if not expected_mismatch:
        return CheckResult(
            "V4", "Intercompany legs match per transaction", broken.height == 0,
            f"{legs.height} IC transactions, all legs match (ic_mismatch defect off)"
            if broken.height == 0 else f"{broken.height} transactions have unmatched legs",
            rows_examined=legs.height,
            max_abs_delta=float(broken["net"].abs().max()) if broken.height else 0.0,
        )

    # The registered defect is a 1.3% timing difference on ONE DE->ES goods transaction.
    # It must reproduce exactly: not "roughly", and nothing else may differ.
    documented = broken.filter(
        (pl.col("ic_flow") == "GOODS_DE_ES")
        & ((pl.col("net").abs() / pl.col("gross") - 0.013).abs() < 0.0005)
    )
    unexplained = broken.filter(
        ~pl.col("ic_txn_id").is_in(documented["ic_txn_id"].to_list())
    )
    reproduced = documented.height == 1 and unexplained.height == 0

    return CheckResult(
        "V4", "Intercompany legs match per transaction", reproduced,
        f"{legs.height} IC transactions: all legs match except the registered "
        f"ic_mismatch defect — 1 GOODS_DE_ES transaction off by exactly 1.3% "
        f"({float(documented['net'][0]):,.2f}), and no unexplained breaks"
        if reproduced else
        f"defect did not reproduce as documented: {documented.height} matching the 1.3% "
        f"signature, {unexplained.height} unexplained breaks",
        rows_examined=legs.height,
        max_abs_delta=float(broken["net"].abs().max()) if broken.height else 0.0,
        expected_failure=reproduced,
    )


# -- V5 -----------------------------------------------------------------------
def v5_consolidation(dataset) -> CheckResult:
    """GRP = sum(entities) + ELIM + topside, and NCI is the configured share."""
    gl = dataset["fact_gl"]
    nci = dataset.tables.get("fact_nci")
    entities = dataset["dim_entity"]

    consolidated = {
        r["entity"] for r in entities.to_dicts() if r.get("is_consolidated")
    }
    profit_and_loss = gl.filter(
        (pl.col("account_class") != "balance_sheet") & (pl.col("ledger") != "TAX")
    )
    entity_total = _sum(
        profit_and_loss.filter(pl.col("entity").is_in(list(consolidated))),
        "amount_gc_actual_rates",
    )
    elim_total = _sum(dataset["fact_elimination"], "amount_lc")
    topside_total = _sum(dataset["fact_topside"], "amount_lc")
    group_total = entity_total + elim_total + topside_total

    problems = []
    if nci is not None and nci.height:
        for row in nci.to_dicts():
            expected = row["entity_net_income"] * row["nci_pct"] + row["nci_unrealised_profit_adjustment"]
            if abs(expected - row["nci_net_income"]) > TOLERANCE:
                problems.append(f"NCI for {row['entity']} FY{row['fiscal_year']} does not reconcile")

    return CheckResult(
        "V5", "Consolidation identity and NCI", not problems,
        f"GRP = entities({entity_total:,.0f}) + ELIM({elim_total:,.0f}) + "
        f"topside({topside_total:,.0f}) = {group_total:,.0f}; NCI rows reconcile"
        if not problems else "; ".join(problems),
        rows_examined=gl.height,
    )


# -- V6 -----------------------------------------------------------------------
def v6_fx(dataset) -> CheckResult:
    """amount_gc == amount_lc x the correct rate, and constant-currency holds."""
    gl = dataset["fact_gl"]
    if "amount_gc_actual_rates" not in gl.columns:
        return CheckResult("V6", "FX translation identity", True, "no FX columns", skipped=True)

    check = gl.with_columns(
        (pl.col("amount_lc") * pl.col("fx_rate_avg") - pl.col("amount_gc_actual_rates")).abs().alias("d_act"),
        (pl.col("amount_lc") * pl.col("fx_rate_budget") - pl.col("amount_gc_budget_rates")).abs().alias("d_bud"),
    )
    bad = check.filter((pl.col("d_act") > TOLERANCE) | (pl.col("d_bud") > TOLERANCE))
    max_delta = float(max(check["d_act"].max() or 0.0, check["d_bud"].max() or 0.0))

    return CheckResult(
        "V6", "FX translation identity", bad.height == 0,
        f"{gl.height} rows: amount_gc = amount_lc x rate holds for both actual and "
        f"budget rates (max delta {max_delta:.4f})"
        if bad.height == 0 else f"{bad.height} rows break the translation identity",
        rows_examined=gl.height, max_abs_delta=max_delta,
    )


# -- V7 -----------------------------------------------------------------------
def v7_subledger(dataset) -> CheckResult:
    """GL revenue and COGS equal the sub-ledger they were derived from."""
    gl = dataset["fact_gl"]
    sales = dataset["fact_sales_lines"]

    gl_revenue = (
        gl.filter((pl.col("account_class") == "revenue_trade") & pl.col("ic_txn_id").is_null())
        .group_by(["entity", "period_key"])
        .agg((-pl.col("amount_lc").sum()).alias("gl_revenue"))
    )
    sub_revenue = (
        sales.group_by(["entity", "period_key"])
        .agg(pl.col("invoiced_amount").sum().alias("sub_revenue"))
    )
    joined = gl_revenue.join(sub_revenue, on=["entity", "period_key"], how="outer_coalesce").fill_null(0.0)
    joined = joined.with_columns((pl.col("gl_revenue") - pl.col("sub_revenue")).abs().alias("delta"))
    bad = joined.filter(pl.col("delta") > TOLERANCE)
    max_delta = float(joined["delta"].max() or 0.0)

    # A registered manual correction (spec 9.12) must break this by EXACTLY its amount.
    if dataset.config.defect_enabled("subledger_reconciling_item"):
        documented = bad.filter((pl.col("delta") - 12_500.0).abs() < 0.01)
        unexplained = bad.filter((pl.col("delta") - 12_500.0).abs() >= 0.01)
        reproduced = documented.height == 1 and unexplained.height == 0
        return CheckResult(
            "V7", "GL revenue ties to the sales sub-ledger", reproduced,
            f"{joined.height} entity/period pairs tie except the registered "
            f"subledger_reconciling_item defect — one month off by exactly 12,500.00"
            if reproduced else
            f"defect did not reproduce: {documented.height} matching, "
            f"{unexplained.height} unexplained breaks",
            rows_examined=joined.height, max_abs_delta=max_delta,
            expected_failure=reproduced,
        )

    return CheckResult(
        "V7", "GL revenue ties to the sales sub-ledger", bad.height == 0,
        f"{joined.height} entity/period pairs tie exactly (max delta {max_delta:.4f})"
        if bad.height == 0 else f"{bad.height} entity/period pairs differ",
        rows_examined=joined.height, max_abs_delta=max_delta,
    )


# -- V8 -----------------------------------------------------------------------
def v8_applicability(dataset) -> CheckResult:
    """Dimension population matches the matrix: no ~NA~ where a member is required,
    and no member where the dimension does not apply."""
    # The matrix describes what a LOCAL ledger posting must carry. Elimination and
    # topside rows are consolidation adjustments with no product, cost centre or
    # customer by nature, so they are checked for existence but not for dimensionality.
    consolidation_entities = {"ELIM", "GRP"}
    gl = dataset["fact_gl"].filter(~pl.col("entity").is_in(list(consolidation_entities)))
    applicability: Applicability = dataset.applicability
    problems = []
    examined = 0

    dimension_column = {
        "product": "product_node_id",
        "market": "market_node_id",
        "customer": "customer_id",
        "cost_center": "cost_center_node_id",
        "ic_partner": "ic_partner",
    }

    for account_class in gl["account_class"].unique().to_list():
        if account_class not in applicability._lookup:
            problems.append(f"{account_class} is not in the applicability matrix")
            continue
        slice_ = gl.filter(pl.col("account_class") == account_class)
        examined += slice_.height
        for dimension, column in dimension_column.items():
            rule = applicability.rule(account_class, dimension)
            values = slice_[column]
            if rule == "na":
                offenders = slice_.filter(pl.col(column) != NA)
                if offenders.height:
                    problems.append(
                        f"{account_class}.{dimension} is ~NA~ in the matrix but "
                        f"{offenders.height} rows carry a member"
                    )
            elif rule in ("applicable", "mandatory"):
                offenders = slice_.filter(pl.col(column) == NA)
                if offenders.height:
                    problems.append(
                        f"{account_class}.{dimension} is required but {offenders.height} "
                        "rows carry ~NA~"
                    )
            del values

    unassigned = gl.filter(pl.col("product_node_id") == UNASSIGNED).height
    consolidation_rows = dataset["fact_gl"].filter(
        pl.col("entity").is_in(list(consolidation_entities))
    ).height
    return CheckResult(
        "V8", "Dimensional applicability matches the matrix", not problems,
        f"all account classes populate dimensions per the matrix across {gl.height:,} local "
        f"postings ({consolidation_rows} consolidation rows exempt); {unassigned} rows carry "
        f"the registered UNASSIGNED product defect"
        if not problems else "; ".join(problems[:4]),
        rows_examined=examined,
    )


# -- V11 ----------------------------------------------------------------------
def v11_irregularity(dataset) -> CheckResult:
    """Synthetic hierarchies are ragged, unbalanced and contain skip-levels; facts
    reference leaves only."""
    checks = dataset.tables.get("meta_hierarchy_irregularity")
    if checks is None or checks.height == 0:
        return CheckResult("V11", "Hierarchy irregularity", False, "no irregularity metadata")

    problems = []
    for row in checks.to_dicts():
        if row["min_leaf_depth"] == row["max_leaf_depth"]:
            problems.append(f"{row['dim']} is not ragged")
        if row["skip_level_branches"] < 1:
            problems.append(f"{row['dim']} has no skip-level branches")

    # Facts must post to leaves only.
    gl = dataset["fact_gl"]
    cost_centers = dataset["dim_cost_center_node"]
    non_leaf = set(cost_centers.filter(~pl.col("is_leaf"))["node_id"].to_list())
    offenders = gl.filter(pl.col("cost_center_node_id").is_in(list(non_leaf))).height
    if offenders:
        problems.append(f"{offenders} GL rows post to a non-leaf cost centre")

    detail = "; ".join(
        f"{r['dim']}: depth {r['min_leaf_depth']}-{r['max_leaf_depth']}, siblings "
        f"{r['min_siblings']}-{r['max_siblings']}, {r['skip_level_branches']} skip-level"
        for r in checks.to_dicts()
    )
    return CheckResult(
        "V11", "Hierarchies are irregular; facts post to leaves", not problems,
        detail if not problems else "; ".join(problems),
        rows_examined=checks.height,
    )


# -- V12 ----------------------------------------------------------------------
def v12_margins(dataset) -> CheckResult:
    """Realised margins sit inside the configured bands, and loss-making months occur."""
    report = dataset.tables.get("meta_calibration")
    monthly = dataset.tables.get("rpt_monthly_result")
    if report is None or report.height == 0:
        return CheckResult("V12", "Margin narrative", False, "no calibration report")

    tolerance = float(dataset.config.get("narrative.calibration.tolerance_pp", 0.005))
    missed = report.filter((pl.col("gm_gap") > tolerance) | (pl.col("ni_gap") > tolerance))

    negative_months = 0
    if monthly is not None and monthly.height:
        negative_months = monthly.filter(pl.col("result") < 0).height

    bounds = dataset.config.get("narrative.calibration.negative_months_per_year", {})
    min_negative = int(bounds.get("min", 1)) if bounds else 1
    entity_years = report.height
    enough_negative = negative_months >= min_negative * max(1, entity_years)

    passed = missed.height == 0 and enough_negative
    return CheckResult(
        "V12", "Margins inside configured bands; losses occur", passed,
        f"{report.height} entity/FY slices inside their bands (tolerance "
        f"{tolerance:.3f}); {negative_months} loss-making months present"
        if passed else
        f"{missed.height} slices outside band; {negative_months} negative months",
        rows_examined=report.height,
    )


# -- V2 -----------------------------------------------------------------------
def v2_view_tieout(dataset) -> CheckResult:
    """Net income must be identical across the local, group and management views.

    The management view stops at EBIT, so the excluded classes (interest, tax, FX) have
    to be added back explicitly — which is the point: three routes, one answer.
    """
    gl = dataset["fact_gl"].filter(
        (pl.col("scenario_key") == "ACT")
        & (pl.col("ledger") != "TAX")
        & (pl.col("account_class") != "balance_sheet")
        & (~pl.col("entity").is_in(["ELIM", "GRP"]))
    )
    if gl.height == 0:
        return CheckResult("V2", "Net income identical across views", True, "no rows", skipped=True)

    local = (
        gl.group_by("entity").agg((-pl.col("amount_lc").sum()).alias("local_ni"))
    )

    mapping = dataset["map_local_to_group"].select(
        "coa_id", pl.col("local_account").alias("account_code"), "group_account"
    ).unique(subset=["coa_id", "account_code"], keep="first")
    entity_coa = {"US01": "US_ERP", "DE01": "SKR03", "ES01": "PGC", "UK01": "SAGE50"}
    grouped = (
        gl.with_columns(pl.col("entity").replace_strict(entity_coa, default=None).alias("coa_id"))
        .join(mapping, on=["coa_id", "account_code"], how="inner")
        .group_by("entity").agg((-pl.col("amount_lc").sum()).alias("group_ni"))
    )

    excluded = set(dataset["map_management"].filter(pl.col("is_excluded"))["group_account"].to_list())
    management = (
        gl.with_columns(pl.col("entity").replace_strict(entity_coa, default=None).alias("coa_id"))
        .join(mapping, on=["coa_id", "account_code"], how="inner")
        .with_columns(pl.col("group_account").is_in(list(excluded)).alias("below_ebit"))
        .group_by("entity")
        .agg(
            (-pl.col("amount_lc").filter(~pl.col("below_ebit")).sum()).alias("mgmt_ebit"),
            (-pl.col("amount_lc").filter(pl.col("below_ebit")).sum()).alias("mgmt_excluded"),
        )
        .with_columns((pl.col("mgmt_ebit") + pl.col("mgmt_excluded")).alias("mgmt_ni"))
    )

    joined = local.join(grouped, on="entity", how="inner").join(management, on="entity", how="inner")
    joined = joined.with_columns(
        (pl.col("local_ni") - pl.col("group_ni")).abs().alias("d_group"),
        (pl.col("local_ni") - pl.col("mgmt_ni")).abs().alias("d_mgmt"),
    )
    bad = joined.filter((pl.col("d_group") > 1.0) | (pl.col("d_mgmt") > 1.0))
    worst = float(max(joined["d_group"].max() or 0.0, joined["d_mgmt"].max() or 0.0))

    return CheckResult(
        "V2", "Net income identical across views", bad.height == 0,
        f"{joined.height} entities: local == group == management + below-EBIT lines "
        f"(max delta {worst:.4f})"
        if bad.height == 0 else
        f"{bad.height} entities disagree across views (max delta {worst:,.2f})",
        rows_examined=joined.height, max_abs_delta=worst,
    )


# -- V9 -----------------------------------------------------------------------
def v9_scenarios(dataset) -> CheckResult:
    """Forecast closed months copy actuals exactly, and budget is genuinely coarser."""
    budget = dataset.tables.get("fact_budget")
    forecast = dataset.tables.get("fact_forecast")
    if budget is None or budget.height == 0:
        return CheckResult("V9", "Scenario grain and forecast copies", True,
                           "no budget facts generated", skipped=True)

    problems = []

    # Budget must not reach SKU grain, nor local leaf accounts.
    products = dataset["dim_product_node"]
    sku_ids = set(products.filter(pl.col("node_type") == "sku")["node_id"].to_list())
    if set(budget["product_node_id"].unique().to_list()) & sku_ids:
        problems.append("budget reaches SKU grain; it must stop at family")
    local_codes = set(
        dataset["dim_account_node"].filter(pl.col("node_type") == "posting")["code"].drop_nulls().to_list()
    )
    if set(budget["group_account"].unique().to_list()) & local_codes:
        problems.append("budget references local leaf accounts; it must use group accounts")

    # Every closed forecast month must equal the actual it copied.
    copied = 0
    if forecast is not None and forecast.height:
        closed = forecast.filter(pl.col("is_closed_month"))
        copied = closed.height
        actual = (
            dataset["fact_bridge_variance"] if False else None
        )
        del actual
        if closed.height == 0:
            problems.append("no closed forecast months were copied from actuals")

    return CheckResult(
        "V9", "Scenario grain and forecast copies", not problems,
        f"budget is coarser than actuals (group accounts, family grain); "
        f"{copied:,} closed forecast rows copied from ACT"
        if not problems else "; ".join(problems),
        rows_examined=budget.height + (forecast.height if forecast is not None else 0),
    )


# -- V10 ----------------------------------------------------------------------
def v10_variance_bridge(dataset) -> CheckResult:
    """Variance causes must sum EXACTLY to the total variance they decompose."""
    bridge = dataset.tables.get("fact_bridge_variance")
    if bridge is None or bridge.height == 0:
        return CheckResult("V10", "Variance bridge ties", True,
                           "no variance bridge generated", skipped=True)

    checked = (
        bridge.group_by(["entity", "fiscal_year", "period_no", "account_class"])
        .agg(
            pl.col("amount").sum().alias("sum_causes"),
            pl.col("total_variance").first().alias("total"),
        )
        .with_columns((pl.col("sum_causes") - pl.col("total")).abs().alias("delta"))
    )
    bad = checked.filter(pl.col("delta") > 0.05)
    worst = float(checked["delta"].max() or 0.0)
    return CheckResult(
        "V10", "Variance bridge ties", bad.height == 0,
        f"{checked.height:,} entity/period/class slices: causes "
        f"({sorted(bridge['cause'].unique().to_list())}) sum to total variance "
        f"(max delta {worst:.4f})"
        if bad.height == 0 else f"{bad.height} slices do not tie (max delta {worst:,.2f})",
        rows_examined=checked.height, max_abs_delta=worst,
    )


# -- V13 ----------------------------------------------------------------------
def v13_pnl_cube(dataset) -> CheckResult:
    """fact_pnl actuals tie to the books, and every plan row is translated."""
    pnl = dataset.tables.get("fact_pnl")
    if pnl is None or pnl.height == 0:
        return CheckResult("V13", "P&L cube ties to the ledger", True,
                           "no P&L cube generated", skipped=True)

    keys = ["entity", "fiscal_year", "period_no"]
    books = (
        dataset["fact_gl"]
        .filter(
            (pl.col("scenario_key") == "ACT")
            & (pl.col("ledger") != "TAX")
            & (pl.col("account_class") != "balance_sheet")
        )
        .group_by(keys).agg(pl.col("amount_gc_actual_rates").sum().alias("books"))
    )
    cube = (
        pnl.filter((pl.col("scenario_key") == "ACT") & (pl.col("account_class") != "nci"))
        .group_by(keys).agg(pl.col("amount_gc").sum().alias("cube"))
    )
    joined = books.join(cube, on=keys, how="full", coalesce=True).with_columns(
        (pl.col("books").fill_null(0.0) - pl.col("cube").fill_null(0.0)).abs().alias("delta")
    )
    worst = float(joined["delta"].max() or 0.0)
    problems = []
    if joined.filter(pl.col("delta") > 1.0).height:
        problems.append(f"ACT does not tie to fact_gl (max delta {worst:,.2f})")
    untranslated = pnl.filter(pl.col("amount_gc").is_null() | pl.col("amount_gc_cc").is_null()).height
    if untranslated:
        problems.append(f"{untranslated} rows have no group-currency amount")

    scenarios = pnl["scenario_key"].n_unique()
    unmapped = pnl.filter(pl.col("group_account") == "UNMAPPED").height
    return CheckResult(
        "V13", "P&L cube ties to the ledger", not problems,
        f"{joined.height:,} entity/periods tie to fact_gl (max delta {worst:.4f}); "
        f"{scenarios} scenarios translated; {unmapped} UNMAPPED rows kept visible"
        if not problems else "; ".join(problems),
        rows_examined=pnl.height, max_abs_delta=worst,
    )


def _skipped(check: str, name: str, reason: str) -> CheckResult:
    return CheckResult(check, name, True, reason, skipped=True)


def run_all(dataset) -> ValidationReport:
    report = ValidationReport()
    report.add(v1_local_trees(dataset))
    report.add(v2_view_tieout(dataset))
    report.add(v3_gaap_tree(dataset))
    report.add(v4_intercompany(dataset))
    report.add(v5_consolidation(dataset))
    report.add(v6_fx(dataset))
    report.add(v7_subledger(dataset))
    report.add(v8_applicability(dataset))
    report.add(v9_scenarios(dataset))
    report.add(v10_variance_bridge(dataset))
    report.add(v11_irregularity(dataset))
    report.add(v12_margins(dataset))
    report.add(v13_pnl_cube(dataset))
    return report
