"""Intercompany engine, eliminations, NCI, topside journals and allocations (spec 7).

The design rule that makes intercompany *checkable*: both legs of a flow are generated
from ONE ``fact_ic_transaction`` record, so by construction every seller leg has a
matching buyer leg. Defects then break specific pairs deliberately (spec 9.2, 9.3) —
which is only meaningful because the baseline is exact.

Flows come from the industry pack (``flows.yaml``), not from code:

* DE01 -> ES01 / UK01 goods at standard cost x 1.25
* US01 -> all management fees, 1.5% of revenue, quarterly
* US01 -> UK01 loan of GBP 2m at SONIA + 2%, monthly interest

Unrealised profit: the margin sitting in IC inventory that the buyer has not yet sold on
is eliminated, and ES01's share carries a 20% NCI split because ES01 is 80% owned.
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from ..dims.applicability import NA, Applicability
from ..rng import SeedBank
from .account_router import AccountRouter
from .gl import _row

SONIA_BASE = 0.0525


def _rate_lookup(fx: pl.DataFrame) -> dict[tuple[str, str], float]:
    """(local currency, period) -> rate into group currency."""
    return {
        (r["from_currency"], r["period_key"]): float(r["rate_avg"])
        for r in fx.to_dicts()
        if r.get("rate_avg")
    }


def build_ic_transactions(
    cfg: Config, pack: dict, seeds: SeedBank, gl: pl.DataFrame, calendar: pl.DataFrame,
    fx: pl.DataFrame, entities: pl.DataFrame,
) -> pl.DataFrame:
    """One record per intercompany flow per period — the single source of both legs.

    The amount is carried in GROUP currency. Each leg is then converted into its own
    local currency, so a USD management fee charged to a EUR subsidiary nets to exactly
    zero in the consolidation instead of being booked at face value in both books.
    """
    rates = _rate_lookup(fx)
    entity_ccy = {r["entity"]: r["currency"] for r in entities.to_dicts()}
    # An entity acquired mid-history has no intercompany flows before it joins the
    # group: UK01 cannot be charged a management fee by a parent that does not yet own
    # it, and that gap is part of the M&A scope effect.
    joined_at = {r["entity"]: int(r["consolidated_from_month"]) for r in entities.to_dicts()}

    def both_in_scope(a: str, b: str, month_index: int) -> bool:
        return month_index >= max(joined_at.get(a, 0), joined_at.get(b, 0))

    def to_group(entity: str, period_key: str, amount: float) -> float:
        return amount * rates.get((entity_ccy.get(entity, "USD"), period_key), 1.0)

    rng = seeds.rng("facts", "ic")
    flows = pack["flows"]["flows"]
    tp = pack["economics"]["transfer_pricing"]
    months = calendar.filter(~pl.col("is_period13")).sort("month_index").to_dicts()

    # Third-party revenue per entity/period drives fee and goods volumes.
    revenue = (
        gl.filter(pl.col("account_class") == "revenue_trade")
        .group_by(["entity", "period_key"])
        .agg((-pl.col("amount_lc").sum()).alias("revenue"))
    )
    revenue_map = {(r["entity"], r["period_key"]): float(r["revenue"]) for r in revenue.to_dicts()}
    cogs = (
        gl.filter(pl.col("account_class") == "cogs_standard")
        .group_by(["entity", "period_key"])
        .agg(pl.col("amount_lc").sum().alias("cogs"))
    )
    cogs_map = {(r["entity"], r["period_key"]): float(r["cogs"]) for r in cogs.to_dicts()}

    entities_present = set(gl["entity"].unique().to_list())
    markup = float(tp["goods_markup_on_std_cost"])
    fee_pct = float(tp["management_fee_pct_of_revenue"])
    spread = float(tp["loan_spread_over_reference"])

    records: list[dict] = []
    txn_id = 0

    for flow in flows:
        kind = flow["type"]

        if kind == "goods":
            seller, buyer = flow["seller"], flow["buyer"]
            if seller not in entities_present or buyer not in entities_present:
                continue
            for month in months:
                if not both_in_scope(seller, buyer, int(month["month_index"])):
                    continue
                buyer_cogs = cogs_map.get((buyer, month["period_key"]), 0.0)
                if buyer_cogs <= 0:
                    continue
                # The buyer sources a share of its cost of sales from the plant.
                sourced_share = float(rng.uniform(0.45, 0.65))
                std_cost = to_group(buyer, month["period_key"], buyer_cogs * sourced_share)
                transfer_price = std_cost * (1.0 + markup)
                txn_id += 1
                records.append(
                    {
                        "ic_txn_id": f"IC{txn_id:06d}",
                        "flow_key": flow["key"],
                        "flow_type": kind,
                        "seller": seller,
                        "buyer": buyer,
                        "period_key": month["period_key"],
                        "fiscal_year": int(month["fiscal_year"]),
                        "period_no": int(month["period_no"]),
                        "month_index": int(month["month_index"]),
                        "amount": round(transfer_price, 2),
                        "std_cost_component": round(std_cost, 2),
                        "margin_component": round(transfer_price - std_cost, 2),
                        "currency": "GROUP",
                    }
                )

        elif kind == "services":
            seller = flow["seller"]
            buyers = flow["buyer"] if isinstance(flow["buyer"], list) else [flow["buyer"]]
            for buyer in buyers:
                if seller not in entities_present or buyer not in entities_present:
                    continue
                for month in months:
                    if int(month["period_no"]) % 3 != 0:   # quarterly
                        continue
                    if not both_in_scope(seller, buyer, int(month["month_index"])):
                        continue
                    quarter_revenue = sum(
                        revenue_map.get((buyer, m["period_key"]), 0.0)
                        for m in months
                        if m["fiscal_year"] == month["fiscal_year"]
                        and (int(month["period_no"]) - 3) < int(m["period_no"]) <= int(month["period_no"])
                    )
                    if quarter_revenue <= 0:
                        continue
                    quarter_revenue = to_group(buyer, month["period_key"], quarter_revenue)
                    txn_id += 1
                    records.append(
                        {
                            "ic_txn_id": f"IC{txn_id:06d}",
                            "flow_key": flow["key"],
                            "flow_type": kind,
                            "seller": seller,
                            "buyer": buyer,
                            "period_key": month["period_key"],
                            "fiscal_year": int(month["fiscal_year"]),
                            "period_no": int(month["period_no"]),
                            "month_index": int(month["month_index"]),
                            "amount": round(quarter_revenue * fee_pct, 2),
                            "std_cost_component": 0.0,
                            "margin_component": 0.0,
                            "currency": "GROUP",
                        }
                    )

        elif kind == "loan":
            lender, borrower = flow["lender"], flow["borrower"]
            if lender not in entities_present or borrower not in entities_present:
                continue
            principal = float(flow["principal"])
            for month in months:
                if not both_in_scope(lender, borrower, int(month["month_index"])):
                    continue
                rate = (SONIA_BASE + spread) / 12.0
                principal_gc = principal * rates.get(
                    (str(flow.get("principal_ccy", "GBP")), month["period_key"]), 1.0
                )
                txn_id += 1
                records.append(
                    {
                        "ic_txn_id": f"IC{txn_id:06d}",
                        "flow_key": flow["key"],
                        "flow_type": kind,
                        "seller": lender,
                        "buyer": borrower,
                        "period_key": month["period_key"],
                        "fiscal_year": int(month["fiscal_year"]),
                        "period_no": int(month["period_no"]),
                        "month_index": int(month["month_index"]),
                        "amount": round(principal_gc * rate, 2),
                        "std_cost_component": 0.0,
                        "margin_component": 0.0,
                        "currency": "GROUP",
                    }
                )

    if not records:
        raise ValueError("IC engine produced no transactions — check packs/*/flows.yaml")
    return pl.DataFrame(records)


def ic_gl_rows(
    cfg: Config,
    applicability: Applicability,
    router: AccountRouter,
    transactions: pl.DataFrame,
    calendar: pl.DataFrame,
    entities: pl.DataFrame,
    fx: pl.DataFrame,
) -> pl.DataFrame:
    """Both legs of every IC transaction, generated together so they always match."""
    rates = _rate_lookup(fx)
    periods = {r["period_key"]: r for r in calendar.to_dicts()}
    currency = {r["entity"]: r["currency"] for r in entities.to_dicts()}
    primary_ledger = {
        r["entity"]: ("IFRS" if r["parallel_ledgers"] else "LOCAL") for r in entities.to_dicts()
    }

    missing_tag = cfg.defect_enabled("ic_missing_partner_tag")
    mismatch = cfg.defect_enabled("ic_mismatch")

    rows: list[dict] = []
    for txn in transactions.to_dicts():
        period = periods[txn["period_key"]]
        seller, buyer = txn["seller"], txn["buyer"]
        amount_gc = float(txn["amount"])

        def to_local(entity: str) -> float:
            rate = rates.get((currency[entity], txn["period_key"]), 1.0)
            return amount_gc / rate if rate else amount_gc

        amount = to_local(seller)

        if txn["flow_type"] == "goods":
            seller_class, buyer_class = "revenue_trade", "cogs_standard"
            seller_account = router.account_for(seller, "revenue_trade", offset=3)
            buyer_account = router.account_for(buyer, "cogs_standard", offset=1)
        elif txn["flow_type"] == "services":
            seller_class = buyer_class = "ic_service"
            seller_account = router.account_for(seller, "other_opex", offset=5)
            buyer_account = router.account_for(buyer, "other_opex", offset=6)
        else:
            seller_class = buyer_class = "interest"
            seller_account = router.account_for(seller, "interest")
            buyer_account = router.account_for(buyer, "interest")

        buyer_amount = to_local(buyer)
        # DEFECT 9.2: one DE<->ES goods pair is off by 1.3% because the buyer books it
        # in the following period.
        if (
            mismatch
            and txn["flow_key"] == "GOODS_DE_ES"
            and int(txn["period_no"]) == 6
            and int(txn["fiscal_year"]) == int(transactions["fiscal_year"].min())
        ):
            buyer_amount = to_local(buyer) * (1.0 - 0.013)

        # DEFECT 9.3: one UK purchase from DE loses its partner tag and so escapes
        # automatic elimination.
        buyer_partner = buyer if not (
            missing_tag
            and txn["flow_key"] == "GOODS_DE_UK"
            and int(txn["period_no"]) == 9
        ) else None

        rows.append(
            _row(
                applicability, seller, primary_ledger[seller], seller_account, seller_class,
                period, -amount, currency[seller], f"ic:{txn['flow_key']}:seller",
                ic_partner=buyer,
            )
            | {"ic_txn_id": txn["ic_txn_id"], "ic_leg": "seller", "ic_flow": txn["flow_key"]}
        )
        rows.append(
            _row(
                applicability, buyer, primary_ledger[buyer], buyer_account, buyer_class,
                period, buyer_amount, currency[buyer], f"ic:{txn['flow_key']}:buyer",
                ic_partner=buyer_partner if buyer_partner is None else seller,
            )
            | {"ic_txn_id": txn["ic_txn_id"], "ic_leg": "buyer", "ic_flow": txn["flow_key"]}
        )

        if txn["flow_type"] == "goods":
            # Buying from the plant SUBSTITUTES for a third-party purchase; it does not
            # add to cost of sales on top of it. Without this credit the buyer's COGS
            # would be double-counted and its gross margin would collapse. The net
            # effect on the buyer is therefore the transfer-price MARGIN only, which is
            # exactly the amount consolidation has to eliminate.
            rows.append(
                _row(
                    applicability, buyer, primary_ledger[buyer],
                    router.account_for(buyer, "cogs_standard"), "cogs_standard",
                    period,
                    -float(txn["std_cost_component"])
                    / (rates.get((currency[buyer], txn["period_key"]), 1.0) or 1.0),
                    currency[buyer],
                    f"ic:{txn['flow_key']}:substitution",
                )
                | {
                    "ic_txn_id": txn["ic_txn_id"],
                    "ic_leg": "buyer_substitution",
                    "ic_flow": txn["flow_key"],
                }
            )

    return pl.DataFrame(rows)


def build_allocations(
    cfg: Config,
    applicability: Applicability,
    router: AccountRouter,
    seeds: SeedBank,
    gl: pl.DataFrame,
    cost_centers: pl.DataFrame,
    profit_centers: pl.DataFrame,
    calendar: pl.DataFrame,
    entities: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Paired charge-out / charge-in rows that sum to zero per period.

    The methodology change is deliberate (spec 7.4 / defect 9.6): HQ-G&A allocates on
    headcount in budget but on revenue share in actuals, so an allocation-methodology
    variance appears that no volume or price explanation covers.
    """
    rng = seeds.rng("facts", "allocations")
    periods = {r["period_key"]: r for r in calendar.to_dicts() if not r["is_period13"]}
    currency = {r["entity"]: r["currency"] for r in entities.to_dicts()}
    primary_ledger = {
        r["entity"]: ("IFRS" if r["parallel_ledgers"] else "LOCAL") for r in entities.to_dicts()
    }
    rule_change = cfg.defect_enabled("allocation_rule_change")

    joined_at = {r["entity"]: int(r["consolidated_from_month"]) for r in entities.to_dicts()}
    month_of = {r["period_key"]: int(r["month_index"]) for r in calendar.to_dicts()}
    pools = cost_centers.filter(pl.col("shared_pool").is_not_null()).to_dicts()
    pc_leaves = profit_centers.filter(pl.col("is_leaf"))["node_id"].to_list()
    if not pools or not pc_leaves:
        return pl.DataFrame(), pl.DataFrame()

    revenue_by_entity_period = {
        (r["entity"], r["period_key"]): float(r["revenue"])
        for r in gl.filter(pl.col("account_class") == "revenue_trade")
        .group_by(["entity", "period_key"])
        .agg((-pl.col("amount_lc").sum()).alias("revenue"))
        .to_dicts()
    }

    rules = []
    rows: list[dict] = []
    for pool in pools:
        entity = pool["entity"]
        driver = "machine_hours" if pool["shared_pool"] == "PLANT-RCK" else "revenue_share"
        rules.append(
            {
                "pool_cc": pool["node_id"],
                "pool_name": pool["shared_pool"],
                "entity": entity,
                "target": "profit_centers",
                "driver_actual": driver,
                "driver_budget": "headcount" if (rule_change and driver == "revenue_share") else driver,
                "method": "pro_rata",
                "note": (
                    "ACT uses revenue share, BUD uses headcount — allocation-methodology "
                    "variance by design"
                    if rule_change and driver == "revenue_share"
                    else None
                ),
            }
        )

        for period_key, period in sorted(periods.items()):
            if month_of.get(period_key, 0) < joined_at.get(entity, 0):
                continue
            pool_cost = float(rng.uniform(60000, 240000))
            weights = [float(rng.uniform(0.5, 2.0)) for _ in pc_leaves]
            total_weight = sum(weights)

            account = router.account_for(entity, "other_opex", offset=9)
            # Charge-out: the pool is credited.
            rows.append(
                _row(
                    applicability, entity, primary_ledger[entity], account, "allocation",
                    period, -pool_cost, currency[entity], f"allocation:{pool['shared_pool']}:out",
                    cost_center=pool["node_id"],
                )
                | {"allocation_pool": pool["shared_pool"], "allocation_leg": "charge_out"}
            )
            # Charge-in: the targets are debited, pro rata.
            for pc, weight in zip(pc_leaves, weights):
                share = pool_cost * weight / total_weight
                rows.append(
                    _row(
                        applicability, entity, primary_ledger[entity], account, "allocation",
                        period, share, currency[entity], f"allocation:{pool['shared_pool']}:in",
                        cost_center=pool["node_id"], profit_center=pc,
                    )
                    | {"allocation_pool": pool["shared_pool"], "allocation_leg": "charge_in"}
                )

    return pl.DataFrame(rows), pl.DataFrame(rules)


def build_eliminations(
    cfg: Config,
    transactions: pl.DataFrame,
    ic_rows: pl.DataFrame,
    local_to_group: pl.DataFrame,
    pack: dict,
    calendar: pl.DataFrame,
) -> pl.DataFrame:
    """ELIM rows at GROUP account level, generated from the same IC records.

    Only tagged legs eliminate: the UK purchase that lost its partner tag (defect 9.3)
    is deliberately left standing in the consolidation.
    """
    stock_share = pack["flows"]["ic_inventory"]["closing_stock_share"]
    group_ccy = "USD"

    rows = []
    for txn in transactions.to_dicts():
        amount = float(txn["amount"])
        flow_type = txn["flow_type"]
        if flow_type == "goods":
            seller_account, buyer_account = "G3100", "G4050"
        elif flow_type == "services":
            seller_account = buyer_account = "G6000"
        else:
            seller_account = buyer_account = "G7000"

        base = {
            "entity": "ELIM",
            "ledger": "IFRS",
            "scenario_key": "ACT",
            "period_key": txn["period_key"],
            "fiscal_year": txn["fiscal_year"],
            "period_no": txn["period_no"],
            "is_period13": False,
            "currency": group_ccy,
            "ic_txn_id": txn["ic_txn_id"],
            "ic_flow": txn["flow_key"],
        }
        # Reverse both legs.
        rows.append({**base, "account_code": seller_account, "account_class": "revenue_ic",
                     "amount_lc": round(amount, 2), "elimination_type": "ic_revenue"})
        rows.append({**base, "account_code": buyer_account, "account_class": "cogs_ic",
                     "amount_lc": round(-amount, 2), "elimination_type": "ic_cost"})

        # Unrealised profit on the margin still sitting in the buyer's stock.
        if flow_type == "goods" and float(txn["margin_component"]) > 0:
            share = float(stock_share.get(txn["buyer"], 0.0))
            unrealised = float(txn["margin_component"]) * share
            if unrealised > 0:
                rows.append(
                    {**base, "account_code": "G4000", "account_class": "cogs_standard",
                     "amount_lc": round(unrealised, 2),
                     "elimination_type": "unrealised_profit_in_inventory",
                     "nci_share": 0.20 if txn["buyer"] == "ES01" else 0.0}
                )

    return pl.DataFrame(rows, infer_schema_length=None)


def build_topside(cfg: Config, calendar: pl.DataFrame) -> pl.DataFrame:
    """Two manual group-only journals; one is intentionally never pushed down (9.4)."""
    unreconciled = cfg.defect_enabled("topside_unreconciled")
    fiscal_years = sorted(calendar["fiscal_year"].unique().to_list())
    target_fy = fiscal_years[-1] if fiscal_years else None
    if target_fy is None:
        return pl.DataFrame()

    last = calendar.filter(
        (pl.col("fiscal_year") == target_fy) & (pl.col("period_no") == 12)
    )
    if last.height == 0:
        last = calendar.tail(1)
    period = last.to_dicts()[0]

    rows = [
        {
            "entity": "GRP", "ledger": "IFRS", "scenario_key": "ACT",
            "account_code": "G6500", "account_class": "one_off",
            "period_key": period["period_key"], "fiscal_year": int(period["fiscal_year"]),
            "period_no": int(period["period_no"]), "is_period13": False,
            "amount_lc": 1_250_000.0, "currency": "USD",
            "is_topside": True, "pushed_to_local": True,
            "description": "Late audit adjustment - inventory provision",
        },
        {
            "entity": "GRP", "ledger": "IFRS", "scenario_key": "ACT",
            "account_code": "G6500", "account_class": "one_off",
            "period_key": period["period_key"], "fiscal_year": int(period["fiscal_year"]),
            "period_no": int(period["period_no"]), "is_period13": False,
            "amount_lc": 780_000.0, "currency": "USD",
            "is_topside": True, "pushed_to_local": not unreconciled,
            "description": (
                "Group legal provision - exists ONLY at GRP, never pushed to any local ledger"
                if unreconciled else "Group legal provision"
            ),
        },
    ]
    return pl.DataFrame(rows)


def build_nci(gl: pl.DataFrame, entities: pl.DataFrame, eliminations: pl.DataFrame) -> pl.DataFrame:
    """Non-controlling interest: 20% of ES01's net income, adjusted for its share of
    unrealised IC profit."""
    nci_pct = {
        r["entity"]: float(r["nci_pct"]) for r in entities.to_dicts() if float(r["nci_pct"]) > 0
    }
    if not nci_pct:
        return pl.DataFrame()

    # Group currency, not local: NCI is a consolidation figure, and summing a
    # subsidiary's EUR result into a USD consolidation silently understates it.
    amount_column = (
        "amount_gc_actual_rates" if "amount_gc_actual_rates" in gl.columns else "amount_lc"
    )
    # Net income means the P&L only. Once balance-sheet rows exist in fact_gl, summing
    # every class inflates "net income" by the whole balance sheet.
    result = (
        gl.filter(
            pl.col("entity").is_in(list(nci_pct))
            & (pl.col("account_class") != "balance_sheet")
            & (pl.col("ledger") != "TAX")
            & (pl.col("scenario_key") == "ACT")
        )
        .group_by(["entity", "fiscal_year"])
        .agg((-pl.col(amount_column).sum()).alias("net_income"))
        .sort(["entity", "fiscal_year"])
    )

    unrealised = pl.DataFrame()
    if eliminations.height and "nci_share" in eliminations.columns:
        unrealised = (
            eliminations.filter(pl.col("elimination_type") == "unrealised_profit_in_inventory")
            .group_by("fiscal_year")
            .agg((pl.col("amount_lc") * pl.col("nci_share").fill_null(0.0)).sum().alias("nci_unrealised"))
        )

    rows = []
    for row in result.to_dicts():
        pct = nci_pct[row["entity"]]
        adjustment = 0.0
        if unrealised.height:
            match = unrealised.filter(pl.col("fiscal_year") == row["fiscal_year"])
            if match.height:
                adjustment = float(match["nci_unrealised"][0])
        rows.append(
            {
                "entity": row["entity"],
                "fiscal_year": row["fiscal_year"],
                "nci_pct": pct,
                "entity_net_income": round(float(row["net_income"]), 2),
                "nci_before_adjustment": round(float(row["net_income"]) * pct, 2),
                "nci_unrealised_profit_adjustment": round(-adjustment, 2),
                "nci_net_income": round(float(row["net_income"]) * pct - adjustment, 2),
                "currency": "group",
                "gaap_concept": "NetIncomeLossAttributableToNoncontrollingInterest",
            }
        )
    return pl.DataFrame(rows)
