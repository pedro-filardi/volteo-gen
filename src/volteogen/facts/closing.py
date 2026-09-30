"""Close mechanics: P13 adjustments, one-offs, the balance sheet, and the TAX ledger.

These are what make the ledger look *closed* rather than merely accumulated:

**P13** (spec 4.6) — a real posting period that is not a month. Close adjustments land
here, so any consumer summing twelve months silently misses them and only a fiscal-year
total is right.

**One-offs** (spec 6.4) — flagged, explainable events rather than noise: DE01's FY25
restructuring, the professional fees US01 paid to acquire UK01, and an insurance claim.
They are *measured* by the calibrator but never *scaled* by it, so they land inside the
configured margin bands instead of on top of them (spec 6.5).

**Balance sheet** (spec 12) — plausible balances, with the one identity that must hold:
retained earnings roll forward by net income. No full articulation is attempted.

**TAX ledger** (spec 12) — simple deltas against the primary ledger on depreciation and
provisions, only for entities whose ERP supports parallel ledgers.
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from ..dims.applicability import Applicability
from ..rng import SeedBank
from .account_router import AccountRouter
from .gl import _row

P13_SHARE_OF_ANNUAL_EXPENSE = 0.003


def build_period13_adjustments(
    cfg: Config,
    applicability: Applicability,
    router: AccountRouter,
    seeds: SeedBank,
    gl: pl.DataFrame,
    calendar: pl.DataFrame,
    entities: pl.DataFrame,
    cost_centers: pl.DataFrame,
) -> pl.DataFrame:
    """Close adjustments posted to P13, concentrated in accruals and provisions."""
    # Spec 9.9: posting close adjustments into P13 rather than into a month IS the
    # defect — any consumer summing twelve months silently misses them.
    if not bool(cfg.get("period.period13", True)):
        return pl.DataFrame()
    if not cfg.defect_enabled("late_period13_adjustments"):
        return pl.DataFrame()
    rng = seeds.rng("facts", "period13")

    p13 = calendar.filter(pl.col("is_period13"))
    if p13.height == 0:
        return pl.DataFrame()

    currency = {r["entity"]: r["currency"] for r in entities.to_dicts()}
    primary_ledger = {
        r["entity"]: ("IFRS" if r["parallel_ledgers"] else "LOCAL") for r in entities.to_dicts()
    }
    cc_by_entity: dict[str, list[str]] = {}
    for row in cost_centers.filter(pl.col("is_leaf")).to_dicts():
        cc_by_entity.setdefault(row["entity"], []).append(row["node_id"])

    expense = (
        gl.filter(
            pl.col("account_class").is_in(
                ["payroll", "facilities", "marketing", "other_opex", "depreciation"]
            )
        )
        .group_by(["entity", "fiscal_year"])
        .agg(pl.col("amount_lc").sum().alias("annual_expense"))
    )
    lookup = {(r["entity"], r["fiscal_year"]): float(r["annual_expense"]) for r in expense.to_dicts()}

    rows = []
    for period in p13.to_dicts():
        for entity, ccs in sorted(cc_by_entity.items()):
            annual = lookup.get((entity, int(period["fiscal_year"])), 0.0)
            if annual <= 0:
                continue
            total = annual * P13_SHARE_OF_ANNUAL_EXPENSE
            # Split across a couple of accrual/provision postings, signs varying — a
            # close produces both charges and releases.
            for index in range(2):
                amount = total * float(rng.uniform(0.3, 0.7)) * (1.0 if index == 0 else -0.45)
                rows.append(
                    _row(
                        applicability, entity, primary_ledger[entity],
                        router.account_for(entity, "other_opex", offset=11 + index),
                        "other_opex", period, amount, currency[entity],
                        "close:period13", cost_center=str(rng.choice(ccs)),
                    )
                )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def build_one_offs(
    cfg: Config,
    applicability: Applicability,
    router: AccountRouter,
    seeds: SeedBank,
    gl: pl.DataFrame,
    calendar: pl.DataFrame,
    entities: pl.DataFrame,
    cost_centers: pl.DataFrame,
) -> pl.DataFrame:
    """Named, explainable one-off events (spec 6.4)."""
    rng = seeds.rng("facts", "one_offs")
    periods = {r["period_key"]: r for r in calendar.to_dicts()}
    months = calendar.filter(~pl.col("is_period13")).sort("month_index").to_dicts()
    if not months:
        return pl.DataFrame()

    currency = {r["entity"]: r["currency"] for r in entities.to_dicts()}
    primary_ledger = {
        r["entity"]: ("IFRS" if r["parallel_ledgers"] else "LOCAL") for r in entities.to_dicts()
    }
    present = set(gl["entity"].unique().to_list())
    cc_by_entity: dict[str, list[str]] = {}
    for row in cost_centers.filter(pl.col("is_leaf")).to_dicts():
        cc_by_entity.setdefault(row["entity"], []).append(row["node_id"])

    fiscal_years = sorted({int(m["fiscal_year"]) for m in months})
    second_fy = fiscal_years[1] if len(fiscal_years) > 1 else fiscal_years[0]

    revenue = (
        gl.filter(pl.col("account_class") == "revenue_trade")
        .group_by(["entity", "fiscal_year"])
        .agg((-pl.col("amount_lc").sum()).alias("revenue"))
    )
    revenue_by_fy = {
        (r["entity"], int(r["fiscal_year"])): float(r["revenue"]) for r in revenue.to_dicts()
    }

    def annual_revenue(entity: str, fiscal_year: int) -> float:
        """Revenue of that entity in that fiscal year — never the whole history."""
        if (entity, fiscal_year) in revenue_by_fy:
            return revenue_by_fy[(entity, fiscal_year)]
        values = [v for (e, _), v in revenue_by_fy.items() if e == entity]
        return sum(values) / len(values) if values else 0.0

    rows = []

    def emit(entity, period, amount, label, cost_center=None):
        rows.append(
            _row(
                applicability, entity, primary_ledger[entity],
                router.account_for(entity, "other_opex", offset=13), "one_off",
                period, amount, currency[entity], f"one_off:{label}",
                cost_center=cost_center, is_one_off=True,
            )
        )

    # DE01 restructuring: two consecutive loss-making months in the second fiscal year.
    if "DE01" in present:
        restructuring_months = [m for m in months if int(m["fiscal_year"]) == second_fy][2:4]
        base = annual_revenue("DE01", second_fy) * 0.02
        for month in restructuring_months:
            emit("DE01", periods[month["period_key"]], base * float(rng.uniform(0.8, 1.2)),
                 "de_restructuring_severance", str(rng.choice(cc_by_entity.get("DE01", [None]))))

    # US01 pays professional fees to acquire UK01, in the acquisition month.
    if "US01" in present and "UK01" in present:
        uk = entities.filter(pl.col("entity") == "UK01")
        acq_index = int(uk["consolidated_from_month"][0]) if uk.height else 0
        acq = next((m for m in months if int(m["month_index"]) == acq_index), months[0])
        emit("US01", periods[acq["period_key"]],
             annual_revenue("US01", int(acq["fiscal_year"])) * 0.004,
             "uk_acquisition_professional_fees", str(rng.choice(cc_by_entity.get("US01", [None]))))

    # An insurance claim: income, so a negative expense.
    if "ES01" in present:
        claim = months[len(months) // 2]
        emit("ES01", periods[claim["period_key"]],
             -annual_revenue("ES01", int(claim["fiscal_year"])) * 0.006,
             "insurance_claim_income", str(rng.choice(cc_by_entity.get("ES01", [None]))))

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def build_balance_sheet(
    cfg: Config,
    applicability: Applicability,
    router: AccountRouter,
    seeds: SeedBank,
    gl: pl.DataFrame,
    calendar: pl.DataFrame,
    entities: pl.DataFrame,
) -> pl.DataFrame:
    """Plausible balances whose retained earnings roll forward by net income.

    Spec 12 explicitly makes full articulation a non-goal, so working-capital balances
    are driven off revenue and cost of sales rather than derived from cash movements.
    The ONE identity that must hold is retained earnings: closing = opening + result.
    """
    rng = seeds.rng("facts", "balance_sheet")
    months = calendar.filter(~pl.col("is_period13")).sort("month_index").to_dicts()
    currency = {r["entity"]: r["currency"] for r in entities.to_dicts()}
    primary_ledger = {
        r["entity"]: ("IFRS" if r["parallel_ledgers"] else "LOCAL") for r in entities.to_dicts()
    }

    monthly = (
        gl.group_by(["entity", "period_key"])
        .agg(
            (-pl.col("amount_lc").filter(pl.col("account_class") == "revenue_trade").sum()).alias("revenue"),
            pl.col("amount_lc").filter(pl.col("account_class") == "cogs_standard").sum().alias("cogs"),
            (-pl.col("amount_lc").sum()).alias("result"),
        )
    )
    facts = {(r["entity"], r["period_key"]): r for r in monthly.to_dicts()}

    # Working-capital intensity, in months of revenue / cost of sales.
    profile = {"receivables": 1.8, "inventory": 2.1, "payables": 1.4}

    rows: list[dict] = []
    retained: dict[str, float] = {}

    for entity in sorted(currency):
        opening_equity = None
        for month in months:
            key = (entity, month["period_key"])
            data = facts.get(key)
            if data is None:
                continue
            revenue = float(data["revenue"] or 0.0)
            cogs = float(data["cogs"] or 0.0)
            result = float(data["result"] or 0.0)
            if opening_equity is None:
                opening_equity = max(revenue, 1.0) * 3.0
                retained[entity] = 0.0

            period = month
            receivables = revenue * profile["receivables"] * float(rng.uniform(0.92, 1.08))
            inventory = cogs * profile["inventory"] * float(rng.uniform(0.9, 1.1))
            payables = -cogs * profile["payables"] * float(rng.uniform(0.9, 1.1))

            # Retained earnings roll forward — the identity the validator checks.
            retained[entity] += result
            cash = -(receivables + inventory + payables + opening_equity + retained[entity])

            for code_offset, account_class, amount, label in (
                (0, "balance_sheet", receivables, "receivables"),
                (1, "balance_sheet", inventory, "inventory"),
                (2, "balance_sheet", payables, "payables"),
                (3, "balance_sheet", cash, "cash"),
                (4, "balance_sheet", -opening_equity, "share_capital"),
                (5, "balance_sheet", -retained[entity], "retained_earnings"),
            ):
                rows.append(
                    _row(
                        applicability, entity, primary_ledger[entity],
                        router.account_for(entity, "balance_sheet", offset=code_offset),
                        account_class, period, amount, currency[entity],
                        f"balance_sheet:{label}",
                    )
                )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def build_tax_ledger(
    cfg: Config,
    applicability: Applicability,
    router: AccountRouter,
    seeds: SeedBank,
    gl: pl.DataFrame,
    entities: pl.DataFrame,
    calendar: pl.DataFrame,
) -> pl.DataFrame:
    """TAX ledger as DELTAS against the primary ledger (spec 12).

    Only entities whose ERP supports parallel ledgers get one, and the deltas are
    confined to depreciation and provisions — tax-basis differences, not a second full
    set of books. A consumer must therefore ADD the TAX rows to the primary ledger, not
    substitute them, which is a trap worth having.
    """
    rng = seeds.rng("facts", "tax_ledger")
    configured = list(cfg.get("scenarios.ledgers"))
    if "TAX" not in configured:
        return pl.DataFrame()

    parallel = {
        r["entity"] for r in entities.to_dicts() if r.get("parallel_ledgers")
    }
    if not parallel:
        return pl.DataFrame()

    periods = {r["period_key"]: r for r in calendar.to_dicts()}
    currency = {r["entity"]: r["currency"] for r in entities.to_dicts()}

    source = gl.filter(
        pl.col("entity").is_in(list(parallel))
        & pl.col("account_class").is_in(["depreciation", "other_opex"])
        & (~pl.col("is_period13"))
    ).group_by(["entity", "period_key", "account_class", "account_code"]).agg(
        pl.col("amount_lc").sum().alias("amount")
    # This loop draws from the RNG per row, and polars group_by does not guarantee row
    # order, so an unsorted result makes the build irreproducible.
    ).sort(["entity", "period_key", "account_class", "account_code"])

    rows = []
    for row in source.to_dicts():
        # Tax depreciation is accelerated; provisions are partly non-deductible.
        factor = 0.18 if row["account_class"] == "depreciation" else -0.06
        delta = float(row["amount"]) * factor * float(rng.uniform(0.7, 1.3))
        if abs(delta) < 1.0:
            continue
        rows.append(
            _row(
                applicability, row["entity"], "TAX", row["account_code"],
                row["account_class"], periods[row["period_key"]], delta,
                currency[row["entity"]], "tax_ledger:delta",
            )
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()
