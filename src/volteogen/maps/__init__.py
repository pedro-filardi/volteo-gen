"""Group mappings, all as versioned data tables (spec 7.1).

Four layers, each a real table rather than code:

``map_local_to_group``  every local account -> a group account. Many-to-one, and the
                        fan-in is severe: SKR03's 1,274 accounts collapse onto a few
                        hundred group lines.
``map_group_to_gaap``   group account -> a US-GAAP concept that must exist in the
                        seeded taxonomy (validated, not assumed).
``map_management``      (group account x cost-centre function) -> management line. This
                        is THE by-nature -> by-function pivot: the same salary account
                        becomes "COGS - labour", "R&D" or "SG&A" depending only on which
                        cost centre it was posted to.
``map_statutory_tags``  local statutory tags -> local statement lines, using the REAL
                        HGB tags shipped in the SKR03 seed.

Coverage gaps are deliberate: the management view stops at EBIT, so interest and tax
have no management line at all, and management-only lines ("HQ allocation", "EBITDA")
have no account behind them.
"""

from __future__ import annotations

import json

import polars as pl

from ..seedload import SeedBundle

GROUP_COA_ID = "GROUP"

# Group chart: one line per (account_class, purpose). Deliberately a simplified
# skeleton — a group chart exists to consolidate, not to bookkeep.
GROUP_LINES = [
    ("G3000", "Revenue - trade", "revenue_trade", "PL", "Revenue"),
    ("G3100", "Revenue - intercompany", "revenue_ic", "PL", "Revenue"),
    ("G3200", "Revenue reductions - rebates", "revenue_reduction", "PL", "Revenue"),
    ("G3210", "Revenue reductions - MDF", "revenue_reduction", "PL", "Revenue"),
    ("G3220", "Revenue reductions - returns", "revenue_reduction", "PL", "Revenue"),
    ("G3230", "Revenue reductions - price protection", "revenue_reduction", "PL", "Revenue"),
    ("G4000", "COGS - standard cost", "cogs_standard", "PL", "COGS"),
    ("G4050", "COGS - intercompany", "cogs_ic", "PL", "COGS"),
    ("G4100", "COGS - production variances", "cogs_variance", "PL", "COGS"),
    ("G4200", "Freight out", "freight_out", "PL", "COGS"),
    ("G5000", "Personnel expense", "payroll", "PL", "Opex"),
    ("G5200", "Facilities", "facilities", "PL", "Opex"),
    ("G5400", "Marketing", "marketing", "PL", "Opex"),
    ("G5600", "Other operating expense", "other_opex", "PL", "Opex"),
    ("G5800", "Depreciation and amortisation", "depreciation", "PL", "Opex"),
    ("G5900", "Allocated costs", "allocation", "PL", "Opex"),
    ("G6000", "Management fees - intercompany", "ic_service", "PL", "Opex"),
    ("G6500", "One-off items", "one_off", "PL", "Opex"),
    ("G7000", "Interest", "interest", "PL", "Financial"),
    ("G7200", "FX result", "fx_result", "PL", "Financial"),
    ("G7500", "Income tax", "tax", "PL", "Tax"),
    ("G1000", "Balance sheet - assets and liabilities", "balance_sheet", "BS", "BalanceSheet"),
]

# Group line -> US-GAAP concept. Every target is asserted to exist in the seeded
# StatementOfIncome network at load time.
GAAP_CONCEPT_BY_GROUP = {
    "G3000": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "G3100": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "G3200": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "G3210": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "G3220": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "G3230": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "G4000": "CostOfGoodsAndServicesSold",
    "G4050": "CostOfGoodsAndServicesSold",
    "G4100": "CostOfGoodsAndServicesSold",
    "G4200": "CostOfGoodsAndServicesSold",
    "G5000": "GeneralAndAdministrativeExpense",
    "G5200": "GeneralAndAdministrativeExpense",
    "G5400": "SellingAndMarketingExpense",
    "G5600": "OtherCostAndExpenseOperating",
    "G5800": "DepreciationAndAmortization",
    "G5900": "OtherCostAndExpenseOperating",
    "G6000": "GeneralAndAdministrativeExpense",
    "G6500": "OtherNonoperatingIncomeExpense",
    "G7000": "InterestExpenseNonoperating",
    "G7200": "OtherNonoperatingIncomeExpense",
    "G7500": "IncomeTaxExpenseBenefit",
}

# THE pivot: the same nature-account lands on a different management line depending on
# the FUNCTION of the cost centre it was posted to.
MANAGEMENT_BY_FUNCTION = {
    ("payroll", "PRD"): "COGS - labour",
    ("payroll", "LOG"): "Logistics",
    ("payroll", "RND"): "R&D",
    ("payroll", "SLS"): "Selling",
    ("payroll", "MKT"): "Marketing",
    ("payroll", "GA"): "G&A",
    ("facilities", "PRD"): "COGS - overhead",
    ("facilities", "LOG"): "Logistics",
    ("facilities", "GA"): "G&A",
    ("facilities", "RND"): "R&D",
    ("facilities", "SLS"): "Selling",
    ("facilities", "MKT"): "Marketing",
    ("other_opex", "PRD"): "COGS - overhead",
    ("other_opex", "LOG"): "Logistics",
    ("other_opex", "RND"): "R&D",
    ("other_opex", "SLS"): "Selling",
    ("other_opex", "MKT"): "Marketing",
    ("other_opex", "GA"): "G&A",
    ("depreciation", "PRD"): "COGS - overhead",
    ("depreciation", "GA"): "G&A",
    ("depreciation", "RND"): "R&D",
    ("depreciation", "LOG"): "Logistics",
    ("depreciation", "SLS"): "Selling",
    ("depreciation", "MKT"): "Marketing",
}

# Classes that need no cost centre to reach a management line.
MANAGEMENT_BY_CLASS = {
    "revenue_trade": "Net revenue",
    "revenue_ic": "Net revenue",
    "revenue_reduction": "Net revenue",
    "cogs_standard": "COGS - material",
    "cogs_ic": "COGS - material",
    "cogs_variance": "COGS - variances",
    "freight_out": "Logistics",
    "marketing": "Marketing",
    "allocation": "HQ allocation",
    "ic_service": "HQ allocation",
    "one_off": "One-offs",
}

# Deliberate coverage gap: the management view stops at EBIT.
MANAGEMENT_EXCLUDED_CLASSES = {"interest", "tax", "fx_result", "balance_sheet"}

# Management-only lines with no account behind them.
MANAGEMENT_ONLY_LINES = ["EBITDA", "EBIT", "HQ allocation", "One-offs"]

FUNCTIONS = ("PRD", "SLS", "MKT", "RND", "GA", "LOG")


def build_group_coa() -> pl.DataFrame:
    rows = []
    for code, name, account_class, income_balance, section in GROUP_LINES:
        rows.append(
            {
                "coa_id": GROUP_COA_ID,
                "node_id": f"{GROUP_COA_ID}:{code}",
                "code": code,
                "name": name,
                "account_class": account_class,
                "income_balance": income_balance,
                "section": section,
            }
        )
    return pl.DataFrame(rows)


def build_local_to_group(
    accounts: pl.DataFrame, group_coa: pl.DataFrame, valid_from: str = "1900-01"
) -> pl.DataFrame:
    """Every local posting account -> exactly one group account, versioned.

    Many-to-one by construction: the group chart has ~20 P&L lines, so a 1,274-account
    local chart fans in hard. Reduction accounts are spread across their four group
    lines by kind so the gross-to-net waterfall survives consolidation.
    """
    by_class: dict[str, list[str]] = {}
    for row in group_coa.to_dicts():
        by_class.setdefault(row["account_class"], []).append(row["code"])

    postings = accounts.filter(pl.col("node_type") == "posting")
    rows = []
    for index, row in enumerate(postings.to_dicts()):
        account_class = row["account_class"]
        targets = by_class.get(account_class) or by_class.get("other_opex")
        if not targets:
            continue
        target = targets[index % len(targets)] if len(targets) > 1 else targets[0]
        rows.append(
            {
                "coa_id": row["coa_id"],
                "local_account": row["code"],
                "local_name": row["name"],
                "account_class": account_class,
                "group_account": target,
                "valid_from": valid_from,
                "valid_to": None,
                "is_defect": False,
            }
        )
    return pl.DataFrame(rows)


def build_group_to_gaap(group_coa: pl.DataFrame, seeds: SeedBundle) -> pl.DataFrame:
    """Group account -> US-GAAP concept, validated against the seeded taxonomy."""
    known = set(seeds.gaap_nodes["code"].drop_nulls().to_list())
    rows = []
    missing = []
    for row in group_coa.to_dicts():
        concept = GAAP_CONCEPT_BY_GROUP.get(row["code"])
        if concept is None:
            continue
        if concept not in known:
            missing.append((row["code"], concept))
            continue
        rows.append(
            {
                "group_account": row["code"],
                "group_name": row["name"],
                "gaap_concept": concept,
                "network": "StatementOfIncome",
            }
        )
    if missing:
        raise ValueError(
            "map_group_to_gaap targets concepts absent from the seeded US-GAAP "
            f"taxonomy: {missing[:5]}"
        )
    if not rows:
        raise ValueError("map_group_to_gaap produced no rows")
    return pl.DataFrame(rows)


def build_management_map(group_coa: pl.DataFrame) -> pl.DataFrame:
    """(group account x cost-centre function) -> management line.

    The gaps are the interesting part: interest and tax appear nowhere, because the
    management P&L stops at EBIT.
    """
    rows = []
    for row in group_coa.to_dicts():
        account_class = row["account_class"]
        if account_class in MANAGEMENT_EXCLUDED_CLASSES:
            rows.append(
                {
                    "group_account": row["code"],
                    "cc_function": "~NA~",
                    "management_line": None,
                    "is_excluded": True,
                    "note": "management view stops at EBIT",
                }
            )
            continue
        if account_class in MANAGEMENT_BY_CLASS:
            rows.append(
                {
                    "group_account": row["code"],
                    "cc_function": "~ANY~",
                    "management_line": MANAGEMENT_BY_CLASS[account_class],
                    "is_excluded": False,
                    "note": None,
                }
            )
            continue
        for function in FUNCTIONS:
            line = MANAGEMENT_BY_FUNCTION.get((account_class, function))
            if line is None:
                continue
            rows.append(
                {
                    "group_account": row["code"],
                    "cc_function": function,
                    "management_line": line,
                    "is_excluded": False,
                    "note": "by-nature to by-function pivot",
                }
            )
    return pl.DataFrame(rows)


def build_statutory_tags(accounts: pl.DataFrame) -> pl.DataFrame:
    """Local statutory tags, taken verbatim from the seeds.

    SKR03 ships the REAL HGB tag ids; the Sage seed ships VAT treatment and HMRC box
    references. Nothing here is invented.
    """
    rows = []
    for row in accounts.filter(pl.col("node_type") == "posting").to_dicts():
        try:
            attrs = json.loads(row["attrs"]) if row["attrs"] else {}
        except json.JSONDecodeError:
            attrs = {}
        for tag in attrs.get("tag_ids") or []:
            rows.append(
                {
                    "coa_id": row["coa_id"],
                    "local_account": row["code"],
                    "tag_system": "HGB" if row["coa_id"] == "SKR03" else row["coa_id"],
                    "tag_id": tag,
                    "statement_line": tag.split(".")[-1],
                }
            )
        if attrs.get("hmrc_box"):
            rows.append(
                {
                    "coa_id": row["coa_id"],
                    "local_account": row["code"],
                    "tag_system": "HMRC",
                    "tag_id": attrs["hmrc_box"],
                    "statement_line": attrs["hmrc_box"],
                }
            )
        if attrs.get("vat_rate"):
            rows.append(
                {
                    "coa_id": row["coa_id"],
                    "local_account": row["code"],
                    "tag_system": "UK_VAT",
                    "tag_id": attrs["vat_rate"],
                    "statement_line": attrs["vat_rate"],
                }
            )
    if not rows:
        raise ValueError("no statutory tags found in the seeds — check the SKR03 loader")
    return pl.DataFrame(rows)


def build_group_coa_hierarchy(group_coa: pl.DataFrame) -> pl.DataFrame:
    """The group chart as a parent-child tree: root -> section -> account.

    The group chart is authored flat (one row per line with a `section`), but spec 8
    wants a bridge over it like every other tree, so the section layer is materialised
    as real nodes rather than left as a string attribute.
    """
    rows = [
        {"node_id": "GROUP", "parent_id": None, "name": "Group chart of accounts",
         "node_type": "root", "code": None, "section": None}
    ]
    for section in sorted(group_coa["section"].unique().to_list()):
        rows.append(
            {"node_id": f"GROUP:{section}", "parent_id": "GROUP", "name": section,
             "node_type": "section", "code": None, "section": section}
        )
    for row in group_coa.to_dicts():
        rows.append(
            {"node_id": f"GROUP:{row['section']}:{row['code']}",
             "parent_id": f"GROUP:{row['section']}", "name": row["name"],
             "node_type": "account", "code": row["code"], "section": row["section"]}
        )
    frame = pl.DataFrame(rows)
    has_child = set(frame["parent_id"].drop_nulls().to_list())
    depth = {"GROUP": 0}
    for row in frame.to_dicts():
        if row["parent_id"]:
            depth[row["node_id"]] = depth.get(row["parent_id"], 0) + 1
    return frame.with_columns(
        pl.col("node_id").is_in(list(has_child)).not_().alias("is_leaf"),
        pl.col("node_id").replace_strict(depth, default=0).alias("depth"),
    )


def build_management_hierarchy(management_map: pl.DataFrame) -> pl.DataFrame:
    """The management P&L as a tree: root -> block -> line.

    Management lines are the by-function view, so they need their own hierarchy —
    including the management-ONLY lines (EBITDA, EBIT, HQ allocation, One-offs) that
    have no account behind them at all.
    """
    blocks = {
        "Net revenue": "Revenue",
        "COGS - material": "COGS", "COGS - labour": "COGS",
        "COGS - overhead": "COGS", "COGS - variances": "COGS",
        "Logistics": "Opex", "Selling": "Opex", "Marketing": "Opex",
        "R&D": "Opex", "G&A": "Opex", "HQ allocation": "Opex", "One-offs": "Opex",
        "EBITDA": "Result", "EBIT": "Result",
    }
    lines = sorted(
        set(management_map["management_line"].drop_nulls().to_list()) | set(MANAGEMENT_ONLY_LINES)
    )
    rows = [{"node_id": "MGMT", "parent_id": None, "name": "Management P&L",
             "node_type": "root", "is_management_only": False}]
    for block in sorted({blocks.get(line, "Other") for line in lines}):
        rows.append({"node_id": f"MGMT:{block}", "parent_id": "MGMT", "name": block,
                     "node_type": "block", "is_management_only": False})
    for line in lines:
        block = blocks.get(line, "Other")
        rows.append({
            "node_id": f"MGMT:{block}:{line}", "parent_id": f"MGMT:{block}", "name": line,
            "node_type": "line", "is_management_only": line in MANAGEMENT_ONLY_LINES,
        })
    frame = pl.DataFrame(rows)
    has_child = set(frame["parent_id"].drop_nulls().to_list())
    depth = {"MGMT": 0}
    for row in frame.to_dicts():
        if row["parent_id"]:
            depth[row["node_id"]] = depth.get(row["parent_id"], 0) + 1
    return frame.with_columns(
        pl.col("node_id").is_in(list(has_child)).not_().alias("is_leaf"),
        pl.col("node_id").replace_strict(depth, default=0).alias("depth"),
    )
