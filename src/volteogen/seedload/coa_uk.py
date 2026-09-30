"""UK nominal ledger (Sage-standard numbering), from the ``uk_coa`` seed package.

Deliberately the *shallow* chart in the group: a two-level category -> nominal tree
against SKR03's four levels and Business Central's six. UK01 therefore genuinely cannot
answer questions the other entities can, which is the point — the group's reporting
must cope with charts of different depth, not just different codes.

The seed is a real Python package (MIT), so it is imported rather than regex-scraped:
that keeps VAT treatment, HMRC box references and tags exactly as published.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .nodes import GROUP, POSTING, AccountNode

COA_ID = "SAGE50"

# Readable names for the seed's AccountType enum. These label the category layer; the
# categories themselves (and which nominal belongs to which) come from the seed.
_CATEGORY_LABELS = {
    "fixed_asset": "Fixed Assets",
    "current_asset": "Current Assets",
    "current_liability": "Current Liabilities",
    "long_term_liability": "Long Term Liabilities",
    "equity": "Capital and Reserves",
    "income": "Income",
    "direct_expense": "Direct Expenses",
    "overhead": "Overheads",
    "tax": "Taxation",
    "control": "Control Accounts",
}

_BALANCE_SHEET_TYPES = {
    "fixed_asset",
    "current_asset",
    "current_liability",
    "long_term_liability",
    "equity",
    "control",
}


def _import_seed_package(seeds_dir: Path) -> ModuleType:
    """Import ``seeds/uk_coa`` without requiring it to be installed."""
    pkg_dir = Path(seeds_dir) / "uk_coa"
    if not (pkg_dir / "data.py").exists():
        raise FileNotFoundError(
            f"missing UK CoA seed package at {pkg_dir}; run seeds/fetch_seeds.sh "
            "(see seeds/SOURCES.md)"
        )
    seeds_str = str(Path(seeds_dir).resolve())
    if seeds_str not in sys.path:
        sys.path.insert(0, seeds_str)
    spec = importlib.util.find_spec("uk_coa.chart")
    if spec is None:  # pragma: no cover - defensive
        raise ImportError("could not import uk_coa.chart from the seeds directory")
    return importlib.import_module("uk_coa.chart")


def load_uk_coa(seeds_dir: Path) -> list[AccountNode]:
    chart_mod = _import_seed_package(seeds_dir)
    chart = chart_mod.ChartOfAccounts()
    accounts = list(chart)
    if not accounts:
        raise ValueError("uk_coa seed package yielded zero accounts")

    root_id = f"{COA_ID}:ROOT"
    nodes: list[AccountNode] = [
        AccountNode(
            coa_id=COA_ID,
            node_id=root_id,
            name="UK Nominal Ledger",
            node_type=GROUP,
            sort_order=-1,
            attrs={"structural": True},
        )
    ]

    used_types = sorted({a.type.value for a in accounts})
    for order, type_key in enumerate(used_types):
        nodes.append(
            AccountNode(
                coa_id=COA_ID,
                node_id=f"{COA_ID}:CAT:{type_key}",
                parent_id=root_id,
                name=_CATEGORY_LABELS.get(type_key, type_key.replace("_", " ").title()),
                node_type=GROUP,
                income_balance="BS" if type_key in _BALANCE_SHEET_TYPES else "PL",
                category=type_key,
                sort_order=order,
                attrs={"category_layer": True},
            )
        )

    for order, account in enumerate(sorted(accounts, key=lambda a: a.code)):
        type_key = account.type.value
        nodes.append(
            AccountNode(
                coa_id=COA_ID,
                node_id=f"{COA_ID}:{account.code}",
                parent_id=f"{COA_ID}:CAT:{type_key}",
                code=str(account.code),
                name=account.name,
                node_type=POSTING,
                income_balance="BS" if type_key in _BALANCE_SHEET_TYPES else "PL",
                category=type_key,
                subcategory=account.vat.value,
                sort_order=order,
                attrs={
                    "vat_rate": account.vat.value,
                    "hmrc_box": account.hmrc_box,
                    "tags": list(account.tags),
                    "debit_increase": account.debit_increase,
                },
            )
        )

    return nodes
