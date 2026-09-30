"""Digit-prefix charts of accounts: SKR03 (Germany) and PGC (Spain).

Both are Odoo locale exports: a flat list of accounts whose hierarchy is carried by the
account *code* itself. SKR03 is uniformly 4-digit; PGC is genuinely ragged (3-, 4-, 5-
and 6-digit accounts coexist), and that raggedness is preserved rather than padded.

Group nodes are only created for prefixes that actually occur in the seed, so the tree
keeps the upstream's real lumpiness — a class with one subgroup sits next to one with
forty.

Group *names*:
  * PGC ships real group names (``account.group-es_common.csv``, keyed by
    ``code_prefix_start``) and those are used verbatim.
  * SKR03 ships no group table, so group nodes are labelled by their code prefix and
    flagged ``synthetic_label: true`` in ``attrs``. No German account names are
    invented — the German statutory hierarchy comes from the real HGB ``tag_ids``
    (see :mod:`volteogen.maps.statutory_tags`), not from these structural nodes.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .nodes import GROUP, POSTING, AccountNode

SKR03_COA_ID = "SKR03"
PGC_COA_ID = "PGC"

# Odoo account_type -> BS/PL. Odoo's own vocabulary, mapped not invented.
_INCOME_BALANCE = {
    "asset_receivable": "BS",
    "asset_cash": "BS",
    "asset_current": "BS",
    "asset_non_current": "BS",
    "asset_prepayments": "BS",
    "asset_fixed": "BS",
    "liability_payable": "BS",
    "liability_credit_card": "BS",
    "liability_current": "BS",
    "liability_non_current": "BS",
    "equity": "BS",
    "equity_unaffected": "BS",
    "income": "PL",
    "income_other": "PL",
    "expense": "PL",
    "expense_depreciation": "PL",
    "expense_direct_cost": "PL",
    "off_balance": "NA",
}


def _prefixes(code: str, max_levels: int) -> list[str]:
    """Ancestor code prefixes, shortest first, excluding the code itself."""
    return [code[:n] for n in range(1, min(max_levels, len(code) - 1) + 1)]


def _load_group_names(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    names: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prefix = (row.get("code_prefix_start") or "").strip()
            if prefix:
                names[prefix] = (row.get("name") or "").strip()
    return names


def _load_group_names_local(path: Path | None, local_col: str) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    names: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prefix = (row.get("code_prefix_start") or "").strip()
            if prefix:
                names[prefix] = (row.get(local_col) or row.get("name") or "").strip()
    return names


def load_digit_coa(
    csv_path: Path,
    coa_id: str,
    local_name_col: str,
    max_group_levels: int = 3,
    groups_path: Path | None = None,
    group_local_col: str | None = None,
    root_name: str = "Chart of accounts",
) -> list[AccountNode]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"missing CoA seed {csv_path.name}; run seeds/fetch_seeds.sh "
            "(see seeds/SOURCES.md)"
        )

    with csv_path.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if (r.get("code") or "").strip()]
    if not rows:
        raise ValueError(f"no accounts parsed from {csv_path}")

    group_names = _load_group_names(groups_path)
    group_names_local = (
        _load_group_names_local(groups_path, group_local_col) if group_local_col else {}
    )

    root_id = f"{coa_id}:ROOT"
    nodes: list[AccountNode] = [
        AccountNode(
            coa_id=coa_id,
            node_id=root_id,
            name=root_name,
            node_type=GROUP,
            sort_order=-1,
            attrs={"structural": True},
        )
    ]

    # Only materialise prefixes that real accounts actually sit under.
    needed: set[str] = set()
    for row in rows:
        needed.update(_prefixes(row["code"].strip(), max_group_levels))

    for prefix in sorted(needed):
        # A group's parents are its own shorter prefixes — never itself.
        parent_prefixes = [prefix[:n] for n in range(1, len(prefix)) if prefix[:n] in needed]
        parent_id = (
            f"{coa_id}:G:{parent_prefixes[-1]}" if parent_prefixes else root_id
        )
        synthetic = prefix not in group_names
        nodes.append(
            AccountNode(
                coa_id=coa_id,
                node_id=f"{coa_id}:G:{prefix}",
                parent_id=parent_id,
                code=prefix,
                name=group_names.get(prefix) or f"{prefix}* accounts",
                name_local=group_names_local.get(prefix) or group_names.get(prefix),
                node_type=GROUP,
                sort_order=int(prefix.ljust(6, "0")) if prefix.isdigit() else 0,
                attrs={"prefix": prefix, "synthetic_label": synthetic},
            )
        )

    seen: set[str] = set()
    for order, row in enumerate(rows):
        code = row["code"].strip()
        if code in seen:
            continue
        seen.add(code)
        parent_prefixes = [p for p in _prefixes(code, max_group_levels) if p in needed]
        parent_id = f"{coa_id}:G:{parent_prefixes[-1]}" if parent_prefixes else root_id
        account_type = (row.get("account_type") or "").strip()
        tags = [t.strip() for t in (row.get("tag_ids") or "").split(",") if t.strip()]
        taxes = [t.strip() for t in (row.get("tax_ids") or "").split(",") if t.strip()]
        nodes.append(
            AccountNode(
                coa_id=coa_id,
                node_id=f"{coa_id}:{code}",
                parent_id=parent_id,
                code=code,
                name=(row.get("name") or "").strip(),
                name_local=(row.get(local_name_col) or row.get("name") or "").strip(),
                node_type=POSTING,
                income_balance=_INCOME_BALANCE.get(account_type, "NA"),
                category=account_type,
                sort_order=order,
                attrs={
                    "odoo_id": (row.get("id") or "").strip(),
                    "odoo_account_type": account_type,
                    "tag_ids": tags,       # SKR03: real HGB statutory tags
                    "tax_ids": taxes,      # SKR03: VAT rate baked into the account
                    "reconcile": (row.get("reconcile") or "").strip() == "True",
                },
            )
        )

    return nodes


def load_skr03(seeds_dir: Path) -> list[AccountNode]:
    """SKR03: 1,274 German accounts with real HGB tags and VAT-rate-bearing accounts."""
    return load_digit_coa(
        csv_path=Path(seeds_dir) / "skr.csv",
        coa_id=SKR03_COA_ID,
        local_name_col="name@de",
        max_group_levels=3,
        root_name="SKR03 Kontenrahmen",
    )


def load_pgc(seeds_dir: Path) -> list[AccountNode]:
    """PGC: 588 Spanish accounts; group names come from Odoo's real account.group table."""
    seeds_dir = Path(seeds_dir)
    return load_digit_coa(
        csv_path=seeds_dir / "pgc_odoo.csv",
        coa_id=PGC_COA_ID,
        local_name_col="name@es",
        max_group_levels=3,
        groups_path=seeds_dir / "pgc_groups.csv",
        group_local_col="name@es",
        root_name="Plan General de Contabilidad",
    )
