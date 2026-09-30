"""US ERP chart of accounts, from the Business Central demo-data codeunits.

Hierarchy comes from **Begin-Total / End-Total code ranges**, which is what actually
defines aggregation in Business Central. The ``Indentation`` column is deliberately
ignored: upstream ships it as 0 almost everywhere and recomputes it at install time via
the ``G/L Account-Indent`` codeunit, so trusting it would flatten the tree.

Five nodes upstream never receive a nominal code (BALANCE SHEET, ASSETS, TOTAL ASSETS,
INCOME STATEMENT, NET INCOME) because the demo tool assigns those numbers at runtime.
They are emitted as structural nodes with ``code = None`` rather than given invented
account numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

from .al_parser import (
    GLAccountSpec,
    parse_gl_account_inserts,
    parse_labels,
    parse_localization_codes,
)
from .nodes import GROUP, HEADING, POSTING, AccountNode

COA_ID = "US_ERP"

W1_FILES = (
    "us_gl_finance.al",
    "us_gl_common.al",
    "us_gl_fa.al",
    "us_gl_hr.al",
    "us_gl_job.al",
    "us_gl_mfg.al",
    "us_gl_svc.al",
)
US_FILE = "us_gl_us.al"


def _camel_to_words(name: str) -> str:
    """Fallback display name for the handful of bases with no upstream label."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).strip()


def _as_int(code: str | None) -> int | None:
    if code is None:
        return None
    text = code.strip()
    return int(text) if text.isdigit() else None


def load_us_coa(seeds_dir: Path) -> list[AccountNode]:
    seeds_dir = Path(seeds_dir)
    us_path = seeds_dir / US_FILE
    label_paths = [seeds_dir / f for f in W1_FILES] + [us_path]
    for path in label_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"missing US CoA seed {path.name}; run seeds/fetch_seeds.sh "
                "(see seeds/SOURCES.md)"
            )

    labels = parse_labels(label_paths)
    codes = parse_localization_codes(us_path)
    specs = parse_gl_account_inserts(us_path)
    if not specs:
        raise ValueError(f"parsed zero InsertGLAccount calls from {us_path}")

    def name_of(spec: GLAccountSpec) -> str:
        return labels.get(spec.name_base) or _camel_to_words(spec.name_base)

    def ib_of(spec: GLAccountSpec) -> str:
        return {"Balance Sheet": "BS", "Income Statement": "PL"}.get(
            spec.income_balance, "NA"
        )

    nodes: list[AccountNode] = []

    # -- Structural headings ---------------------------------------------------
    # Real names, from the seed labels; no invented account codes.
    heading_ids: dict[str, str] = {}
    for key, base, ib in (
        ("BS", "BalanceSheet", "BS"),
        ("PL", "IncomeStatement", "PL"),
    ):
        node_id = f"{COA_ID}:ROOT:{key}"
        heading_ids[key] = node_id
        nodes.append(
            AccountNode(
                coa_id=COA_ID,
                node_id=node_id,
                name=labels.get(base, _camel_to_words(base)),
                node_type=HEADING,
                income_balance=ib,
                sort_order=-1,
                attrs={"structural": True, "reason": "no nominal code upstream"},
            )
        )

    # -- Groups: one per Begin-Total / End-Total range -------------------------
    # The End-Total carries the range; the matching Begin-Total carries the readable
    # name ("Intangible Fixed Assets" vs "Total, Intangible Fixed Assets").
    begin_by_code: dict[int, GLAccountSpec] = {}
    for spec in specs:
        if spec.account_type == "Begin-Total":
            lo = _as_int(codes.get(spec.base))
            if lo is not None:
                begin_by_code[lo] = spec

    groups: list[dict] = []
    for spec in specs:
        if spec.account_type not in ("End-Total", "Total"):
            continue
        if len(spec.totaling_bases) < 2:
            continue
        lo = _as_int(codes.get(spec.totaling_bases[0]))
        hi = _as_int(codes.get(spec.totaling_bases[-1]))
        if lo is None or hi is None or hi < lo:
            continue
        opener = begin_by_code.get(lo)
        groups.append(
            {
                "lo": lo,
                "hi": hi,
                "name": name_of(opener) if opener else name_of(spec),
                "total_name": name_of(spec),
                "total_code": codes.get(spec.base),
                "income_balance": ib_of(spec),
                "category": spec.category,
                "subcategory": spec.subcategory or (opener.subcategory if opener else ""),
                "order": spec.order,
            }
        )

    # Widest range first so containment nesting is unambiguous.
    groups.sort(key=lambda g: (g["lo"], -g["hi"]))
    for group in groups:
        group["node_id"] = f"{COA_ID}:G:{group['lo']}-{group['hi']}"

    def innermost_group(lo: int, hi: int, exclude: str | None = None) -> dict | None:
        best: dict | None = None
        for group in groups:
            if group["node_id"] == exclude:
                continue
            if group["lo"] <= lo and hi <= group["hi"]:
                if group["lo"] == lo and group["hi"] == hi and exclude is None:
                    continue
                if best is None or (group["hi"] - group["lo"]) < (best["hi"] - best["lo"]):
                    best = group
        return best

    for group in groups:
        parent = innermost_group(group["lo"], group["hi"], exclude=group["node_id"])
        parent_id = parent["node_id"] if parent else heading_ids.get(
            group["income_balance"], heading_ids["BS"]
        )
        nodes.append(
            AccountNode(
                coa_id=COA_ID,
                node_id=group["node_id"],
                parent_id=parent_id,
                code=group["total_code"],
                name=group["name"],
                node_type=GROUP,
                income_balance=group["income_balance"],
                category=group["category"],
                subcategory=group["subcategory"],
                range_lo=group["lo"],
                range_hi=group["hi"],
                sort_order=group["order"],
                attrs={"total_name": group["total_name"], "totaling": f"{group['lo']}..{group['hi']}"},
            )
        )

    # -- Posting accounts ------------------------------------------------------
    seen_codes: set[str] = set()
    for spec in specs:
        if spec.account_type != "Posting":
            continue
        code = codes.get(spec.base)
        if not code or code in seen_codes:
            continue  # unmapped in the US locale (e.g. VAT accounts) — not a US account
        seen_codes.add(code)
        numeric = _as_int(code)
        parent = innermost_group(numeric, numeric) if numeric is not None else None
        income_balance = ib_of(spec)
        parent_id = (
            parent["node_id"]
            if parent
            else heading_ids.get(income_balance, heading_ids["BS"])
        )
        nodes.append(
            AccountNode(
                coa_id=COA_ID,
                node_id=f"{COA_ID}:{code}",
                parent_id=parent_id,
                code=code,
                name=name_of(spec),
                node_type=POSTING,
                income_balance=income_balance,
                category=spec.category,
                subcategory=spec.subcategory,
                sort_order=spec.order,
                attrs={"al_base": spec.base},
            )
        )

    return nodes
