"""Canonical account-node schema shared by every chart-of-accounts loader.

Per spec 4.0 the canonical storage for *all* hierarchies is parent-child, one row per
node. Fixed-level tables and bridges are derived later and are never the source of
truth. Each seed CoA has a wildly different native shape (BC nests Begin/End-Total code
ranges, SKR03/PGC nest by code-digit prefix, Sage is a shallow two-level category
layer) and that irregularity is preserved verbatim rather than forced into a common
depth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import polars as pl

# Node types, deliberately independent of depth: consumers must switch on node_type,
# never on level position (spec 4.0).
HEADING = "heading"      # pure presentation node (BALANCE SHEET)
GROUP = "group"          # aggregation node (a Begin/End-Total range, a code prefix)
POSTING = "posting"      # a real postable account — the only valid fact target

ACCOUNT_NODE_SCHEMA = {
    "coa_id": pl.Utf8,
    "node_id": pl.Utf8,
    "parent_id": pl.Utf8,
    "code": pl.Utf8,
    "name": pl.Utf8,
    "name_local": pl.Utf8,
    "node_type": pl.Utf8,
    "is_leaf": pl.Boolean,
    "income_balance": pl.Utf8,   # BS | PL | NA
    "category": pl.Utf8,
    "subcategory": pl.Utf8,
    "range_lo": pl.Int64,
    "range_hi": pl.Int64,
    "depth": pl.Int32,
    "sort_order": pl.Int64,
    "attrs": pl.Utf8,            # JSON blob of seed-specific extras
}


@dataclass
class AccountNode:
    coa_id: str
    node_id: str
    name: str
    node_type: str
    parent_id: str | None = None
    code: str | None = None
    name_local: str | None = None
    income_balance: str = "NA"
    category: str = ""
    subcategory: str = ""
    range_lo: int | None = None
    range_hi: int | None = None
    depth: int = 0
    sort_order: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)

    def as_row(self, is_leaf: bool) -> dict:
        return {
            "coa_id": self.coa_id,
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "code": self.code,
            "name": self.name,
            "name_local": self.name_local or self.name,
            "node_type": self.node_type,
            "is_leaf": is_leaf,
            "income_balance": self.income_balance,
            "category": self.category,
            "subcategory": self.subcategory,
            "range_lo": self.range_lo,
            "range_hi": self.range_hi,
            "depth": self.depth,
            "sort_order": self.sort_order,
            "attrs": json.dumps(self.attrs, sort_keys=True, ensure_ascii=False),
        }


def to_frame(nodes: list[AccountNode]) -> pl.DataFrame:
    """Materialise nodes, computing depth and is_leaf from the parent links.

    Depth is derived here rather than trusted from the loaders, so a loader that
    miscounts levels shows up as a cycle/orphan error instead of silently wrong data.
    """
    by_id = {n.node_id: n for n in nodes}
    if len(by_id) != len(nodes):
        seen: set[str] = set()
        dupes = {n.node_id for n in nodes if n.node_id in seen or seen.add(n.node_id)}  # type: ignore[func-returns-value]
        raise ValueError(f"duplicate account node_id(s): {sorted(dupes)[:10]}")

    parents = {n.node_id for n in nodes if n.parent_id}
    has_child = {n.parent_id for n in nodes if n.parent_id}

    for node in nodes:
        if node.parent_id and node.parent_id not in by_id:
            raise ValueError(
                f"account node {node.node_id!r} references missing parent {node.parent_id!r}"
            )

    depth_cache: dict[str, int] = {}

    def depth_of(node_id: str, guard: int = 0) -> int:
        if guard > 64:
            raise ValueError(f"cycle in account hierarchy at {node_id!r}")
        if node_id in depth_cache:
            return depth_cache[node_id]
        node = by_id[node_id]
        value = 0 if not node.parent_id else depth_of(node.parent_id, guard + 1) + 1
        depth_cache[node_id] = value
        return value

    rows = []
    for node in nodes:
        node.depth = depth_of(node.node_id)
        rows.append(node.as_row(is_leaf=node.node_id not in has_child))
    del parents
    return pl.DataFrame(rows, schema=ACCOUNT_NODE_SCHEMA)


def assert_tree_integrity(frame: pl.DataFrame, coa_id: str) -> None:
    """Structural invariants every loaded CoA must satisfy."""
    sub = frame.filter(pl.col("coa_id") == coa_id)
    if sub.height == 0:
        raise ValueError(f"no nodes loaded for CoA {coa_id!r}")

    roots = sub.filter(pl.col("parent_id").is_null())
    if roots.height == 0:
        raise ValueError(f"CoA {coa_id!r} has no root node (cycle?)")

    # Every posting account must be a leaf: facts post to leaves only (spec 4.0).
    bad = sub.filter((pl.col("node_type") == POSTING) & ~pl.col("is_leaf"))
    if bad.height:
        raise ValueError(
            f"CoA {coa_id!r}: {bad.height} posting account(s) have children, e.g. "
            f"{bad['node_id'].to_list()[:5]}"
        )
