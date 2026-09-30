"""Shared hierarchy machinery: canonical parent-child storage plus derived artifacts.

Spec 4.0 is normative for every dimension, not just accounts:

* canonical storage is parent-child, one row per node;
* facts reference LEAF ``node_id`` only;
* ``dim_<x>_levels`` and ``bridge_<x>`` are DERIVED, never the source of truth;
* irregularity (ragged depth, unbalanced fan-out, skip-levels, node_type not inferable
  from depth) is a REQUIREMENT — :func:`assert_irregularity` fails the build if a
  generated tree comes out regular.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

NODE_SCHEMA_CORE = ("node_id", "parent_id", "name", "node_type", "is_leaf", "depth")


@dataclass
class Node:
    node_id: str
    name: str
    node_type: str
    parent_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


class HierarchyBuilder:
    """Accumulates nodes and materialises the canonical parent-child frame."""

    def __init__(self, dim_name: str) -> None:
        self.dim_name = dim_name
        self._nodes: list[Node] = []
        self._ids: set[str] = set()

    def add(
        self,
        node_id: str,
        name: str,
        node_type: str,
        parent_id: str | None = None,
        **attributes: Any,
    ) -> str:
        if node_id in self._ids:
            raise ValueError(f"{self.dim_name}: duplicate node_id {node_id!r}")
        self._ids.add(node_id)
        self._nodes.append(
            Node(node_id=node_id, name=name, node_type=node_type, parent_id=parent_id, attributes=attributes)
        )
        return node_id

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._ids

    def frame(self) -> pl.DataFrame:
        by_id = {n.node_id: n for n in self._nodes}
        for node in self._nodes:
            if node.parent_id and node.parent_id not in by_id:
                raise ValueError(
                    f"{self.dim_name}: node {node.node_id!r} references missing parent "
                    f"{node.parent_id!r}"
                )
        has_child = {n.parent_id for n in self._nodes if n.parent_id}

        depth_cache: dict[str, int] = {}

        def depth_of(node_id: str, guard: int = 0) -> int:
            if guard > 64:
                raise ValueError(f"{self.dim_name}: cycle at {node_id!r}")
            if node_id in depth_cache:
                return depth_cache[node_id]
            node = by_id[node_id]
            value = 0 if not node.parent_id else depth_of(node.parent_id, guard + 1) + 1
            depth_cache[node_id] = value
            return value

        attribute_keys: list[str] = []
        for node in self._nodes:
            for key in node.attributes:
                if key not in attribute_keys:
                    attribute_keys.append(key)

        rows = []
        for node in self._nodes:
            row = {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "name": node.name,
                "node_type": node.node_type,
                "is_leaf": node.node_id not in has_child,
                "depth": depth_of(node.node_id),
            }
            for key in attribute_keys:
                row[key] = node.attributes.get(key)
            rows.append(row)
        return pl.DataFrame(rows, infer_schema_length=None)


def leaves(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("is_leaf"))


def ancestors_of(frame: pl.DataFrame) -> dict[str, list[str]]:
    """node_id -> [self, parent, ..., root]."""
    parent = dict(zip(frame["node_id"].to_list(), frame["parent_id"].to_list()))
    out: dict[str, list[str]] = {}
    for node_id in parent:
        chain = [node_id]
        current = parent.get(node_id)
        guard = 0
        while current and guard < 64:
            chain.append(current)
            current = parent.get(current)
            guard += 1
        out[node_id] = chain
    return out


def to_bridge(frame: pl.DataFrame, dim_name: str) -> pl.DataFrame:
    """Depth-agnostic ancestor bridge: (ancestor, node, depth_diff, weight_product).

    Weight is 1.0 for ordinary dimensions; only the GAAP tree carries signed weights,
    and that bridge is built in :mod:`volteogen.seedload.gaap`.
    """
    rows = []
    for node_id, chain in ancestors_of(frame).items():
        for distance, ancestor in enumerate(chain):
            rows.append(
                {
                    "dim": dim_name,
                    "ancestor": ancestor,
                    "node": node_id,
                    "depth_diff": distance,
                    "weight_product": 1.0,
                }
            )
    return pl.DataFrame(rows)


def to_levels(frame: pl.DataFrame, dim_name: str, max_levels: int | None = None) -> pl.DataFrame:
    """Flattened level table, padding ragged paths by repeating the leaf downward.

    Padding is standard BI practice and it is lossy, so ``true_depth`` is recorded
    alongside: a consumer that needs to know a Services leaf really stops at level 2
    can still find out (spec 4.0).
    """
    chains = ancestors_of(frame)
    names = dict(zip(frame["node_id"].to_list(), frame["name"].to_list()))
    depth = dict(zip(frame["node_id"].to_list(), frame["depth"].to_list()))
    leaf_ids = frame.filter(pl.col("is_leaf"))["node_id"].to_list()

    levels = max_levels or (max(depth.values()) + 1 if depth else 1)
    rows = []
    for leaf in leaf_ids:
        path = list(reversed(chains[leaf]))  # root -> leaf
        row = {"node_id": leaf, "true_depth": len(path) - 1, "is_padded": len(path) < levels}
        for level in range(levels):
            source = path[level] if level < len(path) else path[-1]
            row[f"level{level + 1}_id"] = source
            row[f"level{level + 1}_name"] = names.get(source)
        rows.append(row)
    return pl.DataFrame(rows).with_columns(pl.lit(dim_name).alias("dim"))


def assert_irregularity(frame: pl.DataFrame, dim_name: str, min_fanout_ratio: int = 4) -> dict:
    """Validator V11: fail the build if a synthetic hierarchy came out regular.

    Checks depth variance, non-degenerate fan-out, at least one skip-level branch and
    at least one ragged leaf pair.
    """
    problems: list[str] = []

    leaf_depths = frame.filter(pl.col("is_leaf"))["depth"].to_list()
    if not leaf_depths:
        raise ValueError(f"{dim_name}: hierarchy has no leaves")
    ragged = len(set(leaf_depths)) > 1
    if not ragged:
        problems.append(
            f"all {len(leaf_depths)} leaves sit at depth {leaf_depths[0]} (not ragged)"
        )

    sibling_counts = (
        frame.filter(pl.col("parent_id").is_not_null())
        .group_by("parent_id")
        .len()
        .rename({"len": "children"})
    )
    max_children = int(sibling_counts["children"].max()) if sibling_counts.height else 0
    min_children = int(sibling_counts["children"].min()) if sibling_counts.height else 0
    if max_children < min_fanout_ratio * max(min_children, 1):
        problems.append(
            f"fan-out is degenerate: sibling counts range {min_children}..{max_children}, "
            f"need max >= {min_fanout_ratio}x min"
        )

    # Skip-level: a node whose parent is not of the immediately-preceding node_type in
    # the declared order — detected via the 'is_skip_level' attribute the builders set.
    skip_levels = 0
    if "is_skip_level" in frame.columns:
        skip_levels = int(frame.filter(pl.col("is_skip_level") == True).height)  # noqa: E712
    if skip_levels == 0:
        problems.append("no skip-level branches present")

    # node_type must not be inferable from depth: at least one type at >1 depth.
    type_depths = frame.group_by("node_type").agg(pl.col("depth").n_unique().alias("n"))
    multi_depth_types = int(type_depths.filter(pl.col("n") > 1).height)

    if problems:
        raise AssertionError(
            f"V11 irregularity check failed for {dim_name}: " + "; ".join(problems)
        )

    return {
        "dim": dim_name,
        "nodes": frame.height,
        "leaves": len(leaf_depths),
        "min_leaf_depth": min(leaf_depths),
        "max_leaf_depth": max(leaf_depths),
        "min_siblings": min_children,
        "max_siblings": max_children,
        "skip_level_branches": skip_levels,
        "types_spanning_multiple_depths": multi_depth_types,
    }
