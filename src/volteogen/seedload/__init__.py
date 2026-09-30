"""Loading of the real-world seed skeletons.

Nothing here invents structure. Every account, code, name, tag and calculation weight
comes from the files documented in ``seeds/SOURCES.md``; a missing seed is a hard
failure with the fetch instructions attached, never a silent synthetic substitute.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .coa_digit import PGC_COA_ID, SKR03_COA_ID, load_pgc, load_skr03
from .coa_uk import COA_ID as SAGE_COA_ID
from .coa_uk import load_uk_coa
from .coa_us import COA_ID as US_COA_ID
from .coa_us import load_us_coa
from .gaap import DEFAULT_NETWORK, build_gaap_bridge, load_gaap_arcs, load_gaap_nodes
from .nodes import assert_tree_integrity, to_frame

__all__ = [
    "SeedBundle",
    "load_seeds",
    "US_COA_ID",
    "SKR03_COA_ID",
    "PGC_COA_ID",
    "SAGE_COA_ID",
    "GAAP_COA_ID",
]

GAAP_COA_ID = "USGAAP"

# CoA per entity (spec 3). Each entity's local world is its own chart.
ENTITY_COA = {
    "US01": US_COA_ID,
    "DE01": SKR03_COA_ID,
    "ES01": PGC_COA_ID,
    "UK01": SAGE_COA_ID,
}


@dataclass
class SeedBundle:
    """Normalised seed data, ready for the dimension layer."""

    accounts: pl.DataFrame        # dim_account_node for all local CoAs
    gaap_nodes: pl.DataFrame      # US-GAAP calc tree as parent-child
    gaap_arcs: pl.DataFrame       # raw calc arcs with signed weights
    gaap_bridge: pl.DataFrame     # ancestor bridge, weights pre-multiplied
    seeds_dir: Path

    def coa(self, coa_id: str) -> pl.DataFrame:
        return self.accounts.filter(pl.col("coa_id") == coa_id)

    def postings(self, coa_id: str) -> pl.DataFrame:
        return self.accounts.filter(
            (pl.col("coa_id") == coa_id) & (pl.col("node_type") == "posting")
        )

    def summary(self) -> pl.DataFrame:
        return (
            self.accounts.group_by("coa_id")
            .agg(
                pl.len().alias("nodes"),
                (pl.col("node_type") == "posting").sum().alias("postings"),
                pl.col("depth").max().alias("max_depth"),
            )
            .sort("coa_id")
        )


def _verify_checksums(seeds_dir: Path) -> None:
    """Warn (never fail) when a local seed drifts from its pinned digest."""
    manifest = seeds_dir / "CHECKSUMS.txt"
    if not manifest.exists():
        return
    for line in manifest.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        digest, name = parts
        path = seeds_dir / name
        if not path.exists():
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            warnings.warn(
                f"seed {name} differs from the pinned checksum in CHECKSUMS.txt "
                "(intentional refresh? re-pin with shasum -a 256)",
                RuntimeWarning,
                stacklevel=2,
            )


def load_seeds(
    seeds_dir: Path | str,
    gaap_network: str = DEFAULT_NETWORK,
    verify: bool = True,
) -> SeedBundle:
    seeds_dir = Path(seeds_dir)
    if not seeds_dir.is_dir():
        raise FileNotFoundError(
            f"seeds directory not found: {seeds_dir}. Run seeds/fetch_seeds.sh "
            "to download the real skeletons (see seeds/SOURCES.md)."
        )
    if verify:
        _verify_checksums(seeds_dir)

    nodes = []
    nodes += load_us_coa(seeds_dir)
    nodes += load_skr03(seeds_dir)
    nodes += load_pgc(seeds_dir)
    nodes += load_uk_coa(seeds_dir)
    accounts = to_frame(nodes)

    for coa_id in (US_COA_ID, SKR03_COA_ID, PGC_COA_ID, SAGE_COA_ID):
        assert_tree_integrity(accounts, coa_id)

    gaap_nodes = to_frame(load_gaap_nodes(seeds_dir, gaap_network))
    gaap_arcs = load_gaap_arcs(seeds_dir, gaap_network)
    gaap_bridge = build_gaap_bridge(gaap_arcs, gaap_network)

    return SeedBundle(
        accounts=accounts,
        gaap_nodes=gaap_nodes,
        gaap_arcs=gaap_arcs,
        gaap_bridge=gaap_bridge,
        seeds_dir=seeds_dir,
    )
