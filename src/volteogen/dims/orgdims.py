"""Cost centres (org tree, versioned) and profit centres (business tree, cross-entity).

Cost centres (spec 4.3): Entity -> Function -> Department (-> Team, but only for US01
and DE01, which is what makes the tree ragged across entities). Two shared pools feed
the allocation engine. DE01 merges two departments at the start of FY25, so
``dim_cost_center_version`` exists and facts always carry the ORIGINAL cost centre —
reports may restate through the version table, or fail to, which is the point.

Profit centres (spec 4.4): Division x Region, ragged because Services exists only in
EMEA. PCs are derived from facts, not posted to directly.
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from ..rng import SeedBank
from .hierarchy import HierarchyBuilder

FUNCTION, DEPARTMENT, TEAM = "function", "department", "team"

FUNCTIONS = {
    "PRD": "Production",
    "SLS": "Sales",
    "MKT": "Marketing",
    "RND": "R&D",
    "GA": "General & Administration",
    "LOG": "Logistics",
}

# Which functions each entity actually runs, and which get a Team layer.
ENTITY_FUNCTIONS = {
    "US01": ["SLS", "MKT", "RND", "GA", "LOG"],
    "DE01": ["PRD", "SLS", "MKT", "RND", "GA", "LOG"],
    "ES01": ["SLS", "MKT", "GA"],
    "UK01": ["SLS", "GA", "LOG"],
}
TEAM_ENTITIES = {"US01", "DE01"}

DEPARTMENTS = {
    "PRD": ["Assembly", "Quality", "Plant Support"],
    "SLS": ["Field Sales", "Key Accounts", "Inside Sales"],
    "MKT": ["Brand", "Digital"],
    "RND": ["Hardware", "Firmware"],
    "GA": ["Finance", "HR", "IT"],
    "LOG": ["Warehouse", "Distribution"],
}

# How many departments each entity actually staffs per function. Small entities run one
# department per function; the parent runs the full set. This is what makes sibling
# fan-out genuinely unbalanced (V11) rather than a uniform 2-3 everywhere.
ENTITY_DEPARTMENT_LIMIT = {"US01": 99, "DE01": 99, "ES01": 1, "UK01": 1}

# US01 carries the group's back office, so G&A is much wider than any other function.
EXTRA_DEPARTMENTS = {
    ("US01", "GA"): ["Legal", "Treasury", "Group Reporting", "Procurement"],
}

TEAMS = {
    "Assembly": ["Line A", "Line B"],
    "Key Accounts": ["Retail Majors", "Distribution"],
    "Hardware": ["Audio Eng", "Power Eng"],
    "Finance": ["Controlling", "Accounting"],
}

MANAGER_NAMES = [
    "R. Alvarez", "S. Becker", "T. Novak", "M. Duarte", "K. Lindqvist", "P. Okafor",
    "J. Whitfield", "A. Moreau", "L. Kaufmann", "N. Ferrari", "D. Hollands", "C. Iversen",
    "E. Sandoval", "H. Brandt", "F. Nakamura", "G. Petrov", "B. Ahmed", "V. Costa",
]

# ---------------------------------------------------------------------------
# Geography lives on the cost-centre MASTER, never as a column on the ledger.
#
# REALISM: this is where real ERPs keep it — an SAP cost centre master carries its
# company code and location, and the posting carries only the cost centre. So a cost
# row has no market column, yet "opex by country" and "opex by site" are both
# answerable with a dimension join. Two different geographies are recorded, because
# they genuinely differ:
#
#   site / country / geo_region  where the cost is INCURRED
#   serves_market_node_id        which market the cost centre SUPPORTS
#
# A DE01 sales team incurred in Germany serving all of EMEA has DE for the first and
# MKT:EMEA for the second. Both point at real nodes of dim_market_node, so the market
# hierarchy and its bridge work unchanged.
# ---------------------------------------------------------------------------

# Physical sites per entity. The plant and the distribution centre are separate
# locations from the office, which is what makes site-level cost reporting possible.
SITES = {
    "US01": {
        "default": {"site": "US-HQ", "city": "Portland, OR"},
        "LOG":     {"site": "US-DC", "city": "Reno, NV"},
    },
    "DE01": {
        "default": {"site": "DE-OFFICE", "city": "Munich"},
        "PRD":     {"site": "DE-PLANT", "city": "Regensburg"},
        "LOG":     {"site": "DE-PLANT", "city": "Regensburg"},
    },
    "ES01": {"default": {"site": "ES-OFFICE", "city": "Barcelona"}},
    "UK01": {"default": {"site": "UK-OFFICE", "city": "Reading"}},
}

ENTITY_GEO = {
    "US01": {"country": "US", "geo_region": "AMER"},
    "DE01": {"country": "DE", "geo_region": "EMEA"},
    "ES01": {"country": "ES", "geo_region": "EMEA"},
    "UK01": {"country": "UK", "geo_region": "EMEA"},
}

# Which market a function supports. Customer-facing functions serve their own region;
# the plant, R&D and group G&A serve every market, so they point at the market root.
FUNCTION_SERVES = {
    "SLS": "home_region",
    "MKT": "home_region",
    "LOG": "home_region",
    "PRD": "all",
    "RND": "all",
    "GA":  "all",
}

# Only cost centres that genuinely belong to ONE division get a home profit centre.
# Sales, marketing, logistics and G&A support every division, so they are left
# unattributed and reach a profit centre through the allocation engine instead —
# which is honest, where forcing them onto a division would not be.
TEAM_DIVISION = {
    "Audio Eng": "AUD",
    "Power Eng": "PWR",
}

SHARED_POOLS = {
    "PLANT-RCK": {"entity": "DE01", "function": "PRD", "driver": "machine_hours"},
    "HQ-G&A": {"entity": "US01", "function": "GA", "driver": "revenue_share"},
}


def build_cost_centers(cfg: Config, seeds: SeedBank, entities: list[str]) -> pl.DataFrame:
    rng = seeds.rng("dims", "cost_centers")
    target_total = int(cfg.get("cardinality.cost_centers_total"))

    builder = HierarchyBuilder("cost_center")
    root = builder.add("CC", "All Cost Centers", "root", None, is_skip_level=False)

    manager_pool = list(MANAGER_NAMES)
    rng.shuffle(manager_pool)
    manager_index = 0
    # target_total is a soft target: the shape comes from the org config above, because
    # truncating branches to hit an exact count is what produces an artificially regular
    # tree. Presets scale the org via ENTITY_DEPARTMENT_LIMIT instead.
    del target_total

    for entity in entities:
        entity_id = builder.add(
            f"CC:{entity}", f"{entity} Cost Centers", "entity", root, is_skip_level=False, entity=entity
        )
        for function in ENTITY_FUNCTIONS.get(entity, ["GA"]):
            function_id = builder.add(
                f"CC:{entity}:{function}",
                FUNCTIONS[function],
                FUNCTION,
                entity_id,
                is_skip_level=False,
                entity=entity,
                function=function,
            )
            departments = list(DEPARTMENTS.get(function, ["General"]))
            departments += EXTRA_DEPARTMENTS.get((entity, function), [])
            departments = departments[: ENTITY_DEPARTMENT_LIMIT.get(entity, 99)]

            for department in departments:
                dept_id = f"CC:{entity}:{function}:{_slug(department)}"
                teams = TEAMS.get(department) if entity in TEAM_ENTITIES else None
                manager = manager_pool[manager_index % len(manager_pool)]
                manager_index += 1
                builder.add(
                    dept_id,
                    department,
                    DEPARTMENT,
                    function_id,
                    is_skip_level=False,
                    entity=entity,
                    function=function,
                    manager_name=manager,
                    home_profit_center=_home_pc(entity, department),
                    shared_pool=None,
                    **_geo_of(entity, function),
                )
                if teams:
                    for team in teams:
                        builder.add(
                            f"{dept_id}:{_slug(team)}",
                            team,
                            TEAM,
                            dept_id,
                            is_skip_level=False,
                            entity=entity,
                            function=function,
                            manager_name=manager_pool[manager_index % len(manager_pool)],
                            home_profit_center=_home_pc(entity, department, team),
                            shared_pool=None,
                            **_geo_of(entity, function),
                        )
                        manager_index += 1

    # Shared pools: the "special projects" skip-level pattern — they report straight to
    # the entity, bypassing the Function layer entirely.
    for pool, spec in SHARED_POOLS.items():
        if spec["entity"] not in entities:
            continue
        builder.add(
            f"CC:{spec['entity']}:POOL:{_slug(pool)}",
            pool,
            DEPARTMENT,
            f"CC:{spec['entity']}",
            is_skip_level=True,
            entity=spec["entity"],
            function=spec["function"],
            manager_name=manager_pool[manager_index % len(manager_pool)],
            home_profit_center=None,
            shared_pool=pool,
            **_geo_of(spec["entity"], spec["function"]),
        )
        manager_index += 1

    return builder.frame()


def build_cost_center_versions(
    cost_centers: pl.DataFrame, cfg: Config, calendar: pl.DataFrame
) -> pl.DataFrame:
    """The FY24 vs FY25 org, with DE01's mid-history department merge.

    Facts keep the ORIGINAL cost centre forever; this table is the only way to restate
    history onto the new org, and only if a consumer remembers to use it.
    """
    fiscal_years = sorted(calendar["fiscal_year"].unique().to_list())
    merge_enabled = cfg.defect_enabled("cc_reorg_midyear")

    leaves = cost_centers.filter(pl.col("is_leaf"))
    rows = []
    for fy in fiscal_years:
        for row in leaves.iter_rows(named=True):
            target = row["node_id"]
            note = None
            # DE01 merges Brand + Digital marketing into one department from FY25.
            if (
                merge_enabled
                and fy >= fiscal_years[0] + 1
                and row["entity"] == "DE01"
                and row["name"] in ("Brand", "Digital")
            ):
                target = "CC:DE01:MKT:BRAND"
                note = "FY25 merge: DE01 Brand + Digital -> Brand"
            rows.append(
                {
                    "cc_hierarchy_version": f"FY{fy}",
                    "fiscal_year": fy,
                    "node_id": row["node_id"],
                    "restated_node_id": target,
                    "entity": row["entity"],
                    "is_restated": target != row["node_id"],
                    "note": note,
                }
            )
    return pl.DataFrame(rows)


def _geo_of(entity: str, function: str) -> dict:
    """Where a cost centre sits, and which market it supports."""
    sites = SITES.get(entity, {})
    site = sites.get(function) or sites.get("default") or {"site": f"{entity}-SITE", "city": ""}
    geo = ENTITY_GEO.get(entity, {"country": "US", "geo_region": "AMER"})
    scope = FUNCTION_SERVES.get(function, "home_region")
    serves = "MKT" if scope == "all" else f"MKT:{geo['geo_region']}"
    return {
        "site": site["site"],
        "site_city": site["city"],
        "country": geo["country"],
        "geo_region": geo["geo_region"],
        "serves_market_node_id": serves,
    }


def _home_pc(entity: str, department: str, team: str | None = None) -> str | None:
    """Profit centre for a cost centre, but ONLY where the division is unambiguous.

    Returning None is the point. A sales team supports every division, so pretending it
    belongs to Audio manufactures a division P&L out of nothing — which is exactly what
    the previous version did, dumping all SG&A onto Audio. Ambiguous costs stay
    unattributed and reach a profit centre through the allocation engine, remaining
    visibly unallocated until they do.
    """
    division = TEAM_DIVISION.get(team or "") or TEAM_DIVISION.get(department)
    if division is None:
        return None
    region = ENTITY_GEO.get(entity, {}).get("geo_region", "AMER")
    return f"PC:{division}:{region}"


def build_profit_centers(cfg: Config, pack: dict) -> pl.DataFrame:
    """Division x Region, ragged: Services exists only in EMEA."""
    builder = HierarchyBuilder("profit_center")
    root = builder.add("PC", "Volteo Group", "root", None, is_skip_level=False)

    divisions = pack["product_taxonomy"]["divisions"]
    regions = ["AMER", "EMEA", "APAC"]

    for division in divisions:
        div_key = division["key"]
        div_id = builder.add(
            f"PC:{div_key}", division["name"], "pc_division", root, is_skip_level=False,
            division=division["name"],
        )
        for region in regions:
            # Ragged by design: Accessories & Services only runs in EMEA, and Smart
            # Home has not reached APAC.
            if div_key == "ACC" and region != "EMEA":
                continue
            if div_key == "SMH" and region == "APAC":
                continue
            builder.add(
                f"PC:{div_key}:{region}",
                f"{division['name']} {region}",
                "pc_division_region",
                div_id,
                is_skip_level=False,
                division=division["name"],
                region=region,
            )
    return builder.frame()


def _slug(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())[:14]
