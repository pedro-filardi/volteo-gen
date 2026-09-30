"""Markets — the demand side of "where", which is NOT the same as legal entity.

Two distinct concepts, both modelled (spec 4.2):

``dim_market``                  Region -> Country -> Subterritory, ragged (the US has
                                states as subterritories; small countries stop at
                                level 2).
``dim_entity_market_coverage``  which entity invoices which market. DE01 invoices
                                Austria and Poland with no local entity there; ES01
                                covers Portugal. So "revenue by country" and "revenue by
                                entity" are genuinely different questions.

Market attaches ONLY to sales-side facts. Payroll and facilities never carry it.
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from ..rng import SeedBank
from .hierarchy import HierarchyBuilder

REGION, COUNTRY, SUBTERRITORY = "region", "country", "subterritory"

# Region -> country -> subterritories. Countries with an empty list stop at level 2,
# which is what makes the tree ragged.
GEOGRAPHY = {
    "EMEA": {
        "DE": ["DACH-North", "DACH-South"],
        "ES": ["Iberia-Central", "Iberia-North"],
        "UK": ["UK-South", "UK-North", "Scotland"],
        "FR": [],
        "IT": [],
        "NL": [],
        "PL": [],
        "AT": [],
        "PT": [],
        "SE": [],
    },
    "AMER": {
        "US": ["US-West", "US-Midwest", "US-Northeast", "US-South"],
        "CA": [],
        "MX": [],
        "BR": [],
    },
    "APAC": {
        "AU": [],
        "JP": [],
        "SG": [],
        "IN": [],
    },
}

# Which entity invoices which country (spec 4.2: many-to-many-ish, not 1:1 with entity).
ENTITY_COVERAGE = {
    "US01": ["US", "CA", "MX", "BR", "AU", "JP", "SG", "IN"],
    "DE01": ["DE", "AT", "PL", "NL", "SE"],
    "ES01": ["ES", "PT", "IT", "FR"],
    "UK01": ["UK"],
}

# Territories that span several countries and therefore cannot sit under one of them.
# These are the skip-level branches: a subterritory reporting straight to the region.
CROSS_BORDER_TERRITORIES = {
    "EMEA": ["Benelux", "Nordics"],
    "APAC": ["ANZ"],
}

COUNTRY_CURRENCY = {
    "DE": "EUR", "ES": "EUR", "FR": "EUR", "IT": "EUR", "NL": "EUR",
    "AT": "EUR", "PT": "EUR", "PL": "EUR", "SE": "EUR",
    "UK": "GBP",
    "US": "USD", "CA": "USD", "MX": "USD", "BR": "USD",
    "AU": "USD", "JP": "USD", "SG": "USD", "IN": "USD",
}


def build_markets(cfg: Config, seeds: SeedBank) -> pl.DataFrame:
    rng = seeds.rng("dims", "markets")
    target_leaves = int(cfg.get("cardinality.markets.target_leaves"))
    skip_share = float(cfg.get("cardinality.markets.skip_level_share", 0.05))

    builder = HierarchyBuilder("market")
    root = builder.add("MKT", "All Markets", "root", None, is_skip_level=False)

    leaf_count = 0
    for region, countries in GEOGRAPHY.items():
        region_id = builder.add(f"MKT:{region}", region, REGION, root, is_skip_level=False, region=region)
        # Skip-level, and a genuinely realistic one: a multi-country territory cannot
        # hang under a single country, so it attaches straight to the region and a
        # subterritory ends up at the depth countries normally occupy. Guaranteed (not
        # sampled) because spec 4.0 makes irregularity a requirement, not an accident.
        for territory in CROSS_BORDER_TERRITORIES.get(region, []):
            builder.add(
                f"MKT:{region}:{_slug(territory)}",
                territory,
                SUBTERRITORY,
                region_id,
                is_skip_level=True,
                region=region,
                country=None,
                currency=None,
            )
            leaf_count += 1

        for country, subterritories in countries.items():
            # Plus sampled skip-levels on top, at the configured share.
            skip = bool(subterritories) and rng.random() < skip_share
            country_id = builder.add(
                f"MKT:{region}:{country}",
                country,
                COUNTRY,
                region_id,
                is_skip_level=skip,
                region=region,
                country=country,
                currency=COUNTRY_CURRENCY.get(country, "USD"),
            )
            if not subterritories or leaf_count >= target_leaves:
                leaf_count += 1
                continue
            for sub in subterritories:
                builder.add(
                    f"MKT:{region}:{country}:{_slug(sub)}",
                    sub,
                    SUBTERRITORY,
                    country_id,
                    is_skip_level=False,
                    region=region,
                    country=country,
                    currency=COUNTRY_CURRENCY.get(country, "USD"),
                )
                leaf_count += 1

    return builder.frame()


def _slug(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())[:14]


def build_entity_market_coverage(markets: pl.DataFrame, entities: list[str]) -> pl.DataFrame:
    """Which entity sells into which market leaf."""
    leaves = markets.filter(pl.col("is_leaf"))
    rows = []
    for entity in entities:
        countries = ENTITY_COVERAGE.get(entity, [])
        for row in leaves.iter_rows(named=True):
            if row["country"] in countries:
                rows.append(
                    {
                        "entity": entity,
                        "market_node_id": row["node_id"],
                        "country": row["country"],
                        "region": row["region"],
                        "is_domestic": _is_domestic(entity, row["country"]),
                    }
                )
    if not rows:
        raise ValueError("entity/market coverage came out empty — check ENTITY_COVERAGE")
    return pl.DataFrame(rows)


def _is_domestic(entity: str, country: str) -> bool:
    return {"US01": "US", "DE01": "DE", "ES01": "ES", "UK01": "UK"}.get(entity) == country
