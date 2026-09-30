"""Legal entities and their local worlds (spec 3).

Each entity owns a different chart of accounts, a different GAAP, a different
functional currency and a different ERP vintage — the asymmetry is the point. UK01 in
particular only enters consolidation from its acquisition month, which creates a real
M&A scope effect in group YoY growth that the variance bridge must attribute.
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from ..seedload import PGC_COA_ID, SAGE_COA_ID, SKR03_COA_ID, US_COA_ID

GROUP_ENTITY = "GRP"
ELIM_ENTITY = "ELIM"

ENTITY_PROFILES = {
    "US01": {
        "name": "Volteo Electronics Inc.",
        "currency": "USD",
        "country": "US",
        "gaap": "US GAAP",
        "coa_id": US_COA_ID,
        "erp_status": "core template",
        "role": "parent",
        "ownership_pct": 1.00,
        "nci_pct": 0.00,
        "parallel_ledgers": True,
        "consolidated_from_month": 0,
        "has_plant": False,
        "notes": "Parent; parallel ledgers; HQ-G&A shared pool.",
    },
    "DE01": {
        "name": "Volteo Elektronik GmbH",
        "currency": "EUR",
        "country": "DE",
        "gaap": "HGB",
        "coa_id": SKR03_COA_ID,
        "erp_status": "core template",
        "role": "plant",
        "ownership_pct": 1.00,
        "nci_pct": 0.00,
        "parallel_ledgers": True,
        "consolidated_from_month": 0,
        "has_plant": True,
        "notes": "Plant; VAT rate baked into the revenue account (8400=19%, 8300=7%).",
    },
    "ES01": {
        "name": "Volteo Iberica S.L.",
        "currency": "EUR",
        "country": "ES",
        "gaap": "PGC",
        "coa_id": PGC_COA_ID,
        "erp_status": "legacy",
        "role": "distributor",
        "ownership_pct": 0.80,
        "nci_pct": 0.20,
        "parallel_ledgers": False,
        "consolidated_from_month": 0,
        "has_plant": False,
        "notes": "80% owned -> NCI 20%; geography baked into accounts (7000/7001/7002).",
    },
    "UK01": {
        "name": "Volteo Electronics UK Ltd.",
        "currency": "GBP",
        "country": "UK",
        "gaap": "FRS 102",
        "coa_id": SAGE_COA_ID,
        "erp_status": "legacy, acquired FY24",
        "role": "acquired",
        "ownership_pct": 1.00,
        "nci_pct": 0.00,
        "parallel_ledgers": False,
        "consolidated_from_month": 8,   # mid-history acquisition -> scope effect
        "has_plant": False,
        "notes": "Acquired mid-history; appears in consolidation only from month 8.",
    },
}


def build_entities(cfg: Config) -> pl.DataFrame:
    keys = cfg.entities
    unknown = [k for k in keys if k not in ENTITY_PROFILES]
    if unknown:
        raise KeyError(f"no profile for configured entities: {unknown}")

    parallel_for = set(cfg.get("scenarios.parallel_ledger_entities", []))
    rows = []
    for key in keys:
        profile = dict(ENTITY_PROFILES[key])
        profile["entity"] = key
        profile["parallel_ledgers"] = key in parallel_for
        profile["is_consolidated"] = True
        rows.append(profile)

    # Derived reporting entities carry the group chart, not a local one.
    rows.append(
        {
            "entity": GROUP_ENTITY,
            "name": "Volteo Electronics Group (consolidated)",
            "currency": cfg.get("fx.group_currency"),
            "country": "US",
            "gaap": "US GAAP",
            "coa_id": "GROUP",
            "erp_status": "derived",
            "role": "group",
            "ownership_pct": 1.00,
            "nci_pct": 0.00,
            "parallel_ledgers": False,
            "consolidated_from_month": 0,
            "has_plant": False,
            "is_consolidated": False,
            "notes": "Derived: sum of entities + eliminations + topside.",
        }
    )
    rows.append(
        {
            "entity": ELIM_ENTITY,
            "name": "Consolidation eliminations",
            "currency": cfg.get("fx.group_currency"),
            "country": "--",
            "gaap": "--",
            "coa_id": "GROUP",
            "erp_status": "derived",
            "role": "elimination",
            "ownership_pct": 1.00,
            "nci_pct": 0.00,
            "parallel_ledgers": False,
            "consolidated_from_month": 0,
            "has_plant": False,
            "is_consolidated": False,
            "notes": "Rule-generated elimination rows only.",
        }
    )
    return pl.DataFrame(rows)


def build_ledgers(cfg: Config, entities: pl.DataFrame) -> pl.DataFrame:
    """Parallel ledgers exist only where the ERP supports them (spec 3)."""
    configured = list(cfg.get("scenarios.ledgers"))
    rows = []
    for row in entities.iter_rows(named=True):
        if row["role"] in ("group", "elimination"):
            ledgers = ["IFRS"]
        elif row["parallel_ledgers"]:
            ledgers = configured
        else:
            ledgers = ["LOCAL"]
        for ledger in ledgers:
            rows.append(
                {
                    "entity": row["entity"],
                    "ledger": ledger,
                    "is_primary": ledger == ("LOCAL" if not row["parallel_ledgers"] else configured[0]),
                    "description": {
                        "IFRS": "Group reporting basis",
                        "LOCAL": "Local statutory basis",
                        "TAX": "Tax basis (deltas on depreciation/provisions only)",
                    }.get(ledger, ledger),
                }
            )
    return pl.DataFrame(rows)
