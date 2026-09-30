"""Customers: named accounts with real concentration, plus the anonymous tail.

REALISM (spec 4.5): revenue concentration is extreme and it matters — the top customer
takes 15-30% of an entity's revenue, and a long tail lands in ``OTHER-RETAIL`` /
``OTHER-ECOM`` buckets that carry no customer identity at all.

Intercompany affiliates ARE customers, flagged ``is_affiliate``. That flag is how
ic_partner tagging happens — and how it fails when a posting misses it (spec 9.3).
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from ..rng import SeedBank

CHANNELS = ["MASS_RETAIL", "DISTRIBUTION", "ECOM_D2C", "B2B"]

# Name banks per country so customers read as plausible local trade accounts.
NAME_BANKS = {
    "US": ["Northbridge Retail", "Cascade Electronics", "Summit Distribution", "BlueRiver Stores",
           "Pacific Tech Mart", "Redwood Wholesale", "Lakeshore Retail", "Ironside Supply"],
    "DE": ["Hansa Elektromarkt", "Rheinland Vertrieb", "Alpenblick Handel", "Nordstern Retail",
           "Bavaria Technik", "Weserland Grosshandel", "Elbtal Distribution", "Schwarzwald Markt"],
    "ES": ["Iberia Electro", "Mediterrania Retail", "Andalus Distribucion", "Catalonia Tech",
           "Levante Comercial", "Meseta Mayorista", "Costa Verde Retail", "Sierra Supply"],
    "UK": ["Thameside Retail", "Pennine Electricals", "Clyde Distribution", "Wessex Tech",
           "Mersey Wholesale", "Chiltern Stores", "Tyne Supply", "Anglia Retail"],
}

PAYMENT_TERMS = ["NET30", "NET45", "NET60", "NET90"]


def build_customers(
    cfg: Config, seeds: SeedBank, entities: list[str], entity_country: dict[str, str]
) -> pl.DataFrame:
    rng = seeds.rng("dims", "customers")
    named_per_entity = int(cfg.get("cardinality.customers.named_per_entity"))
    tail_share = float(cfg.get("cardinality.customers.tail_share"))
    top1_band = dict(cfg.get("cardinality.customers.top1_share"))

    rows: list[dict] = []
    for entity in entities:
        country = entity_country.get(entity, "US")
        bank = list(NAME_BANKS.get(country, NAME_BANKS["US"]))
        rng.shuffle(bank)

        # Named customers share (1 - tail_share) of revenue, with the top account
        # taking a configured 15-30% slice of the ENTITY total.
        top1 = float(rng.uniform(top1_band["min"], top1_band["max"]))
        named_total = max(0.05, 1.0 - tail_share)
        remaining = named_total - top1
        n_named = max(1, named_per_entity)

        if remaining <= 0:
            top1 = named_total * 0.6
            remaining = named_total - top1

        # Zipf-ish decay across the remaining named customers.
        decay = [1.0 / (i + 1) ** 1.25 for i in range(n_named - 1)]
        decay_sum = sum(decay) or 1.0
        shares = [top1] + [remaining * d / decay_sum for d in decay]

        for index in range(n_named):
            name = bank[index % len(bank)]
            if index >= len(bank):
                name = f"{name} {index // len(bank) + 1}"
            channel = (
                "MASS_RETAIL" if index == 0
                else str(rng.choice(CHANNELS, p=[0.3, 0.35, 0.2, 0.15]))
            )
            rows.append(
                {
                    "customer_id": f"CUST:{entity}:{index:03d}",
                    "entity": entity,
                    "customer_name": name,
                    "channel": channel,
                    "country": country,
                    "payment_terms": str(rng.choice(PAYMENT_TERMS)),
                    "customer_tier": "T1" if index == 0 else ("T2" if index < 4 else "T3"),
                    "revenue_share": round(float(shares[index]), 6),
                    "is_affiliate": False,
                    "is_bucket": False,
                }
            )

        # The anonymous tail: real revenue, no customer identity.
        for bucket, channel, split in (
            ("OTHER-RETAIL", "MASS_RETAIL", 0.55),
            ("OTHER-ECOM", "ECOM_D2C", 0.45),
        ):
            rows.append(
                {
                    "customer_id": f"CUST:{entity}:{bucket}",
                    "entity": entity,
                    "customer_name": bucket,
                    "channel": channel,
                    "country": country,
                    "payment_terms": "NET30",
                    "customer_tier": "T3",
                    "revenue_share": round(tail_share * split, 6),
                    "is_affiliate": False,
                    "is_bucket": True,
                }
            )

    # Affiliates are customers of the selling entity and vendors of the buying one.
    for seller in entities:
        for buyer in entities:
            if seller == buyer:
                continue
            rows.append(
                {
                    "customer_id": f"CUST:{seller}:IC:{buyer}",
                    "entity": seller,
                    "customer_name": f"{buyer} (affiliate)",
                    "channel": "B2B",
                    "country": entity_country.get(buyer, "US"),
                    "payment_terms": "NET60",
                    "customer_tier": "IC",
                    "revenue_share": 0.0,   # IC volume is driven by the IC engine
                    "is_affiliate": True,
                    "is_bucket": False,
                    "affiliate_entity": buyer,
                }
            )

    frame = pl.DataFrame(rows, infer_schema_length=None)

    # Normalise third-party shares to exactly 1.0 per entity so revenue splits are exact.
    external = frame.filter(~pl.col("is_affiliate"))
    totals = external.group_by("entity").agg(pl.col("revenue_share").sum().alias("total"))
    frame = frame.join(totals, on="entity", how="left").with_columns(
        pl.when(pl.col("is_affiliate"))
        .then(pl.lit(0.0))
        .otherwise(pl.col("revenue_share") / pl.col("total"))
        .alias("revenue_share")
    ).drop("total")

    return frame
