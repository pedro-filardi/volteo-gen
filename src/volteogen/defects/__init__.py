"""Defect injectors (spec 9).

Every defect is individually toggleable, deterministic, and paired with an entry in the
answer key. The discipline that matters: a defect must be injected in a way the
validators can assert EXACTLY — "off by 1.3%", "3 accounts for 2 months" — because a
defect that only approximately reproduces is indistinguishable from a real bug.

Injectors run late, on assembled frames, so a toggled-off defect leaves the dataset
byte-identical to a clean build.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ..config import Config
from ..rng import SeedBank

COVID_ACCOUNT_NAME = "COVID-19 support costs (discontinued)"


@dataclass
class DefectLog:
    """What was actually injected, so the answer key can state facts not intentions."""

    entries: list[dict] = field(default_factory=list)

    def record(self, key: str, **detail) -> None:
        self.entries.append({"defect": key, **detail})

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.entries) if self.entries else pl.DataFrame(
            {"defect": [], "detail": []}
        )


def inject_unmapped_accounts(
    cfg: Config, seeds: SeedBank, local_to_group: pl.DataFrame, log: DefectLog
) -> pl.DataFrame:
    """Spec 9.1 — 3 local accounts drop out of the group mapping for 2 months, then return.

    Group revenue is understated for exactly those months and restated afterwards, while
    the local trial balance never changes. Modelled with mapping validity windows rather
    than deletion, so the gap is discoverable in the data itself.
    """
    if not cfg.defect_enabled("unmapped_accounts") or local_to_group.height == 0:
        return local_to_group

    picks = []
    for coa_id, account_class in (("SKR03", "revenue_trade"), ("PGC", "revenue_trade"),
                                  ("US_ERP", "other_opex")):
        candidates = local_to_group.filter(
            (pl.col("coa_id") == coa_id) & (pl.col("account_class") == account_class)
        ).sort("local_account")
        if candidates.height:
            picks.append(candidates.row(0, named=True))
    if not picks:
        return local_to_group

    gap_from, gap_to = "2024-09", "2024-11"
    keys = {(p["coa_id"], p["local_account"]) for p in picks}
    frame = local_to_group.with_columns(
        pl.when(
            pl.struct(["coa_id", "local_account"]).map_elements(
                lambda s: (s["coa_id"], s["local_account"]) in keys, return_dtype=pl.Boolean
            )
        )
        .then(pl.lit(gap_from))
        .otherwise(pl.col("valid_from"))
        .alias("mapping_gap_from"),
    ).with_columns(
        pl.when(pl.col("mapping_gap_from") == gap_from)
        .then(pl.lit(gap_to))
        .otherwise(None)
        .alias("mapping_gap_to"),
        pl.when(pl.col("mapping_gap_from") == gap_from)
        .then(pl.lit(True))
        .otherwise(pl.col("is_defect"))
        .alias("is_defect"),
    )
    log.record(
        "unmapped_accounts",
        accounts=", ".join(f"{p['coa_id']}:{p['local_account']}" for p in picks),
        detail=f"unmapped {gap_from}..{gap_to} then restored",
    )
    return frame


def inject_mapping_drift_uk(
    cfg: Config, management_map: pl.DataFrame, log: DefectLog
) -> pl.DataFrame:
    """Spec 9.7 — the UK's direct management mapping contradicts the chained one.

    Two routes to the same management line return different answers, which is only
    findable by computing both.
    """
    if not cfg.defect_enabled("mapping_drift_uk") or management_map.height == 0:
        return management_map

    targets = management_map.filter(
        pl.col("management_line").is_not_null() & (pl.col("cc_function") != "~NA~")
    ).sort(["group_account", "cc_function"]).head(2)
    if targets.height == 0:
        return management_map

    drifted = targets.with_columns(
        pl.lit("UK01").alias("entity_override"),
        pl.when(pl.col("management_line") == "G&A")
        .then(pl.lit("Selling"))
        .otherwise(pl.lit("G&A"))
        .alias("management_line"),
        pl.lit("DEFECT: UK direct mapping disagrees with the chained mapping").alias("note"),
    )
    log.record(
        "mapping_drift_uk",
        accounts=", ".join(targets["group_account"].to_list()),
        detail="UK01 direct mapping overrides disagree with the chained route",
    )
    return pl.concat([management_map.with_columns(pl.lit(None, pl.Utf8).alias("entity_override")),
                      drifted], how="diagonal")


def inject_dormant_accounts(
    cfg: Config, seeds: SeedBank, gl: pl.DataFrame, accounts: pl.DataFrame, log: DefectLog
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Spec 9.8 — ~15 accounts whose history simply stops, one of them COVID-named.

    Only non-revenue accounts are made dormant, so the sub-ledger tie-out (V7) stays
    exact: dormancy is an opex-life story, not a revenue hole.
    """
    if not cfg.defect_enabled("dormant_accounts") or gl.height == 0:
        return gl, accounts
    rng = seeds.rng("defects", "dormant")

    candidates = (
        gl.filter(pl.col("account_class") == "other_opex")
        .select("entity", "account_code").unique().sort(["entity", "account_code"])
    )
    if candidates.height == 0:
        return gl, accounts
    take = min(15, candidates.height)
    chosen = candidates.head(take)
    cutoff = int(gl["fiscal_year"].min())

    keys = {(r["entity"], r["account_code"]) for r in chosen.to_dicts()}
    filtered = gl.filter(
        ~(
            pl.struct(["entity", "account_code"]).map_elements(
                lambda s: (s["entity"], s["account_code"]) in keys, return_dtype=pl.Boolean
            )
            & (pl.col("fiscal_year") > cutoff)
        )
    )

    # One dormant account is renamed to make the reason obvious in hindsight.
    covid_code = chosen["account_code"][0]
    accounts = accounts.with_columns(
        pl.when(pl.col("code") == covid_code)
        .then(pl.lit(COVID_ACCOUNT_NAME))
        .otherwise(pl.col("name"))
        .alias("name")
    )
    log.record(
        "dormant_accounts",
        accounts=f"{take} accounts stop after FY{cutoff}",
        detail=f"{covid_code} renamed to '{COVID_ACCOUNT_NAME}'",
    )
    del rng
    return filtered, accounts


def inject_subledger_reconciling_item(
    cfg: Config, seeds: SeedBank, gl: pl.DataFrame, log: DefectLog
) -> pl.DataFrame:
    """Spec 9.12 — a manual GL correction that makes GL != sub-ledger for one month.

    Booked and documented, so V7 must fail for exactly one entity/month by exactly this
    amount. Anything else is a real break.
    """
    if not cfg.defect_enabled("subledger_reconciling_item") or gl.height == 0:
        return gl

    target = (
        gl.filter(
            (pl.col("account_class") == "revenue_trade")
            & (pl.col("entity") == "DE01")
            & (~pl.col("is_period13"))
            & pl.col("ic_txn_id").is_null()
        )
        .sort(["period_key", "account_code", "amount_lc"])
    )
    if target.height == 0:
        return gl
    template = target.row(target.height // 2, named=True)
    amount = -12_500.0

    # The template row already carries FX columns, so the group-currency measures must
    # be recomputed or the row silently breaks the translation identity (V6).
    row = {**template, "amount_lc": amount, "source": "manual_correction", "quantity": None}
    for measure, rate in (
        ("amount_gc_actual_rates", "fx_rate_avg"),
        ("amount_gc_budget_rates", "fx_rate_budget"),
    ):
        if rate in row and row.get(rate) is not None:
            row[measure] = round(amount * float(row[rate]), 2)
    correction = pl.DataFrame([row])
    log.record(
        "subledger_reconciling_item",
        accounts=f"{template['entity']} {template['account_code']}",
        detail=f"manual GL correction of {amount:,.2f} in {template['period_key']}",
    )
    return pl.concat([gl, correction], how="diagonal")


def inject_sku_relaunch(
    cfg: Config, seeds: SeedBank, products: pl.DataFrame, log: DefectLog
) -> pl.DataFrame:
    """Spec 9.10 / 4.1 — a SKU is discontinued and relaunched under a NEW code.

    Like-for-like SKU trend silently breaks unless the analyst links the codes. The link
    exists in the dimension (`relaunch_of`, `succeeded_by`) but nothing forces its use.
    """
    if not cfg.defect_enabled("sku_relaunch_new_code") or products.height == 0:
        return products

    leaves = products.filter(pl.col("is_leaf") & pl.col("sku_code").is_not_null()).sort("node_id")
    if leaves.height == 0:
        return products
    original = leaves.row(0, named=True)
    new_code = f"{original['sku_code']}-R2"

    products = products.with_columns(
        pl.when(pl.col("node_id") == original["node_id"])
        .then(pl.lit(new_code))
        .otherwise(pl.lit(None, pl.Utf8))
        .alias("succeeded_by"),
        pl.lit(None, pl.Utf8).alias("relaunch_of"),
    )
    successor = products.filter(pl.col("node_id") == original["node_id"]).with_columns(
        pl.lit(f"{original['node_id']}:R2").alias("node_id"),
        pl.lit(new_code).alias("name"),
        pl.lit(new_code).alias("sku_code"),
        pl.lit(None, pl.Utf8).alias("succeeded_by"),
        pl.lit(original["sku_code"]).alias("relaunch_of"),
    )
    log.record(
        "sku_relaunch_new_code",
        accounts=f"{original['sku_code']} -> {new_code}",
        detail="same product, new code; like-for-like trend breaks without the link",
    )
    return pl.concat([products, successor], how="diagonal")
