"""Picks the real local account a posting lands on, per entity.

REALISM: local charts encode business meaning in the account number itself, and each
entity encodes something different. The router is where that shows up:

* **DE01 / SKR03** splits revenue by **VAT rate** (8400 = 19%, 8300 = 7%) — so German
  revenue splits across accounts by tax treatment, NOT by product.
* **ES01 / PGC** splits revenue by **geography** (7000 domestic, 7001 intra-community,
  7002 export) — so part of the market dimension is baked into the chart, and an
  analyst can get "export revenue" two different ways that must agree.
* **UK01 / Sage** has flat Sales Type A/B/C with no such encoding at all.
* **US01 / Business Central** carries the exploded product/channel axes.

Everything is chosen from accounts that actually exist in the loaded seed; the router
never fabricates a code.
"""

from __future__ import annotations

import polars as pl

# Accounts whose meaning we rely on, verified present in the seeds.
DE_REVENUE_BY_VAT = {"standard": "8400", "reduced": "8300"}
ES_REVENUE_BY_GEO = {"domestic": "7000", "eu": "7001", "export": "7002"}

EU_COUNTRIES = {"DE", "ES", "FR", "IT", "NL", "AT", "PT", "PL", "SE"}

# Where a posting goes when the local chart has no dedicated account for its class.
# Ordered most-specific first.
FALLBACK_CHAIN = {
    "cogs_variance": ("cogs_standard", "other_opex"),
    "depreciation": ("other_opex",),
    "tax": ("other_opex",),
    "interest": ("other_opex",),
    "fx_result": ("other_opex",),
    "marketing": ("other_opex",),
    "facilities": ("other_opex",),
    "revenue_reduction": ("revenue_trade",),
}


class AccountRouter:
    """Resolves ``(entity, account_class, context) -> local account code``."""

    def __init__(self, classified_accounts: pl.DataFrame, entity_coa: dict[str, str]) -> None:
        self.entity_coa = entity_coa
        self._by_coa_class: dict[tuple[str, str], list[dict]] = {}
        postings = classified_accounts.filter(pl.col("node_type") == "posting")
        for row in postings.to_dicts():
            key = (row["coa_id"], row["account_class"])
            self._by_coa_class.setdefault(key, []).append(row)
        self.substitutions: dict[tuple[str, str], str] = {}
        self._codes: dict[str, set[str]] = {}
        for row in postings.to_dicts():
            self._codes.setdefault(row["coa_id"], set()).add(row["code"])

    def candidates(self, entity: str, account_class: str) -> list[dict]:
        """Accounts of this class, falling back when the local chart lacks one.

        Charts genuinely differ in what they break out: the Spanish distributor and the
        small UK entity have no production-variance accounts because they do not
        manufacture, and the US locale subset carries no P&L tax account. Falling back
        is realistic; doing it silently is not, so every substitution is recorded in
        :attr:`substitutions` and reported in the dataset README.
        """
        coa = self.entity_coa[entity]
        found = self._by_coa_class.get((coa, account_class), [])
        if found:
            return found

        for fallback in FALLBACK_CHAIN.get(account_class, ()):
            found = self._by_coa_class.get((coa, fallback), [])
            if found:
                self.substitutions.setdefault((entity, account_class), fallback)
                return found

        raise KeyError(
            f"entity {entity} ({coa}) has no posting account classified as "
            f"{account_class!r} and no fallback applies — check "
            "config/account_class_rules.yaml"
        )

    def has_code(self, entity: str, code: str) -> bool:
        return code in self._codes.get(self.entity_coa[entity], set())

    def revenue_account(self, entity: str, country: str | None, category: str | None) -> str:
        """Where trade revenue posts — the entity-specific encoding lives here."""
        if entity == "DE01":
            # VAT rate, not product, decides the account. A small share of the range
            # (services/print-like categories) carries the reduced rate.
            vat = "reduced" if category in ("Services",) else "standard"
            code = DE_REVENUE_BY_VAT[vat]
            if self.has_code(entity, code):
                return code
        if entity == "ES01":
            if country == "ES":
                key = "domestic"
            elif country in EU_COUNTRIES:
                key = "eu"
            else:
                key = "export"
            code = ES_REVENUE_BY_GEO[key]
            if self.has_code(entity, code):
                return code
        return self._first(entity, "revenue_trade")

    def revenue_reduction_account(self, entity: str, kind: str) -> str:
        """Each gross-to-net component gets its own account where the chart allows."""
        options = sorted(self.candidates(entity, "revenue_reduction"), key=lambda r: r["code"])
        index = {"volume_rebate": 0, "mdf": 1, "returns_reserve": 2, "price_protection": 3}.get(kind, 0)
        return options[min(index, len(options) - 1)]["code"]

    def account_for(self, entity: str, account_class: str, offset: int = 0) -> str:
        options = sorted(self.candidates(entity, account_class), key=lambda r: r["code"])
        return options[offset % len(options)]["code"]

    def _first(self, entity: str, account_class: str) -> str:
        return sorted(self.candidates(entity, account_class), key=lambda r: r["code"])[0]["code"]

    def name_of(self, entity: str, code: str) -> str | None:
        coa = self.entity_coa[entity]
        for rows in self._by_coa_class.values():
            for row in rows:
                if row["coa_id"] == coa and row["code"] == code:
                    return row["name"]
        return None

    def class_of(self, entity: str, code: str) -> str | None:
        coa = self.entity_coa[entity]
        for (row_coa, account_class), rows in self._by_coa_class.items():
            if row_coa != coa:
                continue
            for row in rows:
                if row["code"] == code:
                    return account_class
        return None
