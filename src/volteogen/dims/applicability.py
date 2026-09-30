"""The dimensional applicability matrix (spec 5) — the core realism piece.

Answers "which accounts are reported at SKU and which are not, and what goes in the
cell when the dimension does not apply". Both the generator and the validator read this
same table, so the rule is stated exactly once.

Three states must all exist in the output data and must stay distinguishable:

``~NA~``        the dimension structurally does not apply (payroll x product). Correct.
``UNASSIGNED``  it should apply but was not tagged. A DEFECT (spec 9.11).
a real member   applies and is tagged.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from pathlib import Path

import polars as pl
import yaml

NA = "~NA~"
UNASSIGNED = "UNASSIGNED"

DIMENSIONS = ("product", "market", "customer", "cost_center", "ic_partner")

# Applicability verbs used in meta_account_dimensionality.csv
NOT_APPLICABLE = "na"
APPLICABLE = "applicable"
MANDATORY = "mandatory"
CONDITIONAL = "conditional"
CONDITIONAL_AFFILIATE = "conditional_affiliate"
CONDITIONAL_LOAN = "conditional_loan"

_REQUIRED = {APPLICABLE, MANDATORY}


@dataclass
class Applicability:
    """Loaded matrix + per-CoA account classification rules."""

    matrix: pl.DataFrame
    rules: dict
    overrides_applied: list[str]
    # Row-per-class dict view. The matrix is read once per GL row per dimension, so a
    # polars filter here would dominate the whole build.
    _lookup: dict[str, dict] = dataclasses_field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._lookup = {row["account_class"]: row for row in self.matrix.to_dicts()}

    def rule(self, account_class: str, dimension: str) -> str:
        row = self._lookup.get(account_class)
        if row is None:
            raise KeyError(f"account_class {account_class!r} is not in the matrix")
        return str(row[dimension])

    def applies(self, account_class: str, dimension: str) -> bool:
        """True when a real member is expected (so ``~NA~`` would be wrong)."""
        return self.rule(account_class, dimension) in _REQUIRED

    def is_conditional(self, account_class: str, dimension: str) -> bool:
        return self.rule(account_class, dimension).startswith(CONDITIONAL)

    def is_disabled(self, account_class: str) -> bool:
        """True when an industry pack switches a whole account class off.

        A pack sets `posting_grain = none` to say "this industry does not have this at
        all" — professional services has no plant, so no absorption or usage variance
        exists to post. That is different from posting zero, and different again from a
        dimension being `~NA~`.
        """
        row = self._lookup.get(account_class)
        return bool(row) and str(row.get("posting_grain", "")).strip().lower() == "none"

    def gl_product_grain(self, account_class: str) -> str:
        row = self.matrix.filter(pl.col("account_class") == account_class)
        return str(row["gl_product_grain"][0]) if row.height else NOT_APPLICABLE

    def subledger_product_grain(self, account_class: str) -> str:
        row = self.matrix.filter(pl.col("account_class") == account_class)
        return str(row["subledger_product_grain"][0]) if row.height else NOT_APPLICABLE

    @property
    def classes(self) -> list[str]:
        return self.matrix["account_class"].to_list()


def load_applicability(
    config_dir: Path, pack: dict | None = None
) -> Applicability:
    """Load the matrix, then apply the industry pack's deltas.

    An industry that genuinely lacks a concept (professional services has no inventory,
    so no COGS variance rows) expresses that as override rows here — never as an engine
    branch.
    """
    config_dir = Path(config_dir)
    matrix_path = config_dir / "meta_account_dimensionality.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"missing applicability matrix: {matrix_path}")
    matrix = pl.read_csv(matrix_path)

    rules_path = config_dir / "account_class_rules.yaml"
    if not rules_path.exists():
        raise FileNotFoundError(f"missing account class rules: {rules_path}")
    rules = yaml.safe_load(rules_path.read_text()) or {}

    applied: list[str] = []
    override_path = (pack or {}).get("applicability_overrides")
    if override_path and Path(override_path).exists():
        with Path(override_path).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                account_class = (row.get("account_class") or "").strip()
                dimension = (row.get("dimension") or "").strip()
                value = (row.get("override") or "").strip()
                if not (account_class and dimension and value):
                    continue
                if dimension not in matrix.columns:
                    raise ValueError(
                        f"applicability override names unknown dimension {dimension!r}"
                    )
                matrix = matrix.with_columns(
                    pl.when(pl.col("account_class") == account_class)
                    .then(pl.lit(value))
                    .otherwise(pl.col(dimension))
                    .alias(dimension)
                )
                applied.append(f"{account_class}.{dimension}={value}")

    return Applicability(matrix=matrix, rules=rules, overrides_applied=applied)


def classify_accounts(accounts: pl.DataFrame, applicability: Applicability) -> pl.DataFrame:
    """Attach ``account_class`` to every posting account by longest-prefix match.

    Longest-prefix rather than numeric ranges because real charts mix code lengths:
    PGC has 3-, 4-, 5- and 6-digit accounts in the same tree.
    """
    rules = applicability.rules
    known = set(applicability.classes)

    def classify(coa_id: str, code: str | None, income_balance: str) -> str:
        spec = rules.get(coa_id)
        if spec is None:
            raise KeyError(f"no account_class rules configured for CoA {coa_id!r}")
        if code:
            prefixes = spec.get("prefixes") or {}
            best_len = -1
            best_class = None
            for prefix, account_class in prefixes.items():
                prefix = str(prefix)
                if code.startswith(prefix) and len(prefix) > best_len:
                    best_len = len(prefix)
                    best_class = account_class
            if best_class:
                return best_class
        if income_balance == "PL":
            return spec.get("default_pl", "other_opex")
        return spec.get("default_bs", "balance_sheet")

    classified = accounts.with_columns(
        pl.struct(["coa_id", "code", "income_balance"])
        .map_elements(
            lambda s: classify(s["coa_id"], s["code"], s["income_balance"]),
            return_dtype=pl.Utf8,
        )
        .alias("account_class")
    )

    unknown = set(classified["account_class"].unique().to_list()) - known
    if unknown:
        raise ValueError(
            f"account_class_rules.yaml produced classes absent from the applicability "
            f"matrix: {sorted(unknown)}"
        )
    return classified


def resolve_dimension_value(
    applicability: Applicability,
    account_class: str,
    dimension: str,
    value: str | None,
    condition_met: bool = False,
) -> str:
    """Return what actually belongs in a fact cell for this account x dimension.

    This is the single place ``~NA~`` is produced, so a dimension can never quietly
    become NULL or the string "UNKNOWN".
    """
    rule = applicability.rule(account_class, dimension)
    if rule == NOT_APPLICABLE:
        return NA
    if rule.startswith(CONDITIONAL):
        return (value or UNASSIGNED) if condition_met else NA
    if value:
        return value
    return UNASSIGNED if rule in _REQUIRED else NA
