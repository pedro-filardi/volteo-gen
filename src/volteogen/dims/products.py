"""Product hierarchy: Division -> Category -> Family -> Model -> SKU, deliberately ragged.

All industry knowledge (division names, variant axes, lifecycle rates, price erosion)
comes from the industry pack. The engine only knows how to build a ragged, unbalanced,
skip-level tree from shape priors.

Raggedness in the shipped consumer-electronics pack:
  * Audio reaches full depth 5 (SKUs are colour/region variants of models);
  * Smart Home stops at depth 4 (no model layer) and launches mid-history;
  * Accessories' Services branch stops at depth 2 with no SKUs at all.

Electronics lifecycle is harsher than most industries: ~15%/yr EOL, ~18%/yr launches,
street price eroding 15-25%/yr while std_cost falls slower, so margin compresses over
each model's life and is refreshed by the next-gen launch.
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from ..rng import SeedBank, draw_fanout
from .hierarchy import HierarchyBuilder

DIVISION, CATEGORY, FAMILY, MODEL, SKU, SERVICE = (
    "division",
    "category",
    "family",
    "model",
    "sku",
    "service",
)


def _band(rng, band: dict) -> float:
    return float(rng.uniform(float(band["min"]), float(band["max"])))


def build_products(cfg: Config, pack: dict, seeds: SeedBank, months: int) -> pl.DataFrame:
    rng = seeds.rng("dims", "products")
    taxonomy = pack["product_taxonomy"]
    economics = pack["economics"]

    target_leaves = int(cfg.get("cardinality.products.target_leaves"))
    fanout_spec = dict(cfg.get("cardinality.products.fanout"))
    skip_share = float(cfg.get("cardinality.products.skip_level_share"))

    banks = taxonomy["word_banks"]
    lifecycle = taxonomy["lifecycle"]
    erosion = taxonomy["price_erosion"]
    margins = economics["margin_profile"]
    price_bands = economics["list_price_bands"]
    expected_discount = float(economics["expected_blended_channel_discount"])

    builder = HierarchyBuilder("product")
    root = builder.add("PRD", "All Products", "root", None, is_skip_level=False)

    divisions = taxonomy["divisions"]
    total_weight = sum(float(d["weight"]) for d in divisions)

    # Give each division a leaf budget proportional to its demand weight, inflated where
    # the pack says the SKU tail explodes (cables: length x colour x connector).
    budgets: dict[str, int] = {}
    for division in divisions:
        share = float(division["weight"]) / total_weight
        multiplier = float(division.get("sku_tail_multiplier", 1.0))
        budgets[division["key"]] = max(3, int(round(target_leaves * share * multiplier)))
    scale = target_leaves / max(1, sum(budgets.values()))
    budgets = {k: max(2, int(round(v * scale))) for k, v in budgets.items()}

    abc_shares = lifecycle["abc_shares"]
    abc_keys = list(abc_shares)
    abc_probs = [float(abc_shares[k]) for k in abc_keys]
    abc_probs = [p / sum(abc_probs) for p in abc_probs]

    rows_meta: list[dict] = []

    for division in divisions:
        div_key = division["key"]
        div_id = builder.add(
            f"PRD:{div_key}",
            division["name"],
            DIVISION,
            root,
            is_skip_level=False,
            division=division["name"],
        )
        launch_offset = int(division.get("launch_offset_months", 0))
        max_depth = int(division.get("max_depth", 5))
        budget = budgets[div_key]

        categories = division["categories"]
        for category in categories:
            cat_id = builder.add(
                f"PRD:{div_key}:{category['key']}",
                category["name"],
                CATEGORY,
                div_id,
                is_skip_level=False,
                division=division["name"],
            )

            # The Services branch is the ragged one: leaves hang straight off the
            # category with no family/model/sku layers at all.
            if category.get("leaves"):
                for index, leaf_name in enumerate(category["leaves"]):
                    leaf_id = f"PRD:{div_key}:{category['key']}:{index}"
                    builder.add(
                        leaf_id,
                        leaf_name,
                        SERVICE,
                        cat_id,
                        is_skip_level=False,
                        division=division["name"],
                        category=category["name"],
                    )
                    rows_meta.append(
                        _leaf_meta(
                            rng,
                            leaf_id,
                            leaf_name,
                            division,
                            category,
                            family_name=leaf_name,
                            months=months,
                            launch_offset=launch_offset,
                            lifecycle=lifecycle,
                            margins=margins,
                            price_bands=price_bands,
                            erosion=erosion,
                            abc_keys=abc_keys,
                            abc_probs=abc_probs,
                            taxonomy=taxonomy,
                            expected_discount=expected_discount,
                            is_service=True,
                        )
                    )
                continue

            families = category["families"]
            per_family = max(1, budget // max(1, len(categories) * len(families)))

            for family_name in families:
                family_id = builder.add(
                    f"PRD:{div_key}:{category['key']}:{_slug(family_name)}",
                    family_name,
                    FAMILY,
                    cat_id,
                    is_skip_level=False,
                    division=division["name"],
                    category=category["name"],
                )

                if max_depth >= 5:
                    n_models = int(draw_fanout(rng, 1, fanout_spec)[0])
                    n_models = max(1, min(n_models, max(1, per_family)))
                    for model_index in range(n_models):
                        model_name = _model_name(rng, banks, division, family_name, model_index)
                        # Skip-level: some models attach straight to the category,
                        # bypassing the family ("no family" models).
                        skip = rng.random() < skip_share
                        model_parent = cat_id if skip else family_id
                        model_id = builder.add(
                            f"PRD:{div_key}:{category['key']}:{_slug(family_name)}:{model_index}",
                            model_name,
                            MODEL,
                            model_parent,
                            is_skip_level=skip,
                            division=division["name"],
                            category=category["name"],
                        )
                        # Cap derived from the division's leaf budget, not a fixed 8,
                        # so target_leaves genuinely drives SKU cardinality.
                        variant_cap = max(2, min(24, int(round(budget / max(1, n_models * len(categories) * len(families))))))
                        n_variants = max(1, int(draw_fanout(rng, 1, fanout_spec)[0]))
                        n_variants = min(max(n_variants, variant_cap), 24)
                        for variant_index in range(n_variants):
                            sku_code = _sku_code(rng, banks, model_name, division, variant_index)
                            sku_id = f"{model_id}:{variant_index}"
                            builder.add(
                                sku_id,
                                sku_code,
                                SKU,
                                model_id,
                                is_skip_level=False,
                                division=division["name"],
                                category=category["name"],
                            )
                            rows_meta.append(
                                _leaf_meta(
                                    rng,
                                    sku_id,
                                    sku_code,
                                    division,
                                    category,
                                    family_name=family_name,
                                    months=months,
                                    launch_offset=launch_offset,
                                    lifecycle=lifecycle,
                                    margins=margins,
                                    price_bands=price_bands,
                                    erosion=erosion,
                                    abc_keys=abc_keys,
                                    abc_probs=abc_probs,
                                    taxonomy=taxonomy,
                            expected_discount=expected_discount,
                                    model_name=model_name,
                                )
                            )
                else:
                    # Ragged: this division has no model layer — leaves hang off the
                    # family. Some skip the family entirely and attach to the category,
                    # which is the shallow-tree equivalent of a "no family" model.
                    n_leaves = max(1, min(per_family, 24))
                    for variant_index in range(n_leaves):
                        sku_code = _sku_code(rng, banks, family_name, division, variant_index)
                        sku_id = f"PRD:{div_key}:{category['key']}:{_slug(family_name)}:{variant_index}"
                        skip = rng.random() < skip_share
                        builder.add(
                            sku_id,
                            sku_code,
                            SKU,
                            cat_id if skip else family_id,
                            is_skip_level=skip,
                            division=division["name"],
                            category=category["name"],
                        )
                        rows_meta.append(
                            _leaf_meta(
                                rng,
                                sku_id,
                                sku_code,
                                division,
                                category,
                                family_name=family_name,
                                months=months,
                                launch_offset=launch_offset,
                                lifecycle=lifecycle,
                                margins=margins,
                                price_bands=price_bands,
                                erosion=erosion,
                                abc_keys=abc_keys,
                                abc_probs=abc_probs,
                                taxonomy=taxonomy,
                            expected_discount=expected_discount,
                            )
                        )

    frame = builder.frame()
    meta = pl.DataFrame(rows_meta) if rows_meta else pl.DataFrame()
    frame = frame.join(meta, on="node_id", how="left")
    return _apply_lifecycle_events(frame, taxonomy, rng, months)


def _apply_lifecycle_events(
    frame: pl.DataFrame, taxonomy: dict, rng, months: int
) -> pl.DataFrame:
    """Succession chains and the recalled SKU (spec 4.1).

    Succession is what makes electronics margins recover: VX-ANC70 is superseded by
    VX-ANC80, the predecessor is cannibalised, and a like-for-like model trend needs the
    link to stay meaningful. The recall stops one SKU's sales abruptly and books a
    provision — an event no seasonal curve would ever produce.
    """
    lifecycle = taxonomy.get("lifecycle") or {}
    events = taxonomy.get("special_events") or {}

    frame = frame.with_columns(
        pl.lit(None, pl.Utf8).alias("supersedes"),
        pl.lit(False).alias("is_recalled"),
        pl.lit(None, pl.Int64).alias("recall_period"),
    )

    if lifecycle.get("succession_chains"):
        # Within a family, order models by launch and chain each to the next.
        models = frame.filter(
            (pl.col("node_type") == "model") & pl.col("parent_id").is_not_null()
        ).sort(["parent_id", "node_id"])
        successor_of: dict[str, str] = {}
        by_family: dict[str, list[dict]] = {}
        for row in models.to_dicts():
            by_family.setdefault(row["parent_id"], []).append(row)
        for family, group in by_family.items():
            for earlier, later in zip(group, group[1:]):
                successor_of[later["node_id"]] = earlier["name"]
        if successor_of:
            frame = frame.with_columns(
                pl.col("node_id").replace_strict(successor_of, default=None).alias("supersedes")
            )

    recall = events.get("recalled_sku")
    if recall:
        division_key = recall.get("division")
        candidates = frame.filter(
            pl.col("is_leaf")
            & pl.col("sku_code").is_not_null()
            & (pl.col("division_name").is_not_null())
        ).sort("node_id")
        if division_key:
            division_named = candidates.filter(
                pl.col("node_id").str.starts_with(f"PRD:{division_key}:")
            )
            if division_named.height:
                candidates = division_named
        if candidates.height:
            target = candidates.row(0, named=True)["node_id"]
            stop_at = max(1, int(months * 0.6))
            frame = frame.with_columns(
                (pl.col("node_id") == target).alias("is_recalled"),
                pl.when(pl.col("node_id") == target)
                .then(pl.lit(stop_at, pl.Int64))
                .otherwise(None)
                .alias("recall_period"),
                # Sales stop abruptly: the recall date becomes the effective end of life.
                pl.when(pl.col("node_id") == target)
                .then(pl.lit(stop_at, pl.Int64))
                .otherwise(pl.col("eol_period"))
                .alias("eol_period"),
            )
    return frame


def _slug(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())[:12]


def _model_name(rng, banks, division, family_name: str, index: int) -> str:
    prefix = str(rng.choice(banks["series_prefix"]))
    if division["key"] == "AUD":
        series = str(rng.choice(banks["audio_series"]))
        number = 30 + 10 * index + int(rng.integers(0, 5))
        return f"{prefix}-{series}{number}"
    if division["key"] == "PWR":
        token = str(rng.choice(banks["power_tokens"]))
        return f"{prefix}-{token.replace(' ', '')}"
    token = str(rng.choice(banks["smart_tokens"]))
    return f"{prefix}-{_slug(family_name)[:4]}{token}"


def _sku_code(rng, banks, base: str, division, index: int) -> str:
    axes = division.get("variant_axes") or ["color"]
    parts = [base]
    for axis in axes[:2]:
        if axis == "color":
            parts.append(str(rng.choice(banks["colors"])))
        elif axis == "region":
            parts.append(str(rng.choice(banks["regions"])))
        elif axis == "length":
            parts.append(str(rng.choice(banks["cable_lengths"])))
        elif axis == "connector":
            parts.append(str(rng.choice(banks["connectors"])))
    parts.append(f"{index:02d}")
    return "-".join(parts)


def _leaf_meta(
    rng,
    node_id: str,
    name: str,
    division: dict,
    category: dict,
    family_name: str,
    months: int,
    launch_offset: int,
    lifecycle: dict,
    margins: dict,
    price_bands: dict,
    erosion: dict,
    abc_keys: list[str],
    abc_probs: list[float],
    taxonomy: dict,
    expected_discount: float = 0.0,
    model_name: str | None = None,
    is_service: bool = False,
) -> dict:
    """Attributes that make a leaf economically real: price, cost, lifecycle window."""
    cat_name = category["name"]
    band = price_bands.get(cat_name, {"min": 20, "max": 120})
    list_price = round(_band(rng, band), 2)
    gross_margin = float(margins.get(cat_name, margins["default"]))
    # Margin is earned on the price the product actually sells at (MSRP less the blended
    # channel discount), never on MSRP itself.
    expected_net_price = list_price * (1.0 - expected_discount)
    std_cost = round(expected_net_price * (1.0 - gross_margin), 4)

    abc_class = str(rng.choice(abc_keys, p=abc_probs))

    life_band = lifecycle["median_life_months"]
    life = int(rng.integers(int(life_band["min"]), int(life_band["max"]) + 1))

    # Stagger launches across history; the division-level offset is what makes Smart
    # Home a mid-history launch with no prior-year budget structure.
    if launch_offset > 0:
        launch = launch_offset + int(rng.integers(0, max(1, months - launch_offset) // 2 + 1))
    else:
        launch = int(rng.integers(-24, max(1, months - 3)))
    launch = min(launch, months - 1)

    eol = launch + life
    eol_period = eol if eol < months else None

    if launch >= months - 3:
        stage = "launch"
    elif eol_period is not None and eol_period <= months - 1:
        stage = "eol"
    elif launch < 0:
        stage = "mature"
    else:
        stage = "growth"

    div_erosion = float(erosion["annual_pct"].get(division["name"], -0.15))
    trend = float((division.get("trend") or {}).get(family_name, 0.0))

    return {
        "node_id": node_id,
        "sku_code": name,
        "division_name": division["name"],
        "category_name": cat_name,
        "family_name": family_name,
        "model_name": model_name,
        "launch_period": launch,
        "eol_period": eol_period,
        "std_cost": std_cost,
        "list_price": list_price,
        "abc_class": abc_class,
        "lifecycle_stage": stage,
        "price_erosion_annual": div_erosion,
        "std_cost_erosion_annual": float(erosion["std_cost_annual_pct"]),
        "demand_trend_annual": trend,
        "is_service": is_service,
        "abc_weight": float(lifecycle["abc_demand_weight"][abc_class]),
    }
