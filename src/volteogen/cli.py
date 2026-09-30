"""Command-line entry point: build, validate, export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from .config import REPO_ROOT
from .docs import write_dataset_readme
from .export import write_dataset
from .pipeline import build_dataset
from .validate.suite import run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="volteogen",
        description="Generate the Volteo Electronics Group synthetic finance dataset.",
    )
    parser.add_argument("--preset", default=None, help="scale preset: S, M, L, XL")
    parser.add_argument("--seed", type=int, default=None, help="override the master seed")
    parser.add_argument("--out", default=None, help="output directory (default: config output_dir)")
    parser.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE",
        help="override any config path, e.g. --set cardinality.products.target_leaves=400",
    )
    parser.add_argument("--formats", default=None, help="comma-separated: csv,parquet,duckdb,clickhouse_ddl")
    parser.add_argument("--no-export", action="store_true", help="build and validate only")
    parser.add_argument(
        "--fail-on-validation", action="store_true",
        help="exit non-zero if any non-skipped check fails (registered defects still pass)",
    )
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    for item in args.set:
        if "=" not in item:
            parser.error(f"--set expects PATH=VALUE, got {item!r}")
        path, value = item.split("=", 1)
        overrides[path.strip()] = value.strip()
    if args.seed is not None:
        overrides["seed"] = args.seed

    print("Building Volteo dataset...")
    dataset = build_dataset(preset=args.preset, overrides=overrides)

    print("\nTables:")
    pl.Config.set_tbl_rows(60)
    print(dataset.summary())

    print("\nValidation (spec 10):")
    report = run_all(dataset)
    pl.Config.set_fmt_str_lengths(100)
    print(report.frame().select("check", "status", "detail"))

    if dataset.notes:
        print("\nBuild notes:")
        for note in dataset.notes:
            print(f"  - {note}")

    if not args.no_export:
        output_dir = Path(args.out) if args.out else REPO_ROOT / str(dataset.config.get("output_dir"))
        formats = args.formats.split(",") if args.formats else None
        print(f"\nExporting to {output_dir} ...")
        written = write_dataset(dataset, output_dir, formats)
        for fmt, files in written.items():
            if files:
                print(f"  {fmt}: {len(files)} file(s)")
        readme = write_dataset_readme(dataset, report, output_dir)
        print(f"  README: {readme}")

    if args.fail_on_validation and not report.all_green:
        print("\nFAILED checks:", [r.check for r in report.failed], file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
