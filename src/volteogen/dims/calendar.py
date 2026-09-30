"""Fiscal calendar: a non-calendar year (Jul-Jun) plus the P13 close period.

REALISM (spec 4.6): P01 is July, so P05/P06 are November/December — the Black
Friday/Christmas peak lands mid-fiscal-year, not at year end. Any consumer that assumes
period number == calendar month will be wrong, which is intentional.

P13 is a real posting period, not a month: it carries close adjustments dated to the
last day of the fiscal year (spec 4.6, ~0.3% of annual expense).
"""

from __future__ import annotations

from datetime import date

import polars as pl

from ..config import Config


def _parse_start(value) -> tuple[int, int]:
    """Accept ``2023-07`` from YAML as either a string or a parsed date."""
    if isinstance(value, date):
        return value.year, value.month
    text = str(value).strip()
    year, month = text.split("-")[:2]
    return int(year), int(month)


def fiscal_year_of(year: int, month: int, fy_end_month: int) -> int:
    """FY is labelled by the calendar year in which it ENDS (Jul-2023 -> FY2024)."""
    return year + 1 if month > fy_end_month else year


def build_calendar(cfg: Config) -> pl.DataFrame:
    fy_end = str(cfg.get("period.fiscal_year_end"))
    fy_end_month = int(fy_end.split("-")[0])
    start_year, start_month = _parse_start(cfg.get("period.start"))
    months = cfg.months
    include_p13 = bool(cfg.get("period.period13", True))

    rows = []
    year, month = start_year, start_month
    for index in range(months):
        fy = fiscal_year_of(year, month, fy_end_month)
        period_no = (month - fy_end_month - 1) % 12 + 1
        rows.append(
            {
                "period_key": f"{year:04d}-{month:02d}",
                "calendar_year": year,
                "calendar_month": month,
                "fiscal_year": fy,
                "period_no": period_no,
                "period_label": f"P{period_no:02d}",
                "fy_period": f"FY{fy}-P{period_no:02d}",
                "quarter": f"Q{(period_no - 1) // 3 + 1}",
                "is_period13": False,
                "month_index": index,
                "period_start": date(year, month, 1),
                "period_end": date(year + (month == 12), (month % 12) + 1, 1),
            }
        )
        month += 1
        if month > 12:
            month = 1
            year += 1

    if include_p13:
        # One P13 per fiscal year that the window actually completes.
        by_fy: dict[int, dict] = {}
        for row in rows:
            by_fy.setdefault(row["fiscal_year"], {})[row["period_no"]] = row
        for fy, periods in sorted(by_fy.items()):
            if 12 not in periods:
                continue  # partial FY at the end of the window gets no close period
            last = periods[12]
            rows.append(
                {
                    "period_key": f"{fy:04d}-P13",
                    "calendar_year": last["calendar_year"],
                    "calendar_month": last["calendar_month"],
                    "fiscal_year": fy,
                    "period_no": 13,
                    "period_label": "P13",
                    "fy_period": f"FY{fy}-P13",
                    "quarter": "Q4",
                    "is_period13": True,
                    "month_index": last["month_index"],
                    "period_start": last["period_start"],
                    "period_end": last["period_end"],
                }
            )

    return pl.DataFrame(rows).sort(["fiscal_year", "period_no"])


def month_periods(calendar: pl.DataFrame) -> pl.DataFrame:
    """Only the 12 real months per FY — the grain drivers and subledgers use."""
    return calendar.filter(~pl.col("is_period13")).sort("month_index")


def build_scenarios(cfg: Config, calendar: pl.DataFrame) -> pl.DataFrame:
    """dim_scenario: actuals, budget versions, and forecast cycles per fiscal year."""
    rows = [
        {
            "scenario_key": "ACT",
            "scenario_type": "ACT",
            "version": "ACT",
            "fiscal_year": None,
            "as_of_period": None,
            "closed_months": None,
            "description": "Actuals",
        }
    ]

    fiscal_years = sorted(calendar["fiscal_year"].unique().to_list())
    complete_fys = [
        fy
        for fy in fiscal_years
        if calendar.filter(
            (pl.col("fiscal_year") == fy) & (~pl.col("is_period13"))
        ).height
        == 12
    ]

    for fy in complete_fys:
        for version in cfg.get("scenarios.budget_versions"):
            rows.append(
                {
                    "scenario_key": f"{version}_FY{fy}",
                    "scenario_type": "BUD",
                    "version": version,
                    "fiscal_year": fy,
                    "as_of_period": f"FY{fy - 1}-P10",
                    "closed_months": 0,
                    "description": (
                        "Board-approved budget" if version.endswith("V2") else "First-pass budget"
                    ),
                }
            )
        for cycle in cfg.get("scenarios.forecast_cycles"):
            closed = int(str(cycle).split("+")[0].replace("FC", ""))
            rows.append(
                {
                    "scenario_key": f"{cycle}_FY{fy}",
                    "scenario_type": "FC",
                    "version": cycle,
                    "fiscal_year": fy,
                    "as_of_period": f"FY{fy}-P{closed:02d}",
                    "closed_months": closed,
                    "description": f"Forecast after {closed} closed months",
                }
            )

    return pl.DataFrame(rows)
