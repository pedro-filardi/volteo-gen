"""Parser for Microsoft Business Central AL demo-data codeunits.

The upstream US chart of accounts is expressed as executable AL, not a table, and it is
split across two layers that must be joined on the *AL procedure base name*:

* ``Apps/W1/.../Create*GLAccount*.Codeunit.al`` declares the industry-neutral account
  names as labels::

      BalanceSheetLbl: Label 'BALANCE SHEET', MaxLength = 100;

* ``Apps/US/.../CreateUSGLAccounts.Codeunit.al`` declares the US structure::

      ContosoGLAccount.InsertGLAccount(CreateGLAccount.TotalAssets(),
          CreateGLAccount.TotalAssetsName(),
          Enum::"G/L Account Income/Balance"::"Balance Sheet",
          Enum::"G/L Account Category"::Assets, SubCategory,
          Enum::"G/L Account Type"::"End-Total", '', '', 0,
          CreateGLAccount.Assets() + '..' + CreateGLAccount.TotalAssets(), ...);

  and binds names to US nominal codes::

      ContosoGLAccount.AddAccountForLocalization(CommonGLAccount.SalesDomesticName(), '40140');

So ``TotalAssets`` is the join key: label -> name, localization -> code, insert -> shape.
Account numbers are therefore never invented here; they come from the MIT-licensed
upstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Account names are declared as labels, with two suffix conventions upstream:
#   BalanceSheetLbl: Label 'BALANCE SHEET', MaxLength = 100;              (W1 files)
#   AccountReceivableDomesticTok: Label 'Account Receivable, Domestic', … (US file)
_LABEL_RE = re.compile(
    r"^\s*(?P<base>[A-Za-z0-9_]+?)(?:Lbl|Tok)\s*:\s*(?:Label|TextConst)\s*'(?P<text>(?:[^']|'')*)'",
    re.MULTILINE,
)

# AddAccountForLocalization(Qualifier.BaseName(), '40140');
_LOCALIZATION_RE = re.compile(
    r"AddAccountForLocalization\(\s*(?:[A-Za-z0-9_]+\.)?(?P<base>[A-Za-z0-9_]+)Name\(\)\s*,"
    r"\s*'(?P<code>[^']*)'\s*\)"
)

# SubCategory := Format(GLAccountCategoryMgt.GetEquipment(), 80);
_SUBCATEGORY_RE = re.compile(r"SubCategory\s*:=\s*(?P<expr>.+?);\s*$", re.MULTILINE)

_INSERT_RE = re.compile(r"InsertGLAccount\(")

# Enum::"G/L Account Type"::"End-Total"  |  Enum::"G/L Account Category"::Assets
_ENUM_RE = re.compile(r'Enum::"[^"]+"::(?:"(?P<quoted>[^"]*)"|(?P<bare>[A-Za-z0-9_]+))')

_PROC_CALL_RE = re.compile(r"(?:[A-Za-z0-9_]+\.)?(?P<base>[A-Za-z0-9_]+)\(\)")


@dataclass(frozen=True)
class GLAccountSpec:
    """One ``InsertGLAccount`` call, resolved as far as the sources allow."""

    base: str
    name_base: str
    income_balance: str
    category: str
    subcategory: str
    account_type: str
    indentation: int
    totaling_bases: tuple[str, ...]
    order: int


def _unescape(text: str) -> str:
    """AL escapes a literal apostrophe by doubling it."""
    return text.replace("''", "'")


def parse_labels(paths: list[Path]) -> dict[str, str]:
    """Collect ``<Base>Lbl: Label '<text>'`` across every supplied AL file."""
    labels: dict[str, str] = {}
    for path in paths:
        for match in _LABEL_RE.finditer(path.read_text(encoding="utf-8", errors="replace")):
            labels[match.group("base")] = _unescape(match.group("text"))
    return labels


def parse_localization_codes(path: Path) -> dict[str, str]:
    """Collect ``AddAccountForLocalization(<Base>Name(), '<code>')`` -> code.

    Entries with an empty code exist upstream (accounts a locale switches off, e.g. VAT
    accounts in the US); they are dropped so they never masquerade as real accounts.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    codes: dict[str, str] = {}
    for match in _LOCALIZATION_RE.finditer(text):
        code = match.group("code").strip()
        if code:
            codes[match.group("base")] = code
    return codes


def _split_args(arg_text: str) -> list[str]:
    """Split an argument list on top-level commas, respecting quotes and nesting."""
    args: list[str] = []
    depth = 0
    in_quote = False
    current: list[str] = []
    for ch in arg_text:
        if ch == "'":
            in_quote = not in_quote
            current.append(ch)
            continue
        if in_quote:
            current.append(ch)
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        args.append("".join(current).strip())
    return args


def _extract_call(text: str, start: int) -> tuple[str, int]:
    """Return the balanced argument text of the call whose '(' is at ``start``."""
    depth = 0
    in_quote = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx
    raise ValueError("unbalanced InsertGLAccount( call in AL source")


def _enum_value(arg: str) -> str:
    match = _ENUM_RE.search(arg)
    if not match:
        return arg.strip().strip("'")
    return match.group("quoted") or match.group("bare") or ""


def _readable_subcategory(expr: str | None) -> str:
    """Turn ``Format(GLAccountCategoryMgt.GetEquipment(), 80)`` into ``Equipment``."""
    if not expr:
        return ""
    match = re.search(r"Get([A-Za-z0-9_]+)\(\)", expr)
    if match:
        return re.sub(r"(?<!^)(?=[A-Z])", " ", match.group(1)).strip()
    match = re.search(r'::"?([A-Za-z0-9_ ]+)"?', expr)
    if match:
        return match.group(1).strip()
    return expr.strip().strip("'")


def parse_gl_account_inserts(path: Path) -> list[GLAccountSpec]:
    """Parse every ``InsertGLAccount`` call in source order.

    Source order matters: Business Central charts are read top-to-bottom, and the
    Begin-Total / End-Total pairs that define the hierarchy are only meaningful in that
    order.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    specs: list[GLAccountSpec] = []
    subcategory = ""
    order = 0

    pos = 0
    while True:
        match = _INSERT_RE.search(text, pos)
        if not match:
            break
        # Any SubCategory assignment between the previous call and this one applies.
        for sub in _SUBCATEGORY_RE.finditer(text, pos, match.start()):
            subcategory = _readable_subcategory(sub.group("expr"))

        arg_text, end = _extract_call(text, match.end() - 1)
        pos = end + 1
        args = _split_args(arg_text)
        if len(args) < 10:
            continue

        account_call = _PROC_CALL_RE.search(args[0])
        name_call = re.search(r"(?:[A-Za-z0-9_]+\.)?(?P<base>[A-Za-z0-9_]+)Name\(\)", args[1])
        if not account_call or not name_call:
            continue

        try:
            indentation = int(args[8].strip())
        except ValueError:
            indentation = 0

        specs.append(
            GLAccountSpec(
                base=account_call.group("base"),
                name_base=name_call.group("base"),
                income_balance=_enum_value(args[2]),
                category=_enum_value(args[3]),
                subcategory=subcategory,
                account_type=_enum_value(args[5]),
                indentation=indentation,
                totaling_bases=tuple(
                    m.group("base") for m in _PROC_CALL_RE.finditer(args[9])
                ),
                order=order,
            )
        )
        order += 1

    return specs
