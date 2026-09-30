"""Deterministic, module-scoped randomness.

A single ``seed`` in config drives everything, but modules must be *independently*
reproducible: regenerating products must not shift the numbers payroll draws. So each
module derives a child seed from ``sha256(seed | module_name)`` rather than sharing one
global stream.

``hash()`` is deliberately avoided — Python salts string hashing per process, which
would make runs irreproducible across invocations.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(master_seed: int, *parts: str) -> int:
    """Stable 63-bit child seed from a master seed and a dotted module path."""
    key = f"{master_seed}|" + "|".join(parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class SeedBank:
    """Hands out one independent Generator per module path.

    Repeated calls for the same path return the *same* generator instance, so a module
    that draws in a loop keeps a single ordered stream; a fresh SeedBank reproduces it
    exactly.
    """

    def __init__(self, master_seed: int) -> None:
        self.master_seed = int(master_seed)
        self._generators: dict[tuple[str, ...], np.random.Generator] = {}

    def rng(self, *parts: str) -> np.random.Generator:
        if not parts:
            raise ValueError("SeedBank.rng() requires at least one module path part")
        key = tuple(parts)
        gen = self._generators.get(key)
        if gen is None:
            gen = np.random.default_rng(derive_seed(self.master_seed, *parts))
            self._generators[key] = gen
        return gen

    def fresh(self, *parts: str) -> np.random.Generator:
        """A generator that ignores any previously advanced state for this path.

        Use when a caller needs a stream keyed to a specific entity/period and must not
        depend on how many draws earlier callers made.
        """
        return np.random.default_rng(derive_seed(self.master_seed, *parts))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SeedBank(seed={self.master_seed}, streams={len(self._generators)})"


def stable_hash_unit(*parts: str) -> float:
    """Deterministic value in [0, 1) from string parts.

    For per-entity decisions that must not depend on iteration order (e.g. "does this
    SKU get an UNASSIGNED tag") — keyed by identity rather than by draw sequence.
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def zipf_fanout(
    rng: np.random.Generator, n: int, a: float, lo: int, hi: int
) -> np.ndarray:
    """Truncated Zipf draws — the fan-out shape real org/product trees actually have.

    numpy's ``zipf`` is unbounded, so resample the tail into range rather than clipping
    (clipping would pile mass on ``hi`` and flatten the very skew we want).
    """
    out = np.empty(n, dtype=np.int64)
    filled = 0
    while filled < n:
        draw = rng.zipf(a, size=max(16, (n - filled) * 4))
        draw = draw[(draw >= lo) & (draw <= hi)]
        take = min(len(draw), n - filled)
        out[filled : filled + take] = draw[:take]
        filled += take
    return out


def lognormal_fanout(
    rng: np.random.Generator, n: int, mu: float, sigma: float, lo: int, hi: int
) -> np.ndarray:
    """Truncated lognormal fan-out, resampled (not clipped) into ``[lo, hi]``."""
    out = np.empty(n, dtype=np.int64)
    filled = 0
    while filled < n:
        draw = np.rint(rng.lognormal(mu, sigma, size=max(16, (n - filled) * 4)))
        draw = draw[(draw >= lo) & (draw <= hi)]
        take = min(len(draw), n - filled)
        out[filled : filled + take] = draw[:take].astype(np.int64)
        filled += take
    return out


def draw_fanout(
    rng: np.random.Generator, n: int, spec: dict, lo: int | None = None, hi: int | None = None
) -> np.ndarray:
    """Dispatch a ``{dist: ...}`` config block to the right fan-out sampler."""
    dist = str(spec.get("dist", "zipf")).lower()
    lo = int(spec.get("min", 1) if lo is None else lo)
    hi = int(spec.get("max", 12) if hi is None else hi)
    if dist == "zipf":
        return zipf_fanout(rng, n, float(spec.get("a", 1.4)), lo, hi)
    if dist == "lognormal":
        return lognormal_fanout(
            rng, n, float(spec.get("mu", 1.0)), float(spec.get("sigma", 0.6)), lo, hi
        )
    if dist == "uniform":
        return rng.integers(lo, hi + 1, size=n, dtype=np.int64)
    raise ValueError(f"unknown fan-out distribution: {dist!r}")
