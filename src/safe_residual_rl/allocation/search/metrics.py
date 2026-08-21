"""Anytime metrics with preregistered failure semantics for A4b v2."""

from __future__ import annotations

from typing import Mapping, Sequence


def normalized_gap(objective: float, *, target: float, reference: float) -> float:
    if reference <= target:
        raise ValueError("reference must be strictly worse than target")
    return min(1.0, max(0.0, (float(objective) - target) / (reference - target)))


def normalized_primal_integral(
    incumbent_events: Sequence[Mapping[str, float]],
    cutoff_s: float,
    *,
    target: float,
    reference: float,
) -> float:
    """Return time-normalized area under the incumbent-gap step function.

    Gap is one until the first feasible incumbent. Events after the cutoff are
    ignored. Lower is better; a run with no incumbent is exactly one.
    """
    if cutoff_s <= 0:
        raise ValueError("cutoff must be positive")
    eligible = sorted(
        (
            (float(item["elapsed_s"]), float(item["objective"]))
            for item in incumbent_events
            if float(item["elapsed_s"]) <= cutoff_s + 1e-12
        ),
        key=lambda item: (item[0], item[1]),
    )
    area = 0.0
    prior_t = 0.0
    prior_gap = 1.0
    best = float("inf")
    for elapsed, objective in eligible:
        area += prior_gap * max(0.0, elapsed - prior_t)
        best = min(best, objective)
        prior_gap = normalized_gap(best, target=target, reference=reference)
        prior_t = elapsed
    area += prior_gap * max(0.0, cutoff_s - prior_t)
    return area / cutoff_s


def time_to_target(
    incumbent_events: Sequence[Mapping[str, float]],
    cutoff_s: float,
    *,
    target: float,
) -> float | None:
    eligible = [
        float(item["elapsed_s"])
        for item in incumbent_events
        if float(item["elapsed_s"]) <= cutoff_s + 1e-12
        and float(item["objective"]) <= target + 1e-12
    ]
    return None if not eligible else min(eligible)

