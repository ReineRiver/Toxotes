"""Injection-recovery completeness, and the incidence-rate statistics.

You are not entitled to quote a rate - or an upper limit - until you know
what fraction of real variables your pipeline would have found. You measure
that by injecting fake ones and counting how many come back.
"""
from __future__ import annotations
import numpy as np

from .detect import search_star, required_fap
from .simulate import eclipse_shape

# Poisson two-sided 95% confidence limits on a count.
_POISSON_95 = {0: (0.00, 3.00), 1: (0.05, 4.74), 2: (0.36, 6.30), 3: (0.82, 7.75),
               4: (1.37, 9.15), 5: (1.97, 10.51), 6: (2.61, 11.84), 7: (3.29, 13.15),
               8: (3.98, 14.43), 9: (4.70, 15.71), 10: (5.43, 16.96)}


def inject(times: np.ndarray, y: np.ndarray, period_hours: float,
           amplitude_mag: float, phase: float = 0.0, kind: str = "sine") -> np.ndarray:
    ph = (times - times[0]) * 24 / period_hours + phase
    if kind == "sine":
        return y + (amplitude_mag / 2) * np.sin(2 * np.pi * ph)
    return y + (amplitude_mag / 2) * eclipse_shape(ph)


def completeness_map(times: np.ndarray, quiet_curves: np.ndarray, night_id: np.ndarray,
                     amplitudes_mag=(0.002, 0.003, 0.005, 0.010, 0.020, 0.050),
                     periods_hours=(0.25, 0.5, 1.0, 2.0, 4.0, 6.0),
                     n_realisations: int = 8, n_stars_searched: int = 1500,
                     global_fap: float = 0.01, min_nights: int = 2,
                     kind: str = "sine", rng=None, detrend=None,
                     power_threshold: float = 0.0, max_period_h: float | None = None):
    """Fraction of injected signals recovered, on an amplitude x period grid.

    `quiet_curves` should be REAL (or realistic) light curves of stars that
    passed vetting as non-variable, so the noise and systematics are honest.
    `detrend`, if given, is applied after injection - which is the only way
    to find out whether your detrending is eating your science.

    CRITICAL: pass the same `power_threshold` and `max_period_h` that
    `search_field` used. Completeness measured with a weaker criterion than
    detection is worse than no completeness at all - it will inflate your
    efficiency and shrink your quoted rate.
    """
    rng = rng or np.random.default_rng(7)
    thresh = required_fap(n_stars_searched, global_fap)
    grid = np.zeros((len(amplitudes_mag), len(periods_hours)))

    for ia, amp in enumerate(amplitudes_mag):
        for ip, P in enumerate(periods_hours):
            hits = 0
            for k in range(n_realisations):
                base = quiet_curves[rng.integers(len(quiet_curves))]
                y = inject(times, base, P, amp, float(rng.random()), kind)
                if detrend is not None:
                    y = detrend(y)
                kw = {} if max_period_h is None else {"max_period_h": max_period_h}
                Pfit, power, fap, a, present = search_star(times, y, night_id, **kw)
                if (fap < thresh and power > power_threshold and present >= min_nights
                        and abs(Pfit - P) / P < 0.01):
                    hits += 1
            grid[ia, ip] = hits / n_realisations
    return np.array(amplitudes_mag), np.array(periods_hours), grid


def mean_completeness(grid: np.ndarray, amplitudes, periods,
                      amp_min: float = 0.005) -> float:
    """Completeness averaged over the range quoted in the research question."""
    m = np.array(amplitudes) >= amp_min
    return float(grid[m].mean()) if m.any() else 0.0


def incidence_rate(n_detected: int, n_stars: int, efficiency: float):
    """Rate with 95% Poisson confidence limits, corrected for completeness.

    Returns (rate, low, high). For n_detected = 0 the rate is 0 and `high`
    is the upper limit you report.
    """
    eff_n = n_stars * efficiency
    lo, hi = _POISSON_95[min(n_detected, 10)]
    return n_detected / eff_n, lo / eff_n, hi / eff_n


def format_result(n_detected: int, n_stars: int, efficiency: float,
                  amp_min_mmag: float = 5.0, max_period_h: float = 3.0) -> str:
    r, lo, hi = incidence_rate(n_detected, n_stars, efficiency)
    if n_detected == 0:
        return (f"No variables detected among {n_stars} stars. With mean detection "
                f"efficiency {efficiency:.0%}, the incidence of periodic variability "
                f"above {amp_min_mmag:.0f} mmag with P < {max_period_h:.0f} h is "
                f"< {hi:.3%} (95% CL).")
    return (f"{n_detected} variables detected among {n_stars} stars. Corrected for a "
            f"mean detection efficiency of {efficiency:.0%}, the incidence of periodic "
            f"variability above {amp_min_mmag:.0f} mmag with P < {max_period_h:.0f} h is "
            f"{r:.3%} (95% CL: {lo:.3%} - {hi:.3%}).")
