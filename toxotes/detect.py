"""Differential photometry, detrending, and the period search.

The statistics here are the part that most amateur variable-star searches
get wrong. Searching ~1500 stars over ~2400 independent frequencies is
3.6 million trials; a naive "3-sigma" cut yields dozens of confident
detections, all of them noise.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from astropy.timeseries import LombScargle


@dataclass
class Detection:
    star_index: int
    period_hours: float
    power: float
    fap: float
    amplitude_mag: float
    n_nights_present: int


def required_fap(n_stars: int, global_fap: float = 0.01) -> float:
    """Per-star false-alarm probability (Bonferroni across stars).

    astropy's false_alarm_probability already accounts for the frequency
    trials within one star, so only the number of stars remains.
    """
    return global_fap / n_stars


def differential_photometry(mags: np.ndarray, comparison_mask: np.ndarray | None = None,
                            sigma_clip: float = 4.0) -> np.ndarray:
    """Subtract an ensemble comparison light curve from every star.

    A star is never included in its own comparison ensemble.
    """
    n_stars, n = mags.shape
    if comparison_mask is None:
        comparison_mask = np.ones(n_stars, dtype=bool)
    ref = mags[comparison_mask]
    ref = ref - np.median(ref, axis=1, keepdims=True)

    # robust ensemble: clip outlying stars epoch by epoch
    med = np.median(ref, axis=0)
    mad = 1.4826 * np.median(np.abs(ref - med), axis=0) + 1e-9
    good = np.abs(ref - med) < sigma_clip * mad
    ensemble = np.where(good, ref, np.nan)
    ensemble = np.nanmean(ensemble, axis=0)

    out = mags - ensemble[None, :]
    return out - np.median(out, axis=1, keepdims=True)


def common_mode_detrend(resid: np.ndarray, n_components: int = 2,
                        protect: np.ndarray | None = None) -> np.ndarray:
    """Remove the strongest trends shared across all stars (SysRem-lite).

    WARNING: over-detrending eats real signal. Always verify with
    injection-recovery that a signal at your claimed threshold survives
    this step at the amplitude you claim.
    """
    x = resid.copy()
    if protect is None:
        protect = np.zeros(resid.shape[0], dtype=bool)
    for _ in range(n_components):
        basis_rows = x[~protect] if (~protect).any() else x
        mode = np.median(basis_rows, axis=0)
        norm = np.dot(mode, mode)
        if norm <= 0:
            break
        coeff = x @ mode / norm
        x = x - np.outer(coeff, mode)
    return x


def detrend_per_night(resid: np.ndarray, night_id: np.ndarray, order: int = 2) -> np.ndarray:
    """Remove a low-order polynomial from each star on each night.

    Slow per-star drifts - flat-field error as the field walks across the
    chip, dew, differential refraction - are NOT common modes, so the
    ensemble does not touch them. They put enormous power at periods
    comparable to the length of a night, and they are the single most
    common source of spurious "discoveries". Killing them costs you
    sensitivity to periods near the block length, which is why
    `max_searchable_period` exists.
    """
    out = resid.copy()
    for nid in np.unique(night_id):
        m = night_id == nid
        if m.sum() < order + 2:
            continue
        x = np.linspace(-1, 1, m.sum())
        basis = np.vander(x, order + 1)
        coef, *_ = np.linalg.lstsq(basis, out[:, m].T, rcond=None)
        out[:, m] -= (basis @ coef).T
    return out


def max_searchable_period(times: np.ndarray, night_id: np.ndarray,
                          fraction: float = 0.45) -> float:
    """Longest period you can honestly claim, in hours.

    A signal must complete at least ~1/`fraction` cycles inside a single
    continuous block, otherwise it is indistinguishable from the drift
    that `detrend_per_night` just removed.
    """
    blocks = [24 * (times[night_id == n].max() - times[night_id == n].min())
              for n in np.unique(night_id)]
    return float(fraction * np.median(blocks))


def empirical_power_threshold(peak_powers: np.ndarray, percentile: float = 99.0,
                              trim_top: float = 0.05) -> float:
    """Significance threshold taken from the field itself.

    The non-variable stars in your own field are the best possible null
    distribution: they carry exactly your red noise, your systematics and
    your window function. Analytic false-alarm probabilities assume white
    noise and will lie to you by many orders of magnitude.

    But that null distribution is contaminated by the very signals you are
    hunting: if 2% of stars are variable, the 99.5th percentile sits inside
    the variable population and you threshold your own discoveries away.
    So trim the top `trim_top` fraction before estimating the tail.
    """
    cut = np.percentile(peak_powers, 100 * (1 - trim_top))
    bulk = peak_powers[peak_powers <= cut]
    if bulk.size < 20:
        bulk = peak_powers
    return float(np.percentile(bulk, percentile))


def harmonic_match(found: float, injected: float, tol: float = 0.02) -> bool:
    """True if `found` is the injected period or one of its harmonics.

    Eclipsing binaries have two minima per orbit, so a periodogram
    legitimately peaks at P/2 - and a sharp eclipse profile carries power
    at P/3 and P/4 too. This is physics, not an error. Always inspect the
    phase-folded curve at both P and 2P before deciding which is the
    true period.
    """
    r = found / injected
    return min(abs(r - k) for k in (1, 0.5, 2, 1/3, 3, 0.25, 4)) < tol


def rms_vs_magnitude(resid: np.ndarray, mags: np.ndarray):
    """Scatter of each star, the plot that identifies candidates and tests H1."""
    return mags, np.std(resid, axis=1)


def window_function(times: np.ndarray, freqs_per_day: np.ndarray):
    """Periodogram of the sampling itself. Never read a periodogram without it."""
    y = np.ones_like(times)
    return LombScargle(times, y, fit_mean=False, center_data=False).power(freqs_per_day)


def search_star(times: np.ndarray, y: np.ndarray, night_id: np.ndarray,
                min_period_h: float = 0.2, max_period_h: float = 6.0,
                samples_per_peak: int = 8):
    ls = LombScargle(times, y)
    fmin, fmax = 24.0 / max_period_h, 24.0 / min_period_h
    freq, power = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax,
                               samples_per_peak=samples_per_peak)
    i = int(np.argmax(power))
    best_f, best_p = float(freq[i]), float(power[i])
    fap = float(ls.false_alarm_probability(best_p, minimum_frequency=fmin,
                                           maximum_frequency=fmax, method="baluev"))
    # amplitude from a direct sinusoid fit at the best frequency
    ph = 2 * np.pi * best_f * times
    A = np.vstack([np.ones_like(ph), np.sin(ph), np.cos(ph)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    amp = 2 * float(np.hypot(coef[1], coef[2]))

    # persistence: is the signal there on individual nights?
    present = 0
    for nid in np.unique(night_id):
        m = night_id == nid
        if m.sum() < 50:
            continue
        try:
            p_night = LombScargle(times[m], y[m]).power(np.array([best_f]))[0]
            if p_night > 3 * np.median(LombScargle(times[m], y[m]).autopower(
                    minimum_frequency=fmin, maximum_frequency=fmax)[1]):
                present += 1
        except Exception:
            pass
    return 24.0 / best_f, best_p, fap, amp, present


def search_field(times, resid, night_id, n_stars_searched=None,
                 global_fap=0.01, min_nights=2, percentile=99.0, trim_top=0.05,
                 max_period_h=None, verbose=False, **kw) -> list[Detection]:
    """Search every star, using the field itself to set the threshold.

    Two independent cuts must both pass:
      1. the analytic FAP, Bonferroni-corrected across stars (white noise);
      2. the peak power exceeds the `percentile` of all stars' peak powers
         (empirical, and therefore honest about red noise).
    Plus persistence across at least `min_nights` nights.
    """
    n_stars = resid.shape[0]
    if max_period_h is None:
        max_period_h = max_searchable_period(times, night_id)
    kw = {**kw, "max_period_h": max_period_h}
    fap_thresh = required_fap(n_stars_searched or n_stars, global_fap)

    results = [search_star(times, resid[i], night_id, **kw) for i in range(n_stars)]
    search_field.last_power_threshold = None  # set below, for reuse by injection tests
    peak_powers = np.array([r[1] for r in results])
    power_thresh = empirical_power_threshold(peak_powers, percentile, trim_top)
    if verbose:
        print(f"    max searchable period {max_period_h:.2f} h; "
              f"empirical power threshold {power_thresh:.4f} "
              f"({percentile}th percentile of {n_stars} stars, "
              f"top {trim_top:.0%} trimmed)")

    search_field.last_power_threshold = power_thresh
    search_field.last_max_period_h = max_period_h
    out = []
    for i, (P, power, fap, amp, present) in enumerate(results):
        if fap < fap_thresh and power > power_thresh and present >= min_nights:
            out.append(Detection(i, P, power, fap, amp, present))
    return sorted(out, key=lambda d: -d.power)
