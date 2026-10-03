"""The three figures Paper B actually needs.

Saves to disk rather than showing, so this works the same in VS Code, in a
terminal, and on a headless machine.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

from .style import use_paper_style, figure, SERIES, save as _save

use_paper_style()


def _finish(fig, path):
    """Save vector (for the journal) and raster (for drafts) side by side."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout(pad=0.3)
    stem = path[:-4] if path.endswith(".png") else path
    return ", ".join(_save(fig, stem))


def rms_vs_magnitude(mags, rms, model=None, texp=30.0, detections=None,
                     path="figures/rms_vs_mag.png"):
    """Scatter versus magnitude - your H1 test and your candidate finder.

    Real variables sit above the locus. The predicted curve is the photon
    budget; where the points sit above it is your systematics floor.
    """
    fig, ax = figure("a4text", aspect=0.55)
    ax.semilogy(mags, np.asarray(rms) * 1000, ".", ms=4, alpha=0.6,
                label="field stars")
    if model is not None:
        grid = np.linspace(min(mags) - 0.2, max(mags) + 0.2, 100)
        ax.semilogy(grid, [model.sigma_mag(v, texp) * 1000 for v in grid],
                    "-", lw=2, label="predicted (photon budget)")
    if detections:
        idx = [d.star_index for d in detections]
        ax.semilogy(np.asarray(mags)[idx], np.asarray(rms)[idx] * 1000,
                    "o", ms=9, mfc="none", mew=1.8, label="detections")
    ax.set_xlabel("V (mag)")
    ax.set_ylabel("scatter per frame (mmag)")
    ax.set_title("RMS versus magnitude")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return _finish(fig, path)


def periodogram(times, y, min_period_h=0.2, max_period_h=3.0,
                true_period_h=None, path="figures/periodogram.png"):
    """Periodogram with the window function underneath.

    Never read one without the other: the gaps between nights put alias
    structure at 1 cycle/day and its harmonics, and a peak sitting on an
    alias is not a detection.
    """
    ls = LombScargle(times, y)
    fmin, fmax = 24 / max_period_h, 24 / min_period_h
    freq, power = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax,
                               samples_per_peak=10)
    win = LombScargle(times, np.ones_like(times), fit_mean=False,
                      center_data=False).power(freq)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 4.0), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(freq, power, lw=0.8)
    if true_period_h:
        a1.axvline(24 / true_period_h, color="C3", ls="--", lw=1,
                   label=f"P = {true_period_h:.3f} h")
        a1.legend()
    a1.set_ylabel("LS power")
    a1.set_title("Periodogram")
    a2.plot(freq, win, lw=0.8, color="C1")
    a2.set_ylabel("window")
    a2.set_xlabel("frequency (cycles / day)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    return _finish(fig, path)


def phase_fold(times, y, period_hours, n_bins=60, path="figures/folded.png",
               title=None):
    """Phase-folded light curve - the figure that goes in the paper.

    Always plot it at both P and 2P before deciding which is the true
    period: an eclipsing binary folded at half its period superimposes the
    primary and secondary minima and can look convincing.
    """
    ph = ((times - times[0]) * 24 / period_hours) % 1.0
    fig, ax = figure("a4text", aspect=0.52)
    ax.plot(np.concatenate([ph, ph + 1]), np.concatenate([y, y]) * 1000,
            ".", ms=2.5, alpha=0.35, color="0.5")
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.digitize(ph, edges) - 1
    bx, by, be = [], [], []
    for b in range(n_bins):
        m = which == b
        if m.sum() > 2:
            bx.append((edges[b] + edges[b + 1]) / 2)
            by.append(np.median(y[m]) * 1000)
            be.append(1.4826 * np.median(np.abs(y[m] - np.median(y[m]))) * 1000
                      / np.sqrt(m.sum()))
    bx, by, be = np.array(bx), np.array(by), np.array(be)
    ax.errorbar(np.concatenate([bx, bx + 1]), np.concatenate([by, by]),
                yerr=np.concatenate([be, be]), fmt="o", ms=4, color="C0")
    ax.invert_yaxis()
    ax.set_xlabel("phase")
    ax.set_ylabel("differential magnitude (mmag)")
    ax.set_title(title or f"Folded at P = {period_hours:.4f} h")
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def oc_diagram(epochs, oc_days, errors=None, sources=None, fit=None,
               path="figures/oc_diagram.png", ylabel="O$-$C (days)"):
    """The Paper A figure: observed minus computed against epoch.

    `sources` labels where each point came from (your nights, BAV, AAVSO,
    literature) - each source gets its own marker AND colour, so the plot
    still reads in greyscale. `fit` is an optional (epochs, model) pair,
    normally the linear+quadratic ephemeris you fitted.
    """
    epochs = np.asarray(epochs, float)
    oc_days = np.asarray(oc_days, float)
    fig, ax = figure("a4text", aspect=0.5)

    if sources is None:
        sources = np.zeros(len(epochs), dtype=int)
    sources = np.asarray(sources)
    for i, name in enumerate(dict.fromkeys(sources)):
        m = sources == name
        kw = {**SERIES[i % len(SERIES)], "linestyle": "none"}
        ax.errorbar(epochs[m], oc_days[m],
                    yerr=None if errors is None else np.asarray(errors)[m],
                    label=str(name), **kw)
    if fit is not None:
        ax.plot(fit[0], fit[1], "-", color="0.25", lw=1.2, zorder=0,
                label="fitted ephemeris")
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.set_xlabel("epoch (cycles since $T_0$)")
    ax.set_ylabel(ylabel)
    if len(set(sources.tolist())) > 1 or fit is not None:
        ax.legend()
    return _finish(fig, path)
