"""Synthetic campaigns, so the whole pipeline can be built and validated
before a single real photon is collected.

The simulator deliberately includes the systematics that will actually
limit you: airmass extinction, transparency drift, a colour term, and
flat-field error as the field walks across the chip. If your detection
code survives this, it has a chance against real data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from .snr import PhotometryModel
from .observability import usable_hours, target_altaz, PRINCIPIA_LOCATION


@dataclass
class TruthEntry:
    star_index: int
    kind: str          # "sine" or "eclipse"
    period_hours: float
    amplitude_mag: float   # full amplitude
    phase: float


@dataclass
class Campaign:
    times: np.ndarray            # BJD-like, days
    airmass: np.ndarray          # per epoch
    mags: np.ndarray             # (n_stars, n_epochs) instrumental magnitudes
    true_mags: np.ndarray        # (n_stars,) catalogue magnitude
    colours: np.ndarray          # (n_stars,) Gaia BP-RP like
    truth: list = field(default_factory=list)
    night_id: np.ndarray = None

    @property
    def n_stars(self) -> int:
        return self.mags.shape[0]

    @property
    def n_epochs(self) -> int:
        return self.mags.shape[1]

    @property
    def baseline_days(self) -> float:
        return float(self.times[-1] - self.times[0])

    def summary(self) -> str:
        return (f"{self.n_stars} stars x {self.n_epochs} epochs, "
                f"{len(np.unique(self.night_id))} nights, "
                f"baseline {self.baseline_days:.1f} d, "
                f"{self.n_epochs * 32 / 3600:.1f} h on sky, "
                f"{len(self.truth)} injected variables")


def eclipse_shape(phase: np.ndarray, width: float = 0.12) -> np.ndarray:
    """Crude but adequate EW-like light curve: two minima per cycle."""
    p = phase % 1.0
    out = np.zeros_like(p)
    for centre, depth in ((0.0, 1.0), (0.5, 0.92)):
        d = np.abs((p - centre + 0.5) % 1.0 - 0.5)
        inside = d < width
        out[inside] -= depth * np.cos(np.pi * d[inside] / (2 * width)) ** 2
    return out


def simulate_campaign(model: PhotometryModel,
                      target: SkyCoord,
                      dates: list[str],
                      n_stars: int = 300,
                      mag_range: tuple[float, float] = (10.5, 13.5),
                      texp: float = 30.0, overhead: float = 2.0,
                      n_variables: int = 6,
                      rng: np.random.Generator | None = None,
                      systematics: bool = True) -> Campaign:
    rng = rng or np.random.default_rng(42)

    # --- epochs, from the real observability calculation ---
    times, airmass, night_id = [], [], []
    for i, d in enumerate(dates):
        _, blocks = usable_hours(target, d)
        for b in blocks:
            t0, t1 = b[0].jd, b[1].jd
            step = (texp + overhead) / 86400.0
            t = np.arange(t0, t1, step)
            alt, _ = target_altaz(target, __import__("astropy.time", fromlist=["Time"]).Time(t, format="jd"))
            times.append(t)
            airmass.append(1.0 / np.cos(np.radians(90 - np.clip(alt, 5, 89))))
            night_id.append(np.full(t.size, i))
    times = np.concatenate(times)
    airmass = np.concatenate(airmass)
    night_id = np.concatenate(night_id)
    n = times.size

    # --- the stellar population ---
    # more faint stars than bright ones, roughly log N ~ 0.35 V in the plane
    x = rng.random(n_stars)
    lo, hi = mag_range
    true_mags = lo + np.log10(1 + x * (10 ** (0.35 * (hi - lo)) - 1)) / 0.35
    colours = rng.normal(0.9, 0.45, n_stars)          # Gaia BP-RP like

    mags = np.repeat(true_mags[:, None], n, axis=1)

    # --- systematics ---
    if systematics:
        # common-mode transparency: smooth random walk, per night
        common = np.zeros(n)
        for i in np.unique(night_id):
            m = night_id == i
            w = np.cumsum(rng.normal(0, 0.004, m.sum()))
            common[m] = w - w.mean()
        common += 0.20 * (airmass - airmass.mean())            # grey extinction
        mags += common[None, :]

        # second-order extinction: colour couples to airmass
        mags += np.outer(0.03 * (colours - colours.mean()), airmass - airmass.mean())

        # flat-field error as the field drifts (slow, per-star, per-night)
        for i in np.unique(night_id):
            m = night_id == i
            drift = np.linspace(0, 1, m.sum())
            mags[:, m] += np.outer(rng.normal(0, 0.0025, n_stars), drift)

    # --- injected variables ---
    truth = []
    chosen = rng.choice(n_stars, n_variables, replace=False)
    for j, idx in enumerate(chosen):
        kind = "eclipse" if j % 2 else "sine"
        period_h = float(rng.uniform(0.4, 2.8))
        amp = float(10 ** rng.uniform(np.log10(0.006), np.log10(0.09)))   # 6-90 mmag
        ph = float(rng.random())
        phase = (times - times[0]) * 24 / period_h + ph
        if kind == "sine":
            mags[idx] += (amp / 2) * np.sin(2 * np.pi * phase)
        else:
            mags[idx] += (amp / 2) * eclipse_shape(phase)
        truth.append(TruthEntry(int(idx), kind, period_h, amp, ph))

    # --- photon noise, per star, from the real photon budget ---
    for i in range(n_stars):
        sigma = model.sigma_mag(true_mags[i], texp)
        mags[i] += rng.normal(0, sigma, n)

    return Campaign(times, airmass, mags, true_mags, colours, truth, night_id)
