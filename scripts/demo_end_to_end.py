"""End-to-end dress rehearsal on synthetic data.

Simulates three nights on the Sadr field, runs the full detection chain,
compares against the injected truth, measures completeness by injection-
recovery, and prints the incidence-rate sentence.

Run:  python scripts/demo_end_to_end.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from toxotes.snr import PhotometryModel, LX200_10_REDUCED, IMX571, PRINCIPIA
from toxotes.simulate import simulate_campaign
from toxotes.detect import (differential_photometry, common_mode_detrend,
                            detrend_per_night, rms_vs_magnitude, search_field,
                            required_fap, max_searchable_period, harmonic_match)
from toxotes.injection import completeness_map, mean_completeness, format_result
from toxotes import plots

TARGET = SkyCoord(305.5 * u.deg, 40.0 * u.deg)          # Sadr region
NIGHTS = ["2027-07-31", "2027-08-02", "2027-08-04"]     # near new moon
N_STARS = 300                                            # keep the demo quick
N_STARS_IN_REAL_FIELD = 1500                             # what you will really search


def main():
    model = PhotometryModel(LX200_10_REDUCED, IMX571, PRINCIPIA)
    print(model.report(mags=range(10, 15)))
    print()

    print("=== 1. simulate ===")
    camp = simulate_campaign(model, TARGET, NIGHTS, n_stars=N_STARS, n_variables=6)
    print("   ", camp.summary())
    for t in camp.truth:
        print(f"     injected star {t.star_index:4d}  {t.kind:8s} "
              f"P={t.period_hours:5.2f} h  amp={t.amplitude_mag*1000:6.1f} mmag")

    print("\n=== 2. differential photometry + detrending ===")
    resid = differential_photometry(camp.mags)
    print(f"    raw scatter    median {np.median(np.std(camp.mags,axis=1))*1000:6.2f} mmag")
    print(f"    after ensemble median {np.median(np.std(resid,axis=1))*1000:6.2f} mmag")
    resid = common_mode_detrend(resid, n_components=2)
    print(f"    after common   median {np.median(np.std(resid,axis=1))*1000:6.2f} mmag")
    resid = detrend_per_night(resid, camp.night_id, order=2)
    print(f"    after per-night median {np.median(np.std(resid,axis=1))*1000:6.2f} mmag")

    mags, rms = rms_vs_magnitude(resid, camp.true_mags)
    for lo, hi in [(10.5, 11.5), (11.5, 12.5), (12.5, 13.5)]:
        m = (mags >= lo) & (mags < hi)
        if m.any():
            pred = model.sigma_mag((lo + hi) / 2, 30.0)
            print(f"    V {lo}-{hi}: measured {np.median(rms[m])*1000:5.2f} mmag, "
                  f"predicted {pred*1000:5.2f} mmag")

    print("\n=== 3. period search ===")
    thresh = required_fap(N_STARS_IN_REAL_FIELD, 0.01)
    print(f"    searching {N_STARS} stars; threshold FAP < {thresh:.2e} "
          f"(Bonferroni for {N_STARS_IN_REAL_FIELD} stars, 1% global)")
    dets = search_field(camp.times, resid, camp.night_id,
                        n_stars_searched=N_STARS_IN_REAL_FIELD, global_fap=0.01,
                        verbose=True)
    truth_idx = {t.star_index: t for t in camp.truth}
    print(f"    {len(dets)} detections")
    for d in dets:
        t = truth_idx.get(d.star_index)
        if t is None:
            tag = "  <-- FALSE POSITIVE"
        else:
            ok = harmonic_match(d.period_hours, t.period_hours)
            note = "" if abs(d.period_hours / t.period_hours - 1) < 0.02 else \
                   ("  (harmonic - normal for eclipses)" if ok else "  PERIOD MISMATCH")
            tag = f"  <-- real (injected P={t.period_hours:.2f} h, " \
                  f"{t.amplitude_mag*1000:.0f} mmag){note}"
        print(f"      star {d.star_index:4d}  P={d.period_hours:5.2f} h  "
              f"amp={d.amplitude_mag*1000:5.1f} mmag  FAP={d.fap:.1e}  "
              f"nights={d.n_nights_present}{tag}")

    found = {d.star_index for d in dets}
    missed = [t for t in camp.truth if t.star_index not in found]
    print(f"    recovered {len(camp.truth)-len(missed)}/{len(camp.truth)} injected variables")
    for t in missed:
        print(f"      missed: P={t.period_hours:.2f} h amp={t.amplitude_mag*1000:.1f} mmag "
              f"(V={camp.true_mags[t.star_index]:.1f})")

    print("\n=== 4. injection-recovery completeness ===")
    quiet = np.array([resid[i] for i in range(N_STARS) if i not in truth_idx][:60])
    amps, pers, grid = completeness_map(
        camp.times, quiet, camp.night_id,
        amplitudes_mag=(0.002, 0.005, 0.010, 0.030),
        periods_hours=(0.5, 1.0, 2.0),
        n_realisations=6, n_stars_searched=N_STARS_IN_REAL_FIELD,
        power_threshold=search_field.last_power_threshold,
        max_period_h=search_field.last_max_period_h)
    print("    (measured through the identical detection criterion)")
    print("        " + "".join(f"{p:>8.1f}h" for p in pers))
    for i, a in enumerate(amps):
        print(f"    {a*1000:5.1f} mmag" + "".join(f"{grid[i,j]:9.0%}" for j in range(len(pers))))
    eff = mean_completeness(grid, amps, pers, amp_min=0.005)
    print(f"    mean completeness above 5 mmag: {eff:.0%}")

    print("\n=== 5. figures ===")
    print("   ", plots.rms_vs_magnitude(mags, rms, model=model, detections=dets))
    if dets:
        d = dets[0]
        print("   ", plots.periodogram(camp.times, resid[d.star_index],
                                       max_period_h=search_field.last_max_period_h,
                                       true_period_h=d.period_hours))
        print("   ", plots.phase_fold(camp.times, resid[d.star_index], d.period_hours,
                                      title=f"star {d.star_index} folded at "
                                            f"P = {d.period_hours:.4f} h"))

    print("\n=== 6. the sentence you will write ===")
    print("   ", format_result(len(dets), N_STARS_IN_REAL_FIELD, max(eff, 0.01),
                                max_period_h=search_field.last_max_period_h))


if __name__ == "__main__":
    main()
