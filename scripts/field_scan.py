"""Find the best field centre, in two stages.

Gaia's TAP rejects GROUP BY aggregation with HTTP 500, so this uses only
the query form known to work: a plain COUNT(*) with a WHERE clause, and a
plain SELECT of a small area.

    STAGE 1 - coarse map. ~100 small COUNT queries on a wide grid, to find
              where the star density actually peaks.

        python scripts/field_scan.py --coarse

    STAGE 2 - fine scan. Download the stars in the best few degrees only
              (a few tens of thousands of rows) and scan it properly.

        python scripts/field_scan.py --fine 296.0 17.0
        python scripts/field_scan.py                    # re-scan offline

    python scripts/field_scan.py --selftest             # verify, no network
"""
from __future__ import annotations

import os
import sys
import csv
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree

# --- stage 1: where to look ---------------------------------------------------
RA_RANGE = (283.0, 302.0)
DEC_RANGE = (5.0, 28.0)
COARSE_STEP = 2.0            # spacing of the coarse density probes
COARSE_RADIUS = 0.39         # each probe is one field-sized aperture

# --- stage 2: the fine scan ---------------------------------------------------
FINE_HALF_WIDTH = 1.5        # download this many degrees around the chosen centre
STEP_DEG = 0.25              # grid spacing of candidate centres
FIELD_RADIUS = 0.39

BAND = (10.5, 15.0)
SATURATION_G = 7.0
HALO_RADIUS_DEG = 0.45

COARSE_FILE = "results/region_coarse.csv"
FINE_FILE = "results/field_fine.csv"

# Extended objects to avoid. COORDINATES ARE APPROXIMATE - verify in Aladin.
AVOID = [
    ("M27 (Dumbbell Nebula)",     299.90, 22.72, 0.60),
    ("NGC 6823 / 6820 (Vul OB1)", 295.79, 23.30, 0.70),
    ("Cr 399 (Coathanger)",       291.35, 20.18, 0.70),
    ("NGC 6802 (open cluster)",   292.65, 20.27, 0.35),
    ("Stock 1 (open cluster)",    293.90, 25.20, 0.50),
]

COUNT_QUERY = """SELECT COUNT(*) AS n
FROM gaiadr3.gaia_source
WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r})) = 1
  AND phot_g_mean_mag BETWEEN {lo} AND {hi}"""

BOX_QUERY = """SELECT ra, dec, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE ra BETWEEN {ra0} AND {ra1} AND dec BETWEEN {dec0} AND {dec1}
  AND phot_g_mean_mag < {hi}"""


def galactic(ra, dec):
    a, d = np.radians(ra), np.radians(dec)
    rp, dp = math.radians(192.8595), math.radians(27.1284)
    lncp = math.radians(122.9320)
    b = np.arcsin(np.sin(d) * math.sin(dp) + np.cos(d) * math.cos(dp) * np.cos(a - rp))
    l = lncp - np.arctan2(np.cos(d) * np.sin(a - rp),
                          np.sin(d) * math.cos(dp) - np.cos(d) * math.sin(dp) * np.cos(a - rp))
    return np.degrees(l) % 360, np.degrees(b)


# ---------------------------------------------------------------- stage 1 ----
def coarse():
    from astroquery.gaia import Gaia
    os.makedirs("results", exist_ok=True)
    ras = np.arange(RA_RANGE[0] + 1, RA_RANGE[1] - 1, COARSE_STEP)
    decs = np.arange(DEC_RANGE[0] + 1, DEC_RANGE[1] - 1, COARSE_STEP)
    total = len(ras) * len(decs)
    print(f"stage 1: {total} probe queries on a {COARSE_STEP} deg grid "
          f"(a few minutes)\n")
    rows, k = [], 0
    for cdec in decs:
        line = []
        for cra in ras:
            k += 1
            q = COUNT_QUERY.format(ra=cra, dec=cdec, r=COARSE_RADIUS,
                                   lo=BAND[0], hi=BAND[1])
            try:
                n = int(Gaia.launch_job(q).get_results()["n"][0])
            except Exception as exc:
                print(f"  probe {k}/{total} failed: {exc}")
                n = -1
            rows.append(dict(ra=float(cra), dec=float(cdec), n=n))
            line.append(n)
        l, b = galactic(np.array(ras), np.full(len(ras), cdec))
        print(f"  Dec {cdec:+5.1f} (b {b.min():+5.1f} to {b.max():+5.1f}): "
              + " ".join(f"{v:5d}" for v in line))
    with open(COARSE_FILE, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ra", "dec", "n"])
        w.writeheader(); w.writerows(rows)

    good = [r for r in rows if r["n"] > 0]
    good.sort(key=lambda r: -r["n"])
    print(f"\nwritten: {COARSE_FILE}")
    print("\nDensest probes:")
    for r in good[:8]:
        l, b = galactic(np.array([r["ra"]]), np.array([r["dec"]]))
        print(f"  RA {r['ra']:7.2f}  Dec {r['dec']:+6.2f}  "
              f"l {l[0]:5.1f}  b {b[0]:+5.1f}   {r['n']:5d} stars")
    if good:
        best = good[0]
        print(f"\nNext:  python scripts/field_scan.py --fine "
              f"{best['ra']:.1f} {best['dec']:.1f}")


# ---------------------------------------------------------------- stage 2 ----
def fine(cra: float, cdec: float, half: float = FINE_HALF_WIDTH):
    from astroquery.gaia import Gaia
    os.makedirs("results", exist_ok=True)
    cd = math.cos(math.radians(cdec))
    q = BOX_QUERY.format(ra0=cra - half / cd, ra1=cra + half / cd,
                         dec0=cdec - half, dec1=cdec + half, hi=BAND[1])
    print(f"stage 2: stars within {half} deg of {cra} {cdec:+} ...")
    t = Gaia.launch_job_async(q).get_results()
    t.write(FINE_FILE, format="csv", overwrite=True)
    print(f"  {len(t):,} sources -> {FINE_FILE} "
          f"({os.path.getsize(FINE_FILE)/1e6:.1f} MB)")


def load_fine(path=FINE_FILE):
    from astropy.table import Table
    t = Table.read(path, format="ascii.csv", fast_reader=True)
    ra = np.asarray(t["ra"], float)
    dec = np.asarray(t["dec"], float)
    g = np.asarray(t["phot_g_mean_mag"], float)
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(g)
    return ra[ok], dec[ok], g[ok]


def scan(ra, dec, g, top=12):
    dec0 = float(np.mean(dec))
    cosd = math.cos(math.radians(dec0))
    band = (g >= BAND[0]) & (g <= BAND[1])
    bright = g < SATURATION_G
    tb = cKDTree(np.column_stack([ra[band] * cosd, dec[band]]))
    has_bright = bright.any()
    if has_bright:
        tbr = cKDTree(np.column_stack([ra[bright] * cosd, dec[bright]]))
        gbr = g[bright]

    lo_ra, hi_ra = ra.min() + FIELD_RADIUS / cosd, ra.max() - FIELD_RADIUS / cosd
    lo_de, hi_de = dec.min() + FIELD_RADIUS, dec.max() - FIELD_RADIUS
    rows = []
    for cra in np.arange(lo_ra, hi_ra, STEP_DEG):
        for cdec in np.arange(lo_de, hi_de, STEP_DEG):
            p = [cra * cosd, cdec]
            n_band = len(tb.query_ball_point(p, FIELD_RADIUS))
            if has_bright:
                hits = tbr.query_ball_point(p, HALO_RADIUS_DEG)
                n_bright, gmin = len(hits), (float(gbr[hits].min()) if hits else np.nan)
            else:
                n_bright, gmin = 0, np.nan
            avoid = next((nm for nm, ara, ade, arad in AVOID
                          if math.hypot((cra - ara) * cosd, cdec - ade)
                          < arad + FIELD_RADIUS), None)
            rows.append(dict(ra=cra, dec=cdec, n_band=n_band,
                             n_bright=n_bright, gmin=gmin, avoid=avoid))
    clean = [r for r in rows if r["avoid"] is None and r["n_bright"] == 0]
    clean.sort(key=lambda r: -r["n_band"])
    if len(clean) < top:
        rest = sorted((r for r in rows if r not in clean), key=lambda r: -r["n_band"])
        clean += rest[: top - len(clean)]
    return rows, clean[:top]


def report():
    ra, dec, g = load_fine()
    print(f"loaded {len(ra):,} sources")
    rows, best = scan(ra, dec, g)
    print(f"{len(rows)} candidate centres on a {STEP_DEG} deg grid")
    print(f"\nTop candidates ({BAND[0]} < G < {BAND[1]} within {FIELD_RADIUS} deg):")
    print(f"  {'RA':>8}{'Dec':>8}{'l':>7}{'b':>7}{'N_band':>9}{'bright':>8}{'Gmin':>7}  flag")
    for r in best:
        l, b = galactic(np.array([r["ra"]]), np.array([r["dec"]]))
        gm = "  -  " if not np.isfinite(r["gmin"]) else f"{r['gmin']:5.1f}"
        flag = r["avoid"] or ("" if r["n_bright"] == 0 else "bright star on frame")
        print(f"  {r['ra']:8.2f}{r['dec']:8.2f}{l[0]:7.1f}{b[0]:7.1f}"
              f"{r['n_band']:9d}{r['n_bright']:8d}{gm:>7}  {flag}")
    print("\nThen: check the top 3 in Aladin for nebulosity, run archival_gap.py")
    print("on each, and choose on the archival coverage fraction rather than the")
    print("raw count. Confirm each contains a usable EW binary before committing.")


def selftest():
    rng = np.random.default_rng(0)
    n = 40000
    ra = rng.uniform(294.5, 297.5, n); dec = rng.uniform(15.5, 18.5, n)
    ra = np.concatenate([ra, rng.normal(296.0, 0.12, 4000)])
    dec = np.concatenate([dec, rng.normal(17.0, 0.12, 4000)])
    g = rng.uniform(10.5, 15.0, len(ra))
    _, best = scan(ra, dec, g, top=1)
    r = best[0]
    ok = abs(r["ra"] - 296.0) < 0.4 and abs(r["dec"] - 17.0) < 0.4
    print(f"  selftest: planted overdensity at 296.00 +17.00, recovered "
          f"{r['ra']:.2f} {r['dec']:+.2f}  {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", action="store_true", help="stage 1: density map")
    ap.add_argument("--fine", nargs=2, type=float, metavar=("RA", "DEC"),
                    help="stage 2: download and scan around this centre")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return 0
    if a.coarse:
        coarse(); return 0
    if a.fine:
        fine(a.fine[0], a.fine[1])
    if not os.path.exists(FINE_FILE):
        print(f"no catalogue at {FINE_FILE} - run --coarse first, then --fine RA DEC")
        return 1
    report(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
