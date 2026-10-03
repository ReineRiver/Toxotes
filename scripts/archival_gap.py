"""Quantify the survey gap for the target field - prerequisites D2 and D3.

Produces the two measured numbers that turn the Survey Gap section from an
assertion into a result:

  * the fraction of band stars with no high-cadence archival light curve
  * the fraction blended at TESS resolution

THE TRICK: do not query an archive once per star. Pull the whole field once,
then compute the geometry locally with a KD-tree. One network call, a
thousand stars, a few seconds.

    python scripts/archival_gap.py --download      # one Gaia query -> CSV
    python scripts/archival_gap.py                 # analyse the CSV offline
    python scripts/archival_gap.py --selftest      # verify the maths, no network
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

# --- field definition: update once the centre is fixed ------------------------
FIELD_NAME = "Aquila"
RA, DEC = 293.06, 12.02
RADIUS_DEG = 0.39

BAND = (10.5, 15.0)          # the sample Paper B searches
SCIENCE_BIN = (10.5, 13.0)   # where ZTF saturates - the science
CONTROL_BIN = (13.0, 15.0)   # where ZTF has data - the validation sample
FAINT = 18.0                 # contaminants worth counting

TESS_PIXEL_ARCSEC = 21.0     # one TESS pixel
APERTURE_ARCSEC = 6.0        # ~2x your seeing: shares YOUR aperture
DELTA_G_BLEND = 1.5          # a neighbour this close in brightness matters
ZTF_SATURATION_G = 13.0      # ZTF saturates near r ~ 13; G ~ r for these stars

CATALOGUE = "results/field_gaia.csv"

GAIA_QUERY = """SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r})) = 1
  AND phot_g_mean_mag < {faint}"""


def download(path: str = CATALOGUE) -> None:
    """One query. Everything else is local."""
    from astroquery.gaia import Gaia
    adql = GAIA_QUERY.format(ra=RA, dec=DEC, r=RADIUS_DEG, faint=FAINT)
    print("querying Gaia (one call, this may take a minute)...")
    job = Gaia.launch_job_async(adql)
    t = job.get_results()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t.write(path, format="csv", overwrite=True)
    print(f"{len(t)} sources -> {path}")


def load(path: str = CATALOGUE):
    import numpy.lib.recfunctions  # noqa
    ra, dec, g = [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                ra.append(float(row["ra"])); dec.append(float(row["dec"]))
                g.append(float(row["phot_g_mean_mag"]))
            except (ValueError, KeyError, TypeError):
                continue
    return np.array(ra), np.array(dec), np.array(g)


def neighbour_fraction(ra, dec, g, mask, radius_arcsec, delta_g):
    """Fraction of `mask` stars with a neighbour inside `radius_arcsec`
    and within `delta_g` magnitudes.

    Uses a local KD-tree on a tangent-plane projection - accurate to far
    better than an arcsecond over a half-degree field, and instantaneous.
    """
    dec0 = np.radians(np.mean(dec))
    x = (ra - np.mean(ra)) * np.cos(dec0) * 3600.0      # arcsec
    y = (dec - np.mean(dec)) * 3600.0
    tree = cKDTree(np.column_stack([x, y]))
    idx = np.where(mask)[0]
    blended = 0
    for i in idx:
        near = tree.query_ball_point([x[i], y[i]], radius_arcsec)
        for j in near:
            if j != i and abs(g[j] - g[i]) <= delta_g:
                blended += 1
                break
    return blended / max(len(idx), 1), len(idx)


def report(ra, dec, g):
    area = math.pi * RADIUS_DEG ** 2
    band = (g >= BAND[0]) & (g <= BAND[1])
    sci = (g >= SCIENCE_BIN[0]) & (g < SCIENCE_BIN[1])
    ctl = (g >= CONTROL_BIN[0]) & (g <= CONTROL_BIN[1])

    print(f"\nField {FIELD_NAME}  RA {RA}  Dec {DEC}  r={RADIUS_DEG} deg "
          f"({area:.3f} deg^2)")
    print(f"  sources to G={FAINT}      {len(g):6d}   ({len(g)/area:.0f}/deg^2)")
    print(f"  band {BAND[0]}-{BAND[1]}          {band.sum():6d}")
    print(f"    science bin {SCIENCE_BIN[0]}-{SCIENCE_BIN[1]}  {sci.sum():6d}"
          f"   <- ZTF saturated")
    print(f"    control bin {CONTROL_BIN[0]}-{CONTROL_BIN[1]}  {ctl.sum():6d}"
          f"   <- ZTF has light curves")

    print("\n  blending (a neighbour within dG <= "
          f"{DELTA_G_BLEND}):")
    for label, radius in (("TESS pixel (21\")", TESS_PIXEL_ARCSEC),
                          ("your aperture (6\")", APERTURE_ARCSEC)):
        for name, m in (("band", band), ("science bin", sci)):
            f, n = neighbour_fraction(ra, dec, g, m, radius, DELTA_G_BLEND)
            print(f"    {label:20} {name:12} {f:6.1%}  of {n} stars")

    no_ztf = sci.sum()
    print("\n  --- sentences for the Survey Gap section ---")
    print(f"  Of the {band.sum()} stars with {BAND[0]} < G < {BAND[1]} in the target")
    print(f"  field, {no_ztf} ({no_ztf/max(band.sum(),1):.0%}) lie brighter than the ZTF")
    print(f"  saturation limit and therefore have no high-cadence archival")
    print(f"  light curve from that survey.")
    f_tess, _ = neighbour_fraction(ra, dec, g, sci, TESS_PIXEL_ARCSEC, DELTA_G_BLEND)
    print(f"  A further {f_tess:.0%} of these share a 21-arcsecond TESS pixel with a")
    print(f"  neighbour within {DELTA_G_BLEND} magnitudes.")
    print("\n  [Still to add by hand: the ATLAS and ASAS-SN check on a random")
    print("   subsample of ~200 stars, and the VSX cone search. See the docstring")
    print("   of check_subsample() below.]")


def check_subsample(n: int = 200):
    """Why you should NOT query all 1000 stars against every archive.

    ATLAS forced photometry needs a login and is rate-limited; ASAS-SN Sky
    Patrol is fast but still one call per star. Querying 1000 stars against
    both is hours of wall-clock and adds nothing statistically.

    Instead: draw a RANDOM subsample of ~200, check those properly, and quote
    the fraction with its binomial uncertainty, sqrt(p(1-p)/n). At n=200 and
    p=0.6 that is +/- 3.5% - far smaller than any other uncertainty in the
    paper, and completely defensible. Say in the methods that you did this.
    """
    p = 0.6
    print(f"  subsample n={n}: binomial uncertainty on a fraction near {p} is "
          f"+/- {math.sqrt(p*(1-p)/n):.1%}")


def selftest():
    """Verify the neighbour maths on a catalogue with a known answer."""
    rng = np.random.default_rng(0)
    # 500 random stars, plus 50 deliberately paired 3" apart
    n = 500
    ra = RA + (rng.random(n) - 0.5) * 0.6
    dec = DEC + (rng.random(n) - 0.5) * 0.6
    g = rng.uniform(10.5, 15.0, n)
    pra = ra[:50] + 3.0 / 3600 / math.cos(math.radians(DEC))
    pdec = dec[:50]
    pg = g[:50] + 0.5
    ra = np.concatenate([ra, pra]); dec = np.concatenate([dec, pdec])
    g = np.concatenate([g, pg])
    mask = np.ones(len(g), bool)
    f6, _ = neighbour_fraction(ra, dec, g, mask, 6.0, 1.5)
    expected = 100 / len(g)          # 50 pairs -> 100 stars have a close neighbour
    print(f"  selftest: paired fraction measured {f6:.3f}, expected >= {expected:.3f}"
          f"  {'PASS' if f6 >= expected - 0.01 else 'FAIL'}")
    check_subsample()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="run the one Gaia query")
    ap.add_argument("--selftest", action="store_true", help="verify offline")
    ap.add_argument("--catalogue", default=CATALOGUE)
    a = ap.parse_args()

    if a.selftest:
        selftest(); return 0
    if a.download:
        download(a.catalogue)
    if not os.path.exists(a.catalogue):
        print(f"no catalogue at {a.catalogue} - run with --download first")
        return 1
    report(*load(a.catalogue))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
