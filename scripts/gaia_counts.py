"""Star counts for the candidate fields - prerequisite D1.

RUN THIS ON YOUR OWN MACHINE: it is the only script here that needs the
internet. It queries the ESA Gaia archive, so if your network blocks it,
see the fallback at the bottom of this docstring.

    python scripts/gaia_counts.py
    python scripts/gaia_counts.py --dry-run     # check everything but the network

For each candidate field it measures three things:

  cumulative counts from G=10.5 down to several faint limits, so you can
  see exactly how much sample you buy by widening the band - and what
  null-result upper limit each choice would support.

  P_blend   probability that a given star has a neighbour within 6 arcsec
            (about 2x your seeing disc). Faint neighbours do not create
            false variability, but they add constant flux, which DILUTES a
            real variable's amplitude. Correct for it before quoting
            amplitudes.

  hrs       usable hours on one reference night (new moon, 2 Aug 2027),
            equatorial mount, target above 30 degrees.

  Pmax      longest honestly searchable period, ~0.45 x the block length.

  Amin      smallest detectable full amplitude at G=13 over 7 nights, mmag.
            Scales as 1/sqrt(frames), so a field that is richer but lower
            pays for its extra stars in sensitivity AND in period range.

  limit@    the 95% upper limit on the incidence rate you could report if
            you detected nothing, assuming 70% completeness. THIS is the
            quantity to optimise when choosing a field - but only among
            fields whose Pmax and Amin still answer the question.

Results are written to results/gaia_counts.csv so they are reproducible
and can go straight into the proposal.

FALLBACK if the archive is unreachable: go to gea.esac.esa.int/archive in a
browser, sign in (free), and paste the ADQL printed by --dry-run into the
query window by hand. Same numbers, more clicking.
"""
from __future__ import annotations

import os
import sys
import csv
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RADIUS_DEG = 0.39          # matches the f/6.3 reduced field, ~0.478 deg^2
BLEND_RADIUS_ARCSEC = 6.0  # ~2x seeing; anything closer shares your aperture

# Cumulative limits. The band Paper B searches is still being decided: the
# first Gaia counts came in 3-7x below the model estimate, because the Great
# Rift dust sits in FRONT of the Cygnus star clouds and pushes distant stars
# out of a bright magnitude band. Counting in bins shows exactly what
# widening the band buys, and at what cost in photometric precision.
LIMITS = [12.0, 13.5, 15.0, 16.5, 18.0]
BRIGHT = 10.5              # saturation-safe bright end

# All well clear of the Kepler prime field (centre 290.667, +44.5, radius ~8 deg).
# Chosen to span a range of Galactic latitude and dust column.
CANDIDATES = [
    ("Sadr / gamma Cyg",    305.5, 40.0),
    ("Cygnus rift",         310.0, 38.0),
    ("rho Cyg region",      311.7, 45.6),
    ("North America / OB7", 314.5, 44.0),
    ("Cyg, off-plane",      303.0, 45.0),
    ("Vulpecula",           292.5, 23.0),
    ("Vulpecula north",     297.0, 26.0),
    ("Sagitta",             296.5, 18.5),
    ("Lyra / Cyg border",   287.0, 33.0),
    # Richer directions, closer to the Galactic centre. Denser, but lower
    # from Malaga - and fewer hours means fewer frames, a worse amplitude
    # limit, and a shorter maximum searchable period. The table decides.
    ("Aquila (Altair reg)", 286.5,  3.0),
    ("Aquila south",        283.0, -1.0),
    ("Scutum star cloud",   281.5, -7.0),
]

REFERENCE_NIGHT = "2027-08-02"    # new moon
N_NIGHTS = 7
TEXP, OVERHEAD = 30.0, 2.0
LS_THRESHOLD_Z = 22.0
SIGMA_AT_G13 = 0.0083             # mag per 30 s frame, from toxotes.snr

COUNT_QUERY = """SELECT COUNT(*) AS n
FROM gaiadr3.gaia_source
WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r})) = 1
  AND phot_g_mean_mag BETWEEN {lo} AND {hi}"""


def area_deg2(radius_deg: float) -> float:
    return math.pi * radius_deg ** 2


def blend_probability(density_per_deg2: float, radius_arcsec: float) -> float:
    """P(at least one neighbour within `radius_arcsec`), Poisson."""
    r_deg = radius_arcsec / 3600.0
    return 1 - math.exp(-density_per_deg2 * math.pi * r_deg ** 2)


def galactic(ra: float, dec: float) -> tuple[float, float]:
    a, d = math.radians(ra), math.radians(dec)
    rp, dp = math.radians(192.8595), math.radians(27.1284)
    lncp = math.radians(122.9320)
    b = math.asin(math.sin(d) * math.sin(dp) + math.cos(d) * math.cos(dp) * math.cos(a - rp))
    l = lncp - math.atan2(math.cos(d) * math.sin(a - rp),
                          math.sin(d) * math.cos(dp) - math.cos(d) * math.sin(dp) * math.cos(a - rp))
    return math.degrees(l) % 360, math.degrees(b)


def poisson_upper_limit_rate(n_stars: int, efficiency: float = 0.7) -> float:
    """95% upper limit on the incidence rate if you detect nothing."""
    return 3.0 / (n_stars * efficiency) if n_stars else float("nan")


def observability(ra: float, dec: float, date: str = REFERENCE_NIGHT,
                  min_altitude: float = 30.0) -> float:
    """Usable hours on one night, equatorial mount, above `min_altitude`."""
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from toxotes.observability import usable_hours
    hours, _ = usable_hours(SkyCoord(ra * u.deg, dec * u.deg), date,
                            min_altitude=min_altitude, equatorial=True)
    return hours


def amplitude_limit(hours_per_night: float, n_nights: int = N_NIGHTS,
                    sigma: float = SIGMA_AT_G13) -> float:
    """Smallest detectable FULL amplitude at G=13, in magnitudes.

    Scales as 1/sqrt(number of frames), so halving your hours costs you a
    factor 1.4 in sensitivity. This is why a richer field low in the sky is
    not automatically a better field.
    """
    n = hours_per_night * n_nights * 3600 / (TEXP + OVERHEAD)
    return 2 * (2 * sigma * math.sqrt(LS_THRESHOLD_Z / n)) if n > 0 else float("inf")


def max_period(hours_per_night: float, fraction: float = 0.45) -> float:
    """Longest honestly searchable period, hours - see detect.max_searchable_period."""
    return fraction * hours_per_night


def kepler_separation(ra: float, dec: float) -> float:
    kra, kdec = 290.6667, 44.5
    a, b, c, d = map(math.radians, (ra, dec, kra, kdec))
    cosv = math.sin(b) * math.sin(d) + math.cos(b) * math.cos(d) * math.cos(a - c)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def run_query(adql: str) -> int:
    from astroquery.gaia import Gaia
    job = Gaia.launch_job(adql)
    return int(job.get_results()["n"][0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the ADQL and skip the network")
    args = ap.parse_args()

    area = area_deg2(RADIUS_DEG)
    print(f"Gaia DR3 counts, radius {RADIUS_DEG} deg ({area:.3f} deg^2)")
    print(f"cumulative from G={BRIGHT}, blend radius {BLEND_RADIUS_ARCSEC:.0f}\"\n")

    if args.dry_run:
        print("ADQL for the first field (paste into gea.esac.esa.int/archive "
              "if the script cannot reach the network):\n")
        print(COUNT_QUERY.format(ra=CANDIDATES[0][1], dec=CANDIDATES[0][2],
                                r=RADIUS_DEG, lo=BRIGHT, hi=LIMITS[1]))
        print()

    header = (f"{'field':21}{'l':>6}{'b':>6}"
              + "".join(f"{'<'+format(L,'.1f'):>8}" for L in LIMITS)
              + f"{'hrs':>6}{'Pmax':>6}{'Amin':>7}{'lim@13.5':>10}{'lim@15':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for name, ra, dec in CANDIDATES:
        sep = kepler_separation(ra, dec)
        l, b = galactic(ra, dec)
        cum = {}
        for L in LIMITS:
            if args.dry_run:
                cum[L] = 0
                continue
            try:
                cum[L] = run_query(COUNT_QUERY.format(ra=ra, dec=dec, r=RADIUS_DEG,
                                                      lo=BRIGHT, hi=L))
            except Exception as exc:
                print(f"\n  query failed for {name} at G<{L}: {exc}")
                print("  Use --dry-run to get the ADQL and run it in a browser instead.")
                return 1
        pb = blend_probability(cum[LIMITS[-1]] / area, BLEND_RADIUS_ARCSEC)
        hrs = observability(ra, dec)
        amin = amplitude_limit(hrs)
        pmax = max_period(hrs)
        lim135 = poisson_upper_limit_rate(cum[13.5])
        lim15 = poisson_upper_limit_rate(cum[15.0])
        print(f"{name:21}{l:6.1f}{b:6.1f}"
              + "".join(f"{cum[L]:8d}" for L in LIMITS)
              + f"{hrs:6.1f}{pmax:6.1f}{amin*1000:7.1f}{lim135:10.2%}{lim15:8.2%}")
        row = dict(field=name, ra=ra, dec=dec, gal_l=round(l, 2), gal_b=round(b, 2),
                   kepler_sep_deg=round(sep, 2), p_blend_6arcsec=round(pb, 4),
                   usable_hours=round(hrs, 2), max_period_h=round(pmax, 2),
                   amplitude_limit_mmag=round(amin * 1000, 2),
                   null_limit_G13_5=round(lim135, 5), null_limit_G15=round(lim15, 5))
        for L in LIMITS:
            row[f"n_G{L:.1f}"] = cum[L]
        rows.append(row)

    if not args.dry_run:
        os.makedirs("results", exist_ok=True)
        path = os.path.join("results", "gaia_counts.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten: {path}")

        best = max(rows, key=lambda r: r["n_G13.5"])
        widest = max(rows, key=lambda r: r["n_G15.0"])
        print(f"\nMost stars to G=13.5: {best['field']} ({best['n_G13.5']}), "
              f"null-result limit {best['null_limit_G13_5']:.2%}")
        print(f"Most stars to G=15.0: {widest['field']} ({widest['n_G15.0']}), "
              f"null-result limit {widest['null_limit_G15']:.2%}")
        print("\nHow to read this table:")
        print("  Do not pick on star count. A field that is richer because it is")
        print("  closer to the Galactic centre is also lower in the sky from Malaga,")
        print("  which costs you frames (worse Amin) and block length (shorter Pmax).")
        print("  A field with 3x the stars but half the hours has a period range")
        print("  halved and an amplitude limit 1.4x worse - usually a bad trade.")
        print("  Choose the best limit@ among fields where Pmax >= 2.5 h and")
        print("  Amin <= 5 mmag, then confirm with the D2 archival cross-match.")
        print("Do NOT choose on star count alone. Next (prerequisite D2): for the")
        print("top two or three fields, cross-match every band star against ZTF,")
        print("ATLAS, ASAS-SN and VSX, and pick the field that maximises the")
        print("fraction with NO high-cadence archival light curve. That fraction")
        print("is itself a publishable result and the justification for the campaign.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
