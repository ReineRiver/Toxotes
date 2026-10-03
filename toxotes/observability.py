"""Night planning: twilight, target altitude, the moon, and usable hours.

Uses astropy so you learn the library you will need anyway. Runs entirely
offline (IERS auto-download is switched off; the built-in tables are far
more accurate than this project needs).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun, get_body
from astropy.time import Time
from astropy.utils import iers

iers.conf.auto_download = False
iers.conf.auto_max_age = None

PRINCIPIA_LOCATION = EarthLocation(lat=36.73335 * u.deg, lon=-4.42794 * u.deg, height=18 * u.m)
PUERTO_DE_LA_TORRE = EarthLocation(lat=36.735 * u.deg, lon=-4.508 * u.deg, height=180 * u.m)

# Local horizon profile: azimuth -> minimum usable altitude, degrees.
# MEASURE THIS. Until then a flat conservative floor is the honest default:
# it is better to plan against a horizon you might beat than one you might
# not reach. Replace with your Google Earth or on-site measurements as
# {azimuth: altitude} pairs; intermediate azimuths are interpolated.
HORIZON_FLOOR_DEG = 30.0
HORIZON_PROFILE: dict[float, float] | None = None     # e.g. {0: 12, 90: 25, 180: 34, 270: 28}


def horizon_at(az_deg):
    """Minimum usable altitude at a given azimuth."""
    import numpy as _np
    if not HORIZON_PROFILE:
        return _np.full_like(_np.asarray(az_deg, dtype=float), HORIZON_FLOOR_DEG)
    pts = sorted(HORIZON_PROFILE.items())
    xs = _np.array([p[0] for p in pts] + [pts[0][0] + 360])
    ys = _np.array([p[1] for p in pts] + [pts[0][1]])
    return _np.interp(_np.asarray(az_deg, dtype=float) % 360, xs, ys)


# Light domes seen from Principia: (bearing degrees, description).
PRINCIPIA_LIGHT_DOMES = [(156.0, "Malaga city centre, 1.5 km"),
                         (224.0, "airport, 9 km"),
                         (250.0, "Teatinos, 5 km")]


@dataclass
class NightWindow:
    start: Time
    end: Time
    label: str = ""

    @property
    def hours(self) -> float:
        return (self.end - self.start).to(u.hour).value


def _grid(date_local_noon: Time, step_min: float = 2.0) -> Time:
    n = int(24 * 60 / step_min)
    return date_local_noon + np.arange(n + 1) * step_min * u.min


def sun_altitude(times: Time, location=PRINCIPIA_LOCATION) -> np.ndarray:
    frame = AltAz(obstime=times, location=location)
    return get_sun(times).transform_to(frame).alt.deg


def dark_window(date_yyyy_mm_dd: str, location=PRINCIPIA_LOCATION,
                sun_alt_limit: float = -18.0, utc_offset_hours: float = 2.0) -> NightWindow | None:
    """Astronomical darkness for the night *starting* on the given local date."""
    noon = Time(f"{date_yyyy_mm_dd} 12:00:00") - utc_offset_hours * u.hour
    times = _grid(noon, step_min=1.0)
    alt = sun_altitude(times, location)
    below = alt < sun_alt_limit
    if not below.any():
        return None
    idx = np.where(below)[0]
    # contiguous run that contains the middle of the night
    breaks = np.where(np.diff(idx) > 1)[0]
    runs = np.split(idx, breaks + 1)
    run = max(runs, key=len)
    return NightWindow(times[run[0]], times[run[-1]],
                       f"{date_yyyy_mm_dd} sun<{sun_alt_limit:g}deg")


def target_altaz(target: SkyCoord, times: Time, location=PRINCIPIA_LOCATION):
    frame = AltAz(obstime=times, location=location)
    a = target.transform_to(frame)
    return a.alt.deg, a.az.deg


def moon_illumination(t: Time, location=PRINCIPIA_LOCATION) -> float:
    """Fraction of the lunar disc illuminated (0 = new, 1 = full)."""
    sun = get_sun(t)
    moon = get_body("moon", t, location)
    elong = sun.separation(moon).rad
    return float((1 - np.cos(elong)) / 2)


def moon_separation(target: SkyCoord, t: Time, location=PRINCIPIA_LOCATION) -> float:
    return float(target.separation(get_body("moon", t, location)).deg)


def field_rotation_rate(alt_deg, az_deg, latitude_deg: float = 36.733):
    """Alt-az field rotation, degrees per hour.

    Zero on the prime vertical (due east/west); diverges at the zenith.
    Returns zeros for an equatorially mounted telescope - which is what
    Principia has, so this is here for comparison, not for planning.
    """
    alt = np.radians(alt_deg)
    az = np.radians(az_deg)
    return np.abs(15.041 * np.cos(np.radians(latitude_deg)) * np.cos(az) / np.cos(alt))


def usable_hours(target: SkyCoord, date: str, location=PRINCIPIA_LOCATION,
                 min_altitude: float = 0.0, equatorial: bool = True,
                 max_field_rotation: float = 15.0,
                 dome_bearings=PRINCIPIA_LIGHT_DOMES, dome_halfwidth: float = 45.0,
                 dome_min_altitude: float = 40.0, step_min: float = 2.0):
    """Minutes of usable time and the contiguous blocks they fall in.

    Set equatorial=False to apply the field-rotation cut (relevant only if
    you ever go back to an alt-az mount).
    """
    win = dark_window(date, location)
    if win is None:
        return 0.0, []
    n = int(win.hours * 60 / step_min)
    times = win.start + np.arange(n + 1) * step_min * u.min
    alt, az = target_altaz(target, times, location)

    ok = (alt > min_altitude) & (alt > horizon_at(az))
    if not equatorial:
        ok &= field_rotation_rate(alt, az) < max_field_rotation
    for bearing, _ in dome_bearings:
        delta = np.abs((az - bearing + 180) % 360 - 180)
        ok &= ~((delta < dome_halfwidth) & (alt < dome_min_altitude))

    blocks = []
    if ok.any():
        idx = np.where(ok)[0]
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if len(run) * step_min >= 30:          # ignore slivers under 30 min
                blocks.append((times[run[0]], times[run[-1]],
                               float(alt[run[0]]), float(alt[run[-1]])))
    return float(ok.sum() * step_min / 60), blocks


def plan_night(target: SkyCoord, date: str, location=PRINCIPIA_LOCATION,
               utc_offset_hours: float = 2.0, **kw) -> str:
    win = dark_window(date, location)
    if win is None:
        return f"{date}: no astronomical darkness"
    hours, blocks = usable_hours(target, date, location, **kw)
    mid = win.start + (win.end - win.start) / 2
    illum = moon_illumination(mid, location)
    sep = moon_separation(target, mid, location)

    def local(t):
        return (t + utc_offset_hours * u.hour).iso[11:16]

    lines = [f"{date}  dark {local(win.start)}-{local(win.end)} "
             f"({win.hours:.1f} h)   moon {illum:.0%} illuminated, {sep:.0f} deg away",
             f"           usable: {hours:.2f} h"]
    for b in blocks:
        lines.append(f"             {local(b[0])}-{local(b[1])}  alt {b[2]:.0f} -> {b[3]:.0f} deg")
    if illum > 0.5 and sep < 60:
        lines.append("             WARNING: bright moon close to the field")
    return "\n".join(lines)
