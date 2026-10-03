"""Photon budget for the campaign.

Everything here is a prediction you will later test against real data
(hypothesis H1). When the real camera specs arrive from Principia, change
the CAMERA preset and every number in the proposal regenerates.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

# V=0 star delivers ~995 photons/cm2/s/Angstrom; Johnson V is ~890 A wide.
PHOTONS_V0_PER_CM2_S = 995.0 * 890.0


@dataclass
class Telescope:
    aperture_m: float
    obstruction_m: float
    focal_length_mm: float
    optics_throughput: float   # all mirrors/lenses combined
    name: str = ""

    @property
    def area_cm2(self) -> float:
        return math.pi / 4 * ((self.aperture_m * 100) ** 2 - (self.obstruction_m * 100) ** 2)


@dataclass
class Camera:
    pixel_um: float
    read_noise_e: float
    full_well_e: float
    quantum_efficiency: float
    width_px: int
    height_px: int
    name: str = ""


@dataclass
class Site:
    sky_mag_arcsec2: float     # measure this with an SQM; do not trust the default
    seeing_arcsec: float
    extinction_k: float = 0.20
    name: str = ""


# --- presets -----------------------------------------------------------------
# Meade LX200 10 inch. Optics: corrector plate + primary + secondary.
LX200_10 = Telescope(0.254, 0.089, 2540.0, 0.92 * 0.88 * 0.88, "Meade LX200 10\"")
LX200_10_REDUCED = Telescope(0.254, 0.089, 1600.0, 0.92 * 0.88 * 0.88, "LX200 10\" + f/6.3 reducer")

# PLACEHOLDER - replace with the real camera once Principia confirms it (item A1).
IMX571 = Camera(3.76, 1.5, 51000.0, 0.70, 6248, 4176, "IMX571 (assumed)")

PRINCIPIA = Site(18.2, 3.0, 0.20, "Observatorio de Principia (sky assumed, measure it)")


class PhotometryModel:
    """Predicts photometric scatter for one telescope/camera/site combination."""

    def __init__(self, telescope: Telescope, camera: Camera, site: Site,
                 filter_transmission: float = 0.85, binning: int = 3):
        self.tel, self.cam, self.site = telescope, camera, site
        self.filter_transmission = filter_transmission
        self.binning = binning

    # --- geometry ---
    @property
    def plate_scale(self) -> float:
        """Arcsec per unbinned pixel."""
        return 206.265 * self.cam.pixel_um / self.tel.focal_length_mm

    @property
    def binned_scale(self) -> float:
        return self.plate_scale * self.binning

    @property
    def fov_arcmin(self) -> tuple[float, float]:
        w = self.cam.width_px * self.cam.pixel_um / 1000.0    # mm
        h = self.cam.height_px * self.cam.pixel_um / 1000.0
        return (w / self.tel.focal_length_mm * 206265 / 60,
                h / self.tel.focal_length_mm * 206265 / 60)

    @property
    def fov_deg2(self) -> float:
        w, h = self.fov_arcmin
        return w * h / 3600.0

    def sampling_check(self) -> str:
        px_per_fwhm = self.site.seeing_arcsec / self.binned_scale
        if px_per_fwhm < 2:
            return f"UNDERSAMPLED ({px_per_fwhm:.1f} px/FWHM) - reduce binning"
        if px_per_fwhm > 5:
            return f"oversampled ({px_per_fwhm:.1f} px/FWHM) - bin more, save disk"
        return f"well sampled ({px_per_fwhm:.1f} px/FWHM)"

    # --- photons ---
    def throughput(self, airmass: float = 1.05) -> float:
        atmospheric = 10 ** (-0.4 * self.site.extinction_k * airmass)
        return (self.tel.optics_throughput * self.filter_transmission
                * self.cam.quantum_efficiency * atmospheric)

    def rate_v0(self, airmass: float = 1.05) -> float:
        """Detected electrons per second from a V=0 star."""
        return PHOTONS_V0_PER_CM2_S * self.tel.area_cm2 * self.throughput(airmass)

    def aperture_radius(self) -> float:
        return 1.5 * self.site.seeing_arcsec

    def signal(self, V: float, texp: float, airmass: float = 1.05) -> float:
        return self.rate_v0(airmass) * 10 ** (-0.4 * V) * texp

    def sky_in_aperture(self, texp: float, airmass: float = 1.05) -> float:
        area = math.pi * self.aperture_radius() ** 2
        return self.rate_v0(airmass) * 10 ** (-0.4 * self.site.sky_mag_arcsec2) * area * texp

    def sigma_mag(self, V: float, texp: float, airmass: float = 1.05) -> float:
        """Photometric scatter in magnitudes for a single frame."""
        s = self.signal(V, texp, airmass)
        sky = self.sky_in_aperture(texp, airmass)
        npix = math.pi * self.aperture_radius() ** 2 / self.binned_scale ** 2
        noise = math.sqrt(s + sky + npix * self.cam.read_noise_e ** 2)
        return 1.0857 * noise / s

    def snr(self, V: float, texp: float, airmass: float = 1.05) -> float:
        return 1.0857 / self.sigma_mag(V, texp, airmass)

    # --- useful limits ---
    def crossover_magnitude(self, texp: float = 30.0) -> float:
        """Magnitude where star flux equals sky flux in the aperture.

        Brighter than this you are source-limited and light pollution
        barely matters - the core of hypothesis H2.
        """
        area = math.pi * self.aperture_radius() ** 2
        return self.site.sky_mag_arcsec2 - 2.5 * math.log10(area)

    def limiting_magnitude(self, target_sigma: float, texp: float = 30.0) -> float:
        lo, hi = 5.0, 24.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.sigma_mag(mid, texp) < target_sigma:
                lo = mid
            else:
                hi = mid
        return lo

    def saturation_magnitude(self, texp: float = 30.0) -> float:
        """Brightest V that stays inside the full well (unbinned peak pixel)."""
        sigma_px = self.site.seeing_arcsec / 2.355
        for V10 in range(40, 200):
            V = V10 / 10.0
            peak = (self.signal(V, texp) * self.plate_scale ** 2
                    / (2 * math.pi * sigma_px ** 2))
            if peak < self.cam.full_well_e:
                return V
        return 99.0

    def data_volume_gb_per_night(self, hours: float, texp: float, overhead: float = 2.0) -> float:
        frames = hours * 3600 / (texp + overhead)
        mb = (self.cam.width_px * self.cam.height_px / self.binning ** 2) * 2 / 1e6
        return frames * mb / 1000.0

    def report(self, texp: float = 30.0, mags=range(10, 17)) -> str:
        out = [f"{self.tel.name} + {self.cam.name} at {self.site.name}",
               f"  plate scale   {self.plate_scale:.3f}\"/px unbinned, "
               f"{self.binned_scale:.2f}\"/px at {self.binning}x{self.binning}",
               f"  sampling      {self.sampling_check()}",
               f"  field of view {self.fov_arcmin[0]:.1f}' x {self.fov_arcmin[1]:.1f}' "
               f"= {self.fov_deg2:.3f} deg^2",
               f"  throughput    {self.throughput():.3f}",
               f"  saturates at  V = {self.saturation_magnitude(texp):.1f} in {texp:.0f} s",
               f"  source/sky crossover at V = {self.crossover_magnitude(texp):.1f}",
               "",
               f"  {'V':>5}{'SNR':>9}{'sigma/frame':>13}{'sigma/5min':>12}{'sigma/30min':>13}"]
        nb5 = 5 * 60 / texp
        nb30 = 30 * 60 / texp
        for V in mags:
            s = self.sigma_mag(V, texp)
            out.append(f"  {V:5}{self.snr(V, texp):9.1f}{s:13.4f}"
                       f"{s / math.sqrt(nb5):12.4f}{s / math.sqrt(nb30):13.4f}")
        return "\n".join(out)
