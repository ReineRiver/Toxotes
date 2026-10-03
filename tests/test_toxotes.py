"""Sanity tests. Run:  PYTHONPATH=. python -m pytest tests/ -q"""
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from toxotes.snr import PhotometryModel, LX200_10_REDUCED, IMX571, PRINCIPIA
from toxotes.observability import dark_window, usable_hours
from toxotes.simulate import simulate_campaign
from toxotes.detect import (differential_photometry, detrend_per_night,
                            search_field, required_fap, harmonic_match,
                            max_searchable_period)

MODEL = PhotometryModel(LX200_10_REDUCED, IMX571, PRINCIPIA)
TARGET = SkyCoord(305.5 * u.deg, 40.0 * u.deg)


def test_photon_budget_is_monotonic():
    sigmas = [MODEL.sigma_mag(V, 30.0) for V in range(10, 17)]
    assert all(b > a for a, b in zip(sigmas, sigmas[1:]))


def test_crossover_is_where_star_equals_sky():
    Vx = MODEL.crossover_magnitude(30.0)
    assert abs(MODEL.signal(Vx, 30.0) - MODEL.sky_in_aperture(30.0)) / MODEL.signal(Vx, 30.0) < 0.01
    assert 12.5 < Vx < 15.0


def test_saturation_below_full_well():
    assert 6.0 < MODEL.saturation_magnitude(30.0) < 12.0


def test_summer_nights_have_darkness():
    w = dark_window("2027-08-02")
    assert w is not None and 5.0 < w.hours < 9.0


def test_target_is_observable_in_august():
    hours, blocks = usable_hours(TARGET, "2027-08-02")
    assert hours > 4.0 and len(blocks) >= 1


def test_harmonics():
    assert harmonic_match(1.0, 2.0) and harmonic_match(2.0, 1.0)
    assert not harmonic_match(1.7, 1.0)


def test_pipeline_recovers_injected_variables_without_false_positives():
    camp = simulate_campaign(MODEL, TARGET, ["2027-07-31", "2027-08-02"],
                             n_stars=150, n_variables=4,
                             rng=np.random.default_rng(3))
    resid = differential_photometry(camp.mags)
    resid = detrend_per_night(resid, camp.night_id, order=2)
    dets = search_field(camp.times, resid, camp.night_id,
                        n_stars_searched=1500, global_fap=0.01)
    truth = {t.star_index for t in camp.truth}
    found = {d.star_index for d in dets}
    assert len(found & truth) >= 3, "should recover most injected variables"
    assert len(found - truth) == 0, "must produce no false positives"


def test_max_period_shorter_than_a_night():
    camp = simulate_campaign(MODEL, TARGET, ["2027-08-02"], n_stars=20, n_variables=1)
    assert 1.0 < max_searchable_period(camp.times, camp.night_id) < 5.0
