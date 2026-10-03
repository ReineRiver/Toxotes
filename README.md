# Toxotes

Photometry pipeline for a high-cadence campaign on a crowded Cygnus field,
using the Meade LX200 10″ at the Observatorio de Principia, Málaga.

Two papers come out of one campaign:

- **Paper A** — new times of minimum and an O−C period analysis of one known
  short-period eclipsing binary.
- **Paper B** — the incidence rate of short-period variability among stars in
  the range V = 10.5–13.5, where ZTF saturates and only 8–21″/pixel surveys
  remain, plus any new variables found.

Everything here except `scripts/gaia_counts.py` runs **offline**. The point is
to build and validate the entire analysis on synthetic data now, so that in
September 2027 you are pressing "run" on a pipeline you trust.

## Setup (Windows / VS Code)

### 0. Install Python first

If `python --version` in PowerShell says the command is not recognised, Python
is not installed. Either:

```powershell
winget install --id Python.Python.3.12 -e
```

or install "Python 3.12" from the Microsoft Store, or download it from
python.org (**tick "Add python.exe to PATH"** on the first installer screen).

**Then close PowerShell and open a new one.** The PATH is only read when a
terminal starts, so the old window will still say "not recognized". Check:

```powershell
python --version        # should print Python 3.12.x
```

### 1. Virtual environment and install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[archives,dev]"
```

If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled"),
either allow it for this window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

or skip activation entirely and call the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[archives,dev]"
.\.venv\Scripts\python.exe scripts\demo_end_to_end.py
```

Note `.\.venv\...`, with the leading `.\`. Without it PowerShell treats
`.venv\Scripts\activate` as a module name and gives the confusing error
"The module '.venv' could not be loaded".

### 2. VS Code

Install the Python extension, then `Ctrl+Shift+P` -> "Python: Select
Interpreter" -> choose the one inside `.venv`. On macOS or Linux the only
differences are `python3 -m venv .venv` and `source .venv/bin/activate`.

`pip install -e .` installs the package in editable mode, so your edits take
effect immediately and no `PYTHONPATH` juggling is needed. If you would rather
not install anything, the scripts and `pytest` still work from the repository
root - `conftest.py` and a path shim at the top of each script handle it.

## Try it

```bash
python scripts/demo_end_to_end.py   # full dress rehearsal on synthetic data
python scripts/plan_nights.py       # which 2027 nights to ask Sirio for
pytest -q                           # 8 sanity tests, ~8 seconds
python scripts/gaia_counts.py       # the only script needing internet - prerequisite D1
```

The demo writes three figures into `figures/` (git-ignored): the RMS-versus-
magnitude plot, a periodogram with its window function, and a phase-folded
light curve. Those are the three figures Paper B needs.

**Harmless first-run noise:** astropy prints warnings about IERS tables and
"dubious year" ERFA calls. Both are irrelevant at this precision and can be
ignored.

## Modules

| Module | What it does |
|---|---|
| `toxotes/snr.py` | Photon budget. σ(V), source/sky crossover, saturation, field of view, data volume. **Change the `IMX571` and telescope presets the moment Principia confirms the real hardware** — every number in the proposal regenerates. |
| `toxotes/observability.py` | Twilight, target altitude, moon phase and separation, light-dome cuts, usable-hour blocks, night planning. Alt-az field rotation is included for reference but is zero on the Principia wedge. |
| `toxotes/simulate.py` | Synthetic campaigns with the systematics that will really limit you: airmass extinction, transparency drift, a colour term, and per-star flat-field drift. |
| `toxotes/detect.py` | Ensemble differential photometry, common-mode and per-night detrending, Lomb–Scargle, the trials-factor threshold, the empirical field-based threshold, harmonic matching. |
| `toxotes/injection.py` | Injection–recovery completeness maps and the Poisson incidence-rate statistics. |
| `toxotes/plots.py` | The three paper figures: RMS-vs-magnitude, periodogram with window function, phase-folded light curve. |

## Two things the code found that the design documents got wrong

**1. The honest period limit is ~3 hours, not 6.**
Per-star slow drifts (flat-field error as the field walks across the chip)
put enormous power at periods comparable to the length of a night. Removing
them with `detrend_per_night` is mandatory — but it also destroys sensitivity
to periods near the block length. `max_searchable_period()` computes the limit
from your actual data; for ~6.6-hour blocks it returns about 3 hours. **Update
the research question accordingly.**

**2. Analytic false-alarm probabilities will lie to you.**
The first run of the demo produced 144 "detections", nearly all at exactly the
longest period searched, with quoted false-alarm probabilities of 10⁻²⁰⁰. They
were all red noise. Two fixes, both in `search_field`:

- an **empirical threshold** taken from the peak-power distribution of the
  field's own stars, which carry your real noise and your real window function;
- **trimming the top few per cent** before estimating that distribution —
  otherwise the variables you are hunting set the threshold that excludes them.

With both, the demo recovers 6 of 6 injected variables and produces zero false
positives.

## The rule that matters most

Measure completeness through the **identical** criterion you used for
detection — same power threshold, same period cap, same detrending. Pass
`search_field.last_power_threshold` and `search_field.last_max_period_h`
into `completeness_map`. Completeness measured with a weaker criterion than
detection inflates your efficiency and shrinks your quoted rate, which is
worse than quoting no rate at all.

## Status

Placeholders that must be replaced with measurements:

- camera model, pixel size, read noise, full well (prerequisite A1)
- true focal length and whether an f/6.3 reducer exists (A3, A8)
- sky brightness — currently assumed 18.2 mag/arcsec², measure with an SQM (E2)
- field centre — currently the Sadr region, pending the Gaia counts (D1, D4)

Real-image aperture photometry (`photutils`) and Kwee–van Woerden minimum
timing for Paper A are not written yet; they come next, once there are FITS
files to point them at.
