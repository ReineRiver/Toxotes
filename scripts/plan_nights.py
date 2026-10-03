"""Night planner for the 2027 campaign - which dates to ask Sirio for.

    python scripts/plan_nights.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from toxotes.observability import plan_night, dark_window, moon_illumination

TARGET = SkyCoord(305.5 * u.deg, 40.0 * u.deg)      # change once the field is chosen


def main():
    print("Cygnus (Sadr region) from Observatorio de Principia, summer 2027\n")
    best = []
    for month, days in ((7, range(15, 32)), (8, range(1, 32)), (9, range(1, 9))):
        for d in days:
            date = f"2027-{month:02d}-{d:02d}"
            win = dark_window(date)
            if win is None:
                continue
            mid = win.start + (win.end - win.start) / 2
            illum = moon_illumination(mid)
            if illum < 0.25:
                best.append((date, illum))
    print(f"{len(best)} dark-moon nights (moon < 25% illuminated) in the window:\n")
    for date, illum in best:
        print(plan_night(TARGET, date))
    print("\nAsk Sirio for 6-8 of these. Prioritise the ones with the longest")
    print("usable block and the moon furthest from the field.")


if __name__ == "__main__":
    main()
