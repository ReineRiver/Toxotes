"""Publication figure style for the paper.

Journals reproduce figures at a fixed column width and often print in
greyscale. Two rules follow, and they are the ones amateurs break:

  1. Build the figure at its FINAL printed width. Never draw a big figure
     and shrink it in Word - that shrinks the fonts with it and is the
     single most common reason a student figure is unreadable in print.
  2. Identity must never depend on colour alone. Every series gets a
     distinct marker AND line style as well as a colour, so the figure
     survives greyscale printing and colour-blind readers.

Usage:

    from toxotes.style import use_paper_style, figure, SERIES
    use_paper_style()
    fig, ax = figure("single")           # 3.35 in wide, correct aspect
    ax.plot(x, y, **SERIES[0])           # colour + marker + linestyle together
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Printed text-block widths, inches. Check your journal's guide and adjust.
WIDTHS = {
    "single": 3.35,   # one column of a two-column journal (85 mm)
    "wide":   6.90,   # full width across two columns (175 mm)
    "a4text": 6.30,   # full text width of a single-column A4 report
}

# Okabe-Ito subset: colour-blind safe, validated. Ordered so the two
# lowest-contrast hues come last and are rarely reached.
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

#: Ready-made kwargs per series - colour, marker and line style together, so
#: you cannot accidentally encode identity by colour alone.
SERIES = [dict(color=c, marker=m, linestyle=ls, markersize=4, linewidth=1.4)
          for c, m, ls in zip(COLOURS, MARKERS, LINESTYLES)]


def use_paper_style() -> None:
    """Astronomy-journal defaults: serif text, ticks inside on all four sides."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        # Box axes with inward ticks on all four sides - the convention in
        # every astronomy journal.
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.linewidth": 0.8,
        "axes.grid": False,          # journals prefer no grid; add it if it helps
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "errorbar.capsize": 0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.prop_cycle": plt.cycler(color=COLOURS),
    })


def figure(width: str = "single", aspect: float = 0.68, nrows: int = 1,
           ncols: int = 1, **kw):
    """A figure sized for print. `aspect` is height/width."""
    w = WIDTHS.get(width, float(width) if not isinstance(width, str) else 3.35)
    return plt.subplots(nrows, ncols, figsize=(w, w * aspect * nrows), **kw)


def save(fig, stem: str, formats=("pdf", "png")) -> list[str]:
    """Save vector for the journal and raster for drafts, same base name."""
    import os
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    out = []
    for f in formats:
        p = f"{stem}.{f}"
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out


def greyscale_check(path_png: str) -> str:
    """Re-save a figure in greyscale so you can check it still reads.

    Do this for every figure before submission. If two series become
    indistinguishable, you relied on colour alone - fix it with markers.
    """
    try:
        from PIL import Image
    except ImportError:
        return "install pillow to use greyscale_check"
    out = path_png.replace(".png", "_grey.png")
    Image.open(path_png).convert("L").save(out)
    return out
