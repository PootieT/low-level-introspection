"""Shared style for the layer-9 SAE figures. Every SAE plotter takes its fonts, colors,
axes styling, titles and output format from here so the paper figures match.

Colors: one per feature (tab10 order), used in every figure; non-feature distinctions
(task, split, weight) use lightness, hatching or line style so they never reuse a
feature color. Axes, titles, legends and fonts follow the neuron-track plotters
(analysis/plot_neuron_generalization.py): matplotlib defaults -- full box, tick marks,
centered titles, framed legends, light grid -- with the neuron font sizes.
Output: PNG (300 dpi) + PDF, white background, tight crop."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_rgb

ORDER = ["12408", "1568", "978", "4120", "4854"]
CONCEPT = {"12408": "appreciation", "1568": "limitations", "978": "statistics",
           "4120": "affiliations", "4854": "identifiers"}
FEATURE_COLOR = dict(zip(ORDER, sns.color_palette("tab10", len(ORDER))))
BASELINE = "#7f7f7f"
INK, INK2 = "black", "#444444"
SPINE = "#cccccc"
SMALL = 12
DPI = 300


def _fi(fi):
    return str(fi).replace("S9.", "")


def feature_color(fi):
    return FEATURE_COLOR[_fi(fi)]


def feature_label(fi, sep="\n"):
    return f"S9.{_fi(fi)}{sep}{CONCEPT[_fi(fi)]}"


def tint(color, amount):
    """Blend `color` toward white; amount=0 -> color, 1 -> white."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


def feature_cmap(fi):
    return LinearSegmentedColormap.from_list(f"f{_fi(fi)}", ["white", feature_color(fi)])


def apply_rc():
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": SMALL, "ytick.labelsize": SMALL,
        "legend.fontsize": SMALL, "legend.title_fontsize": 13, "figure.titlesize": 18,
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.axisbelow": True,
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style_axes(ax, grid="both"):
    if grid:
        ax.grid(axis=grid, alpha=0.25)


def box(ax):
    """Full black frame -- for axes whose artist hides the spines (seaborn heatmaps)."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)


def chance_line(ax, value=0.5, axis="h"):
    (ax.axhline if axis == "h" else ax.axvline)(value, color=BASELINE, ls="--", lw=1.3)


def fig_title(fig, text):
    fig.suptitle(text)


def save(fig, path):
    path = Path(path)
    if path.suffix in (".png", ".pdf"):
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(f"{path}{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{path}.png", "+ .pdf")
