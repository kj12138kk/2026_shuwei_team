"""Rebuild the paper figures from this checkout's model and results."""

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parent.parent
WIDTH_MM = 180
DPI = 600
FORMATS = ("pdf", "svg", "png")
PLOTS = {
    "scene": "plot_scene",
    "observability": "plot_observability",
    "intersection": "plot_intersection",
    "schedule": "plot_schedule",
    "pointing": "plot_pointing",
    "rescheduling": "plot_rescheduling",
}


def apply_style():
    """Use the landed typography and editable exports from the figure skills."""
    faces = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    face = next((name for name in faces if name in installed), "DejaVu Sans")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": faces,
        "font.size": 9.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "mathtext.fontset": "custom",
        "mathtext.rm": face,
        "mathtext.it": f"{face}:italic",
        "mathtext.bf": f"{face}:bold",
        "mathtext.sf": face,
        "mathtext.default": "regular",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": None,
    })


def audit(fig):
    """Measure final page dimensions, landed type and any content outside the page."""
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    width_in, height_in = fig.get_size_inches()
    width, height = width_in * 25.4, height_in * 25.4
    if abs(width - WIDTH_MM) > 0.05:
        raise ValueError(f"Expected {WIDTH_MM} mm page width, got {width:.2f} mm")
    sizes = [
        item.get_fontsize() * WIDTH_MM / width
        for item in fig.findobj(Text)
        if item.get_visible() and item.get_text().strip()
    ]
    if not sizes or min(sizes) < 5 - 1e-6 or max(sizes) > 12 + 1e-6:
        raise ValueError(f"Text sizes outside 5–12 pt: {sorted(set(sizes))}")
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    tolerance = 0.5 / 25.4
    if (
        bounds.x0 < -tolerance or bounds.y0 < -tolerance
        or bounds.x1 > width_in + tolerance or bounds.y1 > height_in + tolerance
    ):
        raise ValueError(f"Content extends beyond the page: {bounds.bounds}")
    return width, height, min(sizes), max(sizes)


def export(figures, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        try:
            draw_width, draw_height, smallest, largest = audit(fig)
            for fmt in FORMATS:
                path = output / f"{name}.{fmt}"
                fig.savefig(path, dpi=DPI, format=fmt)
                if path.stat().st_size > 50_000_000:
                    raise ValueError(f"{path.name} exceeds 50 MB")
            print(
                f"{name}: {draw_width:.1f} x {draw_height:.1f} mm, "
                f"{smallest:.1f}–{largest:.1f} pt, {', '.join(FORMATS)}",
                flush=True,
            )
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=PLOTS, nargs="+", help="Rebuild selected figures")
    parser.add_argument("--output", type=Path, default=ROOT / "pictures", help="Export directory")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    for topic in args.only or PLOTS:
        with matplotlib.rc_context():
            apply_style()
            module = importlib.import_module(PLOTS[topic])
            export(module.make_figures(ROOT), args.output)


if __name__ == "__main__":
    main()
