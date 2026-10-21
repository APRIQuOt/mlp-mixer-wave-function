"""Render the paper's three-panel architecture overview.

Geometry, shared architecture colours, and Computer Modern typography are
self-contained. SVG text is outlined for portability. Generate this diagram
with ``python scripts/generate_plots.py --diagram-only``.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath

from paper_workflow.plots import (
    IMAGE_DIR, LABELS, colour_for, style_for,
)

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt


OUTPUT = IMAGE_DIR / "diagrams"
INK = "#252525"
MUTED = "#656565"
RULE = "#d7d7d7"
FLOW = to_hex(colour_for("PTF"))


def computer_modern(weight="normal"):
    # Use Matplotlib's bundled TrueType faces. These match the paper's TeX
    # typography and embed cleanly in PDF, without relying on installed fonts.
    name = "cmb10.ttf" if weight == "bold" else "cmr10.ttf"
    path = Path(mpl.get_data_path()) / "fonts" / "ttf" / name
    return font_manager.FontProperties(fname=str(path))




def draw_diagram():
    """Keep the generation/mixing examples at left and architecture choices at right."""
    width, height = 510 / 72.27, 2.55
    fig = plt.figure(figsize=(width, height), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.set_axis_off()
    texts = []
    count = 0

    def identify(artist, name):
        nonlocal count
        count += 1
        artist.set_gid(f"{name}-{count}")
        return artist

    def text(x, y, label, *, size=8, align="center", weight="normal",
             colour=INK, rotation=0, gid="label"):
        artist = ax.text(x, y, label, ha=align, va="center", fontsize=size,
                         fontproperties=computer_modern(weight), color=colour,
                         rotation=rotation, zorder=6)
        texts.append(identify(artist, gid))
        return artist

    def box(x, y, w, h, *, edge=INK, fill="white", gid="box"):
        artist = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                           linewidth=.65, joinstyle="miter", zorder=4)
        ax.add_patch(identify(artist, gid))

    def line(points, *, colour=INK, lw=.6, gid="connector"):
        xs, ys = zip(*points)
        artist, = ax.plot(xs, ys, color=colour, linewidth=lw,
                          solid_capstyle="butt", zorder=2)
        identify(artist, gid)

    def arrow(points, *, colour=INK, lw=.6, scale=4, gid="arrow"):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-1))
        artist = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=scale,
                                 facecolor=colour, edgecolor=colour, linewidth=lw,
                                 joinstyle="miter", capstyle="butt", zorder=4.5)
        ax.add_patch(identify(artist, gid))

    def scan(x, y, size, n=4, *, patches=False, labels=False):
        pitch = size / n
        if patches:
            for row in range(2):
                for col in range(2):
                    box(x+col*size/2, y+row*size/2, size/2, size/2,
                        edge=MUTED, gid="patch-boundary")
        else:
            box(x, y, size, size, edge=MUTED, gid="lattice-boundary")
        for row in range(n):
            for col in range(n):
                dot = Circle((x+(col+.5)*pitch, y+(row+.5)*pitch),
                             .008 if size > .4 else .0045,
                             facecolor=INK, edgecolor="none", zorder=5)
                ax.add_patch(identify(dot, "site"))
        if labels:
            for row in range(2):
                for col in range(2):
                    text(x+(col+.5)*size/2, y+(row+.5)*size/2,
                         rf"$P_{2*row+col+1}$", size=7, gid="patch-label")
            return
        rows = 2 if patches else n
        step = size / rows
        for row in range(rows):
            cy = y+(row+.5)*step
            for col in range(rows-1):
                arrow([(x+(col+.5)*step+.018, cy),
                       (x+(col+1.5)*step-.018, cy)], colour=FLOW, scale=2.5, lw=.45)
            if row < rows-1:
                arrow([(x+size-.5*step+.018, cy), (x+size+.018, cy),
                       (x+size+.018, y+(row+1)*step),
                       (x-.018, y+(row+1)*step), (x-.018, cy+step),
                       (x+.5*step-.018, cy+step)], colour=FLOW, scale=2.5, lw=.45)

    def chip(x, cy, w, alias, mechanism):
        colour = colour_for(alias)
        box(x, cy-.095, w, .19, edge=colour, fill=to_rgba(colour, .075),
            gid=f"architecture-{alias}")
        artist = ax.scatter([x+.085], [cy], s=12, marker=style_for(alias).marker,
                            c=[colour], edgecolors="white", linewidths=.25, zorder=6)
        identify(artist, f"architecture-symbol-{alias}")
        text(x+.17, cy, LABELS[alias], align="left", size=7.7,
             gid=f"architecture-label-{alias}")
        text(x+w/2, cy+.145, mechanism, size=6.1, colour=MUTED,
             gid="mechanism-label")

    text(.12, .12, "(a)", size=9, weight="bold", align="left", gid="panel-label")
    text(.40, .12, "Autoregressive generation", size=9, weight="bold", align="left")
    scan(.16, .39, .46)
    text(.12, .94, "Single-site sequence", size=7.2, align="left")
    arrow([(.72, .62), (1.04, .62)])
    text(1.12, .62, r"$\cdots$", size=9)
    for x, index, previous in ((1.34, "i", "i-1"), (2.72, "i+1", "i")):
        cx = x+.22
        box(x, .49, .44, .26, gid="ann-block")
        text(cx, .62, "ANN", size=8.5)
        arrow([(cx, .49), (cx, .438)], scale=3)
        conditioning = rf"\leq {previous}"
        text(cx, .381, rf"$p(\sigma_{{{index}}}\mid\sigma_{{{conditioning}}})$", size=8.2)
        arrow([(cx, .322), (cx, .282)], scale=3, gid="sample-from-probability")
        text(cx, .229, rf"$\sigma_{{{index}}}$", size=8.5)
        arrow([(cx, .838), (cx, .75)], scale=3)
        text(cx, .913, rf"$\sigma_{{{previous}}}$", size=8.5)
    arrow([(1.20, .62), (1.32, .62)])
    arrow([(1.80, .62), (2.70, .62)])
    arrow([(3.18, .62), (3.88, .62)])
    text(3.98, .62, r"$\cdots$", size=9)
    line([(.12, 1.035), (4.30, 1.035)], colour=RULE, lw=.5, gid="panel-separator")

    text(.12, 1.14, "(b)", size=9, weight="bold", align="left", gid="panel-label")
    text(.40, 1.14, "Global and local mixing", size=9, weight="bold", align="left")
    text(.43, 1.38, "Patch input", size=7.6)
    scan(.16, 1.62, .53, patches=True, labels=True)
    text(.425, 2.28, r"$2\times2$ patches", size=7.2)
    arrow([(.75, 1.89), (.92, 1.89)], colour=MUTED)
    text(1.035, 1.365, "Shifted input", size=6.8)
    text(1.76, 1.365, "Global mixing", size=7.6)
    text(2.49, 1.365, "Local mixing", size=7.6)
    text(3.62, 1.365, "Conditional probabilities", size=7.6)
    ys = [1.58, 1.85, 2.12, 2.39]
    colours = [MUTED] + [colour_for(alias) for alias in ("RNN", "PRNN", "PTF")]
    for target in range(4):
        for source in range(target+1):
            arrow([(1.13, ys[source]), (1.70, ys[target])], colour=colours[source],
                  lw=.45, scale=3, gid="causal-global-mixing")
    for i, cy in enumerate(ys):
        box(.955, cy-.08, .16, .16, edge=colours[i], gid="patch-token")
        text(1.035, cy, r"$s_0$" if i == 0 else rf"$P_{i}$", size=7)
        box(1.72, cy-.08, .16, .16, edge=MUTED, gid="global-features")
        for offset in (-.039, 0, .039):
            arrow([(1.91, cy+offset), (2.29, cy+offset)], colour=colour_for("M"),
                  scale=3, lw=.5, gid="local-feature-mixing")
        box(2.31, cy-.08, .16, .16, edge=MUTED, gid="local-features")
        arrow([(2.49, cy), (2.75, cy)], scale=3)
        arrow([(2.94, cy), (3.10, cy)], scale=3)
        prefix = ",".join(f"P_{j+1}" for j in range(i))
        probability = rf"$p(P_{i+1}" + (rf"\mid {prefix}" if i else "") + ")$"
        text(3.14, cy, probability, size=7.6, align="left", gid="conditional-probability")
    box(2.77, 1.46, .15, 1.05, gid="output-head")
    text(2.845, 1.985, "Output head", size=7, rotation=90)

    line([(4.43, .08), (4.43, 2.48)], colour=RULE, lw=.5, gid="panel-separator")
    text(4.55, .12, "(c)", size=9, weight="bold", align="left", gid="panel-label")
    text(4.83, .12, "Architecture choices", size=9, weight="bold", align="left")
    text(4.60, .43, "Single site", size=8.2, weight="bold", align="left")
    scan(4.61, .59, .34)
    arrow([(5.03, .76), (5.23, .76)], colour=MUTED, scale=3)
    chip(5.30, .76, .75, "RNN", "recurrence")
    chip(6.18, .76, .75, "TF", "attention")

    text(4.60, 1.14, r"$2\times2$ patch", size=8.2, weight="bold", align="left")
    scan(4.61, 1.30, .34, patches=True)
    arrow([(5.03, 1.47), (5.16, 1.47)], colour=MUTED, scale=3)
    chip(5.20, 1.47, .55, "PRNN", "recurrence")
    chip(5.80, 1.47, .55, "PTF", "attention")
    chip(6.40, 1.47, .55, "M", "mixing")

    text(4.60, 1.84, "Hierarchical", size=8.2, weight="bold", align="left")
    scan(4.55, 2.065, .26, n=8, patches=True)
    arrow([(4.85, 2.195), (4.98, 2.195)], scale=3)
    scan(5.01, 2.065, .26, patches=True)
    text(4.68, 2.43, "Large patch", size=6.1)
    text(5.14, 2.43, r"$2\times2$ patch", size=6.1)
    chip(5.42, 2.015, .74, "HRNN", r"$\mathrm{RNN}\rightarrow\mathrm{RNN}$")
    chip(6.21, 2.015, .74, "HM", r"$\mathrm{M}\rightarrow\mathrm{M}$")
    chip(5.42, 2.315, .74, "HTF+R", r"$\mathrm{TF}\rightarrow\mathrm{RNN}$")
    chip(6.21, 2.315, .74, "HTF+M", r"$\mathrm{TF}\rightarrow\mathrm{M}$")
    return fig, texts


def verify_text_bounds(fig, artists):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boundary = fig.bbox
    for artist in artists:
        bounds = artist.get_window_extent(renderer)
        if not (boundary.x0 <= bounds.x0 <= bounds.x1 <= boundary.x1
                and boundary.y0 <= bounds.y0 <= bounds.y1 <= boundary.y1):
            raise ValueError(f"Label is outside the figure: {artist.get_text()}")


def add_svg_description(path):
    contents = path.read_text()
    end = contents.index(">", contents.index("<svg")) + 1
    description = (
        '\n <title>Autoregressive neural quantum state architectures</title>'
        '\n <desc>Three panels: site-wise autoregressive generation, causal global '
        'and local mixing of patches, and the nine architecture families grouped '
        'by input granularity. Clean vector redraw of Sketch_Fig1.pdf using '
        'Computer Modern typography and the shared paper architecture colours.</desc>'
    )
    path.write_text(contents[:end] + description + contents[end:])
    root = ET.parse(path).getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    if root.findall(".//s:image", ns):
        raise ValueError("SVG unexpectedly contains raster images")
    for alias in ("M", "HM", "RNN", "PRNN", "HRNN", "TF", "PTF", "HTF+M", "HTF+R"):
        if to_hex(colour_for(alias)) not in contents.lower():
            raise ValueError(f"Missing shared paper colour for {alias}")


def main():
    with mpl.rc_context({
        "text.usetex": False, "font.family": "serif", "font.serif": ["cmr10"],
        "font.size": 9.5, "mathtext.fontset": "cm", "svg.fonttype": "path",
        "axes.formatter.use_mathtext": True,
        "svg.hashsalt": "tnqs-clean-fig1", "pdf.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }):
        fig, text_artists = draw_diagram()
        verify_text_bounds(fig, text_artists)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        svg = OUTPUT / "Sketch_Fig1.svg"
        fig.savefig(svg, metadata={"Date": None, "Creator": "tNQS paper diagram generator"})
        add_svg_description(svg)
        fig.savefig(OUTPUT / "arch_recipe_standalone_2.pdf", metadata={"Title": "Autoregressive neural quantum state architectures"})
        fig.savefig(OUTPUT / "Sketch_Fig1.png", dpi=220)
        plt.close(fig)
    print(f"Redrew {svg}; exported matching PDF and PNG.")


if __name__ == "__main__":
    main()
