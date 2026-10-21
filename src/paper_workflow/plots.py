"""Generate the paper plots from the result parquet files.

The article includes the PGF files written by this script.  The script keeps
the data reduction in one place so that the figures can be regenerated when a
new parquet file is produced.  Both views of a training curve use
the first (lowest-numbered) seed for each architecture: the main plot shows
its trailing-average error and the inset shows its unsmoothed energy history.
Reusable panels omit legends; shared FLOP keys and an appendix architecture
table use the same marker/color definitions.  The ``*_main`` training assets
retain their legends for the standalone main-text figures.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("pgf")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

from paper_workflow.style import (
    ARCHITECTURE_SPECS,
    ARCH_ALIAS,
    flop_tick_positions,
)


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "build" / "paper"
IMAGE_DIR = PAPER_DIR / "images"
PREVIEW_DIR = PAPER_DIR / "previews"

# Keep the PGF assets small enough for Overleaf's TeX memory limits.  The
# underlying histories contain 15--20k updates, but plotting every update as a
# PGF scatter marker creates tens of thousands of TeX paths in the appendix.
# A fixed, endpoint-preserving decimation keeps the discontinuous error points
# visible while reducing each appendix panel to a few hundred paths.
TRAINING_SCATTER_POINTS = 48
TRAINING_INSET_POINTS = 64
TRAINING_AVERAGE_WINDOW = 1000
FLOPS_COMPARISON_ITERATIONS = 15000

# Computer Modern/LaTeX text is used in the PGF output.  The article already
# loads amsmath, amssymb, siunitx, and defines \mathdefault.
PGF_PREAMBLE = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}\providecommand{\mathdefault}[1]{#1}"
STYLE = {
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": PGF_PREAMBLE,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.6,
    "legend.title_fontsize": 6.6,
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.35,
    "ytick.minor.width": 0.35,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "xtick.direction": "in",
    "ytick.direction": "in",
}

ACADEMIC_FALLBACK = {
    "image.cmap": "viridis",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.spines.left": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "lines.markersize": 6,
}

VIRIDIS = mpl.colormaps["viridis"]

LABELS = {
    "M": r"$\mathrm{M}$",
    "HM": r"$\mathrm{HM}$",
    "RNN": r"$\mathrm{RNN}$",
    "PRNN": r"$\mathrm{PRNN}$",
    "HRNN": r"$\mathrm{HRNN}$",
    "TF": r"$\mathrm{TF}$",
    "PTF": r"$\mathrm{PTF}$",
    "HTF+M": r"$\mathrm{HTF{+}M}$",
    "HTF+R": r"$\mathrm{HTF{+}R}$",
}

ARCHITECTURE_NAMES = {
    "M": "Patched MLP mixer",
    "HM": "Hierarchical MLP mixer",
    "RNN": "Site-wise recurrent neural network",
    "PRNN": "Patched recurrent neural network",
    "HRNN": "Hierarchical recurrent neural network",
    "TF": "Site-wise transformer",
    "PTF": "Patched transformer",
    "HTF+M": "Hierarchical transformer with mixer subsampler",
    "HTF+R": "Hierarchical transformer with RNN subsampler",
}

HIERARCHICAL_ALIASES = ("HM", "HRNN", "HTF+M", "HTF+R")
HIERARCHICAL_SIZES = ("8x8", "16x16", "24x24")


def preferred_aliases(present: Iterable[str]) -> list[str]:
    present = set(present)
    return [
        ARCH_ALIAS[key].identifier
        for key, _, _, _ in ARCHITECTURE_SPECS
        if ARCH_ALIAS[key].identifier in present
    ]




def style_for(alias: str):
    for key, _, _, _ in ARCHITECTURE_SPECS:
        if ARCH_ALIAS[key].identifier == alias:
            return ARCH_ALIAS[key]
    raise KeyError(f"Unknown architecture alias: {alias}")


def colour_for(alias: str):
    aliases = preferred_aliases(ARCH_ALIAS[key].identifier for key, _, _, _ in ARCHITECTURE_SPECS)
    position = aliases.index(alias)
    return VIRIDIS(0.12 + 0.78 * position / max(1, len(aliases) - 1))


def academic_context():
    """Apply the bundled academic style, with a portable fallback."""
    style_file = ROOT / "assets" / "styles" / "academic.mplstyle"
    if style_file.is_file():
        return plt.style.context(str(style_file))
    return mpl.rc_context(ACADEMIC_FALLBACK)


def _set_minimum_log_ticks(axis, lower: float, upper: float, count: int = 5) -> None:
    lower = max(float(lower), np.finfo(float).tiny)
    upper = max(float(upper), lower * 10.0 ** 0.25)
    exponent_low = int(np.floor(np.log10(lower)))
    exponent_high = int(np.ceil(np.log10(upper)))
    candidates = np.asarray(
        [factor * 10.0**exponent for exponent in range(exponent_low, exponent_high + 1) for factor in (1.0, 2.0, 5.0)],
        dtype=float,
    )
    candidates = candidates[(candidates >= lower) & (candidates <= upper)]
    if len(candidates) >= 3:
        if len(candidates) > count:
            indices = np.unique(np.linspace(0, len(candidates) - 1, count).round().astype(int))
            candidates = candidates[indices]
        ticks = candidates
    else:
        ticks = np.geomspace(lower, upper, max(3, count))
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(_format_log_tick))
    axis.set_minor_locator(NullLocator())


def _format_log_tick(value: float, _position: int) -> str:
    exponent = np.log10(max(float(value), np.finfo(float).tiny))
    if abs(exponent - round(exponent)) < 1e-8:
        return rf"$10^{{{int(round(exponent))}}}$"
    return rf"$10^{{{exponent:.1f}}}$"


def _set_decade_y_ticks(ax, values: Iterable[float], *, tight_lower: bool = False) -> None:
    """Use powers of ten; optionally trim the empty margin below the observations."""
    positive = np.asarray([value for value in values if value > 0], dtype=float)
    exponent_low = int(np.floor(np.log10(positive.min())))
    exponent_high = int(np.ceil(np.log10(positive.max())))
    if exponent_high - exponent_low < 2:
        missing = 2 - (exponent_high - exponent_low)
        exponent_low -= (missing + 1) // 2
        exponent_high += missing // 2
    ticks = np.power(10.0, np.arange(exponent_low, exponent_high + 1))
    # Keep the smallest observation visible without reserving an entire empty
    # decade when it lies just below the next power of ten.
    lower = positive.min() / 1.25 if tight_lower else ticks[0]
    ax.set_ylim(lower, ticks[-1])
    ticks = ticks[ticks >= lower]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_log_tick))
    ax.yaxis.set_minor_locator(NullLocator())


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean with the same short-history behavior as Polars rolling_mean."""
    values = np.asarray(values, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    stop = np.arange(1, len(values) + 1)
    start = np.maximum(0, stop - window)
    return (cumulative[stop] - cumulative[start]) / (stop - start)


def _sample_indices(length: int, maximum: int) -> np.ndarray:
    """Return at most ``maximum`` evenly spaced indices, including endpoints."""
    if length <= 0:
        return np.asarray([], dtype=int)
    if length <= maximum:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, maximum).round().astype(int))


def _save(
    fig: mpl.figure.Figure,
    destination: Path,
    *,
    preview_name: str,
    legacy_preview: bool = False,
    tight: bool = True,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    bbox_inches = "tight" if tight else None
    fig.savefig(destination, bbox_inches=bbox_inches)
    # PDF and PNG previews are convenient for visual inspection; the article
    # continues to consume the PGF file at `destination`.
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches=bbox_inches)
    fig.savefig(PREVIEW_DIR / f"{preview_name}.png", dpi=240, bbox_inches=bbox_inches)
    if legacy_preview:
        fig.savefig(
            destination.with_name(f"{destination.stem}-img0.png"),
            dpi=240,
            bbox_inches=bbox_inches,
        )
    plt.close(fig)


def _legend(aliases: list[str]) -> list[Line2D]:
    handles = []
    for alias in aliases:
        spec = style_for(alias)
        handles.append(
            Line2D(
                [],
                [],
                color=colour_for(alias),
                marker=spec.marker,
                linestyle="none",
                linewidth=0.9,
                markersize=3.8,
                markeredgecolor="white",
                markeredgewidth=0.35,
                label=LABELS[alias],
            )
        )
    return handles


def plot_legend(
    aliases: Iterable[str], *, destination: Path, width: float, columns: int
) -> None:
    """Export a shared key independently of the panels it accompanies."""
    aliases = preferred_aliases(aliases)
    rows = (len(aliases) + columns - 1) // columns
    with academic_context(), mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(width, 0.24 * rows), dpi=160)
        fig.legend(
            handles=_legend(aliases),
            loc="center",
            bbox_to_anchor=(0.02, 0.0, 0.96, 1.0),
            mode="expand",
            ncol=columns,
            frameon=False,
            fontsize=8.0,
            handlelength=1.0,
            handletextpad=0.4,
            borderaxespad=0,
        )
        _save(fig, destination, preview_name=destination.stem, tight=False)


def generate_plot_keys(tfim: pl.DataFrame, rydberg: pl.DataFrame) -> None:
    """Use the plot marker/color definitions for the manuscript's architecture key."""
    legend_dir = IMAGE_DIR / "legends"
    aliases = preferred_aliases(
        set(tfim.get_column("arch_alias")) | set(rydberg.get_column("arch_alias"))
    )
    table = [
        "% Generated by scripts/generate_plots.py; symbols match the plots.",
        r"\begin{tabular}{cll}",
        r"Symbol & Label & Architecture \\",
        r"\hline",
    ]
    for alias in aliases:
        spec = style_for(alias)
        destination = legend_dir / f"key_symbol_{alias.replace('+', '_')}.pgf"
        with academic_context(), mpl.rc_context(STYLE):
            fig = plt.figure(figsize=(0.22, 0.14), dpi=160)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.scatter(
                [0.5], [0.5], marker=spec.marker, color=colour_for(alias),
                s=28, edgecolor="white", linewidth=0.35,
            )
            _save(fig, destination, preview_name=destination.stem, tight=False)
        symbol_path = destination.relative_to(PAPER_DIR).as_posix()
        table.append(
            r"\raisebox{-0.25em}{\input{%s}} & %s & %s \\"
            % (symbol_path, LABELS[alias], ARCHITECTURE_NAMES[alias]),
        )
    table.append(r"\end{tabular}")
    table_path = PAPER_DIR / "tables" / "architecture_plot_key.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(table) + "\n")


@dataclass(frozen=True)
class TrainingCurve:
    reference_energy: float
    first_seed_energy: np.ndarray
    first_seed: int


def _select_curve_rows(
    frame: pl.DataFrame, *, size: str, phase: float
) -> dict[str, TrainingCurve]:
    selected = frame.filter(
        (pl.col("size") == size) & (pl.col("phase") == phase)
    )
    curves: dict[str, TrainingCurve] = {}
    for alias in preferred_aliases(selected.get_column("arch_alias").unique().to_list()):
        rows = selected.filter(pl.col("arch_alias") == alias).sort("seed").to_dicts()
        if not rows:
            continue
        references = np.asarray([row["reference_energy"] for row in rows], dtype=float)
        if not np.allclose(references, references[0]):
            raise ValueError(f"Reference energy differs between seeds for {alias}")
        # Select one reproducible run for both the main plot and its inset.
        curves[alias] = TrainingCurve(
            reference_energy=float(references[0]),
            first_seed_energy=np.asarray(rows[0]["energies"], dtype=float),
            first_seed=int(rows[0]["seed"]),
        )
    return curves


def plot_training(
    frame: pl.DataFrame,
    *,
    size: str,
    phase: float,
    destination: Path,
    legend: bool,
    inset: bool,
) -> None:
    curves = _select_curve_rows(frame, size=size, phase=phase)
    aliases = preferred_aliases(curves)
    if not aliases:
        raise ValueError(f"No training curves found for {size}, phase={phase}")

    with academic_context(), mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.40, 2.42), dpi=160)
        displayed_errors = []
        for alias in aliases:
            curve = curves[alias]
            smoothed = rolling_mean(curve.first_seed_energy, TRAINING_AVERAGE_WINDOW)
            error = np.maximum(np.abs(smoothed - curve.reference_energy), 1e-12)
            indices = _sample_indices(len(error), TRAINING_SCATTER_POINTS)
            displayed_errors.extend(error[indices])
            spec = style_for(alias)
            ax.scatter(
                indices + 1,
                error[indices],
                color=colour_for(alias),
                marker=spec.marker,
                s=7.0,
                linewidths=0,
                label=LABELS[alias],
                zorder=3,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Training Iterations")
        ax.set_ylabel(r"$|\langle E\rangle-E_{\mathrm{exact}}|/N$")
        ax.set_xlim(0, max(len(curve.first_seed_energy) for curve in curves.values()))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        _set_decade_y_ticks(ax, displayed_errors)
        ax.grid(axis="y", color="0.86", linewidth=0.45)
        ax.set_axisbelow(True)

        if inset:
            inset_ax = ax.inset_axes([0.56, 0.54, 0.40, 0.39])
            for alias in aliases:
                energy = curves[alias].first_seed_energy
                indices = _sample_indices(len(energy), TRAINING_INSET_POINTS)
                spec = style_for(alias)
                inset_ax.scatter(
                    indices + 1,
                    energy[indices],
                    color=colour_for(alias),
                    marker=spec.marker,
                    s=2.5,
                    linewidths=0,
                    zorder=3,
                )
            reference = curves[aliases[0]].reference_energy
            inset_ax.axhline(reference, color="0.25", linestyle="--", linewidth=0.55)
            inset_ax.set_xlabel("Training Iterations", fontsize=6.2, labelpad=1)
            inset_ax.set_ylabel(r"$\langle E\rangle/N$", fontsize=6.2, labelpad=1)
            inset_ax.tick_params(labelsize=5.3, length=2)
            inset_ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            inset_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            # Keep inset labels readable where a main-curve marker passes
            # behind the inset's exterior tick labels or longer x-axis label.
            for label in (inset_ax.xaxis.label, inset_ax.yaxis.label,
                          *inset_ax.get_xticklabels(), *inset_ax.get_yticklabels()):
                label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": .2})
            inset_ax.grid(axis="y", color="0.88", linewidth=0.35)
            inset_ax.set_axisbelow(True)

        if legend:
            ax.legend(
                handles=_legend(aliases),
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=min(5, len(aliases)),
                frameon=False,
                columnspacing=0.8,
                handletextpad=0.25,
                borderaxespad=0,
            )
        _save(fig, destination, preview_name=destination.stem, legacy_preview=True)


def _format_flops(value: float, _position: int) -> str:
    if value >= 1e6:
        return rf"${value / 1e6:g}\,\mathrm{{M}}$"
    if value >= 1e3:
        return rf"${value / 1e3:g}\,\mathrm{{k}}$"
    return rf"${value:g}$"




def summarise_flops_comparison(
    frame: pl.DataFrame,
    *,
    sizes: Iterable[str],
    flop_column: str,
    training_iterations: int | None = FLOPS_COMPARISON_ITERATIONS,
) -> pl.DataFrame:
    """Keep all per-run errors and summarize them across phases and seeds.

    Histories and reference energies already contain energy densities, so the
    resulting difference is |<E>-E_reference|/N without another division by N.
    FLOP costs remain the supplied forward-pass counts; no new cost formulas
    or averages across different model designs are introduced here.
    """
    selected = frame.filter(pl.col("size").is_in(list(sizes)))
    if flop_column not in selected.columns:
        raise ValueError(f"FLOP column {flop_column!r} is not present")
    records = []
    for row in selected.to_dicts():
        energies = np.asarray(row["energies"], dtype=float)
        stop = len(energies) if training_iterations is None else training_iterations
        if not 1 <= stop <= len(energies):
            raise ValueError(f"Insufficient history for {row['size']} {row['arch_alias']} seed {row['seed']}")
        window = energies[max(0, stop-TRAINING_AVERAGE_WINDOW):stop]
        error = float(abs(window.mean() - row["reference_energy"]))
        flops = int(row[flop_column])
        if not np.isfinite(error) or flops <= 0:
            raise ValueError("FLOP comparison needs finite errors and positive forward costs")
        records.append({
            "size": row["size"], "arch_alias": row["arch_alias"],
            "flops_per_forward_pass": flops, "absolute_error": error,
            "phase": row["phase"], "seed": row["seed"], "training_iterations": stop,
        })
    if not records:
        raise ValueError("No runs available for FLOP comparison")
    runs = pl.DataFrame(records)
    if runs.group_by("size", "arch_alias").agg(
        pl.col("flops_per_forward_pass").n_unique().alias("costs")
    ).filter(pl.col("costs") != 1).height:
        raise ValueError("Expected one forward cost per architecture/system design")
    return runs.group_by("size", "arch_alias", "flops_per_forward_pass").agg(
        pl.col("absolute_error").mean().alias("mean_absolute_error"),
        pl.col("absolute_error").median().alias("median_absolute_error"),
        pl.col("absolute_error").quantile(.25, interpolation="linear").alias("q25_absolute_error"),
        pl.col("absolute_error").quantile(.75, interpolation="linear").alias("q75_absolute_error"),
        pl.col("absolute_error").alias("run_absolute_errors"),
        pl.len().alias("runs"), pl.col("phase").n_unique().alias("phases"),
        pl.col("seed").n_unique().alias("seeds"),
        pl.col("training_iterations").min().alias("min_training_iterations"),
        pl.col("training_iterations").max().alias("max_training_iterations"),
    ).sort("size", "arch_alias")


def plot_flops_grid(
    frame: pl.DataFrame,
    *,
    sizes: tuple[str, ...],
    shape: tuple[int, int],
    destination: Path,
    flop_column: str = "api_flops_per_sample",
    training_iterations: int | None = FLOPS_COMPARISON_ITERATIONS,
    statistic: str = "mean",
) -> pl.DataFrame:
    """Export a figure with independent axes, panel indices, and a bottom key.

    Overall captions and lattice-size captions belong to the manuscript. This
    asset contains only panel indices, axes, observations, and an architecture
    key. The default matches the paper captions: mean of the three phases and
    five seeds at 15,000 iterations, using the trailing 1,000-update estimate.
    """
    if shape[0]*shape[1] != len(sizes) or statistic not in ("mean", "median"):
        raise ValueError("Invalid subplot layout or summary statistic")
    summary = summarise_flops_comparison(
        frame, sizes=sizes, flop_column=flop_column, training_iterations=training_iterations,
    )
    if set(summary.get_column("size")) != set(sizes):
        raise ValueError("Missing system size in combined FLOP figure")
    aliases = preferred_aliases(summary.get_column("arch_alias").unique())
    value_column = f"{statistic}_absolute_error"
    stacked = shape == (2, 1)
    figsize = (3.4, 4.7) if stacked else (6.8, 6.1)
    with academic_context(), mpl.rc_context(STYLE):
        fig, axes = plt.subplots(*shape, figsize=figsize, sharex=False, sharey=False,
                                 squeeze=False, dpi=160)
        if stacked:
            fig.subplots_adjust(left=.18, right=.98, bottom=.23, top=.98, hspace=.28)
        else:
            fig.subplots_adjust(left=.12, right=.985, bottom=.14, top=.975,
                                hspace=.12, wspace=.24)
        for index, (ax, size) in enumerate(zip(axes.flat, sizes)):
            points = summary.filter(pl.col("size") == size)
            displayed_errors = []
            for alias in preferred_aliases(points.get_column("arch_alias")):
                row = points.filter(pl.col("arch_alias") == alias).row(0, named=True)
                flops = row["flops_per_forward_pass"]
                colour = colour_for(alias)
                errors = np.maximum(np.asarray(row["run_absolute_errors"], dtype=float), 1e-12)
                displayed_errors.extend(errors)
                ax.scatter(np.full(len(errors), flops), errors, s=7,
                           color=colour, alpha=.30, edgecolors="none", zorder=2)
                ax.vlines(flops, row["q25_absolute_error"], row["q75_absolute_error"],
                          color=colour, linewidth=2.6, alpha=.78, zorder=3)
                ax.scatter(flops, max(row[value_column], 1e-12),
                           marker=style_for(alias).marker, s=36, color=colour,
                           edgecolor="white", linewidth=.55, zorder=4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            costs = points.get_column("flops_per_forward_pass").to_numpy()
            ax.set_xlim(costs.min()/1.22, costs.max()*1.22)
            ax.xaxis.set_major_locator(FixedLocator(flop_tick_positions(*ax.get_xlim())))
            ax.xaxis.set_major_formatter(FuncFormatter(_format_flops))
            ax.xaxis.set_minor_locator(NullLocator())
            _set_decade_y_ticks(ax, displayed_errors, tight_lower=not stacked and index < shape[1])
            ax.text(.04, .95, f"({chr(ord('a')+index)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}, zorder=5)
            ax.grid(axis="both", color="0.86", linewidth=.45)
            ax.set_axisbelow(True)

        fig.supxlabel("FLOPs per forward pass", fontsize=8, y=.145 if stacked else .064)
        fig.supylabel(f"{statistic.capitalize()} absolute energy-density error", fontsize=8,
                      x=.014, y=.605 if stacked else .552)
        fig.legend(handles=_legend(aliases), loc="lower left",
                   bbox_to_anchor=(.04, .017 if stacked else .013, .92, .09 if stacked else .04),
                   mode="expand", ncol=3 if stacked else len(aliases),
                   frameon=False, borderaxespad=0, handlelength=1,
                   handletextpad=.35, columnspacing=.8)
        _save(fig, destination, preview_name=destination.stem, tight=False)

    metadata = PAPER_DIR / "tables" / f"{destination.stem}.json"
    metadata.write_text(json.dumps({
        "flop_column": flop_column, "statistic": statistic,
        "training_iterations": training_iterations, "trailing_window": TRAINING_AVERAGE_WINDOW,
        "aggregation": "Per-run absolute energy-density errors pooled over phase points and seeds",
        "subplot_order": list(sizes), "shared_axes": [],
        "figure_size_inches": list(figsize),
        "subplot_y_limits": {size: list(ax.get_ylim()) for ax, size in zip(axes.flat, sizes)},
        "observations": "All individual phase/seed run errors",
        "interval": "Interquartile range (25th-75th percentiles)",
        "points": summary.to_dicts(),
    }, indent=2) + "\n")
    return summary


def generate_combined_flops_assets(
    tfim: pl.DataFrame, rydberg: pl.DataFrame, *,
    flop_column: str = "api_flops_per_sample", suffix: str = "",
    training_iterations: int | None = FLOPS_COMPARISON_ITERATIONS, statistic: str = "mean",
) -> None:
    for frame, folder, stem, sizes, shape in (
        (tfim, "tfim", "tfim", ("4x4", "8x8"), (2, 1)),
        (rydberg, "rydberg", "ryd", ("4x4", "8x8", "16x16", "24x24"), (2, 2)),
    ):
        plot_flops_grid(
            frame, sizes=sizes, shape=shape,
            destination=IMAGE_DIR / folder / f"{stem}_flops_combined{suffix}.pgf",
            flop_column=flop_column, training_iterations=training_iterations, statistic=statistic,
        )


def plot_hierarchical_flops_scaling(
    frame: pl.DataFrame,
    *,
    destination: Path,
    flop_column: str = "api_flops_per_sample",
    y_label: str = r"FLOPs per sample",
    legend: bool = False,
) -> pl.DataFrame:
    """Plot the costs of twelve saved designs, without averaging seed duplicates."""
    selected = frame.filter(
        pl.col("size").is_in(HIERARCHICAL_SIZES)
        & pl.col("arch_alias").is_in(HIERARCHICAL_ALIASES)
    )
    if selected.get_column(flop_column).null_count():
        raise ValueError("Hierarchical FLOP counts contain missing values")
    summary = selected.group_by("size", "arch_alias").agg(
        pl.col(flop_column).n_unique().alias("flop_values"),
        pl.col(flop_column).first().alias("flops_per_sample"),
        pl.col("trainable_parameters").n_unique().alias("parameter_values"),
        pl.col("trainable_parameters").first(),
    )
    expected = {(size, alias) for size in HIERARCHICAL_SIZES for alias in HIERARCHICAL_ALIASES}
    if set(summary.select("size", "arch_alias").iter_rows()) != expected:
        raise ValueError("Expected all four hierarchical designs at 8x8, 16x16, and 24x24")
    if summary.filter(
        (pl.col("flop_values") != 1) | (pl.col("parameter_values") != 1)
        | (pl.col("flops_per_sample") <= 0)
    ).height:
        raise ValueError("Expected one positive FLOP cost and parameter count per saved design")
    summary = summary.with_columns(
        pl.col("size").replace_strict(
            {"8x8": 64, "16x16": 256, "24x24": 576}
        ).alias("num_sites")
    ).drop("flop_values", "parameter_values").sort("arch_alias", "num_sites")

    with academic_context(), mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.40, 2.65), dpi=160)
        for alias in HIERARCHICAL_ALIASES:
            points = summary.filter(pl.col("arch_alias") == alias).sort("num_sites")
            spec = style_for(alias)
            ax.plot(
                points.get_column("num_sites").to_numpy(),
                points.get_column("flops_per_sample").to_numpy(),
                color=colour_for(alias), marker=spec.marker,
                linewidth=1.0, markersize=5.0,
                markeredgecolor="white", markeredgewidth=0.4,
                label=LABELS[alias], zorder=3,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(
            [64, 256, 576],
            labels=[rf"${length}\times{length}$" + "\n" + rf"$N={length**2}$"
                    for length in (8, 16, 24)],
        )
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(50, 750)
        # The same 1/2/5 tick convention as the paper's FLOP axes.
        ax.set_ylim(1e5, 5e6)
        ax.yaxis.set_major_locator(FixedLocator([1e5, 2e5, 5e5, 1e6, 2e6, 5e6]))
        ax.yaxis.set_major_formatter(FuncFormatter(_format_flops))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"Lattice size ($N=L^2$)")
        ax.set_ylabel(y_label)
        ax.grid(axis="both", color="0.86", linewidth=0.45)
        ax.set_axisbelow(True)
        if legend:
            ax.legend(
                handles=_legend(list(HIERARCHICAL_ALIASES)),
                loc="upper center", bbox_to_anchor=(0.5, -0.34),
                ncol=4, frameon=False, columnspacing=0.8,
                handletextpad=0.25, borderaxespad=0,
            )
        _save(fig, destination, preview_name=destination.stem)
    return summary


def generate_hierarchical_scaling_assets(
    frame: pl.DataFrame,
    *,
    flop_column: str = "api_flops_per_sample",
) -> None:
    """Export the paper's hierarchical panel, matching key, and source values."""
    suffix = "_2"
    y_label = r"FLOPs per sample"
    stem = "ryd_hierarchical_flops_scaling"
    summary = plot_hierarchical_flops_scaling(
        frame, destination=IMAGE_DIR / "rydberg" / f"{stem}{suffix}.pgf",
        flop_column=flop_column, y_label=y_label,
    )
    plot_legend(
        HIERARCHICAL_ALIASES,
        destination=IMAGE_DIR / "legends" / "hierarchical_scaling.pgf",
        width=3.4, columns=4,
    )
    table_path = PAPER_DIR / "tables" / f"{stem}{suffix}.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps({
        "hamiltonian": "rydberg",
        "flop_column": flop_column,
        "convention": "PyTorch dispatcher-supported log_psi FLOPs for batch size 8, divided by 8",
        "comparison": "Saved experimental designs; parameter budgets vary with lattice size",
        "points": summary.to_dicts(),
    }, indent=2) + "\n")


def generate_training_assets(
    tfim: pl.DataFrame, rydberg: pl.DataFrame, *, suffix: str = ""
) -> None:
    """Render every training panel; the main-paper copies differ only by legend."""
    for frame, folder, prefix, parameter, sizes, phases, main_phase in (
        (tfim, "tfim", "tfim", "h", ("4x4", "8x8"),
         ((1.0, "1"), (3.044, "3_044"), (5.0, "5")), 1.0),
        (rydberg, "rydberg", "ryd", "delta", ("4x4", "8x8", "16x16", "24x24"),
         ((0.4, "0_4"), (1.2, "1_2"), (2.0, "2_0")), 2.0),
    ):
        for size in sizes:
            for phase, phase_name in phases:
                stem = f"{prefix}_{size}_{parameter}_{phase_name}"
                plot_training(
                    frame, size=size, phase=phase,
                    destination=IMAGE_DIR / folder / f"{stem}{suffix}.pgf",
                    legend=False, inset=True,
                )
                if size == "8x8" and phase == main_phase:
                    plot_training(
                        frame, size=size, phase=phase,
                        destination=IMAGE_DIR / folder / f"{stem}_main{suffix}.pgf",
                        legend=True, inset=True,
                    )


def configure_output(path: Path) -> None:
    """Select a generated-artifact directory without changing source files."""
    global PAPER_DIR, IMAGE_DIR, PREVIEW_DIR
    PAPER_DIR = Path(path).resolve()
    IMAGE_DIR = PAPER_DIR / "images"
    PREVIEW_DIR = PAPER_DIR / "previews"


def generate_paper_assets(tfim: pl.DataFrame, rydberg: pl.DataFrame) -> None:
    """Generate the assets referenced by the current manuscript."""
    generate_training_assets(tfim, rydberg)
    generate_plot_keys(tfim, rydberg)
    generate_combined_flops_assets(
        tfim, rydberg, flop_column="api_flops_per_sample", suffix="_2",
    )
    generate_hierarchical_scaling_assets(
        rydberg, flop_column="api_flops_per_sample",
    )
