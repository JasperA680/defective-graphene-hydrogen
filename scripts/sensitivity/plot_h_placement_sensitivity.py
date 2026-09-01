#!/usr/bin/env python3
"""
Supplementary figure: sensitivity of the H assignment to the placement thresholds.

Three panels, one per threshold, each sweeping that threshold while the other two
are held at their production values:

    (a) sp/sp2 angle threshold        150 - 160 deg      production 155
    (b) bounded-face area threshold    10 - 14 A^2       production 12
    (c) steric separation threshold   1.2 - 1.6 A        production 1.40

The y axis is the percentage of candidate carbon sites whose hydrogen assignment
differs from production, where a candidate site is a surviving coordination-1 or
-2 carbon (the only carbons that can receive H under any of these thresholds).
The production value is marked in each panel with a vertical dashed line, where
the curve passes through zero by construction.

A second figure repeats the layout with the number of affected structures out of
95 on the y axis.

The y axis is shared across the three panels of each figure. That is the point of
the figure - the panels are only comparable if the scale is - so a panel that
sits flat on zero is reporting a real insensitivity rather than a missing curve,
and is annotated as such.

Inputs:
    results/sensitivity/h_placement_thresholds/sensitivity_summary.csv

Outputs:
    results/figures/sensitivity/supplementary_h_placement_sensitivity.png/.pdf
    results/figures/sensitivity/supplementary_h_placement_sensitivity_structures.png/.pdf

Run from the repository root, after h_placement_threshold_sensitivity.py:
    .venv-dos/bin/python3 scripts/sensitivity/plot_h_placement_sensitivity.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_FILE = (
    ROOT / "results" / "sensitivity" / "h_placement_thresholds" / "sensitivity_summary.csv"
)
FIGURE_DIR = ROOT / "results" / "figures" / "sensitivity"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# One series per panel, so there is no categorical identity to encode and no
# legend: a single accent carries the data in all three, and the production
# marker is deliberately recessive grey so it reads as an annotation rather than
# as a second series.
LINE_COLOUR = "#1f6fb4"
MARKER_FACE = "#ffffff"
PRODUCTION_COLOUR = "#8a8a8a"
GRID_COLOUR = "#dcdcdc"
ANNOTATION_COLOUR = "#6a6a6a"

PANELS = [
    ("sp_angle_deg", "(a)", "sp/sp$^2$ angle threshold / deg", 155.0, "155$\\degree$"),
    ("max_cage_ring_area", "(b)", "bounded-face area threshold / $\\mathrm{\\AA}^2$", 12.0, "12 $\\mathrm{\\AA}^2$"),
    ("min_steric_distance", "(c)", "steric separation threshold / $\\mathrm{\\AA}$", 1.40, "1.40 $\\mathrm{\\AA}$"),
]


def draw_figure(summary, column, y_label, output_stem, value_format):
    """
    Draw the three-panel sweep for one y quantity.

    value_format formats a y value for the direct labels, which are applied
    selectively - only to the non-zero points, since labelling a run of zeros
    adds nothing and clutters the panel that most needs to look empty.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)

    y_max = float(summary[column].max())
    y_top = 1.0 if y_max <= 0 else y_max * 1.35

    for ax, (parameter, panel_label, x_label, production, production_label) in zip(axes, PANELS):
        block = summary[summary["parameter"] == parameter].sort_values("value")

        x = block["value"].to_numpy(dtype=float)
        y = block[column].to_numpy(dtype=float)

        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.7)

        ax.axvline(
            production,
            color=PRODUCTION_COLOUR,
            linestyle="--",
            linewidth=1.1,
            zorder=1,
        )

        ax.plot(
            x,
            y,
            color=LINE_COLOUR,
            linewidth=2.0,
            marker="o",
            markersize=6.5,
            markerfacecolor=MARKER_FACE,
            markeredgewidth=1.8,
            markeredgecolor=LINE_COLOUR,
            zorder=3,
            clip_on=False,
        )

        # Direct labels on the points that carry information. Values sit in ink,
        # not in the series colour.
        for xi, yi in zip(x, y):
            if yi <= 0:
                continue

            ax.annotate(
                value_format(yi),
                xy=(xi, yi),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color="#222222",
                zorder=4,
            )

        ax.set_xlabel(x_label)
        ax.set_title(panel_label, loc="left", fontsize=12, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xlim(x.min() - 0.12 * (x.max() - x.min()), x.max() + 0.12 * (x.max() - x.min()))
        ax.set_ylim(0.0, y_top)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # The production marker is labelled in every panel rather than in a
        # legend: it sits at a different x in each, so a shared legend entry
        # would be less informative than the value itself.
        ax.annotate(
            "production\n{}".format(production_label),
            xy=(production, y_top),
            xytext=(3, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=ANNOTATION_COLOUR,
        )

        if np.all(y <= 0):
            ax.annotate(
                "no change at any\ntested value",
                xy=(0.5, 0.42),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=9,
                style="italic",
                color=ANNOTATION_COLOUR,
            )

    axes[0].set_ylabel(y_label)

    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for suffix in ("png", "pdf"):
        path = FIGURE_DIR / "{}.{}".format(output_stem, suffix)
        fig.savefig(path, dpi=600 if suffix == "png" else None)
        print("Wrote {}".format(path.relative_to(ROOT)))

    plt.close(fig)


def main():
    if not SUMMARY_FILE.exists():
        raise SystemExit(
            "{} not found - run scripts/sensitivity/"
            "h_placement_threshold_sensitivity.py first".format(
                SUMMARY_FILE.relative_to(ROOT)
            )
        )

    summary = pd.read_csv(SUMMARY_FILE)

    n_candidate = int(summary["n_candidate_sites"].iloc[0])

    draw_figure(
        summary=summary,
        column="percent_changed_sites",
        y_label="changed H assignments / % of {} sites".format(n_candidate),
        output_stem="supplementary_h_placement_sensitivity",
        value_format=lambda v: "{:.2f}".format(v),
    )

    draw_figure(
        summary=summary,
        column="n_structures_affected",
        y_label="structures affected (of 95)",
        output_stem="supplementary_h_placement_sensitivity_structures",
        value_format=lambda v: "{:.0f}".format(v),
    )


if __name__ == "__main__":
    main()
