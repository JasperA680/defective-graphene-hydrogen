#!/usr/bin/env python3
"""
Results figure: ring topology across the 95 experimental structures.

Replaces the multi-line, dual-y-axis ring_count_evolution.png (still written by
plot_n_ring_counts.py, left untouched) with a two-panel figure that separates
the two things that plot was trying to show on one pair of axes:

    (a) a heatmap of the non-hexagonal ring census - structure on x, ring size
        n on y, cell colour = number of n-membered rings - so the ten defect
        ring sizes no longer fight for one crowded y-axis;
    (b) the hexagonal topology of the same structures on the same x-ordering,
        as either the raw hexagon count N_6 or the hexagon fraction
        f_6 = N_6 / sum_n(N_n), on its own axis rather than a twin axis.

The A-J letters are separate experimental sequences, so panel (b) connects
structures only within a sequence: nothing is drawn across a boundary, and no
claim of continuous evolution from A10 into B01 is implied by the geometry.

A third variant adds panel (c): the single-point HSE06 energy per atom of the
same structures, as Delta E against the lowest structure, on the same x-ordering
so ring topology and DFT energetics can be read together. Marker colour there is
the maximum residual force, because these are single points on unrelaxed
experimental frames and the few high-Delta-E structures are the high-force ones.

Ring counts are read as-is from the census table written by
plot_n_membered_rings_all.py, and DFT energies as-is from the existing-runs
summary. Nothing here re-runs ring identification or any calculation.

Inputs:
    results/tables/n_membered_rings/all_structures_ring_summary.csv
    results/tables/existing_aims_runs_summary.csv                (panel (c) only)

Outputs:
    results/figures/results_ring_topology_census.{png,pdf}        panel (b) = N_6
    results/figures/results_ring_topology_census_f6.{png,pdf}     panel (b) = f_6
    results/figures/results_ring_topology_census_f6_dft.{png,pdf} f_6 + panel (c)
    results/tables/n_membered_rings/ring_size_statistics.csv
    results/tables/n_membered_rings/ring_topology_fractions.csv
    results/tables/n_membered_rings/ring_topology_dft_energy.csv

Run from project root:
    .venv-dos/bin/python3 scripts/visualisation/plot_ring_topology_census.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

SUMMARY_FILE = ROOT / "results/tables/n_membered_rings/all_structures_ring_summary.csv"
DFT_SUMMARY_FILE = ROOT / "results/tables/existing_aims_runs_summary.csv"
TABLE_DIR = ROOT / "results/tables/n_membered_rings"
FIGURE_DIR = ROOT / "results/figures"
FIGURE_STEM = "results_ring_topology_census"

# The structure whose 9-membered ring is carried forward into the local DOS
# case study; marked in both panels so the figure and that section line up.
HIGHLIGHT_ID = "H02"
HIGHLIGHT_RING_SIZE = 9

# Figure furniture is set above the matplotlib default of 10 pt because this is
# a full-width two-column figure that still gets read at part-page size.
# fonttype 42 keeps PDF text selectable rather than outlined.
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

# Sequential single hue for the count heatmap (magnitude, not identity), with
# the pale end lifted off white so a count of 1 is still visible.
HEATMAP_CMAP = "Blues"
HEATMAP_RANGE = (0.30, 1.00)
EMPTY_COLOUR = "#f4f4f4"      # cells with no rings of that size
GRID_COLOUR = "white"
SEQUENCE_COLOUR = "#4d4d4d"   # A-J separators
HIGHLIGHT_COLOUR = "#b3450a"  # H02 marker
MARK_COLOUR = "#2c5f8a"       # panel (b) markers

# Panel (c) residual force: a second single hue, clear of both the Blues ring
# counts and the reserved orange, so the two colourbars are never confused.
FORCE_CMAP = "Purples"
FORCE_RANGE = (0.25, 1.00)
ENERGY_LINE_COLOUR = "#9a9a9a"  # marker colour means force, so the line is neutral


def size_columns(df):
    """Ring-size count columns present in the census, ascending by size."""
    excluded = {"n_rings", "n_non_hexagonal_rings"}

    columns = [
        column
        for column in df.columns
        if column.startswith("n_")
        and column.endswith("_rings")
        and column not in excluded
    ]

    return sorted(columns, key=lambda column: int(column.split("_")[1]))


def load_census(summary_file):
    """The successful rows of the ring census, in A01 -> J05 order."""
    df = pd.read_csv(summary_file)
    df = df[df["status"] == "success"].copy()

    df["sequence"] = df["structure_id"].str[0]
    df = df.sort_values("structure_id", kind="mergesort").reset_index(drop=True)

    return df


def load_dft_energies(dft_summary_file):
    """Converged single-point energies keyed on the padded A01 - J05 id.

    The existing-runs summary keeps every structure, including the ones whose
    aims run did not complete, so those are dropped here rather than silently
    renumbering the sequence.
    """
    df = pd.read_csv(dft_summary_file)

    df["structure_id"] = df["family"] + df["site"].astype(int).map("{:02d}".format)
    df = df.dropna(subset=["E_per_atom_eV"])

    return df[["structure_id", "E_per_atom_eV", "max_force_eV_A"]]


def dft_energy_table(table, dft):
    """Topology table with DFT energy joined on, still in A01 -> J05 x order."""
    merged = table.merge(dft, on="structure_id", how="left")
    merged["delta_E_per_atom_eV"] = (
        merged["E_per_atom_eV"] - merged["E_per_atom_eV"].min()
    )

    return merged


def ring_statistics(df, columns):
    """Per-ring-size totals across the whole set, observed sizes only."""
    rows = []

    for column in columns:
        counts = df[column]
        total = int(counts.sum())

        if total == 0:
            continue

        maximum = int(counts.max())
        at_max = df.loc[counts == maximum, "structure_id"].tolist()

        rows.append({
            "ring_size": int(column.split("_")[1]),
            "total_rings": total,
            "n_structures_with_ring": int((counts > 0).sum()),
            "max_in_one_structure": maximum,
            "structures_at_max": ";".join(at_max),
        })

    return pd.DataFrame(rows).sort_values("ring_size").reset_index(drop=True)


def topology_fractions(df, columns):
    """Per-structure hexagon / non-hexagon counts and bounded-face fractions."""
    non_hexagonal = [column for column in columns if column != "n_6_rings"]

    table = pd.DataFrame({
        "structure_id": df["structure_id"],
        "sequence": df["sequence"],
        "n_carbon_atoms": df["n_carbon_atoms"],
        "N_6": df["n_6_rings"],
        "N_nonhex": df[non_hexagonal].sum(axis=1),
    })

    table["N_total"] = table["N_6"] + table["N_nonhex"]
    table["f_6"] = table["N_6"] / table["N_total"]
    table["f_nonhex"] = table["N_nonhex"] / table["N_total"]

    return table


def sequence_blocks(df):
    """(letter, first index, last index) for each A-J sequence, in x order."""
    blocks = []

    for letter, group in df.groupby("sequence", sort=False):
        blocks.append((letter, int(group.index[0]), int(group.index[-1])))

    return blocks


def draw_sequence_furniture(ax, blocks, n_structures, highlight_x, label_axis):
    """Sequence separators, the H02 marker, and A-J ticks shared by both panels."""
    for _, _, last in blocks[:-1]:
        ax.axvline(last + 1.0, color=SEQUENCE_COLOUR, linewidth=0.7, alpha=0.45,
                   zorder=3)

    if highlight_x is not None:
        ax.axvline(highlight_x, color=HIGHLIGHT_COLOUR, linewidth=0.9,
                   linestyle=(0, (3, 2)), alpha=0.85, zorder=4)

    ax.set_xlim(0, n_structures)

    if label_axis:
        ax.set_xticks([(first + last + 1) / 2.0 for _, first, last in blocks])
        ax.set_xticklabels([letter for letter, _, _ in blocks])
        ax.set_xticks([last + 1.0 for _, _, last in blocks[:-1]], minor=True)
        ax.tick_params(axis="x", which="major", length=0)
        ax.tick_params(axis="x", which="minor", length=3, width=0.9)
    else:
        ax.tick_params(axis="x", which="both", labelbottom=False, length=0)


def draw_heatmap(ax, cax, df, columns, blocks, highlight_x):
    """Panel (a): non-hexagonal ring census as structure x ring-size cells."""
    non_hexagonal = [
        column for column in columns
        if column != "n_6_rings" and df[column].sum() > 0
    ]
    ring_sizes = [int(column.split("_")[1]) for column in non_hexagonal]

    counts = df[non_hexagonal].to_numpy(dtype=float).T
    highest = int(counts.max())

    levels = np.arange(0.5, highest + 1.5, 1.0)
    base = matplotlib.colormaps[HEATMAP_CMAP]
    cmap = ListedColormap(
        base(np.linspace(HEATMAP_RANGE[0], HEATMAP_RANGE[1], highest))
    )
    norm = BoundaryNorm(levels, cmap.N)

    ax.set_facecolor(EMPTY_COLOUR)

    mesh = ax.pcolormesh(
        np.arange(counts.shape[1] + 1),
        np.arange(counts.shape[0] + 1),
        np.ma.masked_less(counts, 1.0),
        cmap=cmap,
        norm=norm,
        edgecolors=GRID_COLOUR,
        linewidth=0.25,
    )

    ax.set_yticks(np.arange(len(ring_sizes)) + 0.5)
    ax.set_yticklabels([str(size) for size in ring_sizes])
    ax.set_ylabel("Ring size $n$")
    ax.set_ylim(0, len(ring_sizes))
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    draw_sequence_furniture(ax, blocks, len(df), highlight_x, label_axis=False)

    # Outline the highlighted cell so the ring size behind the case study is
    # findable without hunting along the row.
    if highlight_x is not None and HIGHLIGHT_RING_SIZE in ring_sizes:
        row = ring_sizes.index(HIGHLIGHT_RING_SIZE)
        ax.add_patch(Rectangle(
            (highlight_x - 0.5, row), 1.0, 1.0,
            fill=False, edgecolor=HIGHLIGHT_COLOUR, linewidth=1.4, zorder=5,
        ))

        ax.annotate(
            "{} ({}-ring)".format(HIGHLIGHT_ID, HIGHLIGHT_RING_SIZE),
            xy=(highlight_x, len(ring_sizes)),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=HIGHLIGHT_COLOUR,
            annotation_clip=False,
        )

    bar = ax.figure.colorbar(mesh, cax=cax, extend="neither")
    bar.set_label("Number of rings", fontsize=10)
    bar.set_ticks(np.arange(1, highest + 1, 2 if highest > 6 else 1))
    bar.ax.tick_params(labelsize=9, length=3, width=0.9)
    bar.outline.set_linewidth(0.6)

    return ring_sizes


def draw_hexagon_panel(ax, table, blocks, highlight_x, use_fraction,
                       label_axis=True):
    """Panel (b): hexagon count or hexagon fraction, one axis, no twin."""
    values = table["f_6"] if use_fraction else table["N_6"]
    centres = np.arange(len(table)) + 0.5

    ax.grid(axis="y", color="#cccccc", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Connected within a sequence only: A10 does not evolve into B01, so no
    # line is drawn across a sequence boundary.
    for _, first, last in blocks:
        block = slice(first, last + 1)
        ax.plot(centres[block], values.to_numpy()[block],
                color=MARK_COLOUR, linewidth=1.0, alpha=0.55, zorder=2)

    ax.plot(centres, values, linestyle="none", marker="o", markersize=3.6,
            markerfacecolor=MARK_COLOUR, markeredgecolor="white",
            markeredgewidth=0.5, zorder=3)

    if use_fraction:
        ax.set_ylabel("Hexagon fraction $f_6$", fontsize=9.5)
        low = float(values.min())
        ax.set_ylim(low - 0.10 * (1.0 - low), 1.0 + 0.05 * (1.0 - low))
    else:
        ax.set_ylabel("Hexagons $N_6$")
        span = float(values.max() - values.min())
        ax.set_ylim(float(values.min()) - 0.12 * span,
                    float(values.max()) + 0.12 * span)

    if label_axis:
        ax.set_xlabel("Experimental structure (A01 – J05)")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    draw_sequence_furniture(ax, blocks, len(table), highlight_x, label_axis)


def draw_energy_panel(ax, cax, table, blocks, highlight_x):
    """Panel (c): single-point DFT energy per atom, marker colour = max force."""
    values = table["delta_E_per_atom_eV"].to_numpy(dtype=float)
    forces = table["max_force_eV_A"].to_numpy(dtype=float)
    centres = np.arange(len(table)) + 0.5

    ax.grid(axis="y", color="#cccccc", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Same within-sequence-only connection as panel (b). Structures without a
    # converged energy are NaN, which breaks the line rather than bridging them.
    for _, first, last in blocks:
        block = slice(first, last + 1)
        ax.plot(centres[block], values[block],
                color=ENERGY_LINE_COLOUR, linewidth=1.0, alpha=0.5, zorder=2)

    base = matplotlib.colormaps[FORCE_CMAP]
    cmap = ListedColormap(base(np.linspace(FORCE_RANGE[0], FORCE_RANGE[1], 256)))

    # Max force spans nearly three decades across the set, so the colour scale
    # has to be logarithmic for anything below the worst frames to separate.
    points = ax.scatter(
        centres, values,
        c=forces,
        cmap=cmap,
        norm=LogNorm(vmin=np.nanmin(forces), vmax=np.nanmax(forces)),
        s=14,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    ax.set_ylabel(r"$\Delta E$ per atom / eV", fontsize=9.5)
    span = float(np.nanmax(values) - np.nanmin(values))
    ax.set_ylim(float(np.nanmin(values)) - 0.12 * span,
                float(np.nanmax(values)) + 0.12 * span)

    ax.set_xlabel("Experimental structure (A01 – J05)")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # The gaps in the line are missing runs, not missing structures: say which.
    missing = table.loc[table["E_per_atom_eV"].isna(), "structure_id"].tolist()
    if missing:
        ax.text(0.012, 0.94,
                "no converged DFT: {}".format(", ".join(missing)),
                transform=ax.transAxes, fontsize=8, color="#6f6f6f",
                ha="left", va="top")

    draw_sequence_furniture(ax, blocks, len(table), highlight_x, label_axis=True)

    bar = ax.figure.colorbar(points, cax=cax, extend="neither")
    bar.set_label(r"Max force / eV $\AA^{-1}$", fontsize=10)
    bar.ax.tick_params(labelsize=9, length=3, width=0.9)
    bar.outline.set_linewidth(0.6)

    return points


def build_figure(df, table, columns, blocks, highlight_x, use_fraction,
                 figure_dir, stem, dft_table=None):
    with_dft = dft_table is not None

    fig = plt.figure(figsize=(7.2, 6.0) if with_dft else (7.2, 4.6))

    grid = fig.add_gridspec(
        3 if with_dft else 2, 2,
        width_ratios=[1.0, 0.018],
        height_ratios=[1.45, 1.0, 1.0] if with_dft else [1.45, 1.0],
        hspace=0.17,
        wspace=0.015,
        left=0.105, right=0.93, top=0.90,
        bottom=0.085 if with_dft else 0.115,
    )

    heat_ax = fig.add_subplot(grid[0, 0])
    bar_ax = fig.add_subplot(grid[0, 1])
    line_ax = fig.add_subplot(grid[1, 0], sharex=heat_ax)

    draw_heatmap(heat_ax, bar_ax, df, columns, blocks, highlight_x)
    draw_hexagon_panel(line_ax, table, blocks, highlight_x, use_fraction,
                       label_axis=not with_dft)

    panels = [(heat_ax, "(a)"), (line_ax, "(b)")]

    if with_dft:
        # An empty cell beside panel (b) so all three panels keep the same width
        # as the two that carry a colourbar.
        fig.add_subplot(grid[1, 1]).set_axis_off()

        energy_ax = fig.add_subplot(grid[2, 0], sharex=heat_ax)
        force_bar_ax = fig.add_subplot(grid[2, 1])

        draw_energy_panel(energy_ax, force_bar_ax, dft_table, blocks, highlight_x)
        panels.append((energy_ax, "(c)"))

    for ax, label in panels:
        ax.text(-0.098, 1.03, label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", ha="left", va="bottom")

    paths = []

    for suffix in ("png", "pdf"):
        path = figure_dir / "{}.{}".format(stem, suffix)
        fig.savefig(path, dpi=600 if suffix == "png" else None)
        paths.append(path)

    plt.close(fig)

    return paths


def print_dft_summary(dft_table):
    """Panel (c) numbers: coverage, Delta E span, and the force correlation."""
    energies = dft_table["delta_E_per_atom_eV"]
    have = dft_table[energies.notna()]
    missing = dft_table.loc[energies.isna(), "structure_id"].tolist()

    print()
    print("Single-point DFT energy (panel c)")
    print("-" * 62)
    print("converged energies: {} / {} structures".format(
        len(have), len(dft_table)))

    if missing:
        print("no converged run    : {}".format(", ".join(missing)))

    print("Delta E per atom: {:.3f} eV ({}) to {:.3f} eV ({})".format(
        have["delta_E_per_atom_eV"].min(),
        have.loc[have["delta_E_per_atom_eV"].idxmin(), "structure_id"],
        have["delta_E_per_atom_eV"].max(),
        have.loc[have["delta_E_per_atom_eV"].idxmax(), "structure_id"],
    ))
    print("max force: {:.1f} - {:.1f} eV/A; r(Delta E, max force) = {:.3f}, so the "
          "high-energy frames are the unrelaxed ones.".format(
              have["max_force_eV_A"].min(),
              have["max_force_eV_A"].max(),
              have["delta_E_per_atom_eV"].corr(have["max_force_eV_A"]),
          ))

    highlight = dft_table[dft_table["structure_id"] == HIGHLIGHT_ID]
    if len(highlight) and pd.notna(highlight.iloc[0]["delta_E_per_atom_eV"]):
        row = highlight.iloc[0]
        print("{}: Delta E = {:.3f} eV per atom, max force {:.1f} eV/A.".format(
            HIGHLIGHT_ID, row["delta_E_per_atom_eV"], row["max_force_eV_A"]))


def print_summary(stats, table, df):
    total_faces = int(stats["total_rings"].sum())
    non_hex_total = int(stats.loc[stats["ring_size"] != 6, "total_rings"].sum())

    print("Ring topology across {} experimental structures".format(len(df)))
    print("=" * 62)
    print("{:>4}  {:>7}  {:>11}  {:>7}  {}".format(
        "n", "total", "structures", "max", "structure(s) at max"))

    for _, row in stats.iterrows():
        print("{:>4}  {:>7}  {:>11}  {:>7}  {}".format(
            row["ring_size"],
            row["total_rings"],
            row["n_structures_with_ring"],
            row["max_in_one_structure"],
            row["structures_at_max"],
        ))

    print("-" * 62)
    print("bounded faces in total: {}  ({} hexagonal, {} non-hexagonal)".format(
        total_faces, total_faces - non_hex_total, non_hex_total))

    counts = dict(zip(stats["ring_size"], stats["n_structures_with_ring"]))
    large = stats[stats["ring_size"] > 9]
    n_large = int((df[["n_{}_rings".format(int(size))
                       for size in large["ring_size"]]].sum(axis=1) > 0).sum()) \
        if len(large) else 0

    print()
    print("Structures containing at least one ring of size:")
    for size in (5, 7, 8, 9):
        n_with = counts.get(size, 0)
        print("  n = {:<2} : {:>2} / {} structures ({:.0f}%)".format(
            size, n_with, len(df), 100.0 * n_with / len(df)))
    print("  n > 9  : {:>2} / {} structures ({:.0f}%)  [sizes {}]".format(
        n_large, len(df), 100.0 * n_large / len(df),
        ", ".join(str(int(size)) for size in large["ring_size"]) or "none"))

    print()
    print("Hexagon fraction f_6: min {:.3f} ({}), max {:.3f} ({}), "
          "mean {:.3f}".format(
              table["f_6"].min(),
              table.loc[table["f_6"].idxmin(), "structure_id"],
              table["f_6"].max(),
              table.loc[table["f_6"].idxmax(), "structure_id"],
              table["f_6"].mean()))
    print("N_6 spans {} - {} and correlates with carbon count at r = {:.3f}, "
          "so f_6 is the size-independent measure.".format(
              int(table["N_6"].min()), int(table["N_6"].max()),
              table["N_6"].corr(table["n_carbon_atoms"])))

    highlight = table[table["structure_id"] == HIGHLIGHT_ID]
    if len(highlight):
        row = highlight.iloc[0]
        nine = df.loc[df["structure_id"] == HIGHLIGHT_ID, "n_9_rings"].iloc[0]
        print()
        print("{}: {} hexagons, {} non-hexagonal faces, f_6 = {:.3f}, "
              "{} nine-membered ring(s).".format(
                  HIGHLIGHT_ID, int(row["N_6"]), int(row["N_nonhex"]),
                  row["f_6"], int(nine)))


def main():
    parser = argparse.ArgumentParser(
        description="Two-panel ring topology figure for the results section."
    )
    parser.add_argument("--summary", type=Path, default=SUMMARY_FILE)
    parser.add_argument("--dft-summary", type=Path, default=DFT_SUMMARY_FILE)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--stem", default=FIGURE_STEM)
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    df = load_census(args.summary)
    columns = size_columns(df)

    if not columns:
        raise SystemExit("No ring-size columns in {}".format(args.summary))

    stats = ring_statistics(df, columns)
    table = topology_fractions(df, columns)
    blocks = sequence_blocks(df)

    highlight_index = df.index[df["structure_id"] == HIGHLIGHT_ID]
    highlight_x = float(highlight_index[0]) + 0.5 if len(highlight_index) else None

    stats_path = args.table_dir / "ring_size_statistics.csv"
    fractions_path = args.table_dir / "ring_topology_fractions.csv"
    stats.to_csv(stats_path, index=False)
    table.to_csv(fractions_path, index=False, float_format="%.6f")

    written = []
    written += build_figure(df, table, columns, blocks, highlight_x,
                            use_fraction=False, figure_dir=args.figure_dir,
                            stem=args.stem)
    written += build_figure(df, table, columns, blocks, highlight_x,
                            use_fraction=True, figure_dir=args.figure_dir,
                            stem=args.stem + "_f6")

    tables = [stats_path, fractions_path]
    dft_table = None

    if args.dft_summary.exists():
        dft_table = dft_energy_table(table, load_dft_energies(args.dft_summary))

        energy_path = args.table_dir / "ring_topology_dft_energy.csv"
        dft_table.to_csv(energy_path, index=False, float_format="%.6f")
        tables.append(energy_path)

        written += build_figure(df, table, columns, blocks, highlight_x,
                                use_fraction=True, figure_dir=args.figure_dir,
                                stem=args.stem + "_f6_dft", dft_table=dft_table)
    else:
        print("Warning: {} not found, skipping the DFT energy figure.".format(
            args.dft_summary))

    print_summary(stats, table, df)

    if dft_table is not None:
        print_dft_summary(dft_table)

    print()
    for path in written + tables:
        print("Saved: {}".format(path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
