#!/usr/bin/env python3
"""
Find and plot carbon rings of every size for the defective graphene structures.

This generalises plot_5_7_rings_all.py, which hard-coded a {5, 7} filter and
silently dropped everything else. The A-J set actually contains 3-, 4-, 5-, 6-,
7-, 8-, 9-, 10-, 12-, 13- and 16-membered rings, so most of the non-hexagonal
rings never made it into the figures.

Ring finding is the production planar-face walk, carbon_ring_cycles() in
scripts/hydrogenation/full_saturate_all_structures.py, rather than the networkx
minimum cycle basis the 5/7 scripts used. The face walk enumerates every bounded
face exactly and returns the atoms already in traversal order, it is what the H
placement code already trusts, and it needs no networkx - which is not installed
in the ARCHER2 .venv-dos.

Every ring n != 6 is filled with its own colour and labelled with its size at the
centroid; hexagons get a faint grey wash so the intact lattice reads as
background. Pass --no-hexagons to leave them unfilled.

Inputs:  <ROOT>/{A..J}/{1..10}/geometry.in, or one file via --geometry
Outputs: <ROOT>/results/figures/n_membered_rings/<ID>_rings.png
         <ROOT>/results/tables/n_membered_rings/<ID>_rings.csv
         <ROOT>/results/tables/n_membered_rings/all_structures_ring_summary.csv

Run from the repository root:

    .venv-dos/bin/python3 scripts/visualisation/plot_n_membered_rings_all.py
    .venv-dos/bin/python3 scripts/visualisation/plot_n_membered_rings_all.py --limit 3
    .venv-dos/bin/python3 scripts/visualisation/plot_n_membered_rings_all.py \
        --geometry data/processed/c8_backbone.xyz
"""

from pathlib import Path
import argparse
import sys
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hydrogenation"))

import full_saturate_all_structures as F  # noqa: E402


# Hexagons are the undisturbed lattice, so they are drawn as background rather
# than as a finding. Everything else is a defect ring and gets a real colour:
# warm below six, cool above, so under- and over-coordinated rings separate at a
# glance. 5 and 7 keep the colours plot_5_7_rings_all.py used so the new figures
# can be compared with the old ones.
HEXAGON_COLOUR = ("lightgrey", "grey")
HEXAGON_ALPHA = 0.12

RING_COLOURS = {
    3: ("darkred", "black"),
    4: ("firebrick", "darkred"),
    5: ("orange", "darkorange"),
    6: HEXAGON_COLOUR,
    7: ("deepskyblue", "royalblue"),
    8: ("mediumseagreen", "darkgreen"),
    9: ("mediumpurple", "indigo"),
    10: ("orchid", "purple"),
}

RING_ALPHA = 0.40

# Rings larger than the explicit table are rare pores (one 12-, two 13- and one
# 16-membered face across all 95 structures). Sampled rather than listed so the
# script never runs out of colours on a structure it has not seen.
LARGE_RING_CMAP = "turbo"
LARGE_RING_CMAP_RANGE = (11, 24)


def ring_colour(ring_size):
    """Face and edge colour for a ring of the given size."""
    if ring_size in RING_COLOURS:
        return RING_COLOURS[ring_size]

    low, high = LARGE_RING_CMAP_RANGE
    fraction = (min(max(ring_size, low), high) - low) / float(high - low)
    facecolor = matplotlib.colormaps[LARGE_RING_CMAP](fraction)

    return facecolor, "black"


def find_rings(symbols, positions, neighbour_indices, exclude=None, sizes=None):
    """
    Every bounded carbon ring face, as (cycle, polygon, area), smallest first.

    sizes optionally restricts the result to those ring sizes; the default keeps
    all of them.
    """
    rings = F.carbon_ring_cycles(
        symbols, positions, neighbour_indices, exclude=exclude
    )

    if sizes is not None:
        rings = [ring for ring in rings if len(ring[0]) in sizes]

    return sorted(rings, key=lambda ring: len(ring[0]))


def ring_size_counts(rings):
    """{ring size: count} over a list of (cycle, polygon, area)."""
    counts = {}

    for cycle, _, _ in rings:
        counts[len(cycle)] = counts.get(len(cycle), 0) + 1

    return counts


def census_text(counts):
    """Compact non-hexagonal ring census for the figure title."""
    defects = ", ".join(
        "{}: {}".format(size, counts[size])
        for size in sorted(counts)
        if size != 6
    )

    if not defects:
        defects = "none"

    return "{}  ({} hexagons)".format(defects, counts.get(6, 0))


def plot_structure_with_rings(
    symbols,
    positions,
    neighbour_indices,
    rings,
    output_file,
    title_prefix,
    show_hexagons=True,
):
    """
    Plot carbon atoms, C-C bonds, and every ring found, coloured by size.
    """
    print("Plotting rings...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw C-C bonds first. neighbour_indices is symmetric, so take each pair
    # once.
    for i, neighbours in enumerate(neighbour_indices):
        for j in neighbours:
            if j < i:
                continue

            ax.plot(
                [positions[i][0], positions[j][0]],
                [positions[i][1], positions[j][1]],
                color="lightgrey",
                linewidth=0.8,
                zorder=1,
            )

    ring_rows = []

    # Draw highlighted rings. Rings arrive smallest first, so the legend comes
    # out in ascending size order for free.
    for ring_id, (cycle, polygon, area) in enumerate(rings, start=1):
        ring_size = len(cycle)
        is_hexagon = ring_size == 6

        coords = np.asarray(polygon, dtype=float)
        centroid = coords.mean(axis=0)

        ring_rows.append(
            {
                "ring_id": ring_id,
                "ring_size": ring_size,
                "atom_indices": " ".join(str(i) for i in cycle),
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "area_A2": area,
            }
        )

        if is_hexagon and not show_hexagons:
            continue

        facecolor, edgecolor = ring_colour(ring_size)
        label = "{}-membered rings".format(ring_size)

        existing_labels = ax.get_legend_handles_labels()[1]
        plot_label = label if label not in existing_labels else None

        closed_coords = np.vstack([coords, coords[0]])

        ax.fill(
            closed_coords[:, 0],
            closed_coords[:, 1],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.5,
            alpha=HEXAGON_ALPHA if is_hexagon else RING_ALPHA,
            label=plot_label,
            zorder=2,
        )

        # Labelling every hexagon would bury the defect rings in sixes.
        if not is_hexagon:
            ax.text(
                centroid[0],
                centroid[1],
                str(ring_size),
                fontsize=9,
                ha="center",
                va="center",
                color="black",
                zorder=6,
            )

    # Draw atoms on top.
    carbon_indices = [i for i, sym in enumerate(symbols) if sym == "C"]
    carbon_positions = positions[carbon_indices]

    ax.scatter(
        carbon_positions[:, 0],
        carbon_positions[:, 1],
        s=22,
        c="lightgrey",
        edgecolors="black",
        linewidths=0.4,
        zorder=4,
        label="C atoms",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / Å", fontsize=18)
    ax.set_ylabel("y / Å", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)

    ax.set_title(
        "{}: ring sizes\n{}".format(
            title_prefix, census_text(ring_size_counts(rings))
        ),
        fontsize=22,
        pad=16,
    )

    ax.legend(loc="best", fontsize=12, frameon=True)

    pad = 2.0
    ax.set_xlim(positions[:, 0].min() - pad, positions[:, 0].max() + pad)
    ax.set_ylim(positions[:, 1].min() - pad, positions[:, 1].max() + pad)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(ring_rows)


def rings_for_geometry(geometry_file, sizes=None):
    """
    Parse one geometry and return (symbols, positions, neighbour_indices, rings).

    Same cluster-mode settings as build_ring_face_area_cache.py: no lattice (so
    no minimum image), stretched bonds healed, disconnected fragments excluded.
    Excluding fragments is not optional - without it an island's own outer
    boundary comes back as a bounded face of the whole picture.
    """
    _, symbols, positions = F.read_fhi_aims_geometry(geometry_file)

    coordination, neighbour_indices = F.compute_coordination(
        symbols, positions, None, None
    )
    F.heal_stretched_bonds(
        symbols, positions, coordination, neighbour_indices, None, None
    )

    fragment_atoms = F.disconnected_fragment_atoms(symbols, neighbour_indices)

    rings = find_rings(
        symbols,
        positions,
        neighbour_indices,
        exclude=fragment_atoms,
        sizes=sizes,
    )

    return symbols, positions, neighbour_indices, rings


def path_label(path, root):
    """Repository-relative path where possible, absolute otherwise."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def process_one_structure(
    geom_file,
    root,
    figures_dir,
    tables_dir,
    structure_id,
    sizes=None,
    show_hexagons=True,
):
    """
    Process one A/1/geometry.in style structure.
    """
    output_figure = figures_dir / "{}_rings.png".format(structure_id)
    output_table = tables_dir / "{}_rings.csv".format(structure_id)

    geometry_label = path_label(geom_file, root)

    print()
    print("=" * 80)
    print("Processing {}: {}".format(structure_id, geometry_label))
    print("=" * 80)

    symbols, positions, neighbour_indices, rings = rings_for_geometry(
        geom_file, sizes=sizes
    )

    n_carbon = sum(sym == "C" for sym in symbols)
    n_bonds = sum(len(n) for n in neighbour_indices) // 2

    print("Number of atoms: {}".format(len(symbols)))
    print("Number of carbon atoms: {}".format(n_carbon))
    print("Number of C-C bonds: {}".format(n_bonds))

    counts = ring_size_counts(rings)

    for size in sorted(counts):
        print("Found {} {}-rings.".format(counts[size], size))

    ring_table = plot_structure_with_rings(
        symbols=symbols,
        positions=positions,
        neighbour_indices=neighbour_indices,
        rings=rings,
        output_file=output_figure,
        title_prefix=structure_id,
        show_hexagons=show_hexagons,
    )

    ring_table.insert(0, "structure_id", structure_id)
    ring_table.insert(1, "geometry_file", geometry_label)
    ring_table.to_csv(output_table, index=False)

    print("Saved figure: {}".format(path_label(output_figure, root)))
    print("Saved table:  {}".format(path_label(output_table, root)))

    row = {
        "structure_id": structure_id,
        "geometry_file": geometry_label,
        "n_atoms": len(symbols),
        "n_carbon_atoms": n_carbon,
        "n_bonds": n_bonds,
        "n_rings": len(rings),
        "n_non_hexagonal_rings": len(rings) - counts.get(6, 0),
    }

    for size, count in counts.items():
        row["n_{}_rings".format(size)] = count

    row["figure"] = path_label(output_figure, root)
    row["table"] = path_label(output_table, root)

    return row


def tidy_summary(summary_rows):
    """
    Assemble the summary table with the ring-size columns in ascending order.

    Sizes are the union over all structures, so a structure with no 8-rings gets
    a 0 rather than a blank.
    """
    summary = pd.DataFrame(summary_rows)

    size_columns = sorted(
        (column for column in summary.columns if column.endswith("_rings")
         and column not in {"n_rings", "n_non_hexagonal_rings"}),
        key=lambda column: int(column.split("_")[1]),
    )

    summary[size_columns] = summary[size_columns].fillna(0)

    # Only the successful rows get a real 0; a failed structure has no counts at
    # all and must not read as "no rings found".
    failed = summary["status"] == "failed"
    summary.loc[failed, size_columns] = np.nan

    for column in size_columns:
        summary[column] = summary[column].astype("Int64")

    leading = [
        "structure_id",
        "geometry_file",
        "n_atoms",
        "n_carbon_atoms",
        "n_bonds",
        "n_rings",
        "n_non_hexagonal_rings",
    ]
    trailing = ["figure", "table", "status", "error"]

    return summary[leading + size_columns + trailing]


def parse_sizes(text):
    """Parse --sizes 5,7 into {5, 7}. None means keep every size."""
    if text is None:
        return None

    sizes = {int(part) for part in text.replace(",", " ").split()}

    if not sizes:
        raise ValueError("--sizes was given but parsed to nothing")

    return sizes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot carbon rings of every size for the A-J defective graphene structures."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing A, B, C, ..., J folders. Default: current directory.",
    )

    parser.add_argument(
        "--geometry",
        type=Path,
        default=None,
        help="Plot a single geometry file instead of the A-J batch, "
             "e.g. data/processed/c8_backbone.xyz.",
    )

    parser.add_argument(
        "--structure-id",
        type=str,
        default=None,
        help="Process one A-J structure only, e.g. H02. Case-insensitive.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of structures to process, useful for testing.",
    )

    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated ring sizes to keep, e.g. 5,7 to reproduce the old "
             "five/seven figures. Default: every size found.",
    )

    parser.add_argument(
        "--no-hexagons",
        action="store_true",
        help="Leave 6-membered rings unfilled instead of washing them pale grey.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    root = args.root.resolve()
    sizes = parse_sizes(args.sizes)
    show_hexagons = not args.no_hexagons

    if not root.exists():
        raise FileNotFoundError("Root directory does not exist: {}".format(root))

    figures_dir = root / "results" / "figures" / "n_membered_rings"
    tables_dir = root / "results" / "tables" / "n_membered_rings"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if args.geometry is not None:
        geometry_files = [args.geometry.resolve()]
    else:
        geometry_files = F.find_all_geometry_files(root)

        if args.structure_id is not None:
            geometry_files = F.filter_geometry_files_by_structure_id(
                geometry_files, args.structure_id
            )

        if args.limit is not None:
            geometry_files = geometry_files[:args.limit]

        if len(geometry_files) == 0:
            raise FileNotFoundError(
                "No geometry.in files found in A/1, A/2, ..., J/5 style folders. "
                "Make sure you are running this from the repository root, or pass --root."
            )

    print("Root directory: {}".format(root))
    print("Found {} structures.".format(len(geometry_files)))
    print("C-C cutoff: {:.3f} Å (stretched to {:.3f} Å)".format(
        F.CC_CUTOFF, F.CC_CUTOFF_STRETCHED
    ))
    print("Ring sizes: {}".format("all" if sizes is None else sorted(sizes)))

    summary_rows = []

    for geom_file in geometry_files:
        if args.geometry is not None:
            structure_id = geom_file.stem
        else:
            structure_id = F.structure_id_from_path(geom_file)

        try:
            row = process_one_structure(
                geom_file=geom_file,
                root=root,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
                structure_id=structure_id,
                sizes=sizes,
                show_hexagons=show_hexagons,
            )

            row["status"] = "success"
            row["error"] = ""
            summary_rows.append(row)

        except Exception as exc:
            print()
            print("ERROR processing {}: {}".format(structure_id, exc))
            print(traceback.format_exc())

            summary_rows.append(
                {
                    "structure_id": structure_id,
                    "geometry_file": path_label(geom_file, root),
                    "n_atoms": None,
                    "n_carbon_atoms": None,
                    "n_bonds": None,
                    "n_rings": None,
                    "n_non_hexagonal_rings": None,
                    "figure": "",
                    "table": "",
                    "status": "failed",
                    "error": str(exc),
                }
            )

    summary_table = tidy_summary(summary_rows)

    # A single-structure run is not the A-J set and must not overwrite its
    # summary.
    if args.geometry is None and args.structure_id is None:
        summary_file = tables_dir / "all_structures_ring_summary.csv"
        summary_table.to_csv(summary_file, index=False)
    else:
        summary_file = None

    n_success = int((summary_table["status"] == "success").sum())
    n_failed = int((summary_table["status"] == "failed").sum())

    print()
    print("=" * 80)
    print("Done.")
    print("Processed: {}".format(len(summary_rows)))
    print("Successful: {}".format(n_success))
    print("Failed:     {}".format(n_failed))

    if summary_file is not None:
        print("Saved summary: {}".format(summary_file))

    print("=" * 80)


if __name__ == "__main__":
    main()
