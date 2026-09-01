#!/usr/bin/env python3
"""
Local Mulliken DOS of one non-hexagonal ring in a cluster run.

Companion to plot_pristine_patch_dos.py, which asks whether an interior patch of a torn
flake still looks like graphene. This asks the opposite question about the same kind of
calculation: does a single reconstructed ring carry its own states, and where do they sit
relative to E_F?

The motivating case is H02, whose nanoribbon necks down around a nine-membered ring
(centroid x 24.4, y 12.7, area 13.6 A^2, the largest face in the frame after the
sixteen-carbon pore of I02). That ring is 9 of 379 carbons, so its states are invisible
in the whole-structure DOS and only appear under projection.

Three panels, laid out as in plot_pristine_patch_dos.py:

    (a) the carbon skeleton with the selected ring drawn as a filled green polygon and
        its carbons in green, so the projection is visible rather than asserted. This
        replaces that script's rectangular selection box: a ring is a face of the bond
        graph, not a region of the plane, so there is nothing rectangular to draw.
    (b) the ring LDOS decomposed into C s and C p, per carbon;
    (c) the near-E_F window, which is the panel the figure is for. A pristine patch has
        a density falling towards zero at the Dirac point; weight piling up at 0 eV here
        instead is the ring carrying a localised state.

The ring is found with full_saturate_all_structures.carbon_ring_cycles() via
pristine_patch_regions.carbon_bonding(), i.e. the same planar face walk on the same
cluster-mode bond graph that draws results/figures/n_membered_rings/<ID>_rings.png, so
the ring selected here is by construction the ring shown there.

Reads the per-atom cache from scripts/dos/cache_atom_resolved_dos.py, not Mulliken.out:
the cluster H02 Mulliken output is 122 MB and needs carmm to parse, while the cache is
4.6 MB and needs only numpy and scipy. Broadening is linear, so projecting the cache onto
a subset of atoms is exact rather than an approximation of the original pipeline.

Note the cache root. results/tables/dos_atom_resolved/ holds the *periodic* runs; the
cluster caches live in results/tables/dos_atom_resolved_cluster/, which is what
--cache-root defaults to here.

Caveat carried by --compare. The bulk-hexagon reference is the box-free counterpart of
plot_pristine_patch_dos.py's hand-drawn patch: three-coordinate carbons in hexagons only,
at least --min-edge-hops bonds from any terminated or undercoordinated site. It needs no
per-structure box, which is what lets this script run on a frame that has none.

Inputs:
    calculations/dos_cluster/<ID>/geometry.in
    results/tables/dos_atom_resolved_cluster/<ID>.npz

Outputs:
    results/figures/dos/<ID>_cluster_ring<N>_dos.png
    results/figures/dos/<ID>_cluster_ring<N>_vs_bulk.png   (--compare only)
    results/tables/dos_ring_summary.csv

Usage:
    python scripts/dos/plot_ring_dos.py --structure-id H02 --ring-size 9
    python scripts/dos/plot_ring_dos.py --structure-id H02 --ring-size 9 --compare
    python scripts/dos/plot_ring_dos.py --structure-id H02 --ring-size 5 --list-rings
"""

from pathlib import Path
import argparse
import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pristine_patch_regions as P  # noqa: E402
from cache_atom_resolved_dos import read_cache  # noqa: E402
from crop_regions import check_cache_alignment  # noqa: E402
from dos_common import ROOT, read_fhi_aims_geometry  # noqa: E402
from plot_dos_mulliken import add_carmm_to_path, draw_dos_axes  # noqa: E402
from plot_pristine_patch_dos import (  # noqa: E402
    BOND,
    BULK,
    CLUSTER_CACHE_ROOT,
    DEFAULT_CARMM,
    DEFAULT_XLIM,
    DEFAULT_ZOOM,
    EDGE_DEFECT,
    FIGURES_ROOT,
    HYDROGEN,
    INK_SECONDARY,
    PATCH,
    PATCH_FILL,
    bond_segments,
    carbon_curve,
    draw_zoom,
    near_fermi_metrics,
    occupied_valence_electrons,
    repo_relative,
)
from region_dos import DEFAULT_SIGMA, project, ylabel  # noqa: E402

SUMMARY_FILE = ROOT / "results/tables/dos_ring_summary.csv"

RING_REGION = "ring"
BULK_REGION = "bulk_hexagon"
OUTSIDE_REGION = P.OUTSIDE_REGION


def find_rings(structure_id, calc_root=None):
    """
    Every bounded ring face of the cluster's carbon skeleton, with its geometry.

    Deliberately not P.load_structure(): that one requires a hand-drawn entry in
    PATCH_BOXES and returns a patch mask, neither of which a ring projection has any use
    for. The bonding calls underneath are the same, so the two agree about what is bonded.
    """
    path = P.geometry_path(structure_id, calc_root)
    _lattice, symbols, positions = read_fhi_aims_geometry(path)

    coordination, neighbour_indices, fragment_atoms = P.carbon_bonding(symbols, positions)
    cycles = P.F.carbon_ring_cycles(symbols, positions, neighbour_indices,
                                    exclude=fragment_atoms)

    rings = []
    for cycle, polygon, area in cycles:
        centroid = positions[list(cycle)][:, :2].mean(axis=0)
        rings.append({
            "atoms": list(cycle),
            "polygon": np.asarray(polygon, dtype=float),
            "area": float(area),
            "size": len(cycle),
            "centroid": centroid,
        })

    # Sorted so --ring-index is stable between runs: face-walk order depends on which
    # directed edge the walk happened to start from, which is set iteration order.
    rings.sort(key=lambda r: (r["size"], round(r["centroid"][0], 3),
                              round(r["centroid"][1], 3)))

    return {
        "structure_id": structure_id,
        "path": path,
        "symbols": symbols,
        "positions": positions,
        "coordination": coordination,
        "neighbour_indices": neighbour_indices,
        "fragment_atoms": fragment_atoms,
        "rings": rings,
    }


def rings_of_size(structure, size):
    return [r for r in structure["rings"] if r["size"] == size]


def select_ring(structure, size, ring_index):
    """One ring of the requested size, refusing to guess when there is more than one."""
    matches = rings_of_size(structure, size)

    if not matches:
        census = {}
        for ring in structure["rings"]:
            census[ring["size"]] = census.get(ring["size"], 0) + 1
        raise SystemExit(
            "{} has no {}-membered ring. Census: {}.".format(
                structure["structure_id"], size,
                " ".join("{}:{}".format(k, census[k]) for k in sorted(census)))
        )

    if len(matches) == 1:
        return matches[0]

    if ring_index is None:
        raise SystemExit(
            "{} has {} {}-membered rings; pass --ring-index to choose one, or "
            "--list-rings to see them.".format(
                structure["structure_id"], len(matches), size)
        )

    if not 0 <= ring_index < len(matches):
        raise SystemExit(
            "--ring-index {} is out of range; {} has {} {}-membered rings (0-{}).".format(
                ring_index, structure["structure_id"], len(matches), size,
                len(matches) - 1)
        )

    return matches[ring_index]


def print_rings(structure, size=None):
    """Table of the candidate rings, so --ring-index can be chosen from something."""
    selected = structure["rings"] if size is None else rings_of_size(structure, size)

    print("{}: {} ring(s){}".format(
        structure["structure_id"], len(selected),
        "" if size is None else " of size {}".format(size)))
    print("  {:>5}  {:>4}  {:>16}  {:>7}  atoms".format(
        "index", "size", "centroid x, y / A", "area"))

    for index, ring in enumerate(selected):
        print("  {:>5}  {:>4}  {:>7.2f} {:>8.2f}  {:>7.2f}  {}".format(
            index, ring["size"], ring["centroid"][0], ring["centroid"][1],
            ring["area"], " ".join(str(a) for a in sorted(ring["atoms"]))))


def bulk_hexagon_mask(structure, min_edge_hops):
    """
    The box-free pristine reference: hexagon-only carbons well away from any edge.

    Same three conditions as pristine_patch_regions.audit_patch() asserts of its
    hand-drawn patch - three-coordinate, hexagons only, at least min_edge_hops bonds from
    a terminated or undercoordinated carbon - minus the box, which is the only part of
    that selection that has to be drawn per structure.
    """
    symbols = structure["symbols"]
    positions = structure["positions"]

    non_hexagonal, _rings = P.non_hexagonal_carbons(
        symbols, positions, structure["neighbour_indices"], structure["fragment_atoms"])
    seeds = P.edge_seed_atoms(symbols, positions, structure["coordination"])
    hops = P.edge_hops(symbols, structure["neighbour_indices"], seeds)

    mask = np.zeros(len(symbols), dtype=bool)
    for index, symbol in enumerate(symbols):
        if symbol != "C" or index in non_hexagonal:
            continue
        if structure["coordination"][index] < 3:
            continue
        if hops.get(index, 0) >= min_edge_hops:
            mask[index] = True

    return mask, hops


def rows_for_mask(structure, mask, region, hops=None):
    """Rows in the shape region_dos.project() expects, with `region` on the selection."""
    return [
        {
            "atom_index": index,
            "element": symbol,
            "region": region if mask[index] else OUTSIDE_REGION,
            "edge_hops": int(hops.get(index, 0)) if hops else 0,
            # project() only reads this when region == 'ribbon', so it is inert here.
            "ribbon_width_A": None,
        }
        for index, symbol in enumerate(structure["symbols"])
    ]


def ring_mask(structure, ring):
    mask = np.zeros(len(structure["symbols"]), dtype=bool)
    mask[list(ring["atoms"])] = True
    return mask


def draw_structure(ax, structure, ring, mask, title):
    """Panel (a): skeleton, the ring as a filled polygon, its carbons in green."""
    positions = structure["positions"]
    symbols = np.asarray(structure["symbols"])

    ax.add_collection(
        matplotlib.collections.LineCollection(
            bond_segments(positions, symbols),
            colors=BOND, linewidths=0.6, alpha=0.8, zorder=1,
        )
    )

    # The ring face itself, in place of plot_pristine_patch_dos.py's dashed rectangle.
    # carbon_ring_cycles() returns the polygon in traversal order, so it closes without
    # reordering.
    ax.add_patch(
        plt.Polygon(
            ring["polygon"], closed=True,
            facecolor=PATCH_FILL, edgecolor=PATCH, lw=1.6,
            alpha=0.95, zorder=2,
        )
    )

    carbon = symbols == "C"
    hydrogen = symbols == "H"
    n_carbon = int(carbon.sum())

    ax.scatter(positions[carbon & ~mask, 0], positions[carbon & ~mask, 1],
               s=9, c=BULK, edgecolors="none", zorder=3,
               label="C, rest of flake ({})".format(n_carbon - int(mask.sum())))
    ax.scatter(positions[mask, 0], positions[mask, 1],
               s=30, c=PATCH, edgecolors="none", zorder=5,
               label="C, {}-membered ring ({})".format(ring["size"], int(mask.sum())))
    ax.scatter(positions[hydrogen, 0], positions[hydrogen, 1],
               s=7, c=HYDROGEN, edgecolors="none", zorder=4,
               label="H ({})".format(int(hydrogen.sum())))

    ax.set_aspect("equal")
    ax.set_xlabel(r"x / $\mathrm{\AA}$")
    ax.set_ylabel(r"y / $\mathrm{\AA}$")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left", frameon=False,
              labelcolor=INK_SECONDARY, handletextpad=0.4, borderpad=0.2)


def build_figure(structure, ring, mask, projection, xlim, zoom, output_file):
    fig = plt.figure(figsize=(13.5, 5.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1, 1],
                            wspace=0.24, hspace=0.55)

    ax_structure = fig.add_subplot(grid[:, 0])
    ax_dos = fig.add_subplot(grid[0, 1])
    ax_zoom = fig.add_subplot(grid[1, 1])

    draw_structure(
        ax_structure, structure, ring, mask,
        "(a)  {}  -  {}-membered ring at x {:.1f}, y {:.1f}, area {:.1f} $\\mathrm{{\\AA}}^2$".format(
            structure["structure_id"], ring["size"], ring["centroid"][0],
            ring["centroid"][1], ring["area"]),
    )

    draw_dos_axes(
        ax_dos, projection, xlim, None,
        "(b)  {}-ring LDOS  -  {} C, per carbon".format(ring["size"], projection["n_C"]),
        title_fontsize=10,
    )
    ax_dos.set_ylabel(ylabel(projection["per_atom"]))

    draw_zoom(ax_zoom, projection, zoom)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output_file))


def build_comparison(ring_data, bulk_data, ring_size, zoom, output_file):
    """The ring against the bulk-hexagon carbons of the same run, per carbon."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    labels = {
        RING_REGION: "{}-membered ring".format(ring_size),
        BULK_REGION: "bulk hexagons",
    }

    for ax, window, label in (
        (axes[0], (ring_data["x"][0], 4.0), "full valence range"),
        (axes[1], zoom, "near $E_F$"),
    ):
        for data, colour in ((bulk_data, PATCH), (ring_data, EDGE_DEFECT)):
            x = data["x"]
            y = carbon_curve(data)
            inside = (x >= window[0]) & (x <= window[1])
            ax.plot(x[inside], y[inside], lw=1.5, color=colour,
                    label="{} ({} C)".format(labels[data["region"]], data["n_C"]))

        ax.axvline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.6)
        ax.set_xlim(*window)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(ring_data["graph_xlabel"])
        ax.set_ylabel(ylabel(ring_data["per_atom"]))
        ax.set_title("C s+p per carbon, {}".format(label), fontsize=10)
        ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output_file))


def write_summary(path, row):
    """
    Append or replace one row of the ring summary.

    Not plot_pristine_patch_dos.write_summary(): that one keys on structure and sigma,
    which is enough when a structure has exactly one patch. A structure has many rings,
    so projecting onto the 9-ring and then the 5-ring of the same frame at the same
    broadening would silently drop the first. The key here carries the ring as well.
    """
    key = ("structure_id", "ring_size", "ring_index", "sigma_eV")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if any(r.get(k) != str(row[k]) for k in key)]

    rows.append({k: str(v) for k, v in row.items()})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote {}".format(path))


def main():
    parser = argparse.ArgumentParser(
        description="Project a cluster Mulliken DOS onto one non-hexagonal ring."
    )
    parser.add_argument("--structure-id", default="H02")
    parser.add_argument("--ring-size", type=int, default=9,
                        help="Ring size to project onto (default 9).")
    parser.add_argument("--ring-index", type=int, default=None,
                        help="Which ring of that size, when the structure has more than "
                             "one. See --list-rings.")
    parser.add_argument("--list-rings", action="store_true",
                        help="Print the rings of the requested size and exit.")
    parser.add_argument("--calc-root", default=str(P.CLUSTER_CALC_ROOT),
                        help="Directory holding <ID>/geometry.in for the cluster runs.")
    parser.add_argument("--cache-root", default=str(CLUSTER_CACHE_ROOT),
                        help="Per-atom DOS cache directory (cluster, NOT the periodic one).")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="Gaussian broadening in eV (default {:.4f}).".format(DEFAULT_SIGMA))
    parser.add_argument("--min-edge-hops", type=int, default=P.DEFAULT_MIN_EDGE_HOPS,
                        help="Bulk-hexagon reference only: minimum C-C bonds from any "
                             "edge carbon (default {}).".format(P.DEFAULT_MIN_EDGE_HOPS))
    parser.add_argument("--xlim", nargs=2, type=float, default=list(DEFAULT_XLIM))
    parser.add_argument("--zoom", nargs=2, type=float, default=list(DEFAULT_ZOOM))
    parser.add_argument("--compare", action="store_true",
                        help="Also plot the ring against the bulk-hexagon carbons of "
                             "the same run.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default=str(SUMMARY_FILE))
    parser.add_argument("--carmm-path", default=DEFAULT_CARMM)
    args = parser.parse_args()

    add_carmm_to_path(args.carmm_path)

    structure = find_rings(args.structure_id, args.calc_root)

    if args.list_rings:
        print_rings(structure, args.ring_size)
        return

    ring = select_ring(structure, args.ring_size, args.ring_index)
    mask = ring_mask(structure, ring)

    cache = read_cache(args.cache_root, args.structure_id)
    check_cache_alignment(cache, structure["symbols"], args.structure_id)

    if cache["nkpts"] != 1:
        raise SystemExit(
            "{} cache has {} k-points, so it is a periodic run, not a cluster. Point "
            "--cache-root at {}.".format(
                args.structure_id, cache["nkpts"], CLUSTER_CACHE_ROOT)
        )

    projection = project(cache, rows_for_mask(structure, mask, RING_REGION),
                         RING_REGION, sigma=args.sigma, per_atom=True)

    print("{}: {}-membered ring, {} C, centroid ({:.2f}, {:.2f}), area {:.2f} A^2".format(
        args.structure_id, ring["size"], projection["n_C"],
        ring["centroid"][0], ring["centroid"][1], ring["area"]))

    output = Path(args.output).resolve() if args.output else (
        FIGURES_ROOT / "{}_cluster_ring{}_dos.png".format(
            args.structure_id, ring["size"]))
    build_figure(structure, ring, mask, projection, tuple(args.xlim), tuple(args.zoom),
                 output)

    comparison_file = ""
    bulk = None
    if args.compare:
        bulk_mask, hops = bulk_hexagon_mask(structure, args.min_edge_hops)
        bulk = project(cache, rows_for_mask(structure, bulk_mask, BULK_REGION, hops),
                       BULK_REGION, sigma=args.sigma, per_atom=True)
        print("bulk-hexagon reference: {} C at >= {} bonds from an edge".format(
            bulk["n_C"], args.min_edge_hops))
        comparison_file = FIGURES_ROOT / "{}_cluster_ring{}_vs_bulk.png".format(
            args.structure_id, ring["size"])
        build_comparison(projection, bulk, ring["size"], tuple(args.zoom),
                         comparison_file)

    row = {
        "structure_id": args.structure_id,
        "ring_size": ring["size"],
        "ring_index": args.ring_index if args.ring_index is not None else 0,
        "ring_atoms": " ".join(str(a) for a in sorted(ring["atoms"])),
        "centroid_x_A": round(float(ring["centroid"][0]), 3),
        "centroid_y_A": round(float(ring["centroid"][1]), 3),
        "ring_area_A2": round(ring["area"], 3),
        "n_atoms_total": len(structure["symbols"]),
        "n_C_total": sum(1 for s in structure["symbols"] if s == "C"),
        "n_C_in_ring": projection["n_C"],
    }
    row.update(near_fermi_metrics(projection, tuple(args.zoom)))
    row.update({
        "sigma_eV": round(args.sigma, 6),
        "homo_eV": round(cache["homo"], 6),
        "nkpts": cache["nkpts"],
        "occupied_valence_electrons_per_C": round(
            occupied_valence_electrons(projection, cache["nspin"]), 3),
        "n_C_bulk_reference": bulk["n_C"] if bulk else "",
        "suppression_pm0p5_bulk": (
            near_fermi_metrics(bulk, tuple(args.zoom))["suppression_pm0p5"]
            if bulk else ""),
        "figure": repo_relative(output),
        "comparison_figure": repo_relative(comparison_file) if comparison_file else "",
    })
    write_summary(args.summary, row)


if __name__ == "__main__":
    main()
