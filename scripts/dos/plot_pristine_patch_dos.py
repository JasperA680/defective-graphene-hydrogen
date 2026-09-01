#!/usr/bin/env python3
"""
Local Mulliken DOS of the pristine graphene patch in a cluster run.

The whole-structure DOS of a torn flake (results/figures/dos/<ID>_cluster_mulliken_dos.png)
is dominated by its edges: 85 of D10's 487 atoms are terminating hydrogens and 79 carbons
are edge sites. This projects the same calculation onto the carbons that are none of
those - the top-left patch selected by pristine_patch_regions.py - to see whether an
interior piece of the flake carries a graphene-like local DOS.

Three panels:

    (a) the carbon skeleton, the selection box, and the selected carbons, so the
        projection is visible rather than asserted;
    (b) the patch LDOS decomposed into C s and C p, per carbon;
    (c) the near-E_F window, which is the panel that answers the question. Graphene has
        a density that falls linearly to zero at the Dirac point; a peak at 0 eV instead
        would mean edge or defect states have leaked into the patch.

Reads the per-atom cache from scripts/dos/cache_atom_resolved_dos.py, not Mulliken.out:
the cluster D10 Mulliken output is 134 MB and needs carmm to parse, while the cache is
5 MB and needs only numpy and scipy. Broadening is linear, so projecting the cache onto a
subset of atoms is exact rather than an approximation of the original pipeline.

Note the cache root. results/tables/dos_atom_resolved/ holds the *periodic* runs; the
cluster caches live in results/tables/dos_atom_resolved_cluster/, which is what
--cache-root defaults to here.

Inputs:
    calculations/dos_cluster/<ID>/geometry.in
    results/tables/dos_atom_resolved_cluster/<ID>.npz

Outputs:
    results/figures/dos/<ID>_cluster_pristine_patch_dos.png
    results/figures/dos/<ID>_cluster_pristine_patch_vs_edge.png   (--compare only)
    results/tables/dos_pristine_patch_summary.csv

Usage:
    python scripts/dos/plot_pristine_patch_dos.py --structure-id D10
    python scripts/dos/plot_pristine_patch_dos.py --structure-id D10 --sigma 0.05
    python scripts/dos/plot_pristine_patch_dos.py --structure-id D10 --compare
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
from dos_common import ROOT  # noqa: E402
from plot_dos_mulliken import add_carmm_to_path, draw_dos_axes  # noqa: E402
from region_dos import DEFAULT_SIGMA, project, ylabel  # noqa: E402

CLUSTER_CACHE_ROOT = ROOT / "results/tables/dos_atom_resolved_cluster"
FIGURES_ROOT = ROOT / "results/figures/dos"
SUMMARY_FILE = ROOT / "results/tables/dos_pristine_patch_summary.csv"

# The path the ARCHER2 DOS jobs pass. draw_dos_axes() imports carmm for its axis
# formatting even though the cache does the real work, so it has to be importable.
DEFAULT_CARMM = "/work/e89/e89/jasperallen1/carmm"

# Same cutoff as the bond graph everywhere else in this repo, so panel (a) draws the
# same skeleton as the ring and region figures.
CC_CUTOFF = 1.85

# Upper limit +4 eV, following scripts/visualisation/plot_electronic_structure.py:
# Mulliken weights are not positive-definite above E_F and the C s channel turns
# negative around +4.3 eV, at which point a stacked fill dips below its own baseline.
DEFAULT_XLIM = (-22.0, 4.0)
DEFAULT_ZOOM = (-3.0, 3.0)

INK_SECONDARY = "#52514e"
PATCH = "#2ca02c"
PATCH_FILL = "#d6ecd6"
BULK = "#c8cdd2"
BOND = "#9aa0a6"
HYDROGEN = "#3d7ea6"
EDGE_DEFECT = "#d62728"


def bond_segments(positions, symbols):
    """C-C pairs within CC_CUTOFF as (n, 2, 2) line segments for panel (a)."""
    carbon = np.flatnonzero(np.asarray(symbols) == "C")
    xy = positions[carbon, :2]

    delta = xy[:, None, :] - xy[None, :, :]
    distance = np.sqrt((delta ** 2).sum(axis=-1))

    upper = np.triu(np.ones_like(distance, dtype=bool), k=1)
    i, j = np.nonzero(upper & (distance < CC_CUTOFF))

    return np.stack([xy[i], xy[j]], axis=1)


def carbon_curve(projection):
    """C s + C p for the single spin channel, per carbon when per_atom was set."""
    return projection["c_s"][0] + projection["c_p"][0]


def draw_structure(ax, structure, summary):
    """Panel (a): skeleton, selection box, selected carbons."""
    positions = structure["positions"]
    symbols = np.asarray(structure["symbols"])
    mask = structure["mask"]

    ax.add_collection(
        matplotlib.collections.LineCollection(
            bond_segments(positions, symbols),
            colors=BOND, linewidths=0.6, alpha=0.8, zorder=1,
        )
    )

    x_min, x_max, y_min, y_max = structure["box"]
    ax.add_patch(
        plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            facecolor=PATCH_FILL, edgecolor=PATCH, lw=1.3, ls="--",
            alpha=0.55, zorder=0,
        )
    )

    carbon = symbols == "C"
    hydrogen = symbols == "H"

    ax.scatter(positions[carbon & ~mask, 0], positions[carbon & ~mask, 1],
               s=9, c=BULK, edgecolors="none", zorder=2,
               label="C, rest of flake ({})".format(
                   summary["n_C_total"] - summary["n_C_in_patch"]))
    ax.scatter(positions[carbon & mask, 0], positions[carbon & mask, 1],
               s=17, c=PATCH, edgecolors="none", zorder=4,
               label="C, pristine patch ({})".format(summary["n_C_in_patch"]))
    ax.scatter(positions[hydrogen, 0], positions[hydrogen, 1],
               s=7, c=HYDROGEN, edgecolors="none", zorder=3,
               label="H ({})".format(summary["n_H_total"]))

    ax.set_aspect("equal")
    ax.set_xlabel(r"x / $\mathrm{\AA}$")
    ax.set_ylabel(r"y / $\mathrm{\AA}$")
    ax.set_title(
        "(a)  {}  -  box {}, $\\geq${} bonds from any edge".format(
            structure["structure_id"], P.box_label(structure["box"]),
            structure["min_edge_hops"]),
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="lower left", frameon=False,
              labelcolor=INK_SECONDARY, handletextpad=0.4, borderpad=0.2)


def draw_zoom(ax, projection, zoom):
    """Panel (c): C s+p per carbon near E_F, where the Dirac-point dip would be."""
    x = projection["x"]
    y = carbon_curve(projection)

    window = (x >= zoom[0]) & (x <= zoom[1])

    ax.plot(x[window], y[window], lw=1.6, color=PATCH)
    ax.fill_between(x[window], 0.0, y[window], color=PATCH, alpha=0.18, lw=0)
    ax.axvline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.6)

    ax.set_xlim(*zoom)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(projection["graph_xlabel"])
    ax.set_ylabel(ylabel(projection["per_atom"]))
    ax.set_title(
        "(c)  near $E_F$  -  C s+p, {} C, $\\sigma$ = {:.3f} eV".format(
            projection["n_C"], projection["sigma"]),
        fontsize=10,
    )


# Half-width of the window used to measure the suppression at E_F. Wide enough to average
# over several discrete levels of a flake this size, narrow enough to sit inside the dip.
FERMI_HALF_WIDTH = 0.5


def near_fermi_metrics(projection, zoom):
    """
    Numbers describing panel (c), so the shape does not have to be eyeballed.

    Read dos_at_EF_per_C with care, and do not use it on its own. The axis is referenced
    to the HOMO, so x = 0 is not an arbitrary point in a continuum - it is a discrete
    eigenvalue of a finite flake, and the curve there is that level's own broadened peak.
    Narrowing sigma therefore makes this number go *up* (0.0096 at sigma = 0.141 eV,
    0.0145 at 0.05 eV for D10). That is a property of where the level sits, not of the
    pseudogap, and any threshold-crossing measure built on it inherits the same problem.

    suppression_pm0p5 is the quantity that is about the band structure: mean density over
    |E - E_F| <= FERMI_HALF_WIDTH, divided by the mean over the whole zoom window.
    Broadening conserves the integral, so this is stable in sigma, and it does not care
    whether a level happens to land exactly at zero. Well below 1 is the graphene-like
    answer; at or above 1 would mean states are piling up at E_F instead.
    """
    x = projection["x"]
    y = carbon_curve(projection)

    window = (x >= zoom[0]) & (x <= zoom[1])
    fermi_window = np.abs(x) <= FERMI_HALF_WIDTH

    mean_window = float(y[window].mean())
    mean_fermi = float(y[fermi_window].mean())

    return {
        "dos_at_EF_per_C": round(float(y[np.argmin(np.abs(x))]), 5),
        "mean_dos_pm0p5_per_C": round(mean_fermi, 5),
        "mean_dos_zoom_per_C": round(mean_window, 5),
        "suppression_pm0p5": round(mean_fermi / mean_window, 4) if mean_window else "",
    }


def occupied_valence_electrons(projection, nspin):
    """
    Integrated occupied weight per carbon, in electrons, as a sanity check.

    Should come out near 4: the C 2s2p valence of an sp2 carbon, with no charge transfer
    to speak of inside a pristine patch. The C 1s core sits ~280 eV below E_F, far
    outside the cache window, so it does not contribute.

    Two corrections on the raw integral. The curve is a density of *states*, not of
    electrons, so a closed-shell (nspin = 1) run carries two electrons per state. And
    carbon_curve() is s + p only, so the small l >= 2 weight of the tight basis is
    missing and the answer lands just under 4 rather than on it.
    """
    x = projection["x"]
    y = carbon_curve(projection)
    occupied = x <= 0.0

    states = float(np.trapezoid(y[occupied], x[occupied]))
    return states * (2.0 / nspin)


def build_figure(structure, patch, summary, xlim, zoom, output_file):
    fig = plt.figure(figsize=(13.5, 5.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1, 1],
                            wspace=0.24, hspace=0.55)

    ax_structure = fig.add_subplot(grid[:, 0])
    ax_dos = fig.add_subplot(grid[0, 1])
    ax_zoom = fig.add_subplot(grid[1, 1])

    draw_structure(ax_structure, structure, summary)

    draw_dos_axes(
        ax_dos, patch, xlim, None,
        "(b)  pristine-patch LDOS  -  {} C, per carbon".format(patch["n_C"]),
        title_fontsize=10,
    )
    ax_dos.set_ylabel(ylabel(patch["per_atom"]))

    draw_zoom(ax_zoom, patch, zoom)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output_file))


def build_comparison(patch, defect, zoom, output_file):
    """Optional internal contrast: patch vs the edge and defect carbons of the same run."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    for ax, window, label in (
        (axes[0], (patch["x"][0], 4.0), "full valence range"),
        (axes[1], zoom, "near $E_F$"),
    ):
        for data, colour in ((patch, PATCH), (defect, EDGE_DEFECT)):
            x = data["x"]
            y = carbon_curve(data)
            inside = (x >= window[0]) & (x <= window[1])
            ax.plot(x[inside], y[inside], lw=1.5, color=colour,
                    label="{} ({} C)".format(data["region"], data["n_C"]))

        ax.axvline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.6)
        ax.set_xlim(*window)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(patch["graph_xlabel"])
        ax.set_ylabel(ylabel(patch["per_atom"]))
        ax.set_title("C s+p per carbon, {}".format(label), fontsize=10)
        ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output_file))


def repo_relative(path):
    """Path relative to ROOT when it is inside the repo, else the absolute path."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_summary(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if r.get("structure_id") != row["structure_id"]
                    or r.get("sigma_eV") != str(row["sigma_eV"])]

    rows.append({k: str(v) for k, v in row.items()})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote {}".format(path))


def main():
    parser = argparse.ArgumentParser(
        description="Project a cluster Mulliken DOS onto its pristine graphene patch."
    )
    parser.add_argument("--structure-id", default="D10")
    parser.add_argument("--calc-root", default=str(P.CLUSTER_CALC_ROOT),
                        help="Directory holding <ID>/geometry.in for the cluster runs.")
    parser.add_argument("--cache-root", default=str(CLUSTER_CACHE_ROOT),
                        help="Per-atom DOS cache directory (cluster, NOT the periodic one).")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="Gaussian broadening in eV (default {:.4f}).".format(DEFAULT_SIGMA))
    parser.add_argument("--min-edge-hops", type=int, default=P.DEFAULT_MIN_EDGE_HOPS,
                        help="Minimum C-C bonds from any edge carbon (default {}).".format(
                            P.DEFAULT_MIN_EDGE_HOPS))
    parser.add_argument("--include-h", action="store_true",
                        help="Include hydrogens in the patch projection. Off by default, "
                             "and for a well-chosen patch there are none to include.")
    parser.add_argument("--xlim", nargs=2, type=float, default=list(DEFAULT_XLIM))
    parser.add_argument("--zoom", nargs=2, type=float, default=list(DEFAULT_ZOOM))
    parser.add_argument("--compare", action="store_true",
                        help="Also plot the patch against the edge and defect carbons "
                             "of the same run.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default=str(SUMMARY_FILE))
    parser.add_argument("--carmm-path", default=DEFAULT_CARMM)
    args = parser.parse_args()

    add_carmm_to_path(args.carmm_path)

    structure = P.load_structure(args.structure_id, args.calc_root, args.min_edge_hops)
    audit = P.audit_patch(structure)

    cache = read_cache(args.cache_root, args.structure_id)
    check_cache_alignment(cache, structure["symbols"], args.structure_id)

    if cache["nkpts"] != 1:
        raise SystemExit(
            "{} cache has {} k-points, so it is a periodic run, not a cluster. Point "
            "--cache-root at {}.".format(
                args.structure_id, cache["nkpts"], CLUSTER_CACHE_ROOT)
        )

    patch = project(cache, P.patch_rows(structure, args.include_h), P.PATCH_REGION,
                    sigma=args.sigma, per_atom=True)

    print("Patch: {} C, edge_hops {}-{} (mean {}), rings {}".format(
        audit["n_C_in_patch"], audit["min_edge_hops"], audit["max_edge_hops"],
        audit["mean_edge_hops"], audit["ring_census"]))

    # Resolved, so a relative --output still records cleanly against ROOT below.
    output = Path(args.output).resolve() if args.output else (
        FIGURES_ROOT / "{}_cluster_pristine_patch_dos.png".format(args.structure_id))
    build_figure(structure, patch, audit, tuple(args.xlim), tuple(args.zoom), output)

    comparison_file = ""
    if args.compare:
        defect = project(cache, P.defect_rows(structure), "edge_defect",
                         sigma=args.sigma, per_atom=True)
        comparison_file = FIGURES_ROOT / "{}_cluster_pristine_patch_vs_edge.png".format(
            args.structure_id)
        build_comparison(patch, defect, tuple(args.zoom), comparison_file)

    row = dict(audit)
    row.update(near_fermi_metrics(patch, tuple(args.zoom)))
    row.update({
        "sigma_eV": round(args.sigma, 6),
        "homo_eV": round(cache["homo"], 6),
        "nkpts": cache["nkpts"],
        "occupied_valence_electrons_per_C": round(
            occupied_valence_electrons(patch, cache["nspin"]), 3),
        "figure": repo_relative(output),
        "comparison_figure": repo_relative(comparison_file) if comparison_file else "",
    })
    write_summary(args.summary, row)


if __name__ == "__main__":
    main()
