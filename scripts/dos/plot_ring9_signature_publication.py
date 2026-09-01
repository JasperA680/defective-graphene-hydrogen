#!/usr/bin/env python3
"""
Publication version of results/figures/dos/H02_ring9_local_signature.png.

Same three panels, same regions, same projections, same numbers as
plot_ring9_local_signature.py - this changes presentation only, and imports the region
construction from that module rather than restating it, so the two figures cannot come to
disagree about which atoms are in which region.

What is different, and why:

    * The panels carry no titles at all, only the letters (a), (b), (c). What each one
      shows, the broadening, the ring coordinates, the ring area and the near-E_F
      statistics all belong in the caption and the Results text.
    * Every text size is 1.5x the exploratory figure's, so the canvas survives being
      reduced to a journal column. See FIGURE_WIDTH_IN.
    * Panel (b) drops the block of mean-DOS numbers; panel (c) drops the magnified C s
      trace and its annotation. The C s curve is plotted at its true scale, so the panel
      shows what the data is rather than arguing about it.
    * The energy axis is labelled against the HOMO, not the Fermi level. See below.
    * Every panel is fully boxed, and both PDF (vector, Type 42 fonts, so journal
      submission systems accept it) and PNG at 600 dpi are written.

The panel geometry is the exploratory figure's, copied verbatim - same canvas, same width
ratios, same spacing - so the two are directly comparable. The text is not: see
FIGURE_WIDTH_IN and RC. The canvas is 14 in, so at two-column width it reproduces at about
half size, and the type is set 1.5x larger to survive that.

The energy zero
---------------
The cache is shifted by cache["homo"], which cache_atom_resolved_dos.py takes from
carmm's MullikenData.get_homo(). That routine returns the highest eigenvalue whose
occupancy is at least 0.5, maximised over atoms, spins and k-points. For H02 that is a
molecular orbital of a finite cluster, not a Fermi level: the run is non-periodic (no
lattice_vector lines, one k-point) and closed shell (`spin none`), so there is no Fermi
surface and nothing is thermally populated. The axis is therefore labelled

    epsilon - epsilon_HOMO (eV)

and NOT "epsilon - epsilon_f", which is what cache["graph_xlabel"] carries and what the
exploratory figures in results/figures/dos/ display. That label is a legacy of the
periodic slabs the pipeline was written for, where get_homo() does return the Fermi level;
it is wrong for the cluster runs and is deliberately overridden here rather than inherited.

Two caveats that belong in the caption, not on the axis:

    * FHI-aims reports this cluster's HOMO at -4.3307 eV and its LUMO at -4.3114 eV, an
      overall gap of 0.019 eV that it prints as 0.00000000 eV. With `occupation_type
      gaussian 0.1` the frontier occupations are fractional, so "the HOMO" is a smeared
      quantity here, not a sharp one.
    * carmm's occupancy >= 0.5 test lands on -4.29095 eV, 0.04 eV above the FHI-aims VBM,
      for the same reason. That offset is common to every curve in the figure, so it moves
      the axis zero, not the comparison between regions.

Inputs:
    calculations/dos_cluster/<ID>/geometry.in
    results/tables/dos_atom_resolved_cluster/<ID>.npz

Outputs:
    results/figures/dos/<ID>_ring<N>_signature_publication.pdf
    results/figures/dos/<ID>_ring<N>_signature_publication.png

The exploratory figure is left alone; nothing here writes to
<ID>_ring<N>_local_signature.png.

Usage:
    .venv-dos/bin/python3 scripts/dos/plot_ring9_signature_publication.py
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pristine_patch_regions as P  # noqa: E402
from cache_atom_resolved_dos import read_cache  # noqa: E402
from crop_regions import check_cache_alignment  # noqa: E402
from plot_pristine_patch_dos import (  # noqa: E402
    BOND,
    BULK,
    CLUSTER_CACHE_ROOT,
    DEFAULT_ZOOM,
    FIGURES_ROOT,
    HYDROGEN,
    bond_segments,
    carbon_curve,
)
from plot_ring_dos import find_rings, select_ring, rows_for_mask  # noqa: E402
from plot_ring9_local_signature import (  # noqa: E402
    PATCH_REGION,
    REGION_COLOURS,
    RIBBON_BOX,
    RIBBON_REGION,
    RING_REGION,
    SHELL_REGION,
    build_regions,
)
from plot_n_membered_rings_all import HEXAGON_ALPHA, RING_ALPHA, ring_colour  # noqa: E402
from region_dos import DEFAULT_SIGMA, project, ylabel  # noqa: E402

# The one thing this module refuses to inherit. See the module docstring: for a
# non-periodic, closed-shell cluster the shift is a molecular orbital energy, so calling
# the zero a Fermi level would be wrong on the face of the figure.
ENERGY_LABEL = r"$\epsilon - \epsilon_{\mathrm{HOMO}}$ (eV)"

# Publication labels. "ring complex" reads better than the construction it names, and
# "pristine reference" says what the patch is for rather than how it was selected.
PANEL_LABELS = {
    RING_REGION: "9-ring",
    SHELL_REGION: "ring complex",
    RIBBON_REGION: "nanoribbon",
    PATCH_REGION: "pristine reference",
}

# Innermost first, which is the order the argument runs in and the order the legends read.
PANEL_REGIONS = (RING_REGION, SHELL_REGION, RIBBON_REGION, PATCH_REGION)

# Which side of its box each label sits on. Above is the default and is where the pristine
# box has clear space; the nanoribbon box does not - its top edge runs into the 5|7|5 chain
# at x 22-26, y 28-33 - so that one is labelled underneath, where the frame is empty.
BOX_LABEL_BELOW = {RIBBON_REGION}

# Orbital channels of panel (c). C s+p first so the total reads as the envelope it is.
TOTAL_COLOUR = "#1a1a1a"
P_COLOUR = REGION_COLOURS[RING_REGION]
S_COLOUR = "#2b6ca3"

# Geometry copied from plot_ring9_local_signature.build_figure(), deliberately and
# verbatim, so this figure has the same panel proportions as the exploratory one it
# replaces. Trying to re-derive a layout at 180 mm fought itself: the fixed overhead of
# titles and axis labels is a far larger share of a 7 in canvas than of a 14 in one, so
# panel (a) filling its frame and panels (b)/(c) keeping their shape became mutually
# exclusive. On this canvas both hold at once, because they are the proportions the
# exploratory figure already had.
#
# Consequence to be aware of when placing it: this is a 14 in canvas, so at a 180 mm
# two-column width it is reproduced at about 51 per cent. That is what the 1.5x type in RC
# is for - 15 pt set here prints at roughly 7.5 pt - so place it full width or near it.
FIGURE_WIDTH_IN = 14.0
FIGURE_HEIGHT_IN = 6.4

PANEL_WIDTH_RATIOS = [1.25, 1.0]
PANEL_WSPACE = 0.20
# Cut from the exploratory figure's 0.55 once (b) and (c) went onto a shared energy axis:
# the gap no longer has to hold (b)'s tick labels and axis label, only (c)'s panel letter,
# and leaving it at 0.55 would have left the two panels floating apart on an axis they now
# share. The panels grow into the reclaimed space rather than the canvas shrinking.
PANEL_HSPACE = 0.16

XTICKS = np.arange(-3.0, 3.1, 1.0)

# Horizontal seat of the shared DOS label, in the upper right panel's axes fraction, and
# the one number here that had to be measured rather than derived: at 15 pt the tick
# numbers reach out to -0.128 and the label is 0.06 wide, so its centre clears them from
# about -0.16. See centre_shared_ylabel() for the vertical placement, which is derived.
SHARED_YLABEL_X = -0.17

# Text sized for the canvas above. The exploratory figure's 10 pt is legible on screen at
# 14 in but prints at roughly 5 pt once the canvas is reduced to a 180 mm column, so every
# size here is scaled by 1.5: 15 pt labels and ticks, 13 pt legends, 16 pt panel letters.
# That puts the reproduced text back at 7-8 pt, which is what journals ask for. Applied
# through rc_context so importing this module cannot change how any other figure in the
# repo is drawn.
TITLE_SIZE = 16.0

RC = {
    "font.size": 15.0,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": 15.0,
    "xtick.labelsize": 15.0,
    "ytick.labelsize": 15.0,
    "legend.fontsize": 13.0,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.6,
    "legend.labelspacing": 0.35,
    "lines.solid_capstyle": "round",
    # Embed real fonts in the PDF rather than subsetting to Type 3, which many journal
    # submission systems reject.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Tight, as the exploratory figure saves: on a fixed-geometry layout it only trims the
    # surrounding white margin, it does not move the panels relative to one another, so the
    # proportions copied above survive it.
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

CURVE_LW = 1.5
REFERENCE_LW = 1.0

# Marker areas for panel (a), on the same scale as the exploratory figure so the structure
# reads the same way. Hydrogen is the one departure: kept well below carbon, because the
# brief is that it be visible without competing with the carbon topology.
CARBON_MARKER = 7.0
HYDROGEN_MARKER = 2.5
# Scaled with the rest of the text. The ring digits stay the smallest thing on the canvas
# because they have to sit inside a 5-ring face without covering its edges.
RING_LABEL_SIZE = 11.0
BOX_LABEL_SIZE = 13.0


def panel_label(ax, letter, pad=5):
    """
    The panel's only heading: the bare letter, flush with the left spine.

    What each panel shows belongs in the caption, so the descriptive titles these replace
    are gone rather than shortened. Left-aligned as a title rather than placed inside the
    axes, so it cannot collide with a curve or with panel (a)'s legend.
    """
    ax.set_title(letter, fontsize=TITLE_SIZE, loc="left", pad=pad)


def draw_skeleton(ax, structure):
    """
    Bonds, ring faces and atoms, at publication marker sizes.

    plot_ring9_local_signature.draw_skeleton() scales carbon and hydrogen together, which
    is right for a screen figure and wrong here: the brief is that the hydrogens stay
    visible but subordinate, so the carbon topology reads first. The parts that carry
    meaning - the bond cutoff and the ring-size palette - are the shared ones.
    """
    positions = structure["positions"]
    symbols = np.asarray(structure["symbols"])

    ax.add_collection(
        matplotlib.collections.LineCollection(
            bond_segments(positions, symbols),
            colors=BOND, linewidths=0.6, alpha=0.85, zorder=1,
        )
    )

    for entry in structure["rings"]:
        facecolor, edgecolor = ring_colour(entry["size"])
        hexagon = entry["size"] == 6
        polygon = np.asarray(entry["polygon"], dtype=float)

        ax.fill(polygon[:, 0], polygon[:, 1],
                facecolor=facecolor, edgecolor=edgecolor,
                linewidth=0.6 if hexagon else 1.1,
                alpha=HEXAGON_ALPHA if hexagon else RING_ALPHA,
                zorder=2)

        if not hexagon:
            ax.text(entry["centroid"][0], entry["centroid"][1], str(entry["size"]),
                    fontsize=RING_LABEL_SIZE, ha="center", va="center", color="#111111",
                    zorder=8, clip_on=True)

    carbon = symbols == "C"
    hydrogen = symbols == "H"

    ax.scatter(positions[carbon, 0], positions[carbon, 1],
               s=CARBON_MARKER, c=BULK, edgecolors="none", zorder=3)
    # Small, pale and low: present for completeness, never competing with the carbons.
    ax.scatter(positions[hydrogen, 0], positions[hydrogen, 1],
               s=HYDROGEN_MARKER, c=HYDROGEN, alpha=0.55, edgecolors="none", zorder=3)


def draw_structure_panel(ax, structure, regions):
    """Panel (a): the ring map with the four projection regions on it."""
    positions = structure["positions"]
    by_region = {region["region"]: region for region in regions}

    draw_skeleton(ax, structure)

    marker_size = {RING_REGION: 32.0, SHELL_REGION: 18.0,
                   RIBBON_REGION: 7.0, PATCH_REGION: 7.0}

    # Outermost first, so a carbon shows the innermost region it belongs to.
    for name in reversed(PANEL_REGIONS):
        region = by_region[name]
        colour = REGION_COLOURS[name]
        mask = region["mask"]

        if region.get("box") is not None:
            x_min, x_max, y_min, y_max = region["box"]
            ax.add_patch(
                plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              facecolor="none", edgecolor=colour, lw=1.2, ls=(0, (4, 2)),
                              zorder=5)
            )
            # Labelled where it is drawn rather than in the legend: a dashed rectangle in
            # a legend says nothing about which rectangle it is.
            below = name in BOX_LABEL_BELOW
            ax.text(x_min + 0.4, (y_min - 1.2) if below else (y_max + 1.2),
                    "{} ({} C)".format(PANEL_LABELS[name], int(mask.sum())),
                    fontsize=BOX_LABEL_SIZE, color=colour, ha="left",
                    va="top" if below else "bottom", zorder=9)

        if region["outline"] is not None:
            ax.add_patch(
                plt.Polygon(np.asarray(region["outline"], dtype=float), closed=True,
                            facecolor="none", edgecolor=colour,
                            lw=1.9 if name == RING_REGION else 1.5, zorder=6)
            )

        ax.scatter(positions[mask, 0], positions[mask, 1], s=marker_size[name],
                   color=colour, edgecolors="none", zorder=7,
                   label="{} ({} C)".format(PANEL_LABELS[name], int(mask.sum()))
                   if region["outline"] is not None else None)

    # Plain "equal", as the exploratory figure does it: on this canvas the cell is close
    # enough to the data's own aspect that the frame comes out nearly full, so there is
    # nothing for adjustable="datalim" to fix and it would only pad the limits.
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ / $\mathrm{\AA}$")
    ax.set_ylabel(r"$y$ / $\mathrm{\AA}$")
    panel_label(ax, "(a)", pad=6)

    pad = 1.5
    ax.set_xlim(positions[:, 0].min() - pad, positions[:, 0].max() + pad)
    ax.set_ylim(positions[:, 1].min() - pad, positions[:, 1].max() + pad)

    # Only the two outlined regions; the boxed ones are named where they are drawn.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower left", scatterpoints=1,
              borderpad=0.0, borderaxespad=0.4)


def draw_dos_panel(ax, projections):
    """Panel (b): the four regional LDOS curves, per carbon."""
    for name in reversed(PANEL_REGIONS):
        data = projections[name]
        x = data["x"]
        y = carbon_curve(data)
        window = (x >= DEFAULT_ZOOM[0]) & (x <= DEFAULT_ZOOM[1])

        # No carbon counts here. Panel (a) already gives them next to the regions they
        # belong to, and repeating them only widens this legend into the curves.
        ax.plot(x[window], y[window],
                lw=1.9 if name == RING_REGION else CURVE_LW,
                color=REGION_COLOURS[name], zorder=5 if name == RING_REGION else 3,
                label=PANEL_LABELS[name])

    panel_label(ax, "(b)")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", ncol=2,
              columnspacing=1.0, borderpad=0.0, borderaxespad=0.35)


def draw_orbital_panel(ax, projection):
    """
    Panel (c): the 9-ring decomposed into C s and C p, both at true scale.

    C p is drawn dashed and on top of the solid total. It has to be: C s is so small here
    that C p and C s+p are the same curve to within a line width, so a solid C p simply
    disappears under the total and the panel looks like it is missing a channel. Dashed
    over solid shows the two coinciding, which is the result rather than an artefact of
    the drawing.
    """
    x = projection["x"]
    c_s = projection["c_s"][0]
    c_p = projection["c_p"][0]
    window = (x >= DEFAULT_ZOOM[0]) & (x <= DEFAULT_ZOOM[1])

    ax.plot(x[window], (c_s + c_p)[window], lw=1.8, color=TOTAL_COLOUR, zorder=3,
            label="C s+p")
    ax.plot(x[window], c_p[window], lw=1.3, ls=(0, (3.5, 2.5)), color=P_COLOUR, zorder=5,
            label="C p")
    ax.plot(x[window], c_s[window], lw=CURVE_LW, color=S_COLOUR, zorder=4, label="C s")

    panel_label(ax, "(c)")
    # Tighter than panel (b)'s legend so all three entries finish left of the E = 0 rule
    # rather than straddling it.
    ax.legend(loc="upper left", ncol=3, columnspacing=0.7, handlelength=1.2,
              handletextpad=0.4, borderpad=0.0, borderaxespad=0.35)


def format_energy_axis(ax, ylim, show_xlabel=True, show_ylabel=True):
    """
    The formatting panels (b) and (c) must share, applied to both from one place.

    The two axes are joined with sharex/sharey in build_figure(), so the limits and ticks
    set here propagate; they are set on both anyway, because a shared axis that is only
    ever configured through one of the pair is a trap for whoever edits this next.

    show_xlabel is False for the upper panel: the panels are stacked on one energy axis, so
    the tick labels and the axis label are drawn once, under panel (c). show_ylabel is the
    mirror of that for the DOS axis, and is False for the lower panel.
    """
    ax.axvline(0.0, color="#555555", linestyle=(0, (3, 2)), lw=REFERENCE_LW, zorder=1)
    ax.set_xlim(*DEFAULT_ZOOM)
    ax.set_xticks(XTICKS)
    ax.set_ylim(0.0, ylim)

    if show_xlabel:
        ax.set_xlabel(ENERGY_LABEL)
    else:
        ax.tick_params(labelbottom=False)

    # Both panels keep their tick numbers - they are stacked, so there is no shared column
    # of numbers the way the x axis has a shared row - but only one carries the text; see
    # centre_shared_ylabel().
    if show_ylabel:
        ax.set_ylabel(ylabel(True))
    # Full frame, matching panel (a). All three panels are then boxed the same way.


def centre_shared_ylabel(ax_upper):
    """
    Hang the single y label off the upper panel, centred on the pair.

    Both panels plot the same quantity on the same axis, so labelling each of them is one
    sentence written twice; at PANEL_HSPACE = 0.16 the two copies also very nearly touch.
    Placed by hand rather than with fig.supylabel(), which would centre on the whole figure
    and so on panel (a) as well.

    The y coordinate is derived from PANEL_HSPACE rather than tuned: in the upper panel's
    axes fraction the lower panel runs from -PANEL_HSPACE to -(1 + PANEL_HSPACE), so the
    pair spans 1 down to -(1 + PANEL_HSPACE) and its midpoint is -PANEL_HSPACE / 2. Both
    right-hand panels are the same height, which is what makes that exact.
    """
    # A y label is rotated text anchored at its baseline, so by default the x coordinate
    # given below is the label's right-hand edge and the text sits to the right of it -
    # straight over the tick numbers. Anchoring on the middle of its thickness instead
    # makes SHARED_YLABEL_X mean where the label actually is.
    ax_upper.yaxis.label.set_verticalalignment("center")
    ax_upper.yaxis.set_label_coords(SHARED_YLABEL_X, -PANEL_HSPACE / 2.0)


def shared_ylim(projections):
    """
    One y-limit for panels (b) and (c).

    The C s+p total in panel (c) is the same curve as the 9-ring in panel (b). On different
    scales those two would be the same data drawn at two heights in adjacent panels, which
    is the one misreading this layout invites. build_figure() also joins the two axes with
    sharey; this decides what that shared limit is.
    """
    x = projections[RING_REGION]["x"]
    window = (x >= DEFAULT_ZOOM[0]) & (x <= DEFAULT_ZOOM[1])

    peak = max(float(carbon_curve(projections[name])[window].max())
               for name in PANEL_REGIONS)

    # Headroom for panel (b)'s two-column legend, which sits over the -1.5 eV peak. Raised
    # from 1.42 when the legend text went to 13 pt: at 1.42 the second legend row cleared
    # that peak by well under a line height.
    return 1.55 * peak


def build_figure(structure, regions, projections, output_stem):
    written = []

    with matplotlib.rc_context(RC):
        fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
        grid = fig.add_gridspec(2, 2, width_ratios=PANEL_WIDTH_RATIOS,
                                height_ratios=[1, 1], wspace=PANEL_WSPACE,
                                hspace=PANEL_HSPACE)

        ax_structure = fig.add_subplot(grid[:, 0])
        ax_dos = fig.add_subplot(grid[0, 1])
        # One energy axis and one DOS scale for the pair, enforced by matplotlib rather
        # than by both panels happening to be handed the same numbers. Panel (c)'s C s+p
        # total is panel (b)'s 9-ring curve, so the two must not be able to drift apart.
        ax_orbital = fig.add_subplot(grid[1, 1], sharex=ax_dos, sharey=ax_dos)

        draw_structure_panel(ax_structure, structure, regions)
        draw_dos_panel(ax_dos, projections)
        draw_orbital_panel(ax_orbital, projections[RING_REGION])

        ylim = shared_ylim(projections)
        format_energy_axis(ax_dos, ylim, show_xlabel=False)
        format_energy_axis(ax_orbital, ylim, show_ylabel=False)
        centre_shared_ylabel(ax_dos)

        output_stem = Path(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)

        for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 600})):
            path = output_stem.with_suffix(suffix)
            fig.savefig(path, **kwargs)
            written.append(path)

        plt.close(fig)

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Publication version of the H02 nine-ring local signature figure."
    )
    parser.add_argument("--structure-id", default="H02")
    parser.add_argument("--ring-size", type=int, default=9)
    parser.add_argument("--ring-index", type=int, default=None)
    parser.add_argument("--shell-ring-size", type=int, default=5)
    parser.add_argument("--calc-root", default=str(P.CLUSTER_CALC_ROOT))
    parser.add_argument("--cache-root", default=str(CLUSTER_CACHE_ROOT))
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="Gaussian broadening in eV (default {:.4f}). Belongs in the "
                             "caption, so it is not written on the figure.".format(
                                 DEFAULT_SIGMA))
    parser.add_argument("--min-edge-hops", type=int, default=P.DEFAULT_MIN_EDGE_HOPS)
    parser.add_argument("--ribbon-box", nargs=4, type=float, default=list(RIBBON_BOX))
    parser.add_argument("--output-stem", default=None,
                        help="Path without extension; .pdf and .png are both written.")
    args = parser.parse_args()

    structure = find_rings(args.structure_id, args.calc_root)
    ring = select_ring(structure, args.ring_size, args.ring_index)

    patch = P.load_structure(args.structure_id, args.calc_root, args.min_edge_hops)
    P.audit_patch(patch)

    regions, _shell_faces = build_regions(structure, ring, args.shell_ring_size, patch,
                                          tuple(args.ribbon_box))

    cache = read_cache(args.cache_root, args.structure_id)
    check_cache_alignment(cache, structure["symbols"], args.structure_id)

    if cache["nkpts"] != 1:
        raise SystemExit(
            "{} cache has {} k-points, so it is a periodic run. This figure labels its "
            "energy axis against the cluster HOMO, which would be wrong for a slab; point "
            "--cache-root at {}.".format(
                args.structure_id, cache["nkpts"], CLUSTER_CACHE_ROOT)
        )

    projections = {
        region["region"]: project(
            cache, rows_for_mask(structure, region["mask"], region["region"]),
            region["region"], sigma=args.sigma, per_atom=True)
        for region in regions
    }

    print("{}: {} regions, sigma {:.4f} eV, energy zero at the cluster HOMO "
          "({:.5f} eV)".format(args.structure_id, len(regions), args.sigma,
                               cache["homo"]))
    for name in PANEL_REGIONS:
        print("  {:<20} {:>3} C".format(PANEL_LABELS[name], projections[name]["n_C"]))

    stem = Path(args.output_stem) if args.output_stem else (
        FIGURES_ROOT / "{}_ring{}_signature_publication".format(
            args.structure_id, ring["size"]))

    for path in build_figure(structure, regions, projections, stem):
        print("Saved {}".format(path))


if __name__ == "__main__":
    main()
