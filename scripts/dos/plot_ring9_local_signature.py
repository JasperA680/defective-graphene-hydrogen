#!/usr/bin/env python3
"""
Local electronic signature of the nine-membered ring in H02, and how far it reaches.

plot_ring_dos.py projects a cluster DOS onto one ring and stops there, which shows that
the H02 nine-membered ring has states near E_F but not whether that is unusual or how far
it extends. This figure is built around the question instead:

    Does the experimentally observed 9-membered ring in H02 have a distinct local
    electronic signature compared with pristine-like graphene, and how spatially localised
    is that signature?

Answering it needs a pristine reference measured in *the same calculation* - same
functional, basis, geometry and SCF, so the only difference is which atoms are summed -
and intermediate regions between the ring and the bulk. So the figure projects the one H02
cluster run onto four carbon sets:

    ring9                 9 C   the nine-membered face itself
    ring9_plus_pentagons 23 C   that face plus the five pentagons sharing an edge with it
    nanoribbon           66 C   every carbon in RIBBON_BOX, the whole neck the defect
                                cluster sits in, edges included
    pristine_patch       72 C   the top-left box of pristine_patch_regions.PATCH_BOXES,
                                hexagon-only carbons >= 3 bonds from any termination

The first three are strictly nested, 9 inside 23 inside 66, so their curves read as a decay
series outwards from the ring; the patch is the far end of that series and shares no atom
with any of them. Every projection is carbon-only - the ring and shell contain no hydrogen
at all, and the terminating hydrogens inside the ribbon box are excluded even though the
carbons carrying them are not.

Three panels:

    (a) the structure, drawn in the style of results/figures/n_membered_rings/<ID>_rings.png
        - every ring face filled by size, hexagons washed pale grey - but on the
        H-saturated, bond-healed cluster geometry that the DOS was actually run on, so the
        stretched bonds of the raw A-J frame are gone and the terminating hydrogens are
        present. The four projection regions are overlaid on top of that map, each in a
        saturated version of the ring colour it is made of, drawn largest first so an atom
        shows the innermost region it belongs to. --inset adds a corner zoom on the defect.
    (b) the four LDOS curves over +-3 eV, per carbon. This is the panel that answers both
        halves of the question: how far the ring sits above the pristine reference at E_F,
        and how much of that survives outwards through the pentagon shell and the ribbon.
    (c) the orbital character of the ring LDOS over the same window, so "the ring has
        states at E_F" can be qualified as p-character rather than left unresolved. The C s
        weight is real but tiny - a few per cent of the total at most - so it is also drawn
        magnified, which is honest about the shape without inventing a second axis.

Reads the per-atom cache from scripts/dos/cache_atom_resolved_dos.py, not Mulliken.out:
the cluster H02 Mulliken output is 122 MB and needs carmm to parse, while the cache is
4.6 MB and needs only numpy and scipy. Broadening is linear, so projecting the cache onto
a subset of atoms is exact rather than an approximation of the original pipeline. Unlike
its two companions this script needs no carmm at all - the near-E_F panels are drawn
directly rather than through plot_dos_mulliken.draw_dos_axes - so there is no
--carmm-path.

Note the cache root. results/tables/dos_atom_resolved/ holds the *periodic* runs; the
cluster caches live in results/tables/dos_atom_resolved_cluster/, which is what
--cache-root defaults to here.

--compare adds a second figure, laid out like
results/figures/dos/D10_cluster_pristine_patch_vs_edge.png: the same per-carbon curves over
the full valence range and again near E_F, for five regions rather than four. The extra one
is

    pentagons_excl_ring  14 C   the shell carbons that are *not* in the ring - see
                                shell_only_region() for why the obvious reading of "the
                                pentagons" is not a separate selection at all. With the
                                ring it partitions the 23, so 9 + 14 = 23 exactly.

which is left out of the main figure because it is the one region that neither contains nor
is contained by the others, so it does not belong in a nested series.

Read the nanoribbon curve with its composition in mind. A box across the neck cuts the
ribbon rather than following it, so a third of those 66 carbons are edge - 10 H-terminated
and 12 undercoordinated. That is what "the whole nanoribbon" means here, and it is also why
that curve is not a clean defect-versus-bulk contrast the way the ring and the patch are.
describe_ribbon() counts what the box caught, into the summary CSV and the run log.

Inputs:
    calculations/dos_cluster/<ID>/geometry.in
    results/tables/dos_atom_resolved_cluster/<ID>.npz

Outputs:
    results/figures/dos/<ID>_ring<N>_local_signature.png
    results/figures/dos/<ID>_ring<N>_regions_vs_patch.png   (--compare only)
    results/tables/dos_ring9_signature_summary.csv

Usage:
    .venv-dos/bin/python3 scripts/dos/plot_ring9_local_signature.py
    .venv-dos/bin/python3 scripts/dos/plot_ring9_local_signature.py --compare
    .venv-dos/bin/python3 scripts/dos/plot_ring9_local_signature.py --sigma 0.05
    .venv-dos/bin/python3 scripts/dos/plot_ring9_local_signature.py --no-inset
"""

from pathlib import Path
import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pristine_patch_regions as P  # noqa: E402
from cache_atom_resolved_dos import read_cache  # noqa: E402
from crop_regions import check_cache_alignment  # noqa: E402
from dos_common import ROOT, read_fhi_aims_geometry  # noqa: E402
from plot_pristine_patch_dos import (  # noqa: E402
    BOND,
    BULK,
    CLUSTER_CACHE_ROOT,
    DEFAULT_XLIM,
    DEFAULT_ZOOM,
    FIGURES_ROOT,
    HYDROGEN,
    INK_SECONDARY,
    bond_segments,
    carbon_curve,
    near_fermi_metrics,
    occupied_valence_electrons,
    repo_relative,
)
from plot_ring_dos import find_rings, select_ring, print_rings, rows_for_mask  # noqa: E402
from region_dos import DEFAULT_SIGMA, project, ylabel  # noqa: E402

# The ring palette of panel (a) is imported rather than copied, so this figure and
# results/figures/n_membered_rings/<ID>_rings.png cannot drift apart in colour.
sys.path.insert(0, str(ROOT / "scripts" / "visualisation"))

import plot_n_membered_rings_all as R  # noqa: E402

SUMMARY_FILE = ROOT / "results/tables/dos_ring9_signature_summary.csv"

RING_REGION = "ring9"
SHELL_REGION = "ring9_plus_pentagons"
PENTAGON_REGION = "pentagons_excl_ring"
RIBBON_REGION = "nanoribbon"
PATCH_REGION = "pristine_patch"

# The neck of the flake, as a box in the plane rather than as a set of ring faces: the
# whole nanoribbon the defect cluster sits in, which is the outer end of the decay series
# the comparison figure traces from the ring to the bulk. Contains all 9 ring carbons and
# all 23 shell carbons, the 7-, 8- and 9-rings, the five pentagons and 14 hexagons.
#
# A third of it is edge - 10 of its 66 carbons are H-terminated and 12 undercoordinated -
# because a box across the neck cuts the ribbon rather than following it. That is
# deliberate and is what "the whole nanoribbon" means here, but it is why this curve is
# not a defect-versus-bulk contrast the way the ring and the patch are: part of its
# near-E_F weight is ribbon-edge state, not nine-ring state.
RIBBON_BOX = (10.0, 34.0, 7.0, 26.0)

# The regions of the main figure, innermost first. Panel (b) draws them in reverse so the
# inner curves land on top, then reads the legend and the annotation back in this order,
# which is the order the decay series is argued in.
PANEL_B_REGIONS = (RING_REGION, SHELL_REGION, RIBBON_REGION, PATCH_REGION)

# One colour per region, carried across all three panels. Each is a saturated version of
# the ring colour that region is built out of - purple for the nine-membered ring, orange
# for the pentagons around it - so the overlays in panel (a) read as belonging to the faces
# underneath them rather than as a second, unrelated colour scheme. Green for the pristine
# patch follows plot_pristine_patch_dos.PATCH, which is what the D10 patch figure uses.
REGION_COLOURS = {
    RING_REGION: "#6a3d9a",
    SHELL_REGION: "#d95f02",
    PENTAGON_REGION: "#f1a340",
    # Neutral dark grey, because the nanoribbon is the region that contains all the others
    # rather than a fifth thing alongside them - it should read as context, not as a rival
    # to the ring.
    RIBBON_REGION: "#4d4d4d",
    PATCH_REGION: "#1b7837",
}

REGION_LABELS = {
    RING_REGION: "9-ring",
    SHELL_REGION: "9-ring + 5 pentagons",
    PENTAGON_REGION: "5 pentagons, excl. 9-ring",
    RIBBON_REGION: "nanoribbon",
    PATCH_REGION: "pristine patch",
}

# Panel (c). C s is a few per cent of C p near E_F, so on a shared linear axis it is a flat
# line on zero. Drawn twice - once true, once magnified - rather than on a second axis,
# which would invite the two channels to be read as comparable.
S_COLOUR = "#4575b4"
DEFAULT_S_SCALE = 20.0

# Panel (a) inset. The data window is the defect neighbourhood; the axes fraction is the
# empty wedge to the right of the neck, between the top-left block and the bottom-right
# arm. Both are H02-specific, hence the overrides.
# x_max stops short of 29.8, where the eight-membered ring starts: a ring face sliced by
# the inset frame reads as a drawing error rather than as a ring that is simply elsewhere.
DEFAULT_INSET_WINDOW = (18.0, 29.5, 6.0, 19.6)
DEFAULT_INSET_BOUNDS = (0.60, 0.34, 0.38, 0.38)


def edge_sharing_rings(structure, ring, size):
    """
    The rings of `size` that share a bond - not merely a corner - with `ring`.

    Two atoms in common is the test, which for faces of a planar bond graph means a shared
    edge: two ring faces cannot meet at two separate corners without enclosing something
    between them, and the face walk would have found that instead.
    """
    target = set(ring["atoms"])

    return [
        other for other in structure["rings"]
        if other["size"] == size
        and other is not ring
        and len(target & set(other["atoms"])) >= 2
    ]


def union_boundary(faces):
    """
    The outer boundary of a union of ring faces, as one closed list of atom indices.

    Every interior edge is shared by exactly two of the faces, so counting edges and
    keeping the ones seen once leaves the boundary. Walking those edges then gives the
    outline as a single polygon, which is what panel (a) draws: outlining each face
    separately would put internal lines through a region that is projected as one set.

    Raises if the boundary is not a single simple cycle, which is what a union with a hole,
    or one whose faces only touch at a corner, would produce. Neither happens for the H02
    nine-membered ring and its pentagons, and neither should be drawn as one polygon.
    """
    counts = {}
    for face in faces:
        for a, b in zip(face, list(face[1:]) + [face[0]]):
            counts[(min(a, b), max(a, b))] = counts.get((min(a, b), max(a, b)), 0) + 1

    boundary = [edge for edge, count in counts.items() if count == 1]

    neighbours = {}
    for a, b in boundary:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)

    branching = [atom for atom, linked in neighbours.items() if len(linked) != 2]
    if branching:
        raise ValueError(
            "The union of these {} faces does not have a simple boundary: atom(s) {} sit "
            "on {} boundary edges rather than 2.".format(
                len(faces), branching[:5],
                [len(neighbours[a]) for a in branching[:5]])
        )

    start = boundary[0][0]
    cycle = [start]
    previous = None
    current = start

    while True:
        a, b = neighbours[current]
        nxt = a if a != previous else b
        if nxt == start:
            break
        cycle.append(nxt)
        previous, current = current, nxt

    if len(cycle) != len(boundary):
        raise ValueError(
            "The union of these {} faces has a disconnected boundary: walked {} of {} "
            "boundary edges.".format(len(faces), len(cycle), len(boundary))
        )

    return cycle


def mask_for_atoms(structure, atoms):
    mask = np.zeros(len(structure["symbols"]), dtype=bool)
    mask[sorted(atoms)] = True
    return mask


def build_regions(structure, ring, shell_ring_size, patch, ribbon_box):
    """
    The four carbon selections of the main figure, with the geometry panel (a) draws them from.

    The first three are strictly nested - 9 ring carbons inside the 23 of the shell inside
    the 66 of the ribbon box - so the curves read as a decay series outwards from the ring.
    The pristine patch is the far end of that series and shares no atom with any of them.

    `patch` is the dict from pristine_patch_regions.load_structure(), whose mask is built
    from the same geometry file and so indexes the same atoms.

    Returns the regions and the shell faces they were built from, so the disjoint shell of
    shell_only_region() does not have to find them a second time.
    """
    shell_faces = edge_sharing_rings(structure, ring, shell_ring_size)

    if not shell_faces:
        raise SystemExit(
            "The {}-membered ring at x {:.1f}, y {:.1f} shares no edge with any "
            "{}-membered ring, so there is no shell to project onto. Pass a different "
            "--shell-ring-size.".format(
                ring["size"], ring["centroid"][0], ring["centroid"][1], shell_ring_size)
        )

    shell_atoms = set(ring["atoms"])
    for face in shell_faces:
        shell_atoms.update(face["atoms"])

    faces = [ring["atoms"]] + [face["atoms"] for face in shell_faces]
    positions = structure["positions"]

    regions = [
        {
            "region": RING_REGION,
            "label": REGION_LABELS[RING_REGION],
            "colour": REGION_COLOURS[RING_REGION],
            "mask": mask_for_atoms(structure, ring["atoms"]),
            "outline": ring["polygon"],
            "n_faces": 1,
        },
        {
            "region": SHELL_REGION,
            "label": REGION_LABELS[SHELL_REGION],
            "colour": REGION_COLOURS[SHELL_REGION],
            "mask": mask_for_atoms(structure, shell_atoms),
            "outline": positions[union_boundary(faces)][:, :2],
            "n_faces": len(faces),
        },
        {
            "region": PATCH_REGION,
            "label": REGION_LABELS[PATCH_REGION],
            "colour": REGION_COLOURS[PATCH_REGION],
            "mask": patch["mask"].copy(),
            "outline": None,
            "box": patch["box"],
            "n_faces": 0,
        },
    ]

    # Between the shell and the patch in the drawing order, so panel (a) layers it under
    # the two it contains.
    regions.insert(2, ribbon_region(structure, ribbon_box))

    return regions, shell_faces


def shell_only_region(structure, ring, shell_faces):
    """
    The shell carbons that are NOT in the ring - the first ring of atoms outwards from it.

    The obvious reading of "the pentagons around the nine-membered ring" is the union of
    those five faces, but on H02 that is not a fourth selection at all: every one of the
    nine ring carbons lies on a pentagon, so the union is exactly the 23 carbons of
    SHELL_REGION and its curve would be a duplicate. Removing the ring leaves the 14
    carbons that are pentagon-but-not-ring, which is both distinct and the more useful
    comparison - together with the ring it partitions the shell, 9 + 14 = 23, so the
    combined curve is a weighted mean of the two and the decay outwards can be read off
    directly.
    """
    atoms = set()
    for face in shell_faces:
        atoms.update(face["atoms"])
    atoms -= set(ring["atoms"])

    if not atoms:
        raise SystemExit(
            "Every carbon of the surrounding {} face(s) is also in the {}-membered ring, "
            "so there is no shell left once the ring is removed.".format(
                len(shell_faces), ring["size"])
        )

    return {
        "region": PENTAGON_REGION,
        "label": REGION_LABELS[PENTAGON_REGION],
        "colour": REGION_COLOURS[PENTAGON_REGION],
        "mask": mask_for_atoms(structure, atoms),
        "outline": None,
        "n_faces": len(shell_faces),
    }


def ribbon_region(structure, box):
    """
    Every carbon inside the nanoribbon box, edges and all.

    A plain Cartesian box test on the coordinates as written, as pristine_patch_regions
    .box_mask() does it and for the same reason: the dos_cluster geometries carry no
    lattice, so there is nothing to wrap against. Carbons only, like every other region
    here - the terminating hydrogens that fall inside the box are not projected, though
    the carbons carrying them are.

    Deliberately not filtered by coordination or ring membership. Unlike the pristine
    patch, this region is not claiming to be anything: it is the whole neck as drawn, so
    audit_patch()-style assertions would be the wrong tool. What it contains is recorded
    by describe_ribbon() instead, so the caption can say it rather than the code enforce it.
    """
    positions = structure["positions"]
    inside = P.box_mask(positions, box)

    mask = np.array(
        [inside[i] and symbol == "C" for i, symbol in enumerate(structure["symbols"])],
        dtype=bool,
    )

    if not mask.any():
        raise SystemExit(
            "The nanoribbon box {} contains no carbons.".format(P.box_label(box))
        )

    return {
        "region": RIBBON_REGION,
        "label": REGION_LABELS[RIBBON_REGION],
        "colour": REGION_COLOURS[RIBBON_REGION],
        "mask": mask,
        "outline": None,
        "box": box,
        "n_faces": 0,
    }


def describe_ribbon(structure, region):
    """
    What the ribbon box actually caught, as counts for the summary CSV.

    Reported rather than asserted. A box across the neck cuts the ribbon, so it picks up
    terminated and undercoordinated carbons by construction; the point of counting them is
    that a reader of the curve knows how much of it is edge.
    """
    symbols = structure["symbols"]
    positions = structure["positions"]
    selected = np.flatnonzero(region["mask"])

    terminated = P.hydrogenated_carbons(symbols, positions)
    inside = P.box_mask(positions, region["box"])

    return {
        "n_C_terminated": sum(1 for i in selected if i in terminated),
        "n_C_undercoordinated": sum(1 for i in selected
                                    if structure["coordination"][i] < 3),
        "n_H_in_box": sum(1 for i, s in enumerate(symbols) if s == "H" and inside[i]),
    }


def draw_skeleton(ax, structure, marker_scale=1.0, label_rings=True):
    """
    The ring-census map of plot_n_membered_rings_all.py, on the cluster geometry.

    Same palette and the same face walk, so this is the <ID>_rings.png picture with the
    hydrogens that structure figure has no reason to draw. Bonds and atoms follow
    plot_pristine_patch_dos.draw_structure() rather than that script's per-bond ax.plot
    loop, which is far slower at 379 carbons and looks identical.
    """
    positions = structure["positions"]
    symbols = np.asarray(structure["symbols"])

    ax.add_collection(
        matplotlib.collections.LineCollection(
            bond_segments(positions, symbols),
            colors=BOND, linewidths=0.6, alpha=0.8, zorder=1,
        )
    )

    for entry in structure["rings"]:
        facecolor, edgecolor = R.ring_colour(entry["size"])
        hexagon = entry["size"] == 6
        polygon = np.asarray(entry["polygon"], dtype=float)

        ax.fill(
            polygon[:, 0], polygon[:, 1],
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=0.7 if hexagon else 1.1,
            alpha=R.HEXAGON_ALPHA if hexagon else R.RING_ALPHA,
            zorder=2,
        )

        if label_rings and not hexagon:
            # clip_on, because matplotlib does not clip text by default and the inset
            # draws the whole structure: without it every ring label outside the inset
            # window lands on top of the main panel.
            ax.text(entry["centroid"][0], entry["centroid"][1], str(entry["size"]),
                    fontsize=7.5, ha="center", va="center", color="black", zorder=8,
                    clip_on=True)

    carbon = symbols == "C"
    hydrogen = symbols == "H"

    ax.scatter(positions[carbon, 0], positions[carbon, 1],
               s=7 * marker_scale, c=BULK, edgecolors="none", zorder=3)
    ax.scatter(positions[hydrogen, 0], positions[hydrogen, 1],
               s=4 * marker_scale, c=HYDROGEN, edgecolors="none", zorder=3)


def draw_regions(ax, structure, regions, marker_scale=1.0, legend_counts=True):
    """
    The projections on top of the ring map.

    Drawn outermost first so the nine carbons of the ring end up on top of the shell and
    ribbon markers that also cover them: the regions are nested, and the reader should see
    the innermost one an atom belongs to.
    """
    positions = structure["positions"]

    # Descending with region size, so the nested regions stay visible through each other:
    # the ribbon's 66 markers sit under the shell's 23, which sit under the ring's 9.
    sizes = {RING_REGION: 34.0, SHELL_REGION: 20.0, RIBBON_REGION: 11.0,
             PATCH_REGION: 11.0}

    for region in reversed(regions):
        colour = region["colour"]
        label = region["label"]

        if legend_counts:
            label = "{} ({} C)".format(label, int(region["mask"].sum()))

        if region["outline"] is not None:
            outline = np.asarray(region["outline"], dtype=float)
            ax.add_patch(
                plt.Polygon(outline, closed=True, facecolor="none", edgecolor=colour,
                            lw=2.6 if region["region"] == RING_REGION else 2.0,
                            zorder=6)
            )
        elif region.get("box") is not None:
            x_min, x_max, y_min, y_max = region["box"]
            ax.add_patch(
                plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              facecolor="none", edgecolor=colour, lw=1.3, ls="--",
                              zorder=5)
            )

        mask = region["mask"]
        ax.scatter(positions[mask, 0], positions[mask, 1],
                   s=sizes[region["region"]] * marker_scale, c=colour,
                   edgecolors="none", zorder=7, label=label)


def draw_structure_panel(ax, structure, regions, ring, title, inset=None):
    """Panel (a): the ring map, the three regions, and an optional zoom on the defect."""
    positions = structure["positions"]

    draw_skeleton(ax, structure)
    draw_regions(ax, structure, regions)

    ax.set_aspect("equal")
    ax.set_xlabel(r"x / $\mathrm{\AA}$")
    ax.set_ylabel(r"y / $\mathrm{\AA}$")
    ax.set_title(title, fontsize=10)

    pad = 1.5
    ax.set_xlim(positions[:, 0].min() - pad, positions[:, 0].max() + pad)
    ax.set_ylim(positions[:, 1].min() - pad, positions[:, 1].max() + pad)

    # Bottom left is the empty corner of this flake, and is where the companion figures put
    # their legend. Handles are reversed because the regions are drawn outermost first;
    # the legend reads innermost first, matching panel (b) and the direction of the claim.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              fontsize=8.5, loc="lower left", frameon=False, labelcolor=INK_SECONDARY,
              handletextpad=0.4, borderpad=0.2, scatterpoints=1)

    if inset is None:
        return

    window, bounds = inset
    axins = ax.inset_axes(bounds)

    draw_skeleton(axins, structure, marker_scale=2.6)
    draw_regions(axins, structure, regions, marker_scale=2.6, legend_counts=False)

    axins.set_xlim(window[0], window[1])
    axins.set_ylim(window[2], window[3])
    axins.set_aspect("equal")
    axins.set_xticks([])
    axins.set_yticks([])

    for spine in axins.spines.values():
        spine.set_color(INK_SECONDARY)
        spine.set_linewidth(0.8)

    ax.indicate_inset_zoom(axins, edgecolor=INK_SECONDARY, lw=0.8, alpha=0.7)


def draw_region_comparison(ax, projections, zoom, sigma):
    """
    Panel (b): the three region LDOS over the near-E_F window, per carbon.

    Only the ring is filled. Three filled curves at these overlaps would read as a stack,
    which they are not - the regions are nested, so the curves are three views of
    overlapping sets of atoms and their areas do not add.
    """
    ring = projections[RING_REGION]

    for name in reversed(PANEL_B_REGIONS):
        data = projections[name]
        x = data["x"]
        y = carbon_curve(data)
        window = (x >= zoom[0]) & (x <= zoom[1])
        colour = REGION_COLOURS[name]

        ax.plot(x[window], y[window], lw=1.7, color=colour,
                label="{} ({} C)".format(REGION_LABELS[name], data["n_C"]))

        if name == RING_REGION:
            ax.fill_between(x[window], 0.0, y[window], color=colour, alpha=0.16, lw=0)

    ax.axvline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.6)
    ax.set_xlim(*zoom)

    # Headroom for the legend and the annotation, which both sit at the top of the axes
    # over a curve that peaks in the occupied half of the window. Panel (c) takes this
    # limit too, so it is set from the tallest curve rather than from the ring alone.
    #
    # Over the regions this panel draws, not every region in `projections`: --compare adds
    # one more, and the main figure must not change shape because of a flag that only
    # affects a different figure.
    window = (ring["x"] >= zoom[0]) & (ring["x"] <= zoom[1])
    # 1.45 rather than 1.32: the legend gained a fourth row when the ribbon was added, and
    # at 1.32 that row runs into the ring's peak at -1.5 eV.
    peak = max(float(carbon_curve(projections[name])[window].max())
               for name in PANEL_B_REGIONS)
    ax.set_ylim(0.0, 1.45 * peak)

    ax.set_xlabel(ring["graph_xlabel"])
    ax.set_ylabel(ylabel(ring["per_atom"]))
    ax.set_title(
        "(b)  local DOS near $E_F$  -  C s+p per carbon, $\\sigma$ = {:.3f} eV".format(sigma),
        fontsize=10,
    )
    # Reversed for the same reason as panel (a): drawn back to front, read front to back.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=8.5, frameon=False, loc="upper left")

    # The localisation claim rests on these numbers, so they are on the figure rather than
    # only in the summary CSV.
    lines = ["mean DOS, $|E-E_F| \\leq$ 0.5 eV"]
    for name in PANEL_B_REGIONS:
        lines.append("  {}:  {:.4f}".format(
            REGION_LABELS[name],
            near_fermi_metrics(projections[name], zoom)["mean_dos_pm0p5_per_C"]))

    ax.text(0.985, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
            ha="right", va="top", color=INK_SECONDARY, linespacing=1.5)


def s_free_interval(x, c_s, zoom):
    """
    The widest interval about E_F over which the plotted C s curve is exactly zero.

    Exact zeros are meaningful here rather than lucky. cache_atom_resolved_dos.broaden_hist
    convolves with scipy's gaussian_filter1d, whose kernel is truncated at 4 sigma, so the
    broadened curve is identically zero wherever no raw Mulliken weight lies within 4 sigma.
    An interval of zeros in the plotted curve is therefore a statement about the underlying
    eigenstates: over this range the ring carbons carry no s weight at all, and everything
    is p.

    Conservative by exactly that 4 sigma at each end - the raw gap is wider than the drawn
    one - which is the safe direction for a claim made on the figure.

    Returns None if C s is nonzero at E_F, i.e. if there is no such interval to report.
    """
    window = (x >= zoom[0]) & (x <= zoom[1])
    xw = x[window]
    zero = c_s[window] == 0.0

    centre = int(np.argmin(np.abs(xw)))
    if not zero[centre]:
        return None

    low = centre
    while low > 0 and zero[low - 1]:
        low -= 1

    high = centre
    while high < len(zero) - 1 and zero[high + 1]:
        high += 1

    return float(xw[low]), float(xw[high])


def draw_orbital_panel(ax, projection, zoom, s_scale, ylim=None):
    """
    Panel (c): C s against C p for the ring alone, over the same window as panel (b).

    The magnified C s trace is the compromise the data forces. C s is real - it is not
    numerically zero anywhere in this window - but at a few per cent of C p it has no
    visible shape on the axis that C p needs. A second y-axis would fix that by inviting
    the two channels to be compared as if they were the same size, which is the one
    conclusion this panel exists to rule out.
    """
    x = projection["x"]
    c_s = projection["c_s"][0]
    c_p = projection["c_p"][0]
    total = c_s + c_p

    window = (x >= zoom[0]) & (x <= zoom[1])
    fraction = c_s[window] / np.where(total[window] > 0, total[window], np.nan)
    max_fraction = float(np.nanmax(fraction))

    ax.fill_between(x[window], 0.0, c_p[window], color=REGION_COLOURS[RING_REGION],
                    alpha=0.30, lw=0, label="C p")
    ax.plot(x[window], c_p[window], lw=1.6, color=REGION_COLOURS[RING_REGION])
    ax.plot(x[window], total[window], lw=1.0, color="black", label="C s+p")
    ax.plot(x[window], c_s[window], lw=1.4, color=S_COLOUR, label="C s")

    if s_scale and s_scale != 1.0:
        ax.plot(x[window], c_s[window] * s_scale, lw=1.1, ls="--", color=S_COLOUR,
                alpha=0.85, label=r"C s $\times$ {:g}".format(s_scale))

    ax.axvline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.6)
    ax.set_xlim(*zoom)
    ax.set_ylim(0.0, ylim if ylim else None)
    ax.set_xlabel(projection["graph_xlabel"])
    ax.set_ylabel(ylabel(projection["per_atom"]))
    ax.set_title(
        "(c)  orbital character of the 9-ring  -  {} C, per carbon".format(
            projection["n_C"]),
        fontsize=10,
    )
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    interval = s_free_interval(x, c_s, zoom)

    note = ["C s $\\leq$ {:.1f} % of C s+p across the window".format(100.0 * max_fraction)]
    if interval is not None:
        note.append("C s = 0 from {:+.2f} to {:+.2f} eV".format(*interval))

    ax.text(0.985, 0.97, "\n".join(note), transform=ax.transAxes, fontsize=7.5,
            ha="right", va="top", color=INK_SECONDARY, linespacing=1.5)

    return max_fraction, interval


def build_figure(structure, ring, regions, projections, zoom, sigma, s_scale, inset,
                 output_file):
    fig = plt.figure(figsize=(14.0, 6.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1],
                            wspace=0.20, hspace=0.55)

    ax_structure = fig.add_subplot(grid[:, 0])
    ax_regions = fig.add_subplot(grid[0, 1])
    ax_orbital = fig.add_subplot(grid[1, 1])

    # Carbon counts are on the legend, not here: repeating them in the title only crowds
    # a panel that is already carrying three overlays and an inset.
    draw_structure_panel(
        ax_structure, structure, regions, ring,
        "(a)  {}  -  {}-membered ring at x {:.1f}, y {:.1f}, "
        "area {:.1f} $\\mathrm{{\\AA}}^2$".format(
            structure["structure_id"], ring["size"], ring["centroid"][0],
            ring["centroid"][1], ring["area"]),
        inset=inset,
    )

    draw_region_comparison(ax_regions, projections, zoom, sigma)

    # Same y-limit as panel (b), so the ring curve is literally the same height in both and
    # panel (c) reads as a decomposition of the purple curve above it rather than a new plot.
    orbital = draw_orbital_panel(ax_orbital, projections[RING_REGION], zoom, s_scale,
                                 ylim=ax_regions.get_ylim()[1])

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output_file))

    return orbital


def build_comparison(projections, order, xlim, zoom, output_file):
    """
    The four regions over the whole valence range and again near E_F, per carbon.

    The layout of plot_pristine_patch_dos.build_comparison() - same two windows, same
    per-carbon C s+p curves - with four regions instead of two, so the near-E_F contrast
    of the main figure can be checked against the full valence range it came out of. If
    the ring and the patch only differ near E_F, the deep valence panel is where that
    shows: the sigma and pi manifolds below -5 eV are set by bonding that a nine-membered
    ring changes far less than it changes the frontier states.

    SHELL_REGION is drawn dashed and thinner because it is not independent evidence: it is
    the weighted mean of RING_REGION and PENTAGON_REGION, which partition it 9 + 14.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    for ax, window, label in (
        (axes[0], xlim, "full valence range"),
        (axes[1], zoom, "near $E_F$"),
    ):
        for name in order:
            data = projections[name]
            x = data["x"]
            y = carbon_curve(data)
            inside = (x >= window[0]) & (x <= window[1])
            combined = name == SHELL_REGION

            ax.plot(x[inside], y[inside],
                    lw=1.2 if combined else 1.6,
                    ls="--" if combined else "-",
                    color=REGION_COLOURS[name],
                    label="{} ({} C)".format(REGION_LABELS[name], data["n_C"]))

        ax.axvline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.6)
        ax.set_xlim(*window)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(projections[RING_REGION]["graph_xlabel"])
        ax.set_ylabel(ylabel(projections[RING_REGION]["per_atom"]))
        ax.set_title("C s+p per carbon, {}".format(label), fontsize=10)
        ax.legend(fontsize=8.5, frameon=False)

    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output_file))


def p_fraction_near_fermi(projection, half_width=0.5):
    """Share of the near-E_F weight carried by C p, averaged over |E - E_F| <= half_width."""
    x = projection["x"]
    window = np.abs(x) <= half_width

    c_s = float(projection["c_s"][0][window].mean())
    c_p = float(projection["c_p"][0][window].mean())

    return c_p / (c_s + c_p) if (c_s + c_p) else float("nan")


def write_summary(path, rows):
    """
    Replace this structure's rows in the summary, keyed on structure, ring and broadening.

    One row per region rather than one per run, so the three projections can be compared
    in the table the same way they are in panel (b). plot_ring_dos.write_summary() keys on
    (structure, ring size, ring index, sigma), which here would collapse all three regions
    onto one row; the region is part of the key instead.
    """
    key = ("structure_id", "ring_size", "region", "sigma_eV")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    replaced = {tuple(str(row[k]) for k in key) for row in rows}

    existing = []
    if path.exists():
        with open(path, newline="") as f:
            existing = [r for r in csv.DictReader(f)
                        if tuple(r.get(k) for k in key) not in replaced]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows({k: str(v) for k, v in row.items()} for row in rows)

    print("Wrote {}".format(path))


def main():
    parser = argparse.ArgumentParser(
        description="Local DOS signature of one ring against a pristine patch of the "
                    "same cluster run."
    )
    parser.add_argument("--structure-id", default="H02")
    parser.add_argument("--ring-size", type=int, default=9,
                        help="Ring size to project onto (default 9).")
    parser.add_argument("--ring-index", type=int, default=None,
                        help="Which ring of that size, when the structure has more than "
                             "one. See --list-rings.")
    parser.add_argument("--shell-ring-size", type=int, default=5,
                        help="Ring size making up the shell around it (default 5).")
    parser.add_argument("--list-rings", action="store_true",
                        help="Print the rings of the requested size and exit.")
    parser.add_argument("--calc-root", default=str(P.CLUSTER_CALC_ROOT),
                        help="Directory holding <ID>/geometry.in for the cluster runs.")
    parser.add_argument("--cache-root", default=str(CLUSTER_CACHE_ROOT),
                        help="Per-atom DOS cache directory (cluster, NOT the periodic one).")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="Gaussian broadening in eV (default {:.4f}).".format(DEFAULT_SIGMA))
    parser.add_argument("--min-edge-hops", type=int, default=P.DEFAULT_MIN_EDGE_HOPS,
                        help="Pristine patch only: minimum C-C bonds from any edge carbon "
                             "(default {}).".format(P.DEFAULT_MIN_EDGE_HOPS))
    parser.add_argument("--zoom", nargs=2, type=float, default=list(DEFAULT_ZOOM),
                        help="Energy window about E_F, in eV (default -3 3).")
    parser.add_argument("--s-scale", type=float, default=DEFAULT_S_SCALE,
                        help="Magnification of the C s trace in panel (c); 1 disables it "
                             "(default {:g}).".format(DEFAULT_S_SCALE))
    parser.add_argument("--compare", action="store_true",
                        help="Also plot all four regions - the three above plus the shell "
                             "carbons that are not in the ring - over the full valence "
                             "range and near E_F, as a second figure.")
    parser.add_argument("--ribbon-box", nargs=4, type=float, default=list(RIBBON_BOX),
                        help="--compare only: nanoribbon box, x_min x_max y_min y_max in "
                             "Angstrom (default {:g} {:g} {:g} {:g}).".format(*RIBBON_BOX))
    parser.add_argument("--xlim", nargs=2, type=float, default=list(DEFAULT_XLIM),
                        help="Full-range window of the --compare figure, in eV "
                             "(default {:g} {:g}).".format(*DEFAULT_XLIM))
    parser.add_argument("--inset", action="store_true",
                        help="Add a zoomed inset of the defect neighbourhood to panel (a). "
                             "Off by default: with the nanoribbon box drawn there too, the "
                             "panel carries four overlays and the inset crowds them.")
    parser.add_argument("--inset-window", nargs=4, type=float,
                        default=list(DEFAULT_INSET_WINDOW),
                        help="Inset data window, x_min x_max y_min y_max in Angstrom.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default=str(SUMMARY_FILE))
    args = parser.parse_args()

    zoom = tuple(args.zoom)

    structure = find_rings(args.structure_id, args.calc_root)

    if args.list_rings:
        print_rings(structure, args.ring_size)
        return

    ring = select_ring(structure, args.ring_size, args.ring_index)

    # Built from the same geometry file, so its mask indexes the same atoms as the ring
    # masks above. audit_patch() is the assertion that the patch really is pristine: the
    # whole comparison is meaningless if it quietly picks up a defect ring or an edge.
    patch = P.load_structure(args.structure_id, args.calc_root, args.min_edge_hops)
    audit = P.audit_patch(patch)

    regions, shell_faces = build_regions(structure, ring, args.shell_ring_size, patch,
                                         tuple(args.ribbon_box))
    ribbon = next(r for r in regions if r["region"] == RIBBON_REGION)

    # The four regions of the main figure are nested (or, for the patch, deliberately
    # disjoint), and are the only ones panel (a) draws. The pentagon shell is the odd one
    # out: it overlaps the shell without containing or being contained by the ring, so it
    # is kept apart and used only by --compare. Ordered ring first, then outwards.
    if args.compare:
        extra = shell_only_region(structure, ring, shell_faces)
        all_regions = [regions[0], extra] + regions[1:]
    else:
        all_regions = list(regions)

    cache = read_cache(args.cache_root, args.structure_id)
    check_cache_alignment(cache, structure["symbols"], args.structure_id)

    if cache["nkpts"] != 1:
        raise SystemExit(
            "{} cache has {} k-points, so it is a periodic run, not a cluster. Point "
            "--cache-root at {}.".format(
                args.structure_id, cache["nkpts"], CLUSTER_CACHE_ROOT)
        )

    projections = {
        region["region"]: project(
            cache, rows_for_mask(structure, region["mask"], region["region"]),
            region["region"], sigma=args.sigma, per_atom=True)
        for region in all_regions
    }

    print("{}: {}-membered ring at x {:.2f}, y {:.2f}, area {:.2f} A^2".format(
        args.structure_id, ring["size"], ring["centroid"][0], ring["centroid"][1],
        ring["area"]))
    print("pristine patch: {} C in box {}, {}-{} bonds from an edge (mean {})".format(
        audit["n_C_in_patch"], P.box_label(patch["box"]), audit["min_edge_hops"],
        audit["max_edge_hops"], audit["mean_edge_hops"]))

    ribbon_stats = describe_ribbon(structure, ribbon)
    print("nanoribbon: {} C in box {}, of which {} H-terminated and {} undercoordinated "
          "({} H inside the box, not projected)".format(
              int(ribbon["mask"].sum()), P.box_label(ribbon["box"]),
              ribbon_stats["n_C_terminated"], ribbon_stats["n_C_undercoordinated"],
              ribbon_stats["n_H_in_box"]))
    print()
    print("  {:<26} {:>4} {:>11} {:>13} {:>12} {:>12}".format(
        "region", "n_C", "DOS at E_F", "mean +-0.5 eV", "mean window", "suppression"))

    metrics = {}
    for region in all_regions:
        name = region["region"]
        metrics[name] = near_fermi_metrics(projections[name], zoom)
        print("  {:<26} {:>4} {:>11.4f} {:>13.4f} {:>12.4f} {:>12.3f}".format(
            REGION_LABELS[name], projections[name]["n_C"],
            metrics[name]["dos_at_EF_per_C"], metrics[name]["mean_dos_pm0p5_per_C"],
            metrics[name]["mean_dos_zoom_per_C"], metrics[name]["suppression_pm0p5"]))
    print()

    inset = (tuple(args.inset_window), DEFAULT_INSET_BOUNDS) if args.inset else None

    output = Path(args.output).resolve() if args.output else (
        FIGURES_ROOT / "{}_ring{}_local_signature.png".format(
            args.structure_id, ring["size"]))

    max_s_fraction, s_free = build_figure(structure, ring, regions, projections, zoom,
                                          args.sigma, args.s_scale, inset, output)

    if s_free is not None:
        print("9-ring C s is identically zero from {:+.2f} to {:+.2f} eV: every state the "
              "ring carries in that range is p.".format(*s_free))
        print()

    comparison_file = ""
    if args.compare:
        comparison_file = FIGURES_ROOT / "{}_ring{}_regions_vs_patch.png".format(
            args.structure_id, ring["size"])
        build_comparison(projections, [region["region"] for region in all_regions],
                         tuple(args.xlim), zoom, comparison_file)

    rows = []
    for region in all_regions:
        name = region["region"]
        projection = projections[name]
        row = {
            "structure_id": args.structure_id,
            "ring_size": ring["size"],
            "region": name,
            "sigma_eV": round(args.sigma, 6),
            "n_C": projection["n_C"],
            "n_faces": region["n_faces"],
            "box": P.box_label(region["box"]) if region.get("box") else "",
            # Ribbon only, and reported rather than asserted - see describe_ribbon(). The
            # other regions are defined so that these are zero by construction, which is
            # not the same statement, so they are left blank rather than filled with 0.
            "n_C_terminated": ribbon_stats.get("n_C_terminated", "")
            if name == RIBBON_REGION else "",
            "n_C_undercoordinated": ribbon_stats.get("n_C_undercoordinated", "")
            if name == RIBBON_REGION else "",
            "n_H_in_box": ribbon_stats.get("n_H_in_box", "")
            if name == RIBBON_REGION else "",
            "atoms": " ".join(str(a) for a in np.flatnonzero(region["mask"])),
        }
        row.update(metrics[name])
        row.update({
            "p_fraction_pm0p5": round(p_fraction_near_fermi(projection), 5),
            "max_c_s_fraction_zoom": (
                round(max_s_fraction, 5) if name == RING_REGION else ""),
            "c_s_zero_min_eV": (
                round(s_free[0], 3) if name == RING_REGION and s_free else ""),
            "c_s_zero_max_eV": (
                round(s_free[1], 3) if name == RING_REGION and s_free else ""),
            "occupied_valence_electrons_per_C": round(
                occupied_valence_electrons(projection, cache["nspin"]), 3),
            "homo_eV": round(cache["homo"], 6),
            "nkpts": cache["nkpts"],
            "zoom_min_eV": zoom[0],
            "zoom_max_eV": zoom[1],
            # PENTAGON_REGION and RIBBON_REGION appear only in the comparison figure, so
            # they have no main figure to point at.
            "figure": "" if name in (PENTAGON_REGION, RIBBON_REGION)
            else repo_relative(output),
            "comparison_figure": (
                repo_relative(comparison_file) if comparison_file else ""),
        })
        rows.append(row)

    write_summary(args.summary, rows)


if __name__ == "__main__":
    main()
