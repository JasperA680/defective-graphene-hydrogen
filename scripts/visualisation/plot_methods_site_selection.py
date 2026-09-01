"""
Two methods figures for "Identification and hydrogenation of defect-edge sites".

The figures document the cluster-mode saturation procedure implemented in
scripts/hydrogenation/full_saturate_all_structures.py, using one structure as a
worked example and the whole A-J set to calibrate the two geometric thresholds.

Figure 1 - the worked example:

    (a) Site classification across the whole structure. Every branch of the
        decision tree gets its own marker, and the crop frame plus its
        FRAME_BAND band are drawn, so the internal-edge / crop-rim split is
        visible rather than asserted.
    (b) Zoom on an sp (carbyne-like) chain. Each C-C-C angle is annotated
        against SP_ANGLE_THRESHOLD_DEG, showing the carbons left bare and the
        single axial H on the chain terminus. This is the hybridisation filter.
    (c) Zoom on one site where the missing-bond bisector is blocked. The full
        in-plane tilt sweep is drawn, rejected directions as faint crosses and
        the accepted one as an arrow, with the MIN_NONBONDED_HC_DISTANCE
        exclusion radius on the blocking atom.

The marker legend is a figure-level legend under panel (a) because the same
site classes are used in (a) and (b).

Figure 2 - calibration of the two thresholds over all 95 structures:

    (a) the C-C-C angle distribution at coordination-2 carbons against the
        155 degree cut,
    (b) the bounded ring-face area distribution against the 12 A^2 cut.

J04 is the default worked example because it is the only frame that fires
essentially every branch at once - internal edge, crop rim, coordination-1
terminus, sp exclusion, sp chain terminus, a healed stretched bond, a deleted
disconnected fragment, tilted placements and tight-threshold placements.

Panels 1(a) and 1(b) are drawn from the production per-site table. Panel 1(c) is
NOT read from a table - it replays the placement through the production
functions so the tilt sweep can be shown, and asserts that every replayed H
reproduces the recorded one to within PLACEMENT_TOLERANCE. A figure that
disagrees with the shipped geometry is worse than no figure. The reason the
bisector was rejected is printed to stdout rather than drawn on the panel.

Inputs:
    <ROOT>/results/tables/cluster_saturation_sites/<ID>_saturation_sites.csv
    <ROOT>/results/tables/cluster_saturation_summary.csv
    <ROOT>/results/tables/cluster_ring_face_areas.csv  (build_ring_face_area_cache.py)
    <ROOT>/<Letter>/<N>/geometry.in

Output (300 dpi PNG plus PDF for each):
    <ROOT>/results/figures/methods_h_site_selection_<ID>.png / .pdf
    <ROOT>/results/figures/methods_h_threshold_calibration.png / .pdf

Run from the repository root:
    .venv-dos/bin/python scripts/visualisation/plot_methods_site_selection.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hydrogenation"))

# Panels this dense are read at part-page width, so the axis furniture is set
# above the matplotlib default of 10 pt.
plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14.5,
    "ytick.labelsize": 14.5,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})

import full_saturate_all_structures as F  # noqa: E402

DEFAULT_STRUCTURE_ID = "J04"

SITE_TABLE_DIR = ROOT / "results" / "tables" / "cluster_saturation_sites"
SUMMARY_TABLE = ROOT / "results" / "tables" / "cluster_saturation_summary.csv"
RING_AREA_TABLE = ROOT / "results" / "tables" / "cluster_ring_face_areas.csv"
FIGURE_DIR = ROOT / "results" / "figures"

# Replayed H must match the shipped geometry to this tolerance, in Angstrom.
# The replay uses the same functions in the same order, so the only expected
# difference is floating-point noise from the CSV round trip.
PLACEMENT_TOLERANCE = 1e-6

# Half-width of the zoom windows in panels (b) and (c), in Angstrom.
ZOOM_PAD = 2.6

COLOURS = {
    "bulk": "#c8cdd2",
    "bond": "#9aa0a6",
    "carbon": "#404448",
    "internal": "#2ca02c",
    "crop": "#1f77b4",
    "coord1": "#ff7f0e",
    "sp": "#9467bd",
    "tight": "#d62728",
    "deleted": "#8c1d18",
    "hydrogen": "#3d7ea6",
    "reject": "#d62728",
    "accept": "#1a7f37",
    "frame": "#5f6368",
}


# ----------------------------------------------------------------------------
# Geometry / placement replay
# ----------------------------------------------------------------------------


def geometry_path_for(structure_id):
    """Map a structure ID like 'J04' back onto <ROOT>/J/4/geometry.in."""
    letter = structure_id[0]
    number = int(structure_id[1:])
    return ROOT / letter / str(number) / "geometry.in"


def build_bond_graph(structure_id):
    """
    Rebuild the cluster-mode bond graph exactly as the saturation run does.

    Returns (symbols, positions, neighbour_indices, fragment_atoms,
    ring_polygons, n_healed). lattice is None throughout, which is what makes
    the analysis non-periodic.
    """
    _, symbols, positions = F.read_fhi_aims_geometry(geometry_path_for(structure_id))

    coordination, neighbour_indices = F.compute_coordination(
        symbols, positions, None, None
    )
    n_healed = F.heal_stretched_bonds(
        symbols, positions, coordination, neighbour_indices, None, None
    )

    fragment_atoms = F.disconnected_fragment_atoms(symbols, neighbour_indices)

    ring_polygons = F.carbon_ring_faces(
        symbols,
        positions,
        neighbour_indices,
        exclude=fragment_atoms,
        max_area=F.MAX_CAGE_RING_AREA,
    )

    return symbols, positions, neighbour_indices, fragment_atoms, ring_polygons, n_healed


def placement_order(sites):
    """
    sp chain termini first, then atom index.

    An sp2 site has the full tilt sweep to fall back on while an sp terminus has
    one admissible direction and a 15 degree cap, so the constrained site has to
    claim its space first. This mirrors placement_order() in the saturation
    script; getting it wrong changes which H land where.
    """
    sp_termini = sites.index[sites["site_class"].str.contains("sp_terminus")].tolist()
    others = [i for i in sites.index[sites["n_h_target"].fillna(0) > 0] if i not in sp_termini]
    return sp_termini + sorted(others)


def directions_and_sweep(atom_index, site_row, positions, neighbour_indices):
    """Return (candidate directions, tilt sweep) for one site, as production does."""
    if "sp_terminus" in site_row["site_class"]:
        directions = F.coord1_axial_direction(
            atom_index, positions, neighbour_indices, None, None
        )
        return directions, F.H_TILT_ANGLES_DEG_SP_TERMINUS

    if site_row["coordination"] == 1:
        directions = F.coord1_h_directions(
            atom_index, positions, neighbour_indices, None, None
        )
        return directions, F.H_TILT_ANGLES_DEG_FULL

    return (
        [F.missing_bond_direction(atom_index, positions, neighbour_indices, None, None)],
        F.H_TILT_ANGLES_DEG_FULL,
    )


def replay_placement(sites, symbols, positions, neighbour_indices, ring_polygons):
    """
    Re-run the H placement and record where each H landed and at what tilt.

    Returns {atom_index: [(tilt_deg, position), ...]}. Verified against the
    shipped per-site table by verify_replay().
    """
    new_h_positions = []
    placed = {}

    for atom_index in placement_order(sites):
        row = sites.loc[atom_index]
        directions, tilt_angles = directions_and_sweep(
            atom_index, row, positions, neighbour_indices
        )

        for direction in directions:
            if direction is None:
                continue

            position, tilt, _ = F.place_h_in_plane(
                atom_index,
                direction,
                symbols,
                positions,
                new_h_positions,
                None,
                None,
                tilt_angles=tilt_angles,
                ring_polygons=ring_polygons.get(atom_index),
            )

            if position is None and "sp_terminus" not in row["site_class"]:
                position, tilt, _ = F.place_h_in_plane(
                    atom_index,
                    direction,
                    symbols,
                    positions,
                    new_h_positions,
                    None,
                    None,
                    tilt_angles=tilt_angles,
                    min_hc=F.MIN_NONBONDED_HC_DISTANCE_TIGHT,
                    min_hh=F.MIN_HH_DISTANCE_TIGHT,
                    ring_polygons=ring_polygons.get(atom_index),
                )

            if position is not None:
                new_h_positions.append(position)
                placed.setdefault(atom_index, []).append((tilt, position))

    return placed


def verify_replay(placed, sites):
    """
    Assert the replay reproduces the shipped geometry.

    Panel (c) draws directions the production run never wrote down, so the only
    guarantee that it describes the real procedure is that everything the
    production run *did* write down comes back identical.
    """
    n_checked = 0

    for atom_index, entries in placed.items():
        row = sites.loc[atom_index]

        reference = [(row["h_tilt_deg"], (row["h_x"], row["h_y"], row["h_z"]))]
        if not pd.isna(row.get("h_tilt_deg_2", np.nan)):
            reference.append(
                (row["h_tilt_deg_2"], (row["h_x_2"], row["h_y_2"], row["h_z_2"]))
            )

        for (tilt, position), (ref_tilt, ref_position) in zip(entries, reference):
            offset = np.max(np.abs(np.asarray(position) - np.asarray(ref_position, dtype=float)))
            if abs(tilt - ref_tilt) > 1e-9 or offset > PLACEMENT_TOLERANCE:
                raise AssertionError(
                    "Replay disagrees with the shipped placement at atom {}: "
                    "tilt {} vs {}, offset {:.2e} A".format(
                        atom_index, tilt, ref_tilt, offset
                    )
                )
            n_checked += 1

    n_expected = int(sites["n_h_added"].fillna(0).sum())
    if n_checked != n_expected:
        raise AssertionError(
            "Replayed {} H but the site table records {}.".format(n_checked, n_expected)
        )

    return n_checked


def sweep_trace(atom_index, sites, symbols, positions, neighbour_indices, ring_polygons):
    """
    Every direction tried at one site, in the order the sweep tries them.

    Returns (bisector, trace, prior_h) where trace is
    [(tilt_deg, position, accepted, reason), ...] and prior_h is
    [(parent_carbon, position), ...] for the H already placed when this site was
    reached. Those H are obstacles for this one, so the rejections shown are the
    ones that actually happened rather than a re-derivation on an empty sheet.
    """
    order = placement_order(sites)
    prior_h = []

    for other in order:
        if other == atom_index:
            break

        row = sites.loc[other]
        directions, tilt_angles = directions_and_sweep(
            other, row, positions, neighbour_indices
        )

        for direction in directions:
            if direction is None:
                continue
            placed_so_far = [p for _, p in prior_h]
            position, _, _ = F.place_h_in_plane(
                other, direction, symbols, positions, placed_so_far, None, None,
                tilt_angles=tilt_angles, ring_polygons=ring_polygons.get(other),
            )
            if position is None:
                position, _, _ = F.place_h_in_plane(
                    other, direction, symbols, positions, placed_so_far, None, None,
                    tilt_angles=tilt_angles,
                    min_hc=F.MIN_NONBONDED_HC_DISTANCE_TIGHT,
                    min_hh=F.MIN_HH_DISTANCE_TIGHT,
                    ring_polygons=ring_polygons.get(other),
                )
            if position is not None:
                prior_h.append((other, position))

    row = sites.loc[atom_index]
    directions, tilt_angles = directions_and_sweep(
        atom_index, row, positions, neighbour_indices
    )
    bisector = directions[0]

    trace = []
    parent = positions[atom_index]
    placed_so_far = [p for _, p in prior_h]

    for angle in tilt_angles:
        senses = (0.0,) if angle == 0.0 else (angle, -angle)

        for signed_angle in senses:
            candidate = parent + F.CH_BOND_LENGTH * F.rotate_about_z(bisector, signed_angle)
            accepted, reason = F.check_h_candidate(
                candidate_pos=candidate,
                parent_c_index=atom_index,
                symbols=symbols,
                positions=positions,
                new_h_positions=placed_so_far,
                lattice=None,
                inv_lattice_t=None,
                ring_polygons=ring_polygons.get(atom_index),
            )
            trace.append((float(signed_angle), candidate, accepted, reason))

            if accepted:
                return bisector, trace, prior_h

    return bisector, trace, prior_h


def interpret_rejection(reason, prior_h, positions):
    """
    Turn a check_h_candidate() reason string into something drawable.

    Returns (blocker_xy, exclusion_radius, caption), or None if the reason is not
    a pairwise clash. The reason strings are the production ones, so the caption
    quotes the distance the run itself rejected on rather than a recomputation.
    """
    if reason.startswith("H-H clash") and "newly added H" in reason:
        distance = float(reason.split()[2])
        h_index = int(reason.rsplit(maxsplit=1)[1])
        parent, position = prior_h[h_index]
        return (
            position[:2],
            F.MIN_HH_DISTANCE,
            "H–H clash {:.2f} Å with the H already on C{}".format(distance, parent),
        )

    if reason.startswith("H-H clash"):
        distance = float(reason.split()[2])
        atom_index = int(reason.rsplit(maxsplit=1)[1])
        return (
            positions[atom_index][:2],
            F.MIN_HH_DISTANCE,
            "H–H clash {:.2f} Å with H{}".format(distance, atom_index),
        )

    if reason.startswith("H-C clash"):
        distance = float(reason.split()[2])
        atom_index = int(reason.rsplit(maxsplit=1)[1])
        return (
            positions[atom_index][:2],
            F.MIN_NONBONDED_HC_DISTANCE,
            "H–C clash {:.2f} Å with C{}".format(distance, atom_index),
        )

    if reason.startswith("inside a"):
        return None, None, reason

    return None


# ----------------------------------------------------------------------------
# Site-class presentation
# ----------------------------------------------------------------------------

# (site_class, label, colour, marker, size, z-order). Ordered so the rarer and
# more interesting classes draw on top of the bulk.
SITE_STYLES = [
    ("bulk_coord3", "coordination 3, bulk", COLOURS["bulk"], "o", 7, 1),
    ("external_crop_boundary_coord2", "coordination 2, crop rim (1 H)", COLOURS["crop"], "o", 34, 3),
    ("internal_defect_edge_coord2", "coordination 2, internal edge (1 H)", COLOURS["internal"], "o", 42, 4),
    ("coord1_terminus", "coordination 1, terminus (2 H)", COLOURS["coord1"], "s", 40, 5),
    ("internal_sp_linear_coord2", "coordination 2, sp / near-linear (no H)", COLOURS["sp"], "D", 40, 6),
    ("external_crop_sp_linear_coord2", "coordination 2, sp on crop rim (no H)", COLOURS["sp"], "D", 40, 6),
    ("internal_sp_terminus_coord1", "sp chain terminus (1 axial H)", COLOURS["sp"], "*", 150, 7),
    ("external_crop_sp_terminus_coord1", "sp chain terminus on crop rim", COLOURS["sp"], "*", 150, 7),
    ("disconnected_fragment", "disconnected fragment (deleted)", COLOURS["deleted"], "x", 42, 5),
    ("coord0_deleted", "coordination 0 (deleted)", COLOURS["deleted"], "P", 46, 5),
]


def draw_bonds(ax, positions, neighbour_indices, subset=None, lw=1.0, alpha=1.0):
    """Draw each C-C bond once. subset restricts to bonds with both ends in it."""
    for i, neighbours in enumerate(neighbour_indices):
        for j in neighbours:
            if j <= i:
                continue
            if subset is not None and not (i in subset and j in subset):
                continue
            ax.plot(
                [positions[i][0], positions[j][0]],
                [positions[i][1], positions[j][1]],
                color=COLOURS["bond"], lw=lw, alpha=alpha, zorder=1,
            )


# ----------------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------------


def panel_a(ax, sites, positions, neighbour_indices, zoom_boxes):
    """
    Whole-structure site classification, with the crop frame and its band.

    Returns the legend handles. The same site classes appear in panel (b), so
    the legend is drawn once at figure level rather than inside this panel.
    """
    kept = sites.index[sites["site_class"] != "disconnected_fragment"].tolist()
    draw_bonds(ax, positions, neighbour_indices, subset=set(kept), lw=0.7, alpha=0.55)

    handles = []
    for site_class, label, colour, marker, size, zorder in SITE_STYLES:
        rows = sites[sites["site_class"] == site_class]
        if rows.empty:
            continue

        unfilled = marker == "x"
        ax.scatter(
            rows["x"], rows["y"], s=size, c=colour, marker=marker, zorder=zorder,
            linewidths=1.0 if unfilled else 0.5,
            **({} if unfilled else {"edgecolors": "#33373b"}),
        )
        handles.append(
            Line2D([], [], ls="", marker=marker, ms=np.sqrt(size) * 1.35,
                   mfc=colour, mec=colour if unfilled else "#33373b",
                   mew=0.6, label="{} ({})".format(label, len(rows)))
        )

    # Crop frame: the bounding box of the carbons that survive fragment pruning,
    # and the FRAME_BAND band inside it that defines the crop rim.
    frame = sites.loc[kept]
    x_min, x_max = frame["x"].min(), frame["x"].max()
    y_min, y_max = frame["y"].min(), frame["y"].max()

    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False, ls="--", lw=1.2, ec=COLOURS["frame"], zorder=2,
    ))
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False, lw=0, zorder=2,
    ))
    band = F.FRAME_BAND
    ax.add_patch(Rectangle(
        (x_min + band, y_min + band),
        (x_max - x_min) - 2 * band, (y_max - y_min) - 2 * band,
        fill=False, ls=":", lw=1.0, ec=COLOURS["frame"], alpha=0.8, zorder=2,
    ))
    handles.append(Line2D([], [], ls="--", color=COLOURS["frame"],
                          label="crop frame and {:.1f} A band".format(band)))

    for letter, (cx, cy, pad) in zoom_boxes.items():
        ax.add_patch(Rectangle(
            (cx - pad, cy - pad), 2 * pad, 2 * pad,
            fill=False, lw=1.3, ec="#111418", zorder=8,
        ))
        ax.annotate(
            "({})".format(letter), (cx - pad, cy + pad),
            xytext=(2, 3), textcoords="offset points",
            fontsize=12, fontweight="bold", zorder=9,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x / Å")
    ax.set_ylabel("y / Å")
    ax.set_title("(a)", loc="left", fontsize=16, fontweight="bold")

    return handles


def panel_b(ax, sites, positions, neighbour_indices, placed, centre, pad):
    """Zoom on an sp chain: angles annotated against the 155 degree cut."""
    draw_bonds(ax, positions, neighbour_indices, lw=1.4)

    cx, cy = centre
    in_view = sites[
        (sites["x"] > cx - pad) & (sites["x"] < cx + pad)
        & (sites["y"] > cy - pad) & (sites["y"] < cy + pad)
    ]

    for site_class, _, colour, marker, size, zorder in SITE_STYLES:
        rows = in_view[in_view["site_class"] == site_class]
        if rows.empty:
            continue
        ax.scatter(rows["x"], rows["y"], s=size * 2.2, c=colour, marker=marker,
                   zorder=zorder + 2, linewidths=0.6,
                   edgecolors="none" if marker in ("x", "P") else "#33373b")

    # C-C-C angle at every coordination-2 carbon on show. The sp ones are set in
    # bold rather than in colour; the marker already carries the classification.
    for atom_index, row in in_view.iterrows():
        angle = row["ccc_angle_deg"]
        if pd.isna(angle):
            continue
        is_sp = angle >= F.SP_ANGLE_THRESHOLD_DEG
        ax.annotate(
            "{:.0f}°".format(angle), (row["x"], row["y"]),
            xytext=(0, 11), textcoords="offset points", ha="center",
            fontsize=10.5, zorder=10, color="#33373b",
            fontweight="bold" if is_sp else "normal",
        )

    # The axial H on the chain terminus, and the neighbour angle that earned it.
    for atom_index, row in in_view.iterrows():
        if "sp_terminus" not in row["site_class"]:
            continue
        for _, h_position in placed.get(atom_index, []):
            ax.plot([row["x"], h_position[0]], [row["y"], h_position[1]],
                    color=COLOURS["hydrogen"], lw=1.4, zorder=8)
            ax.scatter([h_position[0]], [h_position[1]], s=52,
                       c=COLOURS["hydrogen"], zorder=9, edgecolors="#1b3b4d", linewidths=0.6)
            ax.annotate(
                "H (axial)\nneighbour {:.0f}°".format(row["terminus_neighbour_angle_deg"]),
                (h_position[0], h_position[1]),
                xytext=(6, -14), textcoords="offset points",
                fontsize=10, color=COLOURS["hydrogen"], zorder=10,
            )

    n_sp = int((in_view["ccc_angle_deg"] >= F.SP_ANGLE_THRESHOLD_DEG).sum())

    ax.set_xlim(cx - pad, cx + pad)
    ax.set_ylim(cy - pad, cy + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x / Å")
    ax.set_ylabel("y / Å")
    ax.set_title("(b)", loc="left", fontsize=16, fontweight="bold")

    return n_sp


def panel_c(ax, atom_index, sites, symbols, positions, neighbour_indices,
            bisector, trace, prior_h, pad):
    """
    Zoom on the in-plane tilt sweep at one blocked site.

    Returns the production reason string for the bisector rejection, which is
    reported on stdout rather than written onto the panel.
    """
    parent = positions[atom_index]
    cx, cy = parent[0], parent[1]

    draw_bonds(ax, positions, neighbour_indices, lw=1.4)

    in_view = sites[
        (sites["x"] > cx - pad - 1) & (sites["x"] < cx + pad + 1)
        & (sites["y"] > cy - pad - 1) & (sites["y"] < cy + pad + 1)
    ]
    ax.scatter(in_view["x"], in_view["y"], s=46, c=COLOURS["carbon"],
               zorder=4, edgecolors="none")

    for _, h_position in prior_h:
        if abs(h_position[0] - cx) < pad + 1.5 and abs(h_position[1] - cy) < pad + 1.5:
            ax.scatter([h_position[0]], [h_position[1]], s=34,
                       facecolors="none", edgecolors=COLOURS["hydrogen"],
                       linewidths=1.1, zorder=5)

    rejected = np.array([p for _, p, ok, _ in trace if not ok])
    if len(rejected):
        ax.scatter(rejected[:, 0], rejected[:, 1], s=17, marker="x",
                   c=COLOURS["reject"], alpha=0.45, linewidths=0.9, zorder=6)

    # The bisector is tried first; here it is rejected, and the reason string is
    # the production one, so the caption quotes the distance the run rejected on.
    untilted = parent + F.CH_BOND_LENGTH * bisector
    first_reason = trace[0][3]

    ax.annotate(
        "", xy=(untilted[0], untilted[1]), xytext=(parent[0], parent[1]),
        arrowprops=dict(arrowstyle="->", color=COLOURS["reject"], lw=1.7, ls="--"),
        zorder=7,
    )

    caption = ""
    blocker_xy = None
    interpreted = interpret_rejection(first_reason, prior_h, positions)
    if interpreted is not None:
        blocker_xy, radius, caption = interpreted
        if blocker_xy is not None:
            ax.add_patch(Circle(
                (blocker_xy[0], blocker_xy[1]), radius,
                fill=True, fc=COLOURS["reject"], alpha=0.05, zorder=2,
            ))
            ax.add_patch(Circle(
                (blocker_xy[0], blocker_xy[1]), radius,
                fill=False, ls=":", lw=1.3, ec=COLOURS["reject"], zorder=3,
            ))

    accepted = [(t, p) for t, p, ok, _ in trace if ok]
    if accepted:
        tilt, position = accepted[0]
        ax.annotate(
            "", xy=(position[0], position[1]), xytext=(parent[0], parent[1]),
            arrowprops=dict(arrowstyle="->", color=COLOURS["accept"], lw=2.1),
            zorder=8,
        )
        ax.scatter([position[0]], [position[1]], s=68, c=COLOURS["hydrogen"],
                   zorder=9, edgecolors="#1b3b4d", linewidths=0.7)
        ax.annotate(
            "accepted {:+.1f}°".format(tilt), (position[0], position[1]),
            xytext=(-4, 9), textcoords="offset points", ha="right",
            fontsize=11, fontweight="bold", color=COLOURS["accept"], zorder=10,
        )

        # The accepted H lands essentially on the constraint boundary. Drawing
        # the separation makes the point that the tilt is set by the steric
        # floor, not chosen for looks.
        if blocker_xy is not None:
            separation = float(np.hypot(position[0] - blocker_xy[0],
                                        position[1] - blocker_xy[1]))
            ax.plot([position[0], blocker_xy[0]], [position[1], blocker_xy[1]],
                    color=COLOURS["accept"], lw=1.0, ls="-", alpha=0.9, zorder=7)
            ax.annotate(
                "{:.2f} Å".format(separation),
                (0.5 * (position[0] + blocker_xy[0]), 0.5 * (position[1] + blocker_xy[1])),
                xytext=(0, 5), textcoords="offset points", ha="center",
                fontsize=10, fontweight="bold", color=COLOURS["accept"], zorder=11,
            )

    ax.scatter([parent[0]], [parent[1]], s=95, c=COLOURS["internal"],
               zorder=10, edgecolors="#1f3d1f", linewidths=0.8)
    ax.annotate("C{}".format(atom_index), (parent[0], parent[1]),
                xytext=(-6, -14), textcoords="offset points", ha="right",
                fontsize=10.5, fontweight="bold", zorder=11)

    ax.set_xlim(cx - pad, cx + pad)
    ax.set_ylim(cy - pad, cy + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x / Å")
    ax.set_ylabel("y / Å")
    ax.set_title("(c)", loc="left", fontsize=16, fontweight="bold")

    step = F.H_TILT_ANGLES_DEG_FULL[1] - F.H_TILT_ANGLES_DEG_FULL[0]
    ax.legend(handles=[
        Line2D([], [], ls="--", color=COLOURS["reject"], label="bisector (rejected)"),
        Line2D([], [], ls="", marker="x", color=COLOURS["reject"],
               label="rejected tilt ({:.1f}° steps)".format(step)),
        Line2D([], [], ls="-", color=COLOURS["accept"], label="accepted direction"),
        Line2D([], [], ls=":", color=COLOURS["reject"], label="exclusion radius"),
        Line2D([], [], ls="", marker="o", mfc="none", mec=COLOURS["hydrogen"],
               label="H already placed"),
    ], fontsize=9.5, loc="lower left", framealpha=0.93)

    return caption


def panel_d_angles(ax, angles):
    """C-C-C angle distribution at coordination-2 carbons vs the sp cut."""
    threshold = F.SP_ANGLE_THRESHOLD_DEG
    upper = 182.5

    # The axis stops at the 180 degree data edge; the sp label is short enough to
    # sit centred inside its own band.
    x_max = 181.0

    ax.hist(angles, bins=np.arange(90, upper, 2.5),
            color=COLOURS["internal"], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvspan(threshold, x_max, color=COLOURS["sp"], alpha=0.16, zorder=0)
    ax.axvline(threshold, color=COLOURS["sp"], lw=1.6, zorder=3)

    ax.set_yscale("log")
    ax.set_xlim(88, x_max)
    ax.set_ylim(0.7, ax.get_ylim()[1] * 6)

    n_sp = int((angles >= threshold).sum())
    top = ax.get_ylim()[1]

    ax.annotate(
        "sp / no H\n({} of {})".format(n_sp, len(angles)),
        (0.5 * (threshold + x_max), top), xytext=(0, -34), textcoords="offset points",
        ha="center", va="top", ma="center", fontsize=14,
        color=COLOURS["sp"], fontweight="bold",
    )
    ax.annotate(
        "sp²  edge sites\n1 H each",
        (threshold - 2.0, top), xytext=(0, -34), textcoords="offset points",
        ha="right", va="top", fontsize=14,
        color=COLOURS["internal"], fontweight="bold",
    )
    ax.text(
        threshold, top, " {:.0f}° ".format(threshold),
        ha="center", va="top", fontsize=15, color="white", fontweight="bold",
        zorder=6, bbox=dict(fc=COLOURS["sp"], ec="none", pad=2.2),
    )

    ax.set_xlabel("C–C–C angle at coordination-2 carbons / degrees")
    ax.set_ylabel("count")
    ax.set_title("(a)", loc="left", fontsize=16, fontweight="bold")


def panel_d_areas(ax, areas, n_vertices):
    """Bounded ring-face area distribution vs the cage cut."""
    threshold = F.MAX_CAGE_RING_AREA
    low, high = areas.min() * 0.85, areas.max() * 1.25

    bins = np.logspace(np.log10(low), np.log10(high), 46)
    ax.hist(areas, bins=bins, color=COLOURS["crop"], alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.axvspan(low, threshold, color=COLOURS["reject"], alpha=0.09, zorder=0)
    ax.axvline(threshold, color=COLOURS["reject"], lw=1.6, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(0.7, ax.get_ylim()[1] * 8)

    n_below = int((areas < threshold).sum())
    top = ax.get_ylim()[1]

    ax.annotate(
        "cage – H rejected\n{} of {} faces".format(n_below, len(areas)),
        (threshold * 0.88, top), xytext=(0, -34), textcoords="offset points",
        ha="right", va="top", fontsize=14,
        color=COLOURS["reject"], fontweight="bold",
    )
    ax.annotate(
        "pore\nH allowed", (threshold * 1.15, top),
        xytext=(0, -34), textcoords="offset points", va="top",
        fontsize=14, color="#33373b", fontweight="bold",
    )
    ax.text(
        threshold, top, " {:.0f} Å² ".format(threshold),
        ha="center", va="top", fontsize=15, color="white", fontweight="bold",
        zorder=6, bbox=dict(fc=COLOURS["reject"], ec="none", pad=2.2),
    )

    # Label the ring sizes so the reader can see what the cut actually separates.
    # They sit on top of the bars, so each gets an opaque plate to read against.
    for size in (5, 6, 7, 8):
        subset = areas[n_vertices == size]
        if len(subset) < 20:
            continue
        ax.annotate(
            "{}-ring".format(size), (np.median(subset), 0.95),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#16191c", rotation=90,
            zorder=7,
            bbox=dict(fc="white", ec="#b8bdc2", lw=0.6, alpha=0.88, pad=1.6),
        )

    ax.set_xlabel("bounded ring-face area / Å²")
    ax.set_ylabel("count")
    ax.set_title("(b)", loc="left", fontsize=16, fontweight="bold")


# ----------------------------------------------------------------------------


def pick_tilt_site(sites):
    """
    Pick the site for panel (c): the largest-tilt internal coordination-2 site.

    Restricted to coordination 2 so the panel shows the bisector construction,
    which is the general case; coordination-1 termini have a different rule.
    """
    candidates = sites[
        (sites["site_class"] == "internal_defect_edge_coord2")
        & sites["h_tilt_deg"].notna()
        & (sites["h_tilt_deg"].abs() > 1e-9)
    ]
    if candidates.empty:
        raise SystemExit("No tilted internal coordination-2 site in this structure.")
    return int(candidates["h_tilt_deg"].abs().idxmax())


def pick_sp_chain(sites):
    """Centre panel (b) on the sp carbons; fall back to the widest edge region."""
    sp = sites[sites["site_class"].str.contains("sp_")]
    if sp.empty:
        raise SystemExit("No sp sites in this structure; panel (b) needs one.")
    pad = max(
        ZOOM_PAD,
        0.5 * max(sp["x"].max() - sp["x"].min(), sp["y"].max() - sp["y"].min()) + 1.2,
    )
    return (sp["x"].mean(), sp["y"].mean()), pad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-id", default=DEFAULT_STRUCTURE_ID)
    args = parser.parse_args()

    structure_id = args.structure_id

    sites = pd.read_csv(
        SITE_TABLE_DIR / "{}_saturation_sites.csv".format(structure_id)
    ).set_index("atom_index")

    symbols, positions, neighbour_indices, fragment_atoms, ring_polygons, n_healed = (
        build_bond_graph(structure_id)
    )

    placed = replay_placement(sites, symbols, positions, neighbour_indices, ring_polygons)
    n_verified = verify_replay(placed, sites)
    print("Replayed and verified {} H against the shipped {} geometry."
          .format(n_verified, structure_id))

    tilt_site = pick_tilt_site(sites)
    bisector, trace, prior_h = sweep_trace(
        tilt_site, sites, symbols, positions, neighbour_indices, ring_polygons
    )
    sp_centre, sp_pad = pick_sp_chain(sites)

    rings = pd.read_csv(RING_AREA_TABLE)
    all_angles = pd.concat([
        pd.read_csv(path)["ccc_angle_deg"]
        for path in sorted(SITE_TABLE_DIR.glob("*_saturation_sites.csv"))
    ]).dropna()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: the worked example. (a) is square and needs the height, so it
    # takes the left column across both rows with the two zooms stacked beside
    # it. The left column is twice as wide as the right so that all three
    # equal-aspect panels fill their axes instead of padding themselves out.
    fig1 = plt.figure(figsize=(12.6, 9.2))
    grid1 = fig1.add_gridspec(
        2, 2, width_ratios=[1.92, 1.0], hspace=0.17, wspace=0.10,
        left=0.062, right=0.99, top=0.975, bottom=0.165,
    )

    ax_a = fig1.add_subplot(grid1[:, 0])
    ax_b = fig1.add_subplot(grid1[0, 1])
    ax_c = fig1.add_subplot(grid1[1, 1])

    handles = panel_a(
        ax_a, sites, positions, neighbour_indices,
        zoom_boxes={
            "b": (sp_centre[0], sp_centre[1], sp_pad),
            "c": (positions[tilt_site][0], positions[tilt_site][1], ZOOM_PAD),
        },
    )
    n_sp_in_zoom = panel_b(
        ax_b, sites, positions, neighbour_indices, placed, sp_centre, sp_pad
    )
    caption = panel_c(ax_c, tilt_site, sites, symbols, positions, neighbour_indices,
                      bisector, trace, prior_h, ZOOM_PAD)

    # Shared by (a) and (b), so it sits at figure level under (a) rather than
    # being repeated in both panels, and is set large enough to read at column
    # width.
    fig1.legend(
        handles=handles, fontsize=11, ncol=3,
        loc="lower center", bbox_to_anchor=(0.5, 0.004),
        framealpha=1.0, borderpad=0.7, labelspacing=0.5,
        columnspacing=1.4, handletextpad=0.6,
    )

    # No suptitle: the structure ID is in the filename and the caption carries
    # the description, so the panels get the full figure height.

    stem1 = FIGURE_DIR / "methods_h_site_selection_{}".format(structure_id)
    fig1.savefig(stem1.with_suffix(".png"), dpi=300)
    fig1.savefig(stem1.with_suffix(".pdf"))
    plt.close(fig1)

    # Figure 2: the two threshold calibrations, over all structures rather than
    # this one, so it carries no structure ID in its name.
    fig2 = plt.figure(figsize=(12.6, 5.0))
    grid2 = fig2.add_gridspec(
        1, 2, wspace=0.16,
        left=0.075, right=0.99, top=0.925, bottom=0.165,
    )

    panel_d_angles(fig2.add_subplot(grid2[0, 0]), all_angles.to_numpy())
    panel_d_areas(fig2.add_subplot(grid2[0, 1]),
                  rings["area_A2"], rings["n_vertices"])

    # No suptitle here either: the caption carries the description.

    stem2 = FIGURE_DIR / "methods_h_threshold_calibration"
    fig2.savefig(stem2.with_suffix(".png"), dpi=300)
    fig2.savefig(stem2.with_suffix(".pdf"))
    plt.close(fig2)

    print("Stretched bonds healed in {}: {}".format(structure_id, n_healed))
    print("Fragment atoms deleted        : {}".format(len(fragment_atoms)))
    print("Panel (b) sp carbons in view : {}".format(n_sp_in_zoom))
    print("Panel (c) site               : C{} at {:+.1f} deg".format(
        tilt_site, sites.loc[tilt_site, "h_tilt_deg"]))
    print("Panel (c) bisector rejected  : {}".format(caption or "not a pairwise clash"))
    print("Wrote {}.png and .pdf".format(stem1))
    print("Wrote {}.png and .pdf".format(stem2))


if __name__ == "__main__":
    main()
