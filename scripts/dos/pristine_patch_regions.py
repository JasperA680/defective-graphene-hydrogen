#!/usr/bin/env python3
"""
Select the patch of a cluster that is genuinely bulk graphene, for a local DOS projection.

Why this exists alongside crop_regions.py and define_dos_regions.py
-------------------------------------------------------------------
Those two answer "where is the ribbon" and "follow the same patch through the tearing
sequence". This one answers a different question: *which carbons in this flake are far
enough from everything to stand in for pristine graphene* - so the DOS projected onto
them can be checked against what graphene is supposed to look like (a linear V through
the Dirac point, van Hove shoulders) rather than against the edge-state signal that
dominates the whole-structure plot.

That needs a stricter selection than a rectangle. A carbon qualifies only if it is

  * inside the hand-drawn box for its structure (PATCH_BOXES below),
  * at least `min_edge_hops` C-C bonds from any edge - meaning any carbon that is
    undercoordinated or carries a terminating H, and
  * three-coordinate and a member of hexagons only, never a 5- or 7-ring.

The last condition is not a filter but an assertion: audit_patch() raises if it is
violated. The whole claim of the resulting figure is that the patch is pristine, so it
must not be allowed to degrade quietly into "mostly pristine".

Cluster bonding
---------------
compute_coordination() and heal_stretched_bonds() are called with lattice=None, i.e. in
cluster mode, exactly as scripts/visualisation/plot_n_membered_rings_all.py does. These
crops fill their cell almost exactly, so applying the minimum image makes the rim bond to
its own periodic image: for D10 that invents 9 four-coordinate carbons and would corrupt
both the coordination test and the ring census. The dos_cluster runs are non-periodic
anyway (no lattice_vector lines, 1 k-point), so cluster mode is also the physical truth here.

Atom ordering
-------------
geometry.in order == Mulliken order == the atom axis of the DOS cache, which is what lets
a mask built from coordinates index the .npz directly. Verified rather than assumed, via
crop_regions.check_cache_alignment().
"""

from pathlib import Path
import sys

import numpy as np

from dos_common import ROOT, bfs_hops, read_fhi_aims_geometry

# The bonding, ring and coordination code lives in the hydrogenation module; the DOS
# scripts import each other by bare module name, so that directory goes on the path the
# same way plot_n_membered_rings_all.py does it.
sys.path.insert(0, str(ROOT / "scripts" / "hydrogenation"))

import full_saturate_all_structures as F  # noqa: E402

CLUSTER_CALC_ROOT = ROOT / "calculations/dos_cluster"

# (x_min, x_max, y_min, y_max) in Angstrom, hand-drawn per structure.
#
# D10: the flake spans x 1.5-47.5, y 0.8-45.9, and the tear plus every non-hexagonal
# ring sits low and central - the three pentagons and three heptagons have centroids at
# x 16-31, y 7-16. The top-left quadrant is the largest piece of unbroken sheet, and
# this box covers it while stopping short of the H-terminated rim on all four sides.
# Bond-hop depth inside it reaches 10, the maximum anywhere in the structure.
#
# The box on its own is not the selection: it is the top-left restriction the question
# asked for, and MIN_EDGE_HOPS below is what makes the result bulk. Box alone would keep
# 3 H-terminated carbons on the rim where the box clips the flake edge.
# H02: the flake spans x 1.3-47.6, y 0.8-46.1 and necks down twice, so the sheet is a
# top-left block joined to a bottom-right arm by a narrow ribbon. Every non-hexagonal ring
# sits on that ribbon or the arm - the nearest to this box are the 5|7|5 chain at x 22-26,
# y 28-33 - so the top-left block is the only piece large enough to stand in for bulk. The
# box covers it while stopping short of the H-terminated rim on all four sides, and of that
# 5|7|5 chain on the low-x, low-y corner nearest the tear.
#
# As for D10, the box is the "top left" restriction and MIN_EDGE_HOPS below is what makes
# the result bulk: this box holds 95 carbons, 23 of which are within 2 bonds of a
# termination. The 72 that survive are hexagon-only, three-coordinate and 3-9 bonds deep
# (mean 5.25) - the same count as the D10 patch, so the two pristine references carry equal
# weight per carbon.
#
# The low-y and high-x edges are the tight ones, and neither has room to spare: y_min 28
# instead of 30, or x_max 24 instead of 22, each pulls one carbon of the 5|7|5 chain into
# the selection and audit_patch() rejects the whole patch.
PATCH_BOXES = {
    "D10": (12.0, 28.0, 26.0, 40.0),
    "H02": (4.0, 22.0, 30.0, 45.0),
}

# Bonds from the nearest edge carbon. 3 puts a whole hexagon between every selected atom
# and any terminated or undercoordinated site, which is where the local DOS stops
# carrying edge character. For D10 this drops 10 of the 82 carbons in the box and leaves
# 72, still at mean depth 6.1 bonds.
DEFAULT_MIN_EDGE_HOPS = 3

# Longest C-H bond counted as a termination. The relaxed C-H bonds in these structures
# are 1.08-1.10 A; the next-nearest C to any H is well beyond 2 A.
CH_BOND_CUTOFF = 1.3

PATCH_REGION = "patch"
OUTSIDE_REGION = "outside"


def patch_box(structure_id):
    """The hand-drawn (x_min, x_max, y_min, y_max) box for a structure."""
    if structure_id not in PATCH_BOXES:
        raise SystemExit(
            "No pristine-patch box for {}. Known: {}. Add one to PATCH_BOXES after "
            "checking it against the structure map that panel (a) of "
            "plot_pristine_patch_dos.py draws.".format(
                structure_id, ", ".join(sorted(PATCH_BOXES))
            )
        )
    return PATCH_BOXES[structure_id]


def box_label(box):
    """'x 12-28, y 26-40', for figure titles."""
    return "x {:g}-{:g}, y {:g}-{:g}".format(*box)


def geometry_path(structure_id, calc_root=None):
    calc_root = Path(calc_root) if calc_root is not None else CLUSTER_CALC_ROOT
    path = calc_root / structure_id / "geometry.in"

    if not path.exists():
        raise FileNotFoundError("No geometry for {}: {}".format(structure_id, path))

    return path


def carbon_bonding(symbols, positions):
    """
    Cluster-mode C-C bonding: coordination, neighbour lists and the fragment set.

    Same three calls, in the same order, as plot_n_membered_rings_all.rings_for_geometry(),
    so the bond graph here is the one the ring figures are drawn from.
    """
    coordination, neighbour_indices = F.compute_coordination(symbols, positions, None, None)
    F.heal_stretched_bonds(symbols, positions, coordination, neighbour_indices, None, None)
    fragment_atoms = F.disconnected_fragment_atoms(symbols, neighbour_indices)

    return coordination, neighbour_indices, fragment_atoms


def hydrogenated_carbons(symbols, positions):
    """Indices of carbons carrying a terminating H."""
    carbons = [i for i, s in enumerate(symbols) if s == "C"]
    hydrogens = [i for i, s in enumerate(symbols) if s == "H"]

    if not carbons or not hydrogens:
        return set()

    carbon_positions = positions[carbons]
    terminated = set()

    for h in hydrogens:
        distances = np.linalg.norm(carbon_positions - positions[h], axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] < CH_BOND_CUTOFF:
            terminated.add(carbons[nearest])

    return terminated


def edge_seed_atoms(symbols, positions, coordination):
    """
    The carbons the bulk is measured away from: undercoordinated, or H-terminated.

    Both matter and neither implies the other. A coordination-2 carbon is a bare dangling
    bond; a three-coordinate carbon bonded to an H is a saturated termination. Either way
    it is where the sheet stops being sheet, so both seed the hop count.
    """
    undercoordinated = {
        i for i, s in enumerate(symbols) if s == "C" and coordination[i] < 3
    }
    return undercoordinated | hydrogenated_carbons(symbols, positions)


def edge_hops(symbols, neighbour_indices, seeds):
    """{carbon index: bonds to the nearest edge carbon}, over the C-C bond graph."""
    adjacency = {
        i: list(neighbour_indices[i]) for i, s in enumerate(symbols) if s == "C"
    }
    return bfs_hops(adjacency, sorted(seeds))


def box_mask(positions, box):
    """
    Atoms inside the box, as a plain Cartesian test on the coordinates as written.

    No minimum-image handling, for the reason crop_regions.crop_mask() records: no
    structure in this set has wrapped coordinates. The dos_cluster geometries carry no
    lattice at all, so there is nothing to wrap against either.
    """
    x_min, x_max, y_min, y_max = box

    return (
        (positions[:, 0] >= x_min)
        & (positions[:, 0] <= x_max)
        & (positions[:, 1] >= y_min)
        & (positions[:, 1] <= y_max)
    )


def non_hexagonal_carbons(symbols, positions, neighbour_indices, fragment_atoms):
    """Carbons belonging to any ring that is not a hexagon."""
    rings = F.carbon_ring_cycles(symbols, positions, neighbour_indices,
                                 exclude=fragment_atoms)

    atoms = set()
    for cycle, _polygon, _area in rings:
        if len(cycle) != 6:
            atoms.update(cycle)

    return atoms, rings


def load_structure(structure_id, calc_root=None, min_edge_hops=DEFAULT_MIN_EDGE_HOPS):
    """
    Geometry, bond graph, hop depths and the patch mask, as one dict.

    The mask covers carbons only. Hydrogen is never in the patch by construction - a
    patch carbon is at least `min_edge_hops` bonds from any H-terminated carbon - so
    there is no H selection to make here.
    """
    path = geometry_path(structure_id, calc_root)
    _lattice, symbols, positions = read_fhi_aims_geometry(path)

    coordination, neighbour_indices, fragment_atoms = carbon_bonding(symbols, positions)

    seeds = edge_seed_atoms(symbols, positions, coordination)
    hops = edge_hops(symbols, neighbour_indices, seeds)

    box = patch_box(structure_id)
    inside_box = box_mask(positions, box)

    mask = np.zeros(len(symbols), dtype=bool)
    for i, symbol in enumerate(symbols):
        if symbol != "C" or not inside_box[i]:
            continue
        if hops.get(i, 0) >= min_edge_hops:
            mask[i] = True

    return {
        "structure_id": structure_id,
        "path": path,
        "symbols": symbols,
        "positions": positions,
        "coordination": coordination,
        "neighbour_indices": neighbour_indices,
        "fragment_atoms": fragment_atoms,
        "edge_seeds": seeds,
        "hops": hops,
        "box": box,
        "inside_box": inside_box,
        "min_edge_hops": min_edge_hops,
        "mask": mask,
    }


def patch_rows(structure, include_h=False):
    """
    Rows in the shape region_dos.project() expects, with region 'patch' for the selection.

    Shaped like crop_regions.crop_rows(), but carrying the real edge_hops per atom
    instead of its placeholder zero, so the same rows can be re-filtered by depth
    without recomputing the bond graph.
    """
    rows = []

    for index, symbol in enumerate(structure["symbols"]):
        selected = bool(structure["mask"][index]) and (include_h or symbol != "H")
        rows.append({
            "atom_index": index,
            "element": symbol,
            "region": PATCH_REGION if selected else OUTSIDE_REGION,
            "edge_hops": int(structure["hops"].get(index, 0)),
            # project() only reads this when region == 'ribbon', so it is inert here.
            "ribbon_width_A": None,
        })

    return rows


def audit_patch(structure):
    """
    Describe the selection, and refuse to return if it is not actually pristine.

    Returns a dict of the counts that go into the summary CSV. Raises ValueError if any
    selected carbon is undercoordinated, H-terminated, or in a 5- or 7-ring: those are
    the three ways the patch could stop being pristine, and each would still produce a
    perfectly plausible-looking DOS curve.
    """
    symbols = structure["symbols"]
    positions = structure["positions"]
    coordination = structure["coordination"]
    mask = structure["mask"]

    selected = [i for i in range(len(symbols)) if mask[i]]

    if not selected:
        raise ValueError(
            "{}: the patch is empty. Box {} with min_edge_hops = {} selects no "
            "carbons.".format(
                structure["structure_id"], box_label(structure["box"]),
                structure["min_edge_hops"],
            )
        )

    non_hex, rings = non_hexagonal_carbons(
        symbols, positions, structure["neighbour_indices"], structure["fragment_atoms"]
    )
    terminated = hydrogenated_carbons(symbols, positions)

    bad_coordination = [i for i in selected if coordination[i] < 3]
    bad_ring = [i for i in selected if i in non_hex]
    bad_terminated = [i for i in selected if i in terminated]

    problems = []
    if bad_coordination:
        problems.append("{} carbon(s) with coordination < 3: {}".format(
            len(bad_coordination), bad_coordination[:10]))
    if bad_ring:
        problems.append("{} carbon(s) in a non-hexagonal ring: {}".format(
            len(bad_ring), bad_ring[:10]))
    if bad_terminated:
        problems.append("{} H-terminated carbon(s): {}".format(
            len(bad_terminated), bad_terminated[:10]))

    if problems:
        raise ValueError(
            "{}: the patch is not pristine, so the projection would not mean what the "
            "figure claims. {}. Tighten the box in PATCH_BOXES or raise "
            "--min-edge-hops.".format(structure["structure_id"], "; ".join(problems))
        )

    hops = [structure["hops"].get(i, 0) for i in selected]
    n_carbon_total = sum(1 for s in symbols if s == "C")

    ring_sizes = {}
    for cycle, _polygon, _area in rings:
        ring_sizes[len(cycle)] = ring_sizes.get(len(cycle), 0) + 1

    return {
        "structure_id": structure["structure_id"],
        "x_min": structure["box"][0],
        "x_max": structure["box"][1],
        "y_min": structure["box"][2],
        "y_max": structure["box"][3],
        "min_edge_hops_setting": structure["min_edge_hops"],
        "n_atoms_total": len(symbols),
        "n_C_total": n_carbon_total,
        "n_H_total": sum(1 for s in symbols if s == "H"),
        "n_C_in_box": int((structure["inside_box"] & np.array(
            [s == "C" for s in symbols])).sum()),
        "n_C_in_patch": len(selected),
        "frac_C_in_patch": round(len(selected) / n_carbon_total, 4),
        "min_edge_hops": int(min(hops)),
        "mean_edge_hops": round(float(np.mean(hops)), 2),
        "max_edge_hops": int(max(hops)),
        "max_edge_hops_structure": int(max(structure["hops"].values())),
        "n_patch_C_non_hexagonal": 0,
        "n_patch_C_undercoordinated": 0,
        "n_patch_C_hydrogenated": 0,
        "n_rings_total": len(rings),
        "ring_census": " ".join(
            "{}:{}".format(size, ring_sizes[size]) for size in sorted(ring_sizes)),
    }


def defect_rows(structure, include_h=True):
    """
    The internal contrast selection: the edge and defect carbons of the same structure.

    Region 'edge_defect' covers every H-terminated or undercoordinated carbon, every
    carbon in a non-hexagonal ring, and (with include_h) the terminating hydrogens. This
    is the opposite end of the same flake from the patch, which is what makes it a
    controlled comparison - identical functional, basis, cell and SCF.
    """
    symbols = structure["symbols"]
    positions = structure["positions"]

    non_hex, _rings = non_hexagonal_carbons(
        symbols, positions, structure["neighbour_indices"], structure["fragment_atoms"]
    )
    selected = set(structure["edge_seeds"]) | non_hex

    if include_h:
        selected |= {i for i, s in enumerate(symbols) if s == "H"}

    return [
        {
            "atom_index": index,
            "element": symbol,
            "region": "edge_defect" if index in selected else OUTSIDE_REGION,
            "edge_hops": int(structure["hops"].get(index, 0)),
            "ribbon_width_A": None,
        }
        for index, symbol in enumerate(symbols)
    ]
