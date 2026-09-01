#!/usr/bin/env python3
"""
A fixed rectangular crop per structure family, used to project the DOS onto one patch.

Why this exists alongside define_dos_regions.py
-----------------------------------------------
define_dos_regions.py finds the nanoribbon geometrically, from facing edge pairs, so it
only reports a ribbon once a narrow neck actually exists: across the finished structures
it finds ribbon carbons in the I and J families and essentially none in A-H. That is the
right answer to "where is the ribbon now", but it is the wrong tool for following the
*same patch of sheet* through the tearing sequence, because the selection changes shape
as the tear develops.

The boxes below are hand-chosen per family for exactly that purpose: one rectangle
anchored at the origin covering the area that becomes the nanoribbon, applied unchanged
to every site of that family (A01 ... A10 all use the A box). Projecting the DOS onto the
carbons inside it gives a local signal that is comparable across the whole A -> J series.

The crop is an additional, independent selection - it does not replace or modify the
ribbon/edge/bulk classification, and the two are cross-referenced in the region maps.

Atom ordering
-------------
geometry.in order == Mulliken order == the atom axis of the DOS cache. That is what lets
a mask built from coordinates index results/tables/dos_atom_resolved/<ID>.npz directly.
It is asserted rather than assumed - see check_cache_alignment().
"""

from pathlib import Path

import numpy as np

from dos_common import ROOT, CALC_ROOT, read_fhi_aims_geometry

# (x_limit, y_limit) in Angstrom; every box starts at (0, 0). Supplied by hand to track
# the patch of sheet that becomes the nanoribbon, not derived from the geometry.
CROP_BOXES = {
    "A": (34.0, 18.0),
    "B": (33.0, 20.0),
    "C": (31.0, 24.0),
    "D": (31.0, 24.0),
    "E": (31.0, 26.0),
    "F": (32.0, 28.0),
    "G": (31.0, 28.0),
    "H": (30.0, 28.0),
    "I": (31.0, 24.0),
    "J": (32.0, 24.0),
}

# Structures with no DFT run still get a region map, from the geometry that was staged.
RELAXED_ROOT = ROOT / "data/h_saturated/relaxed"

CROP_REGION = "crop"
OUTSIDE_REGION = "outside"


def crop_box(structure_id):
    """The (x_limit, y_limit) box for a structure, from its family letter."""
    letter = structure_id[0].upper()

    if letter not in CROP_BOXES:
        raise KeyError(
            "No crop box for family {!r} (structure {}). Known families: {}.".format(
                letter, structure_id, ", ".join(sorted(CROP_BOXES))
            )
        )

    return CROP_BOXES[letter]


def box_label(box):
    """'(0,0)-(34,18)', for figure titles."""
    return "(0,0)-({:g},{:g})".format(*box)


def geometry_path(structure_id, calc_root=None):
    """
    The geometry to crop: the DFT input if there is one, else the staged relaxed geometry.

    Nine structures have no DOS calculation but do have a relaxed geometry, and they
    should still appear in the visual check so a family's contact sheet has no holes.
    """
    calc_root = Path(calc_root) if calc_root is not None else CALC_ROOT

    path = calc_root / structure_id / "geometry.in"
    if path.exists():
        return path, True

    fallback = RELAXED_ROOT / structure_id / "geometry.in"
    if fallback.exists():
        return fallback, False

    raise FileNotFoundError(
        "No geometry for {}: tried {} and {}.".format(structure_id, path, fallback)
    )


def crop_mask(positions, box):
    """
    Boolean mask over atom order for the atoms inside the box.

    A plain Cartesian test on the coordinates as written in geometry.in. The cell is
    non-orthogonal, but no structure has wrapped coordinates - every atom sits in
    x 1.9-46.6, y 1.9-45.4 - so there is no minimum-image subtlety here.
    """
    positions = np.asarray(positions, dtype=float)
    x_limit, y_limit = box

    return (
        (positions[:, 0] >= 0.0)
        & (positions[:, 0] <= x_limit)
        & (positions[:, 1] >= 0.0)
        & (positions[:, 1] <= y_limit)
    )


def load_structure(structure_id, calc_root=None):
    """Geometry plus the crop mask, as one dict."""
    path, has_dos = geometry_path(structure_id, calc_root)
    lattice, symbols, positions = read_fhi_aims_geometry(path)

    box = crop_box(structure_id)
    mask = crop_mask(positions, box)

    return {
        "structure_id": structure_id,
        "path": path,
        "has_dos": has_dos,
        "lattice": lattice,
        "symbols": symbols,
        "positions": positions,
        "box": box,
        "mask": mask,
    }


def crop_rows(structure_id, calc_root=None, include_h=False, structure=None):
    """
    Rows in the shape region_dos.project() expects, with region 'crop' inside the box.

    Hydrogens are left outside the crop by default: the projection asked for is C s and
    C p only, and zeroing h_s this way makes render_dos_plot() draw the two carbon
    channels and nothing else, rather than needing a special case downstream.
    """
    structure = structure or load_structure(structure_id, calc_root)

    rows = []
    for index, (symbol, inside) in enumerate(zip(structure["symbols"], structure["mask"])):
        selected = bool(inside) and (include_h or symbol != "H")
        rows.append({
            "atom_index": index,
            "element": symbol,
            "region": CROP_REGION if selected else OUTSIDE_REGION,
            # region_dos.read_region_table supplies these; project() only reads
            # ribbon_width_A when region == 'ribbon', so the values are inert here.
            "edge_hops": 0,
            "ribbon_width_A": None,
        })

    return rows


def check_cache_alignment(cache, symbols, structure_id):
    """
    Fail loudly if the DOS cache and the geometry disagree about the atom order.

    region_dos.region_mask() only bounds-checks the atom index. The crop mask comes from
    a different file than the cache, so the correspondence is verified explicitly: a
    silent mismatch would project the DOS onto the wrong atoms and still produce a
    plausible-looking plot.
    """
    cached = list(cache["symbols"])

    if len(cached) != len(symbols):
        raise ValueError(
            "{}: DOS cache has {} atoms but geometry.in has {}.".format(
                structure_id, len(cached), len(symbols)
            )
        )

    for index, (a, b) in enumerate(zip(cached, symbols)):
        if a != b:
            raise ValueError(
                "{}: atom {} is {} in the DOS cache but {} in geometry.in; the cache and "
                "the geometry are not in the same order.".format(structure_id, index, a, b)
            )
