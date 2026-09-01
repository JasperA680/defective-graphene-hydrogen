#!/usr/bin/env python3
"""
Helpers shared by the DOS post-processing scripts in this directory.

These were originally defined in plot_dos_movie_frames.py; cache_atom_resolved_dos.py
and the region scripts need the same structure ordering and readiness checks, so they
live here rather than being copied.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CALC_ROOT = ROOT / "calculations/dos"


# read_fhi_aims_geometry and minimum_image_displacement are re-implemented here rather
# than imported from scripts/structure_prep or scripts/hydrogenation: every copy over
# there lives in a module that imports pandas at module scope, and the DOS environment
# (.venv-dos) has numpy/scipy/matplotlib/ase only.

def read_fhi_aims_geometry(path):
    """
    Read lattice vectors, symbols and positions from an FHI-aims geometry.in.

    A cluster geometry carries no lattice_vector lines, and lattice comes back as
    None for it. The minimum-image helpers below accept that and fall through to
    plain Cartesian differences, so callers do not need to branch: passing the
    lattice straight through from here does the right thing for both.
    """
    lattice = []
    symbols = []
    positions = []

    for line in Path(path).read_text().splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()

        if parts[0] == "lattice_vector":
            lattice.append([float(parts[1]), float(parts[2]), float(parts[3])])

        elif parts[0] == "atom":
            positions.append([float(v) for v in parts[1:4]])
            symbols.append(parts[4])

    return (
        np.array(lattice, dtype=float) if lattice else None,
        symbols,
        np.array(positions, dtype=float),
    )


def minimum_image_displacement(ri, rj, lattice):
    """
    Displacement rj - ri under the minimum-image convention.

    lattice=None means the structure is a cluster: the plain Cartesian
    difference is returned. Applying the minimum image to these crops is exactly
    the artefact the cluster treatment removes - they fill their cell almost
    exactly, so the rim otherwise bonds to its own image.
    """
    delta = np.asarray(rj, dtype=float) - np.asarray(ri, dtype=float)

    if lattice is None:
        return delta

    frac = np.linalg.solve(np.asarray(lattice, dtype=float).T, delta)
    frac -= np.round(frac)
    return np.asarray(lattice, dtype=float).T @ frac


def bfs_hops(adjacency, sources, max_hops=None):
    """
    Bond-hop distance from a set of source atoms over an adjacency mapping.

    Returns {atom_index: hops}; atoms not reached (or beyond max_hops) are absent.
    """
    hops = {s: 0 for s in sources if s in adjacency}
    frontier = list(hops)
    depth = 0

    while frontier and (max_hops is None or depth < max_hops):
        depth += 1
        nxt = []
        for node in frontier:
            for neighbour in adjacency[node]:
                if neighbour not in hops:
                    hops[neighbour] = depth
                    nxt.append(neighbour)
        frontier = nxt

    return hops


def minimum_image_matrix(positions_a, positions_b, lattice):
    """
    All minimum-image displacements from each of positions_a to each of positions_b.

    Returns an (len(a), len(b), 3) array. Vectorised equivalent of calling
    minimum_image_displacement() over every pair, and takes lattice=None for a
    cluster on the same terms.
    """
    delta = np.asarray(positions_b)[None, :, :] - np.asarray(positions_a)[:, None, :]

    if lattice is None:
        return delta

    lattice = np.asarray(lattice, dtype=float)

    frac = delta @ np.linalg.inv(lattice.T).T
    frac -= np.round(frac)
    return frac @ lattice


def ordered_structure_ids():
    """Every A-J structure ID in tearing-sequence order (A01 ... I10, J01 ... J05)."""
    ids = []
    for letter in "ABCDEFGHI":
        for number in range(1, 11):
            ids.append("{}{:02d}".format(letter, number))
    for number in range(1, 6):
        ids.append("J{:02d}".format(number))
    return ids


def structure_ready(calc_dir):
    """True if this structure has a finished aims.out plus a Mulliken.out."""
    calc_dir = Path(calc_dir)
    aims_file = calc_dir / "aims.out"
    mulliken_file = calc_dir / "Mulliken.out"

    if not aims_file.exists() or not mulliken_file.exists():
        return False

    with open(aims_file) as f:
        return "Have a nice day" in f.read()


def ready_structure_ids(calc_root=None):
    """
    Structure IDs with a finished DFT run, in sequence order.

    G07 and D08 are staged but never finished (no Mulliken.out), so they drop out here
    rather than being excluded by name.
    """
    calc_root = Path(calc_root) if calc_root is not None else CALC_ROOT
    return [s for s in ordered_structure_ids() if structure_ready(calc_root / s)]


def strided_chunk(structure_ids, index, num_chunks):
    """
    The `index`-th of `num_chunks` strided chunks of structure_ids.

    Strided rather than sliced: the A/B structures are the largest and slowest, so
    slicing would pile them all into the first array task.
    """
    if not 0 <= index < num_chunks:
        raise SystemExit(
            "--index {} is out of range for --num-chunks {} (0-{}).".format(
                index, num_chunks, num_chunks - 1
            )
        )
    return structure_ids[index::num_chunks]
