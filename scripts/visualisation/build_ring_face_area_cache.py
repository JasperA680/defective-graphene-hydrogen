"""
Enumerate the bounded ring faces of every A-J carbon skeleton and cache their areas.

This is the calibration evidence behind MAX_CAGE_RING_AREA in
scripts/hydrogenation/full_saturate_all_structures.py. That constant decides
whether an in-plane H direction pointing into a ring is a placement inside a
cage (rejected) or into a genuine reconstructed pore (allowed), and the only
honest way to justify the number is to show where it sits in the distribution of
face areas actually present in the structures.

The face enumeration is the production one - carbon_ring_faces() is imported
rather than reimplemented, and is called with the same cluster-mode settings the
saturation run uses: no lattice (so no minimum image), stretched bonds healed,
disconnected fragments excluded. max_area is opened up to infinity here because
the point is to see the whole distribution, not just the sub-threshold part.

Inputs:  <ROOT>/{A..J}/{1..10}/geometry.in
Output:  <ROOT>/results/tables/cluster_ring_face_areas.csv
         columns: structure_id, n_vertices, area_A2, centroid_x, centroid_y

Run from the repository root:
    .venv-dos/bin/python scripts/visualisation/build_ring_face_area_cache.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hydrogenation"))

import full_saturate_all_structures as F  # noqa: E402

OUTPUT_TABLE = ROOT / "results" / "tables" / "cluster_ring_face_areas.csv"


def ring_faces_for_structure(geometry_file):
    """
    Return every bounded carbon ring face of one structure as (area, polygon).

    Faces are keyed by bordering carbon inside carbon_ring_faces(), so the same
    face comes back once per carbon on its perimeter. Rounded vertex tuples are
    used to deduplicate.
    """
    _, symbols, positions = F.read_fhi_aims_geometry(geometry_file)

    coordination, neighbour_indices = F.compute_coordination(
        symbols, positions, None, None
    )
    F.heal_stretched_bonds(symbols, positions, coordination, neighbour_indices, None, None)

    fragment_atoms = F.disconnected_fragment_atoms(symbols, neighbour_indices)

    faces_by_carbon = F.carbon_ring_faces(
        symbols,
        positions,
        neighbour_indices,
        exclude=fragment_atoms,
        max_area=np.inf,
    )

    unique = {}
    for polygons in faces_by_carbon.values():
        for polygon in polygons:
            polygon = np.asarray(polygon, dtype=float)
            key = tuple(map(tuple, np.round(polygon, 4)))
            unique[key] = polygon

    return unique.values()


def main():
    rows = []

    for geometry_file in F.find_all_geometry_files(ROOT):
        structure_id = F.structure_id_from_path(geometry_file)

        for polygon in ring_faces_for_structure(geometry_file):
            centroid = polygon.mean(axis=0)
            rows.append(
                {
                    "structure_id": structure_id,
                    "n_vertices": len(polygon),
                    "area_A2": F.polygon_area(polygon),
                    "centroid_x": centroid[0],
                    "centroid_y": centroid[1],
                }
            )

        print("{}: {} faces".format(structure_id, len(rows)), end="\r", flush=True)

    df = pd.DataFrame(rows)
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_TABLE, index=False)

    n_below = int((df["area_A2"] < F.MAX_CAGE_RING_AREA).sum())

    print()
    print("Structures      : {}".format(df["structure_id"].nunique()))
    print("Bounded faces   : {}".format(len(df)))
    print(
        "Below {:.0f} A^2   : {} ({} at or above)".format(
            F.MAX_CAGE_RING_AREA, n_below, len(df) - n_below
        )
    )
    print("Largest face    : {:.1f} A^2 ({} carbons, {})".format(
        df["area_A2"].max(),
        int(df.loc[df["area_A2"].idxmax(), "n_vertices"]),
        df.loc[df["area_A2"].idxmax(), "structure_id"],
    ))
    print("Wrote {}".format(OUTPUT_TABLE))


if __name__ == "__main__":
    main()
