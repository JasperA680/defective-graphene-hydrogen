#!/usr/bin/env python3
"""
Label every atom of each A-J structure with a DOS projection region.

Motivation
----------
Projecting the Mulliken DOS onto all ~500 atoms buries the interesting states: by the
late frames the sheet has necked down to a narrow hydrogen-terminated carbon strip - the
nanoribbon - of ~40 atoms, whose states are swamped by the ~250 bulk carbons around it.
This script defines which atoms belong to that strip so the DOS can be projected onto it
alone.

Region definition
-----------------
Everything is derived from results/tables/full_saturation_sites/<ID>_saturation_sites.csv,
whose `neighbours` column already holds the healed C-C bond graph used to choose the
hydrogenation sites, so the regions stay consistent with that classification and no
geometry is re-analysed. Its site_class column distinguishes real torn edges
(`internal_defect_edge_coord2`) from atoms merely sliced by the TEM crop frame
(`external_crop_boundary_coord2`) - see scripts/hydrogenation/full_saturate_all_structures.py.
Each has an `*_sp_linear_*` counterpart for the near-linear sp carbons that take no
hydrogen; those are region-equivalent to the class they were split from, so the sp
exclusion does not move any atom between regions.

The ribbon is found by casting a ray from each real edge atom *into the material*:

  1. Each internal defect-edge carbon has an inward normal, the normalised sum of the
     unit vectors to its bonded neighbours (the opposite of the missing-bond direction
     used to place its hydrogen).
  2. Two edge atoms `i` and `j` face each other across the material when they lie within
     RIBBON_MAX_WIDTH of each other and *each* lies inside the other's inward cone. The
     mutual test is what makes this work: two rims facing each other across a void fail
     it, because from either one the other lies opposite the inward normal.
  3. Every carbon inside the slab spanned by such a pair - projecting between them, and
     within RIBBON_LATERAL of the connecting line - is ribbon.

Simpler definitions were tried and do not work. Connected components of the edge atoms
are not rims, because consecutive coordination-2 carbons along an edge are usually
separated by coordination-3 carbons, so the induced subgraph is mostly singletons.
Linking edge atoms within a few bond hops then merges the two facing rims of a narrow
ribbon into one. Purely angular tests fail too: an atom just inside a long straight rim
sees edge atoms in opposite directions *along* that rim and looks two-sided. Only the
inward normal distinguishes "material between two edges" from "material beside one edge".

This degrades gracefully: a frame with no neck simply has an empty ribbon, which is
itself the result that the ribbon forms partway through the tearing sequence.

Remaining carbons are `edge` (on a real tear edge but not in the ribbon),
`crop_boundary` (a TEM crop artefact, never a real edge) or `bulk`. Each hydrogen
inherits the region of the carbon it is bonded to.

Outputs:
    results/tables/dos_regions/<ID>_atom_regions.csv
    results/tables/dos_region_summary.csv

Usage:
    python scripts/dos/define_dos_regions.py
    python scripts/dos/define_dos_regions.py --structure-id J05 --verbose
"""

from pathlib import Path
import argparse
import csv

import numpy as np

from dos_common import (
    ROOT,
    CALC_ROOT,
    bfs_hops,
    minimum_image_matrix,
    read_fhi_aims_geometry,
    ready_structure_ids,
)

# Thickest strip of material that still counts as a ribbon, in Angstrom - roughly three
# hexagon rows. Measured over all 86 structures, the narrowest strip in each one is
# either <= 7.3 A (a real neck) or >= 9.2 A (ordinary graphene that happens to sit
# between two tears), with nothing in between. A 10 A cut sits inside that upper cluster
# and flickers on and off between consecutive frames of the same tearing sequence
# (C01-C03 in, C04-C06 out) purely from sub-Angstrom geometry changes; 8 A sits in the
# empty gap, so the classification is stable and every structure it keeps has a strip
# genuinely narrower than bulk. Per-atom widths are written to the region table, so a
# tighter cut can be applied afterwards without recomputing anything.
RIBBON_MAX_WIDTH = 8.0

# Half-angle of the inward cone within which an opposing edge atom must lie. Generous
# enough to tolerate ragged edges, tight enough that atoms along the same rim (which sit
# perpendicular to the inward normal) are not counted as opposing.
RIBBON_CONE_DEG = 45.0

# How far from the line joining two facing edge atoms a carbon can sit and still count
# as being in that strip.
RIBBON_LATERAL = 2.0

# A facing pair must be joined by material, not by empty space. Walking the segment
# between the two atoms, no sample may be further than this from the nearest carbon.
# A line through the centre of a hexagon sits 1.42 A from its six carbons, so 1.8 A
# accepts any path through intact graphene and rejects one crossing a tear.
RIBBON_VOID_TOLERANCE = 1.8
RIBBON_SAMPLE_STEP = 0.4

# Longest plausible C-H bond; used only to attach each H to its carbon.
CH_BOND_MAX = 1.3

SITES_ROOT = ROOT / "results/tables/full_saturation_sites"
REGIONS_ROOT = ROOT / "results/tables/dos_regions"
SUMMARY_FILE = ROOT / "results/tables/dos_region_summary.csv"
OVERRIDES_FILE = ROOT / "results/tables/dos_region_overrides.csv"

INTERNAL_EDGE = "internal_defect_edge_coord2"
CROP_BOUNDARY = "external_crop_boundary_coord2"

# Coordination-2 carbons that full_saturate_all_structures.py leaves bare because
# they are sp rather than sp2. Not being hydrogenated says nothing about which
# region they belong to, so each is grouped with the non-sp class it shares a
# frame-band side with. The internal ones are real torn-edge atoms - in fact the
# innermost ones in the I and J frames, where they form the chains bounding the
# central tear - so they count as edge here exactly as they did before the sp
# exclusion existed; treating them as bulk would delete the atoms that define the
# ribbon.
SP_LINEAR_EDGE = "internal_sp_linear_coord2"
SP_LINEAR_CROP_BOUNDARY = "external_crop_sp_linear_coord2"

# The coordination-1 carbons on the ends of those same chains, which full_saturate
# _all_structures.py does cap (one H along the chain axis). They are grouped the
# same way and for the same reason: an sp terminus sits at the innermost point of
# the tear, so it is edge, not bulk. Being hydrogenated where its chain is not
# makes no difference to which region it belongs to.
SP_TERMINUS_EDGE = "internal_sp_terminus_coord1"
SP_TERMINUS_CROP_BOUNDARY = "external_crop_sp_terminus_coord1"

INTERNAL_EDGE_CLASSES = (INTERNAL_EDGE, SP_LINEAR_EDGE, SP_TERMINUS_EDGE)
CROP_BOUNDARY_CLASSES = (
    CROP_BOUNDARY, SP_LINEAR_CROP_BOUNDARY, SP_TERMINUS_CROP_BOUNDARY,
)

REGIONS = ["ribbon", "edge", "bulk", "crop_boundary", "unassigned"]


def read_site_table(path):
    """Read the saturation-site CSV into a list of per-carbon dicts."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "atom_index": int(row["atom_index"]),
                "element": row["element"],
                "coordination": int(row["coordination"]),
                "position": np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                "neighbours": [int(v) for v in row["neighbours"].split()],
                "site_class": row["site_class"],
            })
    return rows


def read_overrides(path):
    """Optional manual corrections: structure_id, atom_index, region."""
    if not Path(path).exists():
        return {}

    overrides = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            region = row["region"].strip()
            if region not in REGIONS:
                raise ValueError(
                    "Override for {} atom {} has unknown region {!r}; expected one of {}".format(
                        row["structure_id"], row["atom_index"], region, REGIONS
                    )
                )
            overrides.setdefault(row["structure_id"].strip(), {})[int(row["atom_index"])] = region
    return overrides


def map_site_rows_to_geometry(structure_id, rows, symbols, positions, tol=1e-3):
    """
    Re-index the site table onto the atom order of the DOS calculation.

    The site table indexes carbons by position in the original geometry, while the DOS
    run indexes them by position in calculations/dos/<ID>/geometry.in, and the two are
    not always the same list: E01 had an isolated carbon removed by hand because it
    stopped the SCF converging, which shifts every index after it. Every region-resolved
    projection depends on getting this right, so match carbons by coordinate rather than
    assuming the orders agree. Carbons are frozen during the H relaxation, so their
    coordinates are identical between the two files and the match is exact.

    Returns (rows in DOS order with remapped neighbours, n_dropped, max coordinate offset).
    """
    carbon_positions = np.array([p for p, s in zip(positions, symbols) if s == "C"])
    by_table_index = sorted(rows, key=lambda r: r["atom_index"])
    table_positions = np.array([r["position"] for r in by_table_index])

    if len(carbon_positions) > len(table_positions):
        raise ValueError(
            "{}: the DOS geometry has {} carbons but the site table only has {}; the "
            "table cannot describe atoms that are not in it.".format(
                structure_id, len(carbon_positions), len(table_positions)
            )
        )

    distances = np.linalg.norm(
        carbon_positions[:, None, :] - table_positions[None, :, :], axis=2
    )
    nearest = distances.argmin(axis=1)
    offsets = distances.min(axis=1)

    if offsets.max() > tol:
        raise ValueError(
            "{}: {} DOS carbon(s) have no match in the site table (worst offset "
            "{:.4f} A). Atom indices would not line up, so the region projection would "
            "be meaningless.".format(structure_id, int((offsets > tol).sum()), offsets.max())
        )

    if len(set(nearest.tolist())) != len(nearest):
        raise ValueError(
            "{}: two DOS carbons matched the same site-table row; the coordinates are "
            "degenerate and the mapping is ambiguous.".format(structure_id)
        )

    table_to_dos = {int(t): int(g) for g, t in enumerate(nearest)}

    mapped = []
    for dos_index, table_index in enumerate(nearest):
        row = dict(by_table_index[table_index])
        row["atom_index"] = dos_index
        # Drop bonds to carbons that are not in this calculation.
        row["neighbours"] = [
            table_to_dos[n] for n in row["neighbours"] if n in table_to_dos
        ]
        mapped.append(row)

    return mapped, len(table_positions) - len(carbon_positions), float(offsets.max())


def inward_normals(edge_atoms, by_index, carbon_positions, lattice):
    """
    Unit vector pointing from each real-edge carbon into the material.

    This is the normalised sum of the unit vectors to its bonded neighbours, i.e. the
    opposite of the missing-bond direction along which its hydrogen was placed.
    """
    normals = {}
    for i in edge_atoms:
        neighbours = by_index[i]["neighbours"]
        if not neighbours:
            continue

        deltas = minimum_image_matrix(
            carbon_positions[i][None, :], carbon_positions[neighbours], lattice
        )[0]
        lengths = np.linalg.norm(deltas, axis=1)
        good = lengths > 1e-6
        if not good.any():
            continue

        total = (deltas[good] / lengths[good, None]).sum(axis=0)
        norm = np.linalg.norm(total)
        if norm < 1e-6:
            # Neighbours are symmetric about the atom (e.g. a 2-coordinate carbon in a
            # straight chain); there is no well-defined inward direction.
            continue

        normals[i] = total / norm

    return normals


def segment_crosses_material(start, direction, length, carbon_positions, lattice,
                             tolerance=RIBBON_VOID_TOLERANCE, step=RIBBON_SAMPLE_STEP):
    """
    True if every point along a segment has a carbon nearby.

    The inward-cone test alone can pair two atoms on the same ragged rim whose normals
    happen to converge, joining them by a line that leaves the sheet. Requiring the
    segment to stay in contact with carbon enforces the thing actually meant by ribbon
    width: the two edges are separated by material, not by a tear.
    """
    n_samples = max(int(np.ceil(length / step)), 2)
    offsets = np.linspace(0.0, length, n_samples)[:, None] * direction[None, :]
    samples = start[None, :] + offsets

    gaps = np.linalg.norm(
        minimum_image_matrix(samples, carbon_positions, lattice), axis=2
    ).min(axis=1)

    return bool(gaps.max() <= tolerance)


def find_ribbon_atoms(edge_atoms, normals, carbon_positions, lattice,
                      max_width, cone_deg, lateral):
    """
    Carbons lying in a strip between two real edges that face each other.

    Returns ({atom index: narrowest containing strip width}, list of (i, j, width) pairs).
    Recording the width per atom rather than a bare membership set means the threshold
    can be tightened afterwards from the region table, without recomputing anything.
    """
    cone_cos = np.cos(np.radians(cone_deg))

    facing = []
    indexed = [i for i in edge_atoms if i in normals]

    if len(indexed) >= 2:
        deltas = minimum_image_matrix(
            carbon_positions[indexed], carbon_positions[indexed], lattice
        )
        distances = np.linalg.norm(deltas, axis=2)

        for a, i in enumerate(indexed):
            for b in range(a + 1, len(indexed)):
                j = indexed[b]
                d = distances[a, b]
                if d < 1e-6 or d > max_width:
                    continue

                u = deltas[a, b] / d
                # j must lie inside i's inward cone and i inside j's: two rims facing
                # each other across a void fail this, because from either one the other
                # lies opposite the inward normal.
                if np.dot(u, normals[i]) < cone_cos:
                    continue
                if np.dot(-u, normals[j]) < cone_cos:
                    continue
                if not segment_crosses_material(
                    carbon_positions[i], u, d, carbon_positions, lattice
                ):
                    continue

                facing.append((i, j, float(d)))

    ribbon = {}
    all_indices = np.arange(len(carbon_positions))
    for i, j, d in facing:
        u = minimum_image_matrix(
            carbon_positions[i][None, :], carbon_positions[j][None, :], lattice
        )[0, 0] / d

        rel = minimum_image_matrix(
            carbon_positions[i][None, :], carbon_positions, lattice
        )[0]
        along = rel @ u
        lateral_dist = np.linalg.norm(rel - along[:, None] * u[None, :], axis=1)

        inside = (along >= -1e-6) & (along <= d + 1e-6) & (lateral_dist <= lateral)
        for k in all_indices[inside]:
            k = int(k)
            ribbon[k] = min(ribbon.get(k, np.inf), d)

    return ribbon, facing


def classify_structure(structure_id, calc_root, sites_root,
                       max_width=RIBBON_MAX_WIDTH, cone_deg=RIBBON_CONE_DEG,
                       lateral=RIBBON_LATERAL, overrides=None):
    """Assign a region to every atom of one structure."""
    sites_file = Path(sites_root) / "{}_saturation_sites.csv".format(structure_id)
    geom_file = Path(calc_root) / structure_id / "geometry.in"

    if not sites_file.exists():
        raise FileNotFoundError("No site table for {}: {}".format(structure_id, sites_file))
    if not geom_file.exists():
        raise FileNotFoundError("No DOS geometry for {}: {}".format(structure_id, geom_file))

    rows = read_site_table(sites_file)
    lattice, symbols, positions = read_fhi_aims_geometry(geom_file)
    rows, n_dropped, offset = map_site_rows_to_geometry(
        structure_id, rows, symbols, positions
    )

    by_index = {r["atom_index"]: r for r in rows}
    adjacency = {r["atom_index"]: r["neighbours"] for r in rows}

    n_carbons = len(rows)
    carbon_positions = positions[:n_carbons]

    edge_atoms = [
        i for i, r in by_index.items() if r["site_class"] in INTERNAL_EDGE_CLASSES
    ]

    # Bond-hop distance to the nearest real tear edge. Not used to define the ribbon,
    # but a useful per-atom diagnostic and the natural handle for edge-shell projections.
    edge_hops = bfs_hops(adjacency, edge_atoms)

    normals = inward_normals(edge_atoms, by_index, carbon_positions, lattice)
    ribbon, facing = find_ribbon_atoms(
        edge_atoms, normals, carbon_positions, lattice, max_width, cone_deg, lateral,
    )

    region = {}
    for index, row in by_index.items():
        if index in ribbon:
            region[index] = "ribbon"
        elif row["site_class"] in INTERNAL_EDGE_CLASSES:
            region[index] = "edge"
        elif row["site_class"] in CROP_BOUNDARY_CLASSES:
            region[index] = "crop_boundary"
        else:
            region[index] = "bulk"

    # Hydrogens inherit from the carbon they passivate.
    bonded_c = {}
    if len(symbols) > n_carbons:
        h_to_c = np.linalg.norm(
            minimum_image_matrix(positions[n_carbons:], carbon_positions, lattice), axis=2
        )
        for offset_index, index in enumerate(range(n_carbons, len(symbols))):
            nearest = int(h_to_c[offset_index].argmin())

            if h_to_c[offset_index, nearest] > CH_BOND_MAX:
                region[index] = "unassigned"
                bonded_c[index] = -1
            else:
                region[index] = region[nearest]
                bonded_c[index] = nearest

    if overrides:
        for index, forced in overrides.items():
            if index not in region:
                raise ValueError(
                    "{}: override targets atom {} which does not exist".format(structure_id, index)
                )
            region[index] = forced

    records = []
    for index in range(len(symbols)):
        row = by_index.get(index)
        records.append({
            "atom_index": index,
            "element": symbols[index],
            "x": positions[index][0],
            "y": positions[index][1],
            "z": positions[index][2],
            "site_class": row["site_class"] if row else "",
            "coordination": row["coordination"] if row else -1,
            "edge_hops": edge_hops.get(index, -1) if row else -1,
            "ribbon_width_A": round(ribbon[index], 3) if index in ribbon else "",
            "bonded_c": bonded_c.get(index, -1),
            "region": region[index],
        })

    widths = [d for _, _, d in facing]

    return records, facing, {
        "structure_id": structure_id,
        "n_atoms": len(symbols),
        "n_C": n_carbons,
        "n_H": len(symbols) - n_carbons,
        "n_C_dropped_vs_site_table": n_dropped,
        "n_edge_atoms": len(edge_atoms),
        "n_facing_pairs": len(facing),
        "min_ribbon_width_A": round(min(widths), 3) if widths else "",
        "median_ribbon_width_A": round(float(np.median(widths)), 3) if widths else "",
        "max_coord_offset_A": offset,
    }


def write_region_table(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["atom_index", "element", "x", "y", "z", "site_class", "coordination",
              "edge_hops", "ribbon_width_A", "bonded_c", "region"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def summarise(records, meta):
    """Per-region atom counts, split by element, plus the C sum rule."""
    row = dict(meta)

    for region in REGIONS:
        row["C_" + region] = sum(
            1 for r in records if r["region"] == region and r["element"] == "C"
        )
        row["H_" + region] = sum(
            1 for r in records if r["region"] == region and r["element"] == "H"
        )

    carbon_total = sum(row["C_" + region] for region in REGIONS)
    if carbon_total != row["n_C"]:
        raise ValueError(
            "{}: regions cover {} carbons but the structure has {}".format(
                meta["structure_id"], carbon_total, row["n_C"]
            )
        )

    row["ribbon_frac_C"] = row["C_ribbon"] / row["n_C"] if row["n_C"] else 0.0
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Assign DOS projection regions (ribbon / edge / bulk / crop) to every atom."
    )
    parser.add_argument("--calc-root", default=str(CALC_ROOT))
    parser.add_argument("--sites-root", default=str(SITES_ROOT))
    parser.add_argument("--regions-root", default=str(REGIONS_ROOT))
    parser.add_argument("--summary", default=str(SUMMARY_FILE))
    parser.add_argument("--overrides", default=str(OVERRIDES_FILE),
                        help="Optional CSV of manual region corrections.")
    parser.add_argument("--max-width", type=float, default=RIBBON_MAX_WIDTH,
                        help="Widest strip counted as ribbon, in A (default {}).".format(
                            RIBBON_MAX_WIDTH))
    parser.add_argument("--cone-deg", type=float, default=RIBBON_CONE_DEG,
                        help="Inward cone half-angle in degrees (default {}).".format(
                            RIBBON_CONE_DEG))
    parser.add_argument("--lateral", type=float, default=RIBBON_LATERAL,
                        help="Lateral tolerance about a facing pair, in A (default {}).".format(
                            RIBBON_LATERAL))
    parser.add_argument("--structure-id", default=None, help="Process only this structure.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    overrides = read_overrides(args.overrides)
    if overrides:
        print("Loaded manual overrides for {} structure(s) from {}".format(
            len(overrides), args.overrides))

    if args.structure_id:
        structure_ids = [args.structure_id]
    else:
        structure_ids = ready_structure_ids(args.calc_root)

    summary_rows = []
    for structure_id in structure_ids:
        records, _facing, meta = classify_structure(
            structure_id, args.calc_root, args.sites_root,
            max_width=args.max_width, cone_deg=args.cone_deg, lateral=args.lateral,
            overrides=overrides.get(structure_id),
        )

        out_file = Path(args.regions_root) / "{}_atom_regions.csv".format(structure_id)
        write_region_table(out_file, records)

        row = summarise(records, meta)
        summary_rows.append(row)

        print("{:<5} {:>4} C {:>3} H | ribbon {:>3} C {:>3} H ({:>4.1%})  edge {:>3}  "
              "bulk {:>3}  crop {:>3} | {:>3} pairs, width {}".format(
                  structure_id, row["n_C"], row["n_H"],
                  row["C_ribbon"], row["H_ribbon"], row["ribbon_frac_C"], row["C_edge"],
                  row["C_bulk"], row["C_crop_boundary"], row["n_facing_pairs"],
                  "{} - {} A".format(row["min_ribbon_width_A"], row["median_ribbon_width_A"])
                  if row["n_facing_pairs"] else "-"))

        if args.verbose:
            print("    coordinate check: max offset {:.2e} A".format(row["max_coord_offset_A"]))

    if summary_rows:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        print()
        print("Wrote {} structures to {}".format(len(summary_rows), summary_path))

        with_ribbon = sum(1 for r in summary_rows if r["C_ribbon"] > 0)
        print("{} of {} structures have a non-empty ribbon.".format(with_ribbon, len(summary_rows)))


if __name__ == "__main__":
    main()
