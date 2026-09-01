#!/usr/bin/env python3
"""
Relax the hydrogens of the non-periodic cluster structures with GFN2-xTB.

This is the cluster counterpart of scripts/mlip/relax_saturated_structures.py and
keeps its shard/merge/resume/audit machinery unchanged. Two things differ:

    1. The calculator is GFN2-xTB (via xtb_ase) instead of MACE. GFN2 is an
       actual electronic-structure method, so it handles the radical / open-shell
       character of a torn graphene edge sensibly, and it hands back molecular
       orbital energies, which gives the HOMO-LUMO gap of every frame for free.
       That gap is what decides whether the FHI-aims run for that frame needs
       `spin collinear` rather than `spin none`.
    2. There are two relaxation modes, selected with --mode:
         planar - H relax in x and y only, held in the sheet plane (the same
                  constraint set as the MACE runs).
         free   - H relax in all three directions.
       The carbon backbone is frozen at the experimental TEM coordinates in both,
       so the geometry stays tied to the measurement. Comparing
       max_H_out_of_plane_A between the two modes tests whether the planar
       assumption actually holds.

Two properties of xtb_ase drive the rest of the design:

    - It is a file-I/O calculator: it writes an input, runs the xtb executable,
      and parses the output. Every optimizer step is therefore a fresh process
      and a fresh SCF, with no wavefunction reuse. Cost scales with the *number*
      of steps, which is why this uses LBFGS rather than the FIRE used for MACE -
      FIRE is robust but step-hungry, and the placed geometry is already a decent
      guess.
    - Its working files are written into the calculator's `directory`. A SLURM
      array running 16 tasks at once would have them all writing xtb.out into the
      same place, so each structure gets its own scratch directory - the same
      reasoning as the per-structure CSV shards.

Inputs:
    data/h_saturated/cluster_full_saturation/<ID>/geometry.in

Outputs:
    data/h_saturated/cluster_relaxed_<mode>/<ID>/{geometry.in,relaxation.traj,relaxation.log}
    results/tables/cluster_relaxation_<mode>_parts/<ID>.csv   (one row per structure)
    results/tables/cluster_relaxation_<mode>_summary.csv      (merged)

Example usage:
    # One structure
    python scripts/mlip/relax_clusters_xtb.py --mode planar --structure-id J05

    # One array task (implies --no-merge)
    python scripts/mlip/relax_clusters_xtb.py --mode planar --index $SLURM_ARRAY_TASK_ID

    # Merge the shards once the array has finished
    python scripts/mlip/relax_clusters_xtb.py --mode planar --merge

Requires `source load_xtb_modules.sh` for the xtb executable and the .venv-xtb
environment.
"""

from pathlib import Path
from subprocess import check_call
import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd

from ase.constraints import FixAtoms, FixCartesian
from ase.io import read, write
from ase.optimize import LBFGS


ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

METHOD = "gfn2-xtb"

FMAX = 0.05          # eV / Angstrom
MAX_STEPS = 300

# Hard-clash thresholds, matching relax_saturated_structures.py. Deliberately
# looser than the thresholds used to *place* the H: the relaxation is allowed to
# bring H closer than placement would permit, and these are the distances at
# which a contact is physically wrong rather than merely tight.
MIN_HH_DISTANCE = 0.80
MIN_NONBONDED_HC_DISTANCE = 0.90

CC_CUTOFF_STRETCHED = 2.05
CH_BOND_MAX = 1.35

# GFN2 valence electrons per element, used to locate the HOMO from the orbital
# energies. cclib's own `homos` attribute does not parse reliably from xtb
# output, so the occupied count is derived from the composition instead.
VALENCE_ELECTRONS = {"C": 4, "H": 1}

SUMMARY_COLUMNS = [
    "structure_id",
    "mode",
    "n_atoms",
    "n_C",
    "n_H",
    "energy_initial_eV",
    "energy_final_eV",
    "delta_energy_eV",
    "fmax_initial_eV_per_A",
    "fmax_final_eV_per_A",
    "n_steps",
    "converged",
    "homo_lumo_gap_eV",
    "n_valence_electrons",
    "odd_electron_count",
    "min_CH_bond_A",
    "max_CH_bond_A",
    "min_HH_distance_A",
    "min_HH_distance_initial_A",
    "min_HC_nonbonded_A",
    "max_C_displacement_A",
    "max_H_displacement_A",
    "max_H_out_of_plane_A",
    "n_H_desorbed",
    "n_C_under_coordinated",
    "elapsed_s",
    "input_geometry",
    "output_geometry",
    "method",
    "status",
    "error",
]


def relpath(path):
    """Return a project-relative path when possible."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------

def make_profile():
    """
    Build an XTBProfile that writes xtb.out into the calculation directory.

    xtb_ase 0.0.1 opens the log file by bare name in the *calling process's*
    working directory while running xtb with cwd=directory, so with any
    directory other than "." the output lands in one place and the parser looks
    for it in another, and every calculation fails with FileNotFoundError. Since
    a per-structure directory is exactly what keeps concurrent array tasks from
    clobbering each other, this fix is required rather than cosmetic.
    """
    from xtb_ase import XTBProfile

    class DirectoryAwareXTBProfile(XTBProfile):
        def run(self, directory, input_filename, geom_filename, output_filename):
            cmd = [str(arg) for arg in self.argv] + [
                "--input", input_filename, geom_filename, "--json",
            ]

            with open(Path(directory) / output_filename, "w") as handle:
                check_call(cmd, stdout=handle, cwd=directory)

    return DirectoryAwareXTBProfile()


def make_calculator(scratch_dir):
    """Build the GFN2-xTB calculator for one structure."""
    from xtb_ase import XTB

    scratch_dir.mkdir(parents=True, exist_ok=True)

    return XTB(profile=make_profile(), method=METHOD, directory=str(scratch_dir))


def homo_lumo_gap(atoms, scratch_dir):
    """
    Return (gap_eV, n_valence_electrons) from the last GFN2 calculation.

    Read from xtb's own xtbout.json rather than from the ASE `attributes`.
    cclib parses the orbital table printed in xtb.out, but xtb prints only an
    abbreviated window around the frontier with the middle elided as "...", so
    that array is not the full spectrum and indexing it by the true HOMO position
    silently returns an unrelated pair of levels. For J05 that produced 8.5 eV
    against a real gap of 0.002 eV - not a small error but a meaningless number,
    and it feeds the spin decision in stage_cluster_dos_calculations.py, so it
    has to come from the authoritative source.

    The electron count is still derived from the composition, and is checked
    against the count xtb reports so a disagreement cannot pass unnoticed.
    """
    n_electrons = sum(
        VALENCE_ELECTRONS.get(symbol, 0) for symbol in atoms.get_chemical_symbols()
    )

    json_file = Path(scratch_dir) / "xtbout.json"

    if not json_file.exists():
        return None, n_electrons

    with open(json_file) as handle:
        data = json.load(handle)

    reported = data.get("number of electrons")

    if reported is not None and int(reported) != n_electrons:
        print(
            "  WARNING: valence-electron count {} from composition disagrees with "
            "xtb's {}; using xtb's.".format(n_electrons, int(reported))
        )
        n_electrons = int(reported)

    gap = data.get("HOMO-LUMO gap / eV")

    return (None if gap is None else float(gap)), n_electrons


# ---------------------------------------------------------------------
# Structure discovery
# ---------------------------------------------------------------------

def all_structure_ids(input_root):
    """Every structure with a cluster geometry, sorted so A01 < A02 < ... < J05."""
    ids = sorted(path.parent.name for path in input_root.glob("*/geometry.in"))

    if not ids:
        raise RuntimeError(
            "No cluster geometries under {}. Run "
            "scripts/hydrogenation/full_saturate_all_structures.py --cluster "
            "first.".format(relpath(input_root))
        )

    return ids


# ---------------------------------------------------------------------
# Per-structure result shards
# ---------------------------------------------------------------------

def shard_file(parts_root, structure_id):
    return parts_root / "{}.csv".format(structure_id)


def write_shard(parts_root, row):
    """Write one structure's result as its own CSV. Array-safe: no shared file."""
    parts_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=SUMMARY_COLUMNS).to_csv(
        shard_file(parts_root, row["structure_id"]), index=False
    )


def shard_succeeded(parts_root, structure_id):
    """True if this structure already has a successful shard on disk."""
    path = shard_file(parts_root, structure_id)

    if not path.exists():
        return False

    try:
        existing = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return False

    if "status" not in existing.columns or len(existing) == 0:
        return False

    return existing["status"].astype(str).str.lower().eq("success").any()


def merge_shards(parts_root, summary_file, all_ids):
    """Concatenate the per-structure shards into the master summary CSV."""
    frames = []

    for structure_id in all_ids:
        path = shard_file(parts_root, structure_id)

        if not path.exists():
            continue

        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            print("Warning: empty shard {}".format(relpath(path)))

    if not frames:
        print("No shards found under {}; nothing to merge.".format(relpath(parts_root)))
        return None

    summary = pd.concat(frames, ignore_index=True)
    summary = summary.sort_values("structure_id").reset_index(drop=True)

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_file, index=False)

    print("Merged {} shard(s) -> {}".format(len(summary), relpath(summary_file)))

    return summary


# ---------------------------------------------------------------------
# Geometry audit
# ---------------------------------------------------------------------

def audit_geometry(atoms, initial_positions):
    """
    Post-relaxation geometry checks.

    All distances are plain Cartesian (mic=False): these are clusters, and the
    minimum image is exactly the artefact the cluster treatment removes.

    The nearest carbon to each H is taken to be its parent (the C-H bond), so the
    closest non-bonded H-C contact is the *second* nearest carbon.

    max_C_displacement_A must be 0 if FixAtoms held the backbone.
    max_H_out_of_plane_A must be 0 in planar mode if FixCartesian held the H in
    the sheet plane, and is the physically interesting number in free mode.
    Both are regression checks on the constraints first, physics second.

    n_C_under_coordinated should be 0 for a cluster. Unlike the periodic run,
    where crop-boundary and coordination-0/1 carbons were deliberately left bare
    and a non-zero count was expected, every carbon here is supposed to have been
    terminated, so any non-zero value means the cluster still carries a
    dangling-bond radical.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    h_idx = np.where(symbols == "H")[0]
    c_idx = np.where(symbols == "C")[0]

    audit = {key: "" for key in (
        "min_CH_bond_A", "max_CH_bond_A", "min_HH_distance_A",
        "min_HC_nonbonded_A", "max_C_displacement_A", "max_H_displacement_A",
        "max_H_out_of_plane_A", "n_H_desorbed", "n_C_under_coordinated",
    )}

    positions = atoms.get_positions()

    audit["max_C_displacement_A"] = float(
        np.linalg.norm(positions[c_idx] - initial_positions[c_idx], axis=1).max()
    ) if len(c_idx) else 0.0

    if len(h_idx) == 0:
        return audit

    audit["max_H_displacement_A"] = float(
        np.linalg.norm(positions[h_idx] - initial_positions[h_idx], axis=1).max()
    )

    # The carbon backbone is frozen and planar, so its median z is the sheet plane.
    plane_z = float(np.median(positions[c_idx, 2])) if len(c_idx) else 0.0
    audit["max_H_out_of_plane_A"] = float(np.abs(positions[h_idx, 2] - plane_z).max())

    distances = atoms.get_all_distances(mic=False)

    if len(h_idx) > 1:
        hh = distances[np.ix_(h_idx, h_idx)]
        audit["min_HH_distance_A"] = float(hh[np.triu_indices(len(h_idx), k=1)].min())

    if len(c_idx) > 1:
        hc = np.sort(distances[np.ix_(h_idx, c_idx)], axis=1)

        audit["min_CH_bond_A"] = float(hc[:, 0].min())
        audit["max_CH_bond_A"] = float(hc[:, 0].max())
        audit["min_HC_nonbonded_A"] = float(hc[:, 1].min())
        audit["n_H_desorbed"] = int((hc[:, 0] > CH_BOND_MAX).sum())

        cc = distances[np.ix_(c_idx, c_idx)]
        np.fill_diagonal(cc, np.inf)

        n_c_neighbours = (cc < CC_CUTOFF_STRETCHED).sum(axis=1)
        n_h_neighbours = (distances[np.ix_(c_idx, h_idx)] < CH_BOND_MAX).sum(axis=1)

        audit["n_C_under_coordinated"] = int(
            ((n_c_neighbours + n_h_neighbours) < 3).sum()
        )

    return audit


def initial_min_hh(atoms):
    """Closest H-H contact before relaxation, for a before/after comparison."""
    symbols = np.array(atoms.get_chemical_symbols())
    h_idx = np.where(symbols == "H")[0]

    if len(h_idx) < 2:
        return ""

    hh = atoms.get_all_distances(mic=False)[np.ix_(h_idx, h_idx)]

    return float(hh[np.triu_indices(len(h_idx), k=1)].min())


# ---------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------

def relax_one(structure_id, input_file, output_dir, scratch_dir, mode,
              fmax=FMAX, max_steps=MAX_STEPS):
    """Relax the hydrogens of one cluster."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "geometry.in"
    traj_file = output_dir / "relaxation.traj"
    log_file = output_dir / "relaxation.log"

    atoms = read(input_file, format="aims")

    # xtb_ase branches on atoms.pbc.all(): a periodic structure is written as a
    # POSCAR and silently switched to the tblite backend. The cluster geometries
    # carry no lattice vectors so this should always hold, but a stray cell would
    # change the method with no error at all, so check rather than assume.
    if atoms.pbc.any():
        raise ValueError(
            "{} has periodic boundary conditions set ({}). Cluster relaxation "
            "requires a geometry with no lattice vectors.".format(
                relpath(input_file), atoms.pbc
            )
        )

    symbols = atoms.get_chemical_symbols()
    n_h = symbols.count("H")
    n_c = symbols.count("C")

    if n_h == 0:
        raise ValueError("No hydrogens to relax in {}".format(relpath(input_file)))

    unexpected = sorted(set(symbols) - {"C", "H"})
    if unexpected:
        print("  WARNING: unexpected species found: {}".format(unexpected))

    initial_positions = atoms.get_positions().copy()
    min_hh_initial = initial_min_hh(atoms)

    atoms.calc = make_calculator(scratch_dir)

    # The carbon backbone is frozen in both modes: it holds the experimentally
    # measured TEM coordinates, and relaxing it would break the link to the
    # measurement that makes the A->J series a record of the tearing.
    h_indices = [i for i, symbol in enumerate(symbols) if symbol == "H"]

    constraints = [FixAtoms(mask=[symbol != "H" for symbol in symbols])]

    if mode == "planar":
        # Zero the z-force on every H so planarity is a constraint rather than a
        # coincidence of the mirror symmetry of a perfectly flat input.
        constraints.append(FixCartesian(h_indices, mask=(False, False, True)))

    atoms.set_constraint(constraints)

    energy_initial = float(atoms.get_potential_energy())
    fmax_initial = float(np.linalg.norm(atoms.get_forces(), axis=1).max())

    opt = LBFGS(atoms, trajectory=str(traj_file), logfile=str(log_file))
    opt.run(fmax=fmax, steps=max_steps)

    energy_final = float(atoms.get_potential_energy())
    fmax_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())

    gap, n_electrons = homo_lumo_gap(atoms, scratch_dir)

    write(output_file, atoms, format="aims")

    row = {
        "structure_id": structure_id,
        "mode": mode,
        "n_atoms": len(atoms),
        "n_C": n_c,
        "n_H": n_h,
        "energy_initial_eV": energy_initial,
        "energy_final_eV": energy_final,
        "delta_energy_eV": energy_final - energy_initial,
        "fmax_initial_eV_per_A": fmax_initial,
        "fmax_final_eV_per_A": fmax_final,
        "n_steps": int(opt.get_number_of_steps()),
        "converged": fmax_final <= fmax,
        "homo_lumo_gap_eV": "" if gap is None else gap,
        "n_valence_electrons": n_electrons,
        "odd_electron_count": bool(n_electrons % 2),
        "min_HH_distance_initial_A": min_hh_initial,
        "input_geometry": relpath(input_file),
        "output_geometry": relpath(output_file),
        "method": METHOD,
        "status": "success",
        "error": "",
    }
    row.update(audit_geometry(atoms, initial_positions))

    return row


def failed_row(structure_id, mode, input_file, output_dir, error):
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update({
        "structure_id": structure_id,
        "mode": mode,
        "input_geometry": relpath(input_file),
        "output_geometry": relpath(output_dir / "geometry.in"),
        "method": METHOD,
        "status": "failed",
        "error": str(error),
    })

    return row


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Relax cluster hydrogens with GFN2-xTB."
    )

    parser.add_argument(
        "--mode",
        choices=["planar", "free"],
        required=True,
        help=(
            "planar: H relax in-plane only (H z frozen). "
            "free: H relax in all three directions. Carbon is frozen in both."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "data/h_saturated/cluster_full_saturation",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--parts-root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help="Where each structure's xtb working files go. Defaults to "
             "<output-root>/<ID>/xtb_scratch.",
    )
    parser.add_argument("--structure-id", type=str, default=None)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Process one structure by position in the sorted list "
             "(SLURM array task). Implies --no-merge.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fmax", type=float, default=FMAX)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Only merge existing shards into the summary CSV, then exit.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Redo structures that already have a successful shard.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    mode = args.mode

    input_root = args.input_root
    output_root = args.output_root or (
        ROOT / "data/h_saturated/cluster_relaxed_{}".format(mode)
    )
    parts_root = args.parts_root or (
        ROOT / "results/tables/cluster_relaxation_{}_parts".format(mode)
    )
    summary_file = args.summary or (
        ROOT / "results/tables/cluster_relaxation_{}_summary.csv".format(mode)
    )

    all_ids = all_structure_ids(input_root)

    if args.merge:
        merge_shards(parts_root, summary_file, all_ids)
        return

    if args.index is not None:
        if not 0 <= args.index < len(all_ids):
            raise SystemExit(
                "--index {} out of range for {} structures.".format(
                    args.index, len(all_ids)
                )
            )
        selected = [all_ids[args.index]]

    elif args.structure_id is not None:
        structure_id = args.structure_id.upper()

        if structure_id not in all_ids:
            raise SystemExit(
                "Unknown structure '{}'. Available: {} ...".format(
                    structure_id, ", ".join(all_ids[:10])
                )
            )
        selected = [structure_id]

    else:
        selected = all_ids[:args.limit] if args.limit else all_ids

    print("Mode:        ", mode)
    print("Method:      ", METHOD)
    print("Input root:  ", relpath(input_root))
    print("Output root: ", relpath(output_root))
    print("Structures:  ", len(selected))

    n_success = 0
    n_failed = 0
    n_skipped = 0

    for structure_id in selected:
        input_file = input_root / structure_id / "geometry.in"
        output_dir = output_root / structure_id
        scratch_dir = (
            args.scratch_root / structure_id if args.scratch_root
            else output_dir / "xtb_scratch"
        )

        if not args.no_resume and shard_succeeded(parts_root, structure_id):
            print("{}: already done, skipping.".format(structure_id))
            n_skipped += 1
            continue

        print()
        print("=" * 70)
        print("{} ({})".format(structure_id, mode))
        print("=" * 70)

        started = time.time()

        try:
            row = relax_one(
                structure_id=structure_id,
                input_file=input_file,
                output_dir=output_dir,
                scratch_dir=scratch_dir,
                mode=mode,
                fmax=args.fmax,
                max_steps=args.max_steps,
            )
            n_success += 1

        except Exception as exc:
            print("ERROR: {}".format(exc))
            print(traceback.format_exc())

            row = failed_row(structure_id, mode, input_file, output_dir, exc)
            n_failed += 1

        row["elapsed_s"] = round(time.time() - started, 2)
        write_shard(parts_root, row)

        if row["status"] == "success":
            print("  steps {:>3}  dE {:+.4f} eV  fmax {:.4f}  gap {}  {:.1f} s".format(
                row["n_steps"], row["delta_energy_eV"], row["fmax_final_eV_per_A"],
                "{:.3f} eV".format(row["homo_lumo_gap_eV"])
                if row["homo_lumo_gap_eV"] != "" else "n/a",
                row["elapsed_s"],
            ))
            print("  min H-H {:.3f} -> {:.3f} Å   min non-bonded H-C {:.3f} Å".format(
                float(row["min_HH_distance_initial_A"]),
                float(row["min_HH_distance_A"]),
                float(row["min_HC_nonbonded_A"]),
            ))
            print("  max C displacement {:.2e} Å   max H out of plane {:.4f} Å".format(
                float(row["max_C_displacement_A"]),
                float(row["max_H_out_of_plane_A"]),
            ))

            if int(row["n_C_under_coordinated"]) > 0:
                print("  WARNING: {} carbon(s) still under-coordinated".format(
                    row["n_C_under_coordinated"]
                ))

    print()
    print("=" * 70)
    print("Succeeded: {}  Failed: {}  Skipped: {}".format(n_success, n_failed, n_skipped))

    # --index means "one task of an array"; the shared summary is written by a
    # separate --merge run once every task has finished.
    if args.index is None:
        merge_shards(parts_root, summary_file, all_ids)


if __name__ == "__main__":
    main()
