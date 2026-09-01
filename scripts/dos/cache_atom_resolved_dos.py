#!/usr/bin/env python3
"""
Parse one Mulliken.out into a per-atom, *unbroadened* DOS histogram cache.

Why this exists
---------------
plot_dos_mulliken.py sums the Mulliken weights over all atoms of an element while
parsing, so the cached curves in results/tables/dos_movie_frames_parts/ cannot be
re-projected onto a subset of atoms (e.g. the nanoribbon). Extracting any new region
would mean re-reading all 86 Mulliken.out files, which are 285-600 MB each (~50 GB).

Gaussian broadening is linear, so

    broaden(sum_over_atoms(w)) == sum_over_atoms(broaden(w))

which means a per-atom histogram of the *unbroadened* weights is a sufficient
statistic: any region grouping and any broadening width can be chosen afterwards
for free. This script pays the parse once; everything downstream reads the cache.

Energy window
-------------
fast_get_plot_data() in plot_dos_mulliken.py windows the eigenvalues to an
*absolute* +-20 eV. E_F sits near -4.9 eV in these runs, so the existing frames only
cover about -15.1 to +24.9 eV relative to E_F: the bottom of the C 2s sigma manifold
is cut off, and the left edge of those plots is the data boundary rather than a band
edge. This cache windows relative to E_F instead (EMIN_REL/EMAX_REL below) so the
full valence manifold is present and valence bandwidth metrics are meaningful.

Output
------
results/tables/dos_atom_resolved/<StructureID>.npz

    x         (nbins,)                    bin-centre energies, relative to E_F
                                          (clusters included - see the shift in
                                          atom_resolved_histograms)
    hist_s    (nspin, n_atoms, nbins)     per-atom l=0 weight, k-point weighted
    hist_p    (nspin, n_atoms, nbins)     per-atom l=1 weight, k-point weighted
    hist_all  (nspin, nbins)              whole-structure total over all l
    symbols   (n_atoms,)                  chemical symbols, in Mulliken atom order
    homo, nspin, nkpts, nstates, graph_xlabel, shifted, bin_width

Roughly 22 MB per structure, ~1.9 GB for the full set.

Usage:
    python scripts/dos/cache_atom_resolved_dos.py --structure-id J05
    python scripts/dos/cache_atom_resolved_dos.py --index 0 --num-chunks 16
    python scripts/dos/cache_atom_resolved_dos.py --count
"""

from pathlib import Path
import argparse

import numpy as np

from ase.io import read

from plot_dos_mulliken import add_carmm_to_path, _ANGULAR_MOMENTA
from dos_common import ROOT, CALC_ROOT, ready_structure_ids, strided_chunk

# Energy window relative to E_F. Wide enough on the low side to contain the whole
# graphene sigma manifold (bottom around -20 eV) with room for the broadening tail,
# and wide enough on the high side to strictly contain the old absolute +-20 eV window
# (E_F sits near -5 eV) so the two pipelines can be compared without an edge artefact.
EMIN_REL = -30.0
EMAX_REL = 30.0

# 5 meV bins, matching the internal fine grid that _broadened_sum() uses in
# plot_dos_mulliken.py. Binning snaps each eigenvalue to a bin centre, and that jitter
# is the only measurable difference between this cache and the original pipeline
# (see check_cache_against_shards.py); at 5 meV it is ~1e-4 of peak after broadening at
# sigma = 0.14 eV, and still small at the sigma = 0.05 eV used to check that the band
# metrics are not broadening-limited. Costs ~7 MB per structure compressed.
NBINS = 12000

DEFAULT_OUT_ROOT = ROOT / "results/tables/dos_atom_resolved"


def broaden_hist(hist, bin_width, sigma):
    """
    Gaussian-broaden a binned weight histogram into a density, along the last axis.

    This is the convolution from plot_dos_mulliken._broadened_sum() applied to an
    already-binned array: dividing by the bin width turns weights into a density,
    and gaussian_filter1d convolves with a normalised Gaussian of the given sigma.
    Linear in `hist`, so summing rows before or after broadening gives the same
    answer - that is what makes region selection a pure post-processing step.
    """
    from scipy.ndimage import gaussian_filter1d

    density = np.asarray(hist, dtype=np.float64) / bin_width
    return gaussian_filter1d(density, sigma=sigma / bin_width, mode="constant", axis=-1)


def atom_resolved_histograms(mulliken_data, emin_rel, emax_rel, nbins):
    """
    Bin every atom's Mulliken weights onto a shared energy grid, without broadening.

    Mirrors the inner loop of plot_dos_mulliken.fast_get_plot_data(), but accumulates
    one histogram row per atom instead of concatenating across atoms.
    """
    natoms = mulliken_data.get_natoms()
    nspin = mulliken_data.get_nspin()
    nkpts = mulliken_data.get_nkpts()

    if mulliken_data.homo is None:
        # get_homo() returns 0.0 for periodic systems by design, but caches the real
        # value on .homo as a side effect. That cached value is the shift we need.
        mulliken_data.get_homo()

    # Always shift, so E_F sits at zero for a cluster as well as a slab. This used to
    # be `nkpts > 1`, mirroring carmm, but that is the same bug that
    # plot_dos_mulliken.homo_reference() documents and fixes: a cluster has exactly
    # one k-point, so the old test silently left it on the raw eigenvalue scale.
    # Everything downstream assumes the shift has happened - draw_dos_axes() draws its
    # E_F line at x = 0 for that reason - so an unshifted cluster cache would put E_F
    # near -4.3 eV and mislabel the figure without failing.
    #
    # Safe for the caches already on disk: they are all periodic (nkpts = 4), so they
    # were shifted under the old test too and are unchanged by this.
    shifted = True
    shift = mulliken_data.homo

    edges = np.linspace(emin_rel, emax_rel, nbins + 1)
    bin_width = (emax_rel - emin_rel) / nbins
    x = 0.5 * (edges[:-1] + edges[1:])

    hist_s = np.zeros((nspin, natoms, nbins), dtype=np.float64)
    hist_p = np.zeros((nspin, natoms, nbins), dtype=np.float64)
    hist_all = np.zeros((nspin, nbins), dtype=np.float64)

    l_s = _ANGULAR_MOMENTA["s"]
    l_p = _ANGULAR_MOMENTA["p"]

    for sp in range(nspin):
        for atom in range(natoms):
            for kpt in range(nkpts):
                k = mulliken_data.atoms[atom].spin[sp].kpts[kpt]

                energies = np.asarray(k.energies, dtype=np.float64) - shift
                mask = (energies > emin_rel) & (energies < emax_rel)
                if not mask.any():
                    continue

                # Same binning as np.histogram's edges, but reused across the three
                # channels below instead of re-deriving it each time.
                idx = ((energies[mask] - emin_rel) / bin_width).astype(np.intp)
                np.clip(idx, 0, nbins - 1, out=idx)

                weight = k.weight

                s_w = np.asarray(k.orbitals[l_s], dtype=np.float64)[mask] * weight
                p_w = np.asarray(k.orbitals[l_p], dtype=np.float64)[mask] * weight
                a_w = np.asarray(k.all_mulliken, dtype=np.float64)[mask] * weight

                hist_s[sp, atom] += np.bincount(idx, weights=s_w, minlength=nbins)
                hist_p[sp, atom] += np.bincount(idx, weights=p_w, minlength=nbins)
                hist_all[sp] += np.bincount(idx, weights=a_w, minlength=nbins)

    return {
        "x": x,
        "bin_width": bin_width,
        "hist_s": hist_s,
        "hist_p": hist_p,
        "hist_all": hist_all,
        "shifted": shifted,
    }


def cache_file(out_root, structure_id):
    return Path(out_root) / "{}.npz".format(structure_id)


def build_cache(calc_dir, carmm_path, emin_rel=EMIN_REL, emax_rel=EMAX_REL, nbins=NBINS):
    add_carmm_to_path(carmm_path)

    from carmm.analyse.mulliken import parse_mulliken_file

    calc_dir = Path(calc_dir).resolve()

    aims_file = calc_dir / "aims.out"
    mulliken_file = calc_dir / "Mulliken.out"

    if not aims_file.exists():
        raise FileNotFoundError("Could not find {}".format(aims_file))
    if not mulliken_file.exists():
        raise FileNotFoundError("Could not find {}".format(mulliken_file))

    print("Reading atoms from: {}".format(aims_file))
    atoms = read(str(aims_file))
    symbols = np.array(atoms.get_chemical_symbols())

    print("Reading Mulliken DOS from: {} ({:.1f} MB)".format(
        mulliken_file, mulliken_file.stat().st_size / 1e6
    ))
    mulliken_data = parse_mulliken_file(str(mulliken_file))

    natoms = mulliken_data.get_natoms()
    if natoms != len(symbols):
        raise ValueError(
            "Atom count mismatch for {}: Mulliken.out has {}, aims.out has {}. "
            "The region tables index atoms by position, so this must line up.".format(
                calc_dir.name, natoms, len(symbols)
            )
        )

    print("Atoms: {} ({} C, {} H)".format(
        natoms, int((symbols == "C").sum()), int((symbols == "H").sum())
    ))
    print("Spins: {}  k-points: {}  states: {}".format(
        mulliken_data.get_nspin(), mulliken_data.get_nkpts(), mulliken_data.get_nstates()
    ))

    result = atom_resolved_histograms(mulliken_data, emin_rel, emax_rel, nbins)

    result["name"] = calc_dir.name
    result["symbols"] = symbols
    result["homo"] = float(mulliken_data.homo)
    result["nspin"] = mulliken_data.get_nspin()
    result["nkpts"] = mulliken_data.get_nkpts()
    result["nstates"] = mulliken_data.get_nstates()
    # Not get_graph_xlabel(): carmm picks the label off the same `nkpts > 1` test that
    # atom_resolved_histograms() no longer uses, so for a cluster it returns a bare
    # "epsilon (eV)" while this cache's axis is in fact referenced to E_F. Store the
    # label that matches the data.
    result["graph_xlabel"] = "$\\epsilon - \\epsilon_{f}$ (eV)"

    return result


def write_cache(out_root, structure_id, result):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    path = cache_file(out_root, structure_id)

    np.savez_compressed(
        path,
        name=result["name"],
        x=result["x"],
        bin_width=result["bin_width"],
        # float32 halves the cache; Mulliken weights carry ~6 significant figures in
        # the source file, so this loses nothing real.
        hist_s=result["hist_s"].astype(np.float32),
        hist_p=result["hist_p"].astype(np.float32),
        hist_all=result["hist_all"].astype(np.float32),
        symbols=result["symbols"],
        homo=result["homo"],
        nspin=result["nspin"],
        nkpts=result["nkpts"],
        nstates=result["nstates"],
        graph_xlabel=result["graph_xlabel"],
        shifted=result["shifted"],
    )
    return path


def read_cache(out_root, structure_id):
    """Load one cache shard. Histograms come back as float64 for downstream maths."""
    with np.load(cache_file(out_root, structure_id), allow_pickle=False) as z:
        return {
            "name": str(z["name"]),
            "x": z["x"],
            "bin_width": float(z["bin_width"]),
            "hist_s": z["hist_s"].astype(np.float64),
            "hist_p": z["hist_p"].astype(np.float64),
            "hist_all": z["hist_all"].astype(np.float64),
            "symbols": [str(s) for s in z["symbols"]],
            "homo": float(z["homo"]),
            "nspin": int(z["nspin"]),
            "nkpts": int(z["nkpts"]),
            "nstates": int(z["nstates"]),
            "graph_xlabel": str(z["graph_xlabel"]),
            "shifted": bool(z["shifted"]),
        }


def run_one(structure_id, calc_root, out_root, carmm_path, force, nbins):
    path = cache_file(out_root, structure_id)
    if not force and path.exists():
        print("Cache already exists for {}; skipping (use --force to redo).".format(structure_id))
        return

    print("=== {} ===".format(structure_id))
    result = build_cache(Path(calc_root) / structure_id, carmm_path, nbins=nbins)
    path = write_cache(out_root, structure_id, result)
    print("Wrote {} ({:.1f} MB)".format(path, path.stat().st_size / 1e6))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Cache per-atom unbroadened Mulliken DOS histograms for A-J structures."
    )
    parser.add_argument("--carmm-path", default="~/carmm",
                        help="Path to local CARMM source directory.")
    parser.add_argument("--calc-root", default=str(CALC_ROOT),
                        help="Directory holding calculations/dos/<StructureID>.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT),
                        help="Directory to write the per-structure .npz cache into.")
    parser.add_argument("--nbins", type=int, default=NBINS,
                        help="Energy bins across the window (default {}).".format(NBINS))
    parser.add_argument("--structure-id", default=None,
                        help="Cache a single named structure, e.g. --structure-id J05.")
    parser.add_argument("--index", type=int, default=None,
                        help="0-based index into the ready-structure list, or into the chunking.")
    parser.add_argument("--num-chunks", type=int, default=None,
                        help="With --index, evaluate only that strided chunk of structures.")
    parser.add_argument("--count", action="store_true",
                        help="Print the number of ready structures and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild a cache shard even if it already exists.")
    args = parser.parse_args()

    calc_root = Path(args.calc_root)
    out_root = Path(args.out_root)

    structure_ids = ready_structure_ids(calc_root)

    if args.count:
        print(len(structure_ids))
        return

    if args.structure_id is not None:
        run_one(args.structure_id, calc_root, out_root, args.carmm_path, args.force, args.nbins)
        return

    if args.num_chunks is not None:
        if args.index is None:
            raise SystemExit("--num-chunks requires --index (which chunk to run).")
        chunk = strided_chunk(structure_ids, args.index, args.num_chunks)
        print("Chunk {}/{}: {} structure(s): {}".format(
            args.index, args.num_chunks, len(chunk), " ".join(chunk)
        ))
        for structure_id in chunk:
            run_one(structure_id, calc_root, out_root, args.carmm_path, args.force, args.nbins)
        return

    if args.index is not None:
        if not 0 <= args.index < len(structure_ids):
            raise SystemExit(
                "--index {} is out of range; there are {} ready structures (0-{}).".format(
                    args.index, len(structure_ids), len(structure_ids) - 1
                )
            )
        run_one(structure_ids[args.index], calc_root, out_root,
                args.carmm_path, args.force, args.nbins)
        return

    for structure_id in structure_ids:
        run_one(structure_id, calc_root, out_root, args.carmm_path, args.force, args.nbins)


if __name__ == "__main__":
    main()
