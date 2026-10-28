# defective-graphene-hydrogen

Analysis code and processed structural data for *Atomistic reconstruction and electronic
structure of IL-STEM-derived defective graphene* (Jasper Allen, 2026).

The repository contains the 95 experimentally derived carbon structures (A01–J05), the
geometry-aware workflow that identifies and hydrogen-terminates their internal defect edges,
the ring-topology analysis, and the atom-resolved DOS analysis of the H02 nine-membered ring
case study.

## Quick start

Python 3.10 or newer. No compiled dependencies.

```bash
bash setup_venv.sh
```

If `python3` on your PATH is older than 3.10, the script stops and says so. Point it at a
newer interpreter with `PYTHON=python3.11 bash setup_venv.sh`, or load one first (on
ARCHER2: `module load cray-python`).

Then reproduce Figure 3 of the report directly from the 95 experimental carbon structures:

```bash
.venv-dos/bin/python3 scripts/visualisation/plot_n_membered_rings_all.py
.venv-dos/bin/python3 scripts/visualisation/plot_ring_topology_census.py
```

The first command runs the ring identification of Methods 2.2 on every structure and writes
the dataset summary `results/tables/n_membered_rings/all_structures_ring_summary.csv`, along
with a per-structure ring table and figure. The second builds the census figure from that
summary. Figure 3 of the report is

```
results/figures/results_ring_topology_census_f6.pdf
```

Both commands together take a few minutes on one core and need no HPC resources, no external
executables and no network access. Reference copies of every output are tracked in this
repository, so a regenerated `all_structures_ring_summary.csv` can be diffed against the
published one.

The second command also writes `results_ring_topology_census.pdf` (the same figure with the
raw hexagon count `N_6` in panel (b) instead of the fraction `f_6`) and
`results_ring_topology_census_f6_dft.pdf` (a third panel carrying the single-point HSE06
energies of the unrelaxed frames). Neither appears in the report.

## Which script makes which figure

| Report figure | Command | Inputs |
|---|---|---|
| Fig. 1, 2 — site classification and threshold calibration | `scripts/visualisation/plot_methods_site_selection.py` | `results/tables/cluster_saturation_sites/`, `cluster_saturation_summary.csv`, `cluster_ring_face_areas.csv`, `<Letter>/<N>/geometry.in` |
| **Fig. 3 — ring topology census** | `scripts/visualisation/plot_n_membered_rings_all.py` then `plot_ring_topology_census.py` | `<Letter>/<N>/geometry.in` only |
| Fig. 4 — H02 local electronic structure | `scripts/dos/plot_ring9_signature_publication.py` | `calculations/dos_cluster/H02/geometry.in`, `results/tables/dos_atom_resolved_cluster/H02.npz` |
| §3.3 sensitivity analysis | `scripts/sensitivity/h_placement_threshold_sensitivity.py` then `plot_h_placement_sensitivity.py` | `<Letter>/<N>/geometry.in` |

Every script is run from the repository root, not from inside its own directory:

```bash
.venv-dos/bin/python3 scripts/<subdir>/<script>.py
```

Each resolves its paths from the repository root via `ROOT = Path(__file__).resolve().parents[N]`,
and several add `scripts/hydrogenation` to `sys.path`, so the `scripts/` layout must be kept
as it is. Pass `--help` to any of them for its options.

## Layout

```
A/1/geometry.in … J/5/geometry.in     the 95 experimental carbon structures (A01–J05)
scripts/
  hydrogenation/                      Methods 2.1: bond graph, hybridisation filter,
                                      H placement, steric and topological criteria.
                                      full_saturate_all_structures.py is the shared
                                      dependency of every other script here.
  visualisation/                      Figures 1, 2, 3
  sensitivity/                        §3.3 threshold sensitivity analysis
  dos/                                Figure 4 and the atom-resolved DOS machinery
  mlip/relax_clusters_xtb.py          Methods 2.3: GFN2-xTB hydrogen relaxation
data/h_saturated/
  cluster_full_saturation/<ID>/       hydrogen-terminated structures as placed (Methods 2.1)
  cluster_relaxed_planar/<ID>/        after in-plane GFN2-xTB H relaxation — the model
                                      used for the DFT in the report
  cluster_relaxed_free/<ID>/          after unconstrained GFN2-xTB H relaxation
calculations/dos_cluster/H02/         geometry and control.in of the FHI-aims run behind Fig. 4
results/tables/                       reference outputs (see below)
results/figures/                      the report figures, PDF and PNG
```

Geometries are in FHI-aims `geometry.in` format throughout. Structure IDs are the family
letter plus a zero-padded site number: `A01`, `H02`, `J05`.

## Provenance and caveats

**The ring census (Fig. 3) is fully reproducible from this repository.** It is computed from
`geometry.in` alone and is the example the paper's Data and Code Availability section names.
The remaining figures start from processed intermediates whose upstream calculations are too
large to distribute; those intermediates are shipped and documented below.

**The stored hydrogenation tree does not re-derive from a fresh run.**
`results/tables/cluster_saturation_sites/` and `data/h_saturated/cluster_full_saturation/`
are the production tables behind Figure 1 and the counts quoted in Methods 2.1, but they are a
mixed regeneration: H02 was regenerated with stretched-coordination-1 stub deletion enabled and
the other 94 structures predate that change. The stored tree carries 8394 hydrogens. Re-running
`scripts/hydrogenation/full_saturate_all_structures.py --cluster` gives

- 8340 H at current defaults, differing from the stored tree at 116 carbons across 40 structures;
- 8397 H with `--keep-stretched-coord1`, matching the stored tree exactly on 94 of 95 structures
  and differing only on H02.

Each difference appears as a *pair* of changed sites, because removing a stub also drops its
partner carbon back to coordination 2, which then becomes a real internal-edge site and gains a
hydrogen. Treat the stored tables as the record of the production run, not as a regression
target. The sensitivity analysis in `scripts/sensitivity/` sidesteps this entirely by
generating its own baseline alongside each test variant.

**Figure 4 starts from a cached projection.** `results/tables/dos_atom_resolved_cluster/H02.npz`
holds the atom-resolved Mulliken weights. The step that produces it,
`scripts/dos/cache_atom_resolved_dos.py`, needs the `carmm` package and a 122 MB `Mulliken.out`
from the FHI-aims run; the run's `geometry.in` and `control.in` are included under
`calculations/dos_cluster/H02/` so the settings are on the record, but its output is not
distributed. Replotting from the cache needs neither carmm nor FHI-aims.

**Not every structure has a relaxation.** `cluster_relaxed_planar/` and `cluster_relaxed_free/`
hold 36 structures each. `cluster_relaxation_planar_summary.csv` and its `_free` counterpart
have 43 rows: the other 7 (A10, B02, C07, C08, D02, D05, D09) are `status = failed` with the
xtb error recorded in the `error` column. They are kept in the summary rather than dropped, so
the attempted set is visible.

**J05 relaxation rows are stale.** The `J05` rows of both relaxation summaries report 402 atoms,
while the current `cluster_full_saturation/J05/geometry.in` has 395. Those relaxations predate
the 2026-08-09 regeneration of the saturation tree and were not re-run. Do not read a J05
relaxation row as current without checking `n_atoms == 395`.

## External codes

Two stages of the workflow call executables that are not redistributed here:

- **FHI-aims** v260530 — the single-point DFT and Mulliken projection of Methods 2.4. Settings
  are in `calculations/dos_cluster/H02/control.in`: HSE06 with MBD-NL, atomic ZORA, 2020 *tight*
  defaults for C and H.
- **xtb** v6.7.1 (GFN2-xTB) — the hydrogen relaxations of Methods 2.3, driven through ASE by
  `xtb_ase`. The executable must be on `PATH` with `XTBPATH` pointing at its parameter files.

The calculations in the report were run on ARCHER2 (EPCC) and on Avon (University of Warwick
SCRTP). The analysis scripts in this repository have no HPC dependency: they run anywhere the
four packages in `requirements.txt` are installed.

## Citation

Please cite the report. If you use the structural data, note that the carbon coordinates derive
from a high-speed TEM experiment in which atomic positions were identified by a convolutional
neural network; see the report's Introduction and its reference [5].

## Licence

Code is released under the MIT Licence (see `LICENSE`). The structural data under `A/`–`J/`,
`data/` and `results/` is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
