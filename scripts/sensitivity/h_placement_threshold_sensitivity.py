#!/usr/bin/env python3
"""
One-at-a-time sensitivity analysis of the three H-placement geometric thresholds.

The hydrogen assignment in scripts/hydrogenation/full_saturate_all_structures.py
is gated by three hand-tuned numbers:

    SP_ANGLE_THRESHOLD_DEG      155.0 deg   sp/sp2 discriminator
    MAX_CAGE_RING_AREA           12.0 A^2   bounded-face cage cut
    MIN_HH_DISTANCE /            1.40 A     steric floors of the normal
    MIN_NONBONDED_HC_DISTANCE               placement pass

This script perturbs one of them at a time, holding the other two at their
production values, and compares the resulting per-carbon hydrogen assignment
against production site by site. It is a sensitivity analysis of the initial
placement/classification stage only: no relaxation and no DFT is rerun.

Nothing here reimplements the placement logic. Every variant is produced by
calling full_saturate_all_structures.process_one_structure with its output roots
redirected into a sandbox under results/sensitivity/, so the code path is
identical to production in all respects except the one number under test.

Production itself is recomputed as the "production" variant inside the same
sandbox rather than read from results/tables/cluster_saturation_sites/. That way
the baseline and the tests differ in exactly one number and nothing else. The
stored production tables are used once, by --check-baseline, to prove the
sandbox baseline reproduces the real Aug 20 run.

The tight retry (MIN_HH_DISTANCE_TIGHT / MIN_NONBONDED_HC_DISTANCE_TIGHT, 1.00
and 1.10 A) is deliberately held at its production values throughout, including
during the steric sweep. Those are separate production parameters, and this is a
strict one-at-a-time analysis. The consequence is that raising the main floor
pushes sites into the retry where many still succeed, so the steric panel
understates the raw effect of the filter; n_h_tight_total is reported per row so
that masking can be quantified rather than left implicit.

A "changed site" is a carbon whose n_h_added differs from production. That is the
realised count actually written into the geometry, so it captures classification
flips and steric rejections alike, and a site that swaps its H with a neighbour
at constant structure total still counts.

Inputs:
    A/1/geometry.in ... J/5/geometry.in            (95 structures)
    results/tables/cluster_saturation_sites/*.csv  (--check-baseline only)

Outputs, all under results/sensitivity/h_placement_thresholds/:
    variants.csv                 one row per variant, the three threshold values
    sensitivity_summary.csv      one row per (parameter, value) - the tidy table
    changed_sites.csv            one row per changed carbon
    per_structure_deltas.csv     one row per (parameter, value, structure)
    site_tables/<variant>/<ID>_saturation_sites.csv
    geometries/<variant>/<ID>/geometry.in
    summaries/<variant>_summary.csv

Run from the repository root:
    .venv-dos/bin/python3 scripts/sensitivity/h_placement_threshold_sensitivity.py
or, on ARCHER2, sbatch jobs/archer2_h_placement_sensitivity.slurm
"""

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hydrogenation"))

import full_saturate_all_structures as F  # noqa: E402

OUTPUT_ROOT = ROOT / "results" / "sensitivity" / "h_placement_thresholds"
PRODUCTION_SITE_TABLES = ROOT / "results" / "tables" / "cluster_saturation_sites"

# Production values, taken from the module rather than hard-coded here so this
# analysis can never silently disagree with the pipeline it is measuring.
PROD_SP_ANGLE = float(F.SP_ANGLE_THRESHOLD_DEG)
PROD_CAGE_AREA = float(F.MAX_CAGE_RING_AREA)
PROD_MIN_HH = float(F.MIN_HH_DISTANCE)
PROD_MIN_HC = float(F.MIN_NONBONDED_HC_DISTANCE)

PRODUCTION_VARIANT = "production"

# The sweeps. The production value appears in each list and is served by the
# single shared "production" variant, so 13 runs cover all 15 table rows.
SWEEPS = {
    "sp_angle_deg": [150.0, 152.5, 155.0, 157.5, 160.0],
    "max_cage_ring_area": [10.0, 11.0, 12.0, 13.0, 14.0],
    "min_steric_distance": [1.20, 1.30, 1.40, 1.50, 1.60],
}

PRODUCTION_VALUE = {
    "sp_angle_deg": PROD_SP_ANGLE,
    "max_cage_ring_area": PROD_CAGE_AREA,
    "min_steric_distance": PROD_MIN_HH,
}

PARAMETER_LABEL = {
    "sp_angle_deg": "sp/sp2 angle threshold / deg",
    "max_cage_ring_area": "bounded-face area threshold / $\\mathrm{\\AA}^2$",
    "min_steric_distance": "steric separation threshold / $\\mathrm{\\AA}$",
}


def variant_name(parameter, value):
    """
    Stable directory-safe name for one variant.

    The production point of every sweep maps onto the single shared production
    variant, so it is computed once and compared against itself.
    """
    if value == PRODUCTION_VALUE[parameter]:
        return PRODUCTION_VARIANT

    return "{}_{}".format(parameter, ("%g" % value).replace(".", "p"))


def variant_thresholds(parameter, value):
    """
    The (sp_angle, cage_area, min_hh, min_hc) tuple for one variant.

    One parameter takes the test value, the other two stay at production. The
    two steric floors move together: they are one physical threshold in the
    production settings and the user's sweep treats them as one.
    """
    thresholds = {
        "sp_angle_deg": PROD_SP_ANGLE,
        "max_cage_ring_area": PROD_CAGE_AREA,
        "min_hh": PROD_MIN_HH,
        "min_hc": PROD_MIN_HC,
    }

    if parameter == "min_steric_distance":
        thresholds["min_hh"] = float(value)
        thresholds["min_hc"] = float(value)
    else:
        thresholds[parameter] = float(value)

    return thresholds


def build_variant_table():
    """
    Every unique variant to run, in a deterministic order with production first.
    """
    rows = []
    seen = set()

    for parameter, values in SWEEPS.items():
        for value in values:
            name = variant_name(parameter, value)

            if name in seen:
                continue

            seen.add(name)
            thresholds = variant_thresholds(parameter, value)
            rows.append(
                {
                    "variant": name,
                    "parameter": "production" if name == PRODUCTION_VARIANT else parameter,
                    "value": np.nan if name == PRODUCTION_VARIANT else float(value),
                    **thresholds,
                }
            )

    rows.sort(key=lambda row: (row["variant"] != PRODUCTION_VARIANT, row["variant"]))

    return pd.DataFrame(rows)


def run_variant(row, geometry_files, resume=True, quiet=True):
    """
    Run one variant over every structure, writing its site tables and geometries.

    process_one_structure does all the work; this only redirects its outputs and
    collects the per-structure summaries. Plots are off - 1235 site maps would
    dominate the runtime and none of them is used.
    """
    name = row["variant"]

    site_table_root = OUTPUT_ROOT / "site_tables" / name
    geometry_root = OUTPUT_ROOT / "geometries" / name
    figure_root = OUTPUT_ROOT / "figures_unused" / name
    summary_file = OUTPUT_ROOT / "summaries" / "{}_summary.csv".format(name)

    if resume and summary_file.exists():
        print("  {}: already done, skipping".format(name))
        return pd.read_csv(summary_file)

    site_table_root.mkdir(parents=True, exist_ok=True)
    geometry_root.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    start = time.time()

    for geom_file in geometry_files:
        # process_one_structure is chatty by design; at 95 structures x 13
        # variants its output would bury everything that matters here.
        stream = io.StringIO()
        context = contextlib.redirect_stdout(stream) if quiet else contextlib.nullcontext()

        with context:
            summary = F.process_one_structure(
                geom_file=geom_file,
                root=ROOT,
                output_root=geometry_root,
                site_table_root=site_table_root,
                figure_root=figure_root,
                reject_clashes=True,
                make_plots=False,
                cluster=True,
                sp_angle_deg=row["sp_angle_deg"],
                max_cage_ring_area=row["max_cage_ring_area"],
                min_hh=row["min_hh"],
                min_hc=row["min_hc"],
            )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(summary_file, index=False)

    failed = summary_df[summary_df["status"] != "success"]

    if len(failed):
        raise RuntimeError(
            "{}: {} structures failed: {}".format(
                name, len(failed), ", ".join(failed["structure_id"])
            )
        )

    print(
        "  {}: {} structures, {} H, {:.0f} s".format(
            name,
            len(summary_df),
            int(summary_df["n_H_added"].sum()),
            time.time() - start,
        )
    )

    return summary_df


def load_site_tables(variant):
    """
    Read one variant's per-structure site tables into a single frame.

    Only the columns the comparison needs are kept - the full table is 30 columns
    wide and 13 of them at 37815 rows each is a lot of memory for nothing.
    """
    columns = [
        "atom_index",
        "coordination",
        "site_class",
        "ccc_angle_deg",
        "n_h_target",
        "n_h_added",
        "n_h_tight",
        "delete_atom",
        "addition_status",
    ]

    frames = []

    for path in sorted((OUTPUT_ROOT / "site_tables" / variant).glob("*_saturation_sites.csv")):
        df = pd.read_csv(path, usecols=columns)
        df.insert(0, "structure_id", path.name.split("_")[0])
        frames.append(df)

    if not frames:
        raise FileNotFoundError("no site tables found for variant {}".format(variant))

    return pd.concat(frames, ignore_index=True)


def compare_to_production(test, production, parameter, value):
    """
    Site-by-site comparison of one variant against production.

    Returns (summary_row, changed_sites_df, per_structure_df).

    The merge is an inner join on (structure_id, atom_index) - the input-geometry
    index, which is stable across variants because the three deletions depend
    only on CC_CUTOFF and CC_CUTOFF_STRETCHED and neither is being varied. That
    stability is asserted rather than assumed.
    """
    merged = production.merge(
        test,
        on=["structure_id", "atom_index"],
        how="outer",
        suffixes=("_production", "_test"),
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"]

    if len(unmatched):
        raise AssertionError(
            "{}={}: atom_index set differs from production on {} rows "
            "(structures: {}). The comparison assumes deletions are "
            "threshold-independent.".format(
                parameter,
                value,
                len(unmatched),
                ", ".join(sorted(unmatched["structure_id"].unique())),
            )
        )

    merged = merged.drop(columns="_merge")

    changed_mask = merged["n_h_added_production"] != merged["n_h_added_test"]
    changed = merged[changed_mask].copy()

    changed_sites = pd.DataFrame(
        {
            "parameter": parameter,
            "value": value,
            "structure_id": changed["structure_id"],
            "atom_index": changed["atom_index"],
            "H_production": changed["n_h_added_production"].astype(int),
            "H_test": changed["n_h_added_test"].astype(int),
            "coordination": changed["coordination_production"].astype(int),
            "ccc_angle_deg": changed["ccc_angle_deg_production"],
            "site_class_production": changed["site_class_production"],
            "site_class_test": changed["site_class_test"],
            "addition_status_production": changed["addition_status_production"],
            "addition_status_test": changed["addition_status_test"],
        }
    ).sort_values(["structure_id", "atom_index"])

    per_structure = (
        merged.groupby("structure_id")
        .apply(
            lambda g: pd.Series(
                {
                    "total_H_production": int(g["n_h_added_production"].sum()),
                    "total_H_test": int(g["n_h_added_test"].sum()),
                    "n_changed_sites": int(
                        (g["n_h_added_production"] != g["n_h_added_test"]).sum()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    per_structure["delta_H"] = (
        per_structure["total_H_test"] - per_structure["total_H_production"]
    )
    per_structure.insert(0, "value", value)
    per_structure.insert(0, "parameter", parameter)

    affected = per_structure[
        (per_structure["n_changed_sites"] > 0) | (per_structure["delta_H"] != 0)
    ]

    # Candidate denominator: carbons that can receive H under any of these
    # thresholds, i.e. surviving coordination-1 and -2. Coordination 0 and 3 are
    # always 0 H regardless. Taken from production so it is one fixed number
    # across every row of the table and the percentages are comparable.
    n_candidate = int(
        (
            production["coordination"].isin([1, 2])
            & ~production["delete_atom"].astype(bool)
        ).sum()
    )

    total_h_production = int(production["n_h_added"].sum())
    total_h_test = int(test["n_h_added"].sum())
    n_changed = int(changed_mask.sum())

    summary_row = {
        "parameter": parameter,
        "value": value,
        "total_H": total_h_test,
        "delta_H": total_h_test - total_h_production,
        "n_changed_sites": n_changed,
        "fraction_changed_sites": n_changed / n_candidate if n_candidate else np.nan,
        "percent_changed_sites": 100.0 * n_changed / n_candidate if n_candidate else np.nan,
        "n_structures_affected": int(len(affected)),
        "max_delta_H_per_structure": (
            int(per_structure["delta_H"].abs().max()) if len(per_structure) else 0
        ),
        "affected_structures": ";".join(sorted(affected["structure_id"])),
        # Extras that make the numbers above interpretable in the write-up.
        "n_candidate_sites": n_candidate,
        "n_sites_gained_H": int(
            (merged["n_h_added_test"] > merged["n_h_added_production"]).sum()
        ),
        "n_sites_lost_H": int(
            (merged["n_h_added_test"] < merged["n_h_added_production"]).sum()
        ),
        "n_h_tight_total": int(test["n_h_tight"].sum()),
        "n_sites_unplaceable": int((test["addition_status"] == "unplaceable").sum()),
        "n_sites_partial": int((test["addition_status"] == "partial").sum()),
        "is_production": bool(value == PRODUCTION_VALUE[parameter]),
    }

    return summary_row, changed_sites, per_structure


def check_baseline(production):
    """
    Compare the sandbox baseline against the stored cluster_saturation_sites tree.

    This is NOT a plain equality check, because the stored tree does not
    reproduce under the current script. Its site tables date from 2026-08-09 and
    predate the stretched-coord1 deletion (full_saturate_all_structures.py,
    stretched_coord1_carbons): 94 of the 95 rows in the stored summary carry no
    n_stretched_coord1_deleted value at all, and only H02 was regenerated with
    the feature on. Deleting one of those stubs also drops its partner carbon
    from coordination 3 back to 2, which turns the partner into a real
    defect-edge site, so each deletion shows up as a pair of changed rows.

    That difference is a property of the stored tree, not of this analysis: the
    sensitivity numbers are all computed against the sandbox baseline, which is
    generated by the current code alongside every test variant. This check exists
    to confirm the discrepancy is entirely the known cascade and nothing else, so
    a genuine regression would still be caught.
    """
    print("Comparing sandbox baseline against the stored production site tables")

    columns = ["atom_index", "coordination", "n_h_added", "site_class", "addition_status"]
    frames = []

    for path in sorted(PRODUCTION_SITE_TABLES.glob("*_saturation_sites.csv")):
        df = pd.read_csv(path, usecols=columns)
        df.insert(0, "structure_id", path.name.split("_")[0])
        frames.append(df)

    stored = pd.concat(frames, ignore_index=True)

    merged = stored.merge(
        production[["structure_id"] + columns],
        on=["structure_id", "atom_index"],
        how="outer",
        suffixes=("_stored", "_sandbox"),
        indicator=True,
    )

    differing = merged[
        (merged["_merge"] != "both")
        | (merged["n_h_added_stored"] != merged["n_h_added_sandbox"])
        | (merged["site_class_stored"] != merged["site_class_sandbox"])
        | (merged["addition_status_stored"] != merged["addition_status_sandbox"])
    ]

    # One stretched-coord1 deletion shows up in three ways, so all three are
    # accounted for here rather than only the obvious one:
    #   1. the deleted stub itself;
    #   2. a surviving carbon whose coordination fell because the stub was its
    #      neighbour - 3->2 turns a bulk carbon into a defect-edge site, and
    #      2->1 turns an edge site into a coord1_terminus taking two H;
    #   3. a neighbour whose H assignment is unchanged but which no longer needs
    #      the tight retry, because the stub's H are no longer in its way. Same
    #      site_class and same n_h_added, only addition_status moves.
    is_deleted_stub = differing["site_class_sandbox"] == "stretched_coord1_deleted"

    is_cascade_partner = (
        differing["coordination_sandbox"] != differing["coordination_stored"]
    )

    is_status_only = (
        (differing["n_h_added_stored"] == differing["n_h_added_sandbox"])
        & (differing["site_class_stored"] == differing["site_class_sandbox"])
        & (differing["coordination_stored"] == differing["coordination_sandbox"])
    )

    unexplained = differing[
        ~(is_deleted_stub | is_cascade_partner | is_status_only)
    ]

    if len(unexplained):
        raise AssertionError(
            "sandbox baseline differs from the stored tree in {} rows that the "
            "known stretched-coord1 cascade does not explain (structures: {}). "
            "This is a real regression - investigate before trusting the "
            "sweep.".format(
                len(unexplained), ", ".join(sorted(unexplained["structure_id"].unique()))
            )
        )

    print(
        "  Stored tree: {} H. Sandbox baseline: {} H. {} differing carbons in {} "
        "structures, all attributable to the stretched-coord1 deletion that the "
        "stored 2026-08-09 tables predate: {} deleted stubs, {} carbons whose "
        "coordination the deletion changed, {} tight-retry knock-ons with the "
        "same H assignment. No unexplained differences.".format(
            int(stored["n_h_added"].sum()),
            int(production["n_h_added"].sum()),
            len(differing),
            differing["structure_id"].nunique(),
            int(is_deleted_stub.sum()),
            int((is_cascade_partner & ~is_deleted_stub).sum()),
            int((is_status_only & ~is_deleted_stub & ~is_cascade_partner).sum()),
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One-at-a-time sensitivity analysis of the sp/sp2 angle, "
            "bounded-face area and steric separation thresholds used by the "
            "hydrogen-placement stage."
        )
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Run only this variant and stop, without comparing or writing the "
            "summary tables. For debugging or for driving a SLURM array; see "
            "variants.csv for the names."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of structures, for a quick smoke test.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute every variant even if its summary already exists.",
    )

    parser.add_argument(
        "--skip-baseline-check",
        action="store_true",
        help=(
            "Skip the comparison of the sandbox production variant against the "
            "stored cluster_saturation_sites tables. Only meaningful with "
            "--limit, where the two cannot match."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    geometry_files = F.find_all_geometry_files(ROOT)

    if args.limit is not None:
        geometry_files = geometry_files[: args.limit]

    print("Structures: {}".format(len(geometry_files)))
    print(
        "Production thresholds: sp angle {} deg, cage area {} A^2, "
        "steric {} / {} A".format(
            PROD_SP_ANGLE, PROD_CAGE_AREA, PROD_MIN_HH, PROD_MIN_HC
        )
    )
    print(
        "Tight retry held fixed at H-H {} / H-C {} A throughout".format(
            F.MIN_HH_DISTANCE_TIGHT, F.MIN_NONBONDED_HC_DISTANCE_TIGHT
        )
    )

    variants = build_variant_table()
    variants.to_csv(OUTPUT_ROOT / "variants.csv", index=False)

    if args.variant is not None:
        selected = variants[variants["variant"] == args.variant]

        if not len(selected):
            raise SystemExit(
                "unknown variant {!r}; see {}".format(
                    args.variant, OUTPUT_ROOT / "variants.csv"
                )
            )

        run_variant(selected.iloc[0], geometry_files, resume=not args.no_resume)
        return

    print()
    print("Running {} unique variants".format(len(variants)))

    for _, row in variants.iterrows():
        run_variant(row, geometry_files, resume=not args.no_resume)

    print()
    print("Comparing every variant against production site by site")

    production = load_site_tables(PRODUCTION_VARIANT)

    if not args.skip_baseline_check:
        check_baseline(production)

    summary_rows = []
    changed_frames = []
    per_structure_frames = []

    for parameter, values in SWEEPS.items():
        for value in values:
            test = load_site_tables(variant_name(parameter, value))

            summary_row, changed, per_structure = compare_to_production(
                test=test, production=production, parameter=parameter, value=value
            )

            summary_rows.append(summary_row)
            changed_frames.append(changed)
            per_structure_frames.append(per_structure)

    summary = pd.DataFrame(summary_rows)
    changed_sites = pd.concat(changed_frames, ignore_index=True)
    per_structure = pd.concat(per_structure_frames, ignore_index=True)

    summary.to_csv(OUTPUT_ROOT / "sensitivity_summary.csv", index=False)
    changed_sites.to_csv(OUTPUT_ROOT / "changed_sites.csv", index=False)
    per_structure.to_csv(OUTPUT_ROOT / "per_structure_deltas.csv", index=False)

    # Sanity checks that would otherwise need eyeballing. Each is cheap and each
    # has a specific failure it protects against.
    assert len(summary) == 15, "expected 15 rows, got {}".format(len(summary))

    production_rows = summary[summary["is_production"]]
    assert len(production_rows) == 3, "expected 3 production rows"
    assert (production_rows["delta_H"] == 0).all(), "production row has non-zero delta_H"
    assert (
        production_rows["n_changed_sites"] == 0
    ).all(), "production row has changed sites"

    for _, row in variants.iterrows():
        variant_summary = pd.read_csv(
            OUTPUT_ROOT / "summaries" / "{}_summary.csv".format(row["variant"])
        )
        table_total = int(load_site_tables(row["variant"])["n_h_added"].sum())
        assert table_total == int(variant_summary["n_H_added"].sum()), (
            "{}: site tables and run summary disagree on total H".format(row["variant"])
        )
        assert (variant_summary["n_H_inside_ring"] == 0).all(), (
            "{}: the cage filter was bypassed - a written H sits inside a "
            "ring".format(row["variant"])
        )

    print()
    print("Wrote:")
    for name in (
        "variants.csv",
        "sensitivity_summary.csv",
        "changed_sites.csv",
        "per_structure_deltas.csv",
    ):
        print("  {}".format((OUTPUT_ROOT / name).relative_to(ROOT)))

    print()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(
            summary[
                [
                    "parameter",
                    "value",
                    "total_H",
                    "delta_H",
                    "n_changed_sites",
                    "percent_changed_sites",
                    "n_structures_affected",
                    "max_delta_H_per_structure",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
