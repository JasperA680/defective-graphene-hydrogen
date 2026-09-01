#!/usr/bin/env python3
"""
Turn the atom-resolved DOS cache plus a region assignment into projected DOS curves.

Shared by plot_region_dos.py, plot_region_dos_movie_frames.py and analyse_band_metrics.py
so all three project, normalise and broaden identically.
"""

from pathlib import Path
import csv

import numpy as np

from cache_atom_resolved_dos import read_cache, broaden_hist, DEFAULT_OUT_ROOT
from define_dos_regions import REGIONS_ROOT
from dos_common import ROOT

# Every carbon belongs to exactly one of these, so "all" is their union.
CARBON_REGIONS = ["ribbon", "edge", "bulk", "crop_boundary"]

DEFAULT_SIGMA = float(np.sqrt(0.02))


def read_region_table(regions_root, structure_id):
    path = Path(regions_root) / "{}_atom_regions.csv".format(structure_id)
    if not path.exists():
        raise FileNotFoundError(
            "No region table for {}: {}. Run scripts/dos/define_dos_regions.py first.".format(
                structure_id, path
            )
        )

    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "atom_index": int(row["atom_index"]),
                "element": row["element"],
                "region": row["region"],
                "edge_hops": int(row["edge_hops"]),
                "ribbon_width_A": float(row["ribbon_width_A"]) if row["ribbon_width_A"] else None,
            })
    return rows


def region_mask(rows, n_atoms, region, element=None, max_ribbon_width=None):
    """Boolean mask over cache atom order for one region (or 'all')."""
    mask = np.zeros(n_atoms, dtype=bool)

    for row in rows:
        if row["atom_index"] >= n_atoms:
            raise ValueError(
                "Region table refers to atom {} but the DOS cache has {} atoms.".format(
                    row["atom_index"], n_atoms
                )
            )
        if element is not None and row["element"] != element:
            continue
        if region != "all" and row["region"] != region:
            continue
        if max_ribbon_width is not None and region == "ribbon":
            width = row["ribbon_width_A"]
            if width is None or width > max_ribbon_width:
                continue
        mask[row["atom_index"]] = True

    return mask


def project(cache, rows, region, sigma=DEFAULT_SIGMA, per_atom=False,
            max_ribbon_width=None):
    """
    Projected DOS for one region.

    Returns a dict with the energy grid, the C s / C p / H s curves, and the atom counts
    behind them. With per_atom=True each curve is divided by the number of atoms of that
    element in the region, which is what makes a 40-atom ribbon comparable with a
    250-atom bulk.
    """
    n_atoms = cache["hist_s"].shape[1]

    c_mask = region_mask(rows, n_atoms, region, "C", max_ribbon_width)
    h_mask = region_mask(rows, n_atoms, region, "H", max_ribbon_width)

    n_c = int(c_mask.sum())
    n_h = int(h_mask.sum())

    bw = cache["bin_width"]

    def curve(hist_key, mask, count):
        if not mask.any():
            return np.zeros_like(cache[hist_key][:, 0, :])
        summed = cache[hist_key][:, mask].sum(axis=1)
        if per_atom and count:
            summed = summed / count
        return broaden_hist(summed, bw, sigma)

    total = c_mask | h_mask
    all_data = curve("hist_s", total, n_c + n_h) + curve("hist_p", total, n_c + n_h)

    return {
        "name": cache["name"],
        "region": region,
        "x": cache["x"],
        "nspin": cache["nspin"],
        "homo": 0.0 if cache["shifted"] else cache["homo"],
        "graph_xlabel": cache["graph_xlabel"],
        "c_s": curve("hist_s", c_mask, n_c),
        "c_p": curve("hist_p", c_mask, n_c),
        "h_s": curve("hist_s", h_mask, n_h),
        # render_dos_plot() expects these; there is no oxygen in any of these runs.
        "o_s": np.zeros_like(cache["hist_s"][:, 0, :]),
        "o_p": np.zeros_like(cache["hist_s"][:, 0, :]),
        "h_present": n_h > 0,
        "o_present": False,
        "all_data": all_data,
        "n_C": n_c,
        "n_H": n_h,
        "per_atom": per_atom,
        "sigma": sigma,
    }


def load_projection(structure_id, region, cache_root=DEFAULT_OUT_ROOT,
                    regions_root=REGIONS_ROOT, **kwargs):
    cache = read_cache(cache_root, structure_id)
    rows = read_region_table(regions_root, structure_id)
    return project(cache, rows, region, **kwargs)


def ylabel(per_atom):
    if per_atom:
        return "DOS / states eV$^{-1}$ atom$^{-1}$"
    return "DOS / states eV$^{-1}$"
