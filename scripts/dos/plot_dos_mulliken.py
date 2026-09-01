#!/usr/bin/env python3

from pathlib import Path
import argparse
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read


def add_carmm_to_path(carmm_path):
    carmm_path = Path(carmm_path).expanduser().resolve()

    if not carmm_path.exists():
        raise FileNotFoundError(
            "CARMM directory not found: {}\n"
            "Check that ~/carmm exists on Avon.".format(carmm_path)
        )

    sys.path.insert(0, str(carmm_path))


def get_element_indices(symbols, element):
    element = element.lower()
    return [i for i, symbol in enumerate(symbols) if symbol.lower() == element]


_ANGULAR_MOMENTA = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}


def _broadened_sum(x, energies, weights, sigma, oversample=8):
    """
    Approximate sum_i weights[i] * norm.pdf(x, energies[i], sigma).

    carmm.analyse.mulliken.MullikenData.get_plot_data() computes this exact
    sum with one scipy.stats.norm.pdf() call per (atom, spin, kpt, state) -
    O(N * npoints) work that, for a ~350-atom/~2000-state structure (~2.7M
    states), takes minutes; larger structures are worse, and load_dos_data()
    needs this six times per structure across ~95 structures.

    Binning the weighted eigenvalues into a fine histogram and convolving
    once with a Gaussian kernel is O(N + npoints) instead, and converges to
    the same result as the histogram bin width -> 0. An oversampled internal
    grid (default 8x the requested resolution) keeps the discretization error
    negligible before downsampling back to the caller's grid.
    """
    from scipy.ndimage import gaussian_filter1d

    npoints = len(x)
    dx_out = x[1] - x[0]
    xmin = x[0] - dx_out / 2.0
    xmax = x[-1] + dx_out / 2.0

    n_fine = npoints * oversample
    dx_fine = (xmax - xmin) / n_fine
    edges = np.linspace(xmin, xmax, n_fine + 1)

    hist, _ = np.histogram(energies, bins=edges, weights=weights)
    density_fine = hist / dx_fine

    sigma_bins = sigma / dx_fine
    density_fine = gaussian_filter1d(density_fine, sigma=sigma_bins, mode="constant")

    return density_fine.reshape(npoints, oversample).mean(axis=1)


def homo_reference(mulliken_data):
    """
    Return the energy the DOS x-axis should be shifted by, so every spectrum is
    plotted relative to its own HOMO.

    This used to be applied only when nkpts > 1. That was fine while every
    calculation was a periodic slab with a 2x2x1 k-grid, but a cluster has
    exactly one k-point, so the old test silently skipped the shift and left the
    spectrum on the raw eigenvalue scale. Absolute eigenvalues are not comparable
    between structures - they float with composition and size, and the A-J series
    changes both - so without this every cluster DOS in the series would sit on
    its own arbitrary zero and the whole point of the comparison would be lost.

    get_homo() returns 0.0 when nkpts > 1 by design, but caches the real value on
    mulliken_data.homo as a side effect, which is the value wanted here.
    """
    if mulliken_data.homo is None:
        mulliken_data.get_homo()

    return 0.0 if mulliken_data.homo is None else mulliken_data.homo


def fast_get_plot_data(mulliken_data, atom_ind, spin, kpts, angular,
                        xmin=-20, xmax=20, npoints=1000, variance=0.02):
    """
    Drop-in replacement for MullikenData.get_plot_data(), vectorized with
    numpy instead of a pure-Python loop over every (atom, spin, kpt, state)
    combination. Reproduces the same math (see _broadened_sum) exactly.
    """
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    sigma = np.sqrt(variance)
    x = np.linspace(xmin, xmax, npoints)

    if mulliken_data.homo is None:
        # get_homo() returns 0.0 when nkpts > 1 (by design, for the axis
        # shift below) but as a side effect caches the real value on
        # mulliken_data.homo, which is what we actually need here.
        mulliken_data.get_homo()

    atom_ind = list(atom_ind)
    spin = list(spin)
    kpts = list(kpts)

    data = [np.zeros(npoints) for _ in spin]

    for si, sp in enumerate(spin):
        energies_all = []
        weights_all = []
        for atom in atom_ind:
            for kpt in kpts:
                k = mulliken_data.atoms[atom].spin[sp].kpts[kpt]
                energies = np.asarray(k.energies)
                mask = (energies > xmin) & (energies < xmax)
                if not mask.any():
                    continue

                if angular == "all":
                    coeff = np.asarray(k.all_mulliken)[mask]
                else:
                    coeff = np.zeros(int(mask.sum()))
                    for n in angular:
                        coeff = coeff + np.asarray(k.orbitals[_ANGULAR_MOMENTA[n]])[mask]

                energies_all.append(energies[mask])
                weights_all.append(coeff * k.weight)

        if energies_all:
            data[si] = _broadened_sum(
                x, np.concatenate(energies_all), np.concatenate(weights_all), sigma
            )

    return x - homo_reference(mulliken_data), data


def fast_get_all_plot_data(mulliken_data, xmin=-20, xmax=20, npoints=1000, variance=0.02):
    return fast_get_plot_data(
        mulliken_data,
        range(mulliken_data.get_natoms()),
        range(mulliken_data.get_nspin()),
        range(mulliken_data.get_nkpts()),
        "all",
        xmin, xmax, npoints, variance,
    )


def safe_get_plot_data(mulliken_data, indices, orbital):
    """
    Return Mulliken DOS for a set of atom indices and orbital type.

    If there are no matching atoms, return zeros with the right shape.
    """
    x = np.linspace(-20, 20, 1000) - homo_reference(mulliken_data)

    nspin = mulliken_data.get_nspin()

    if len(indices) == 0:
        return x, [np.zeros_like(x) for _ in range(nspin)]

    x, data = fast_get_plot_data(
        mulliken_data,
        indices,
        range(nspin),
        range(mulliken_data.get_nkpts()),
        orbital,
    )

    return x, data


def load_dos_data(calc_dir, carmm_path):
    """
    Read aims.out + Mulliken.out for one structure and split the DOS into
    per-element/orbital contributions.

    Returns a dict of arrays/metadata consumed by render_dos_plot().
    """
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

    print("Reading Mulliken DOS from: {}".format(mulliken_file))
    mulliken_data = parse_mulliken_file(str(mulliken_file))

    symbols = atoms.get_chemical_symbols()

    c_indices = get_element_indices(symbols, "C")
    h_indices = get_element_indices(symbols, "H")
    o_indices = get_element_indices(symbols, "O")

    print("Number of C atoms:", len(c_indices))
    print("Number of H atoms:", len(h_indices))
    print("Number of O atoms:", len(o_indices))
    print("Number of spins:", mulliken_data.get_nspin())
    print("Number of k-points:", mulliken_data.get_nkpts())

    x, all_data = fast_get_all_plot_data(mulliken_data)

    x, c_s = safe_get_plot_data(mulliken_data, c_indices, "s")
    x, c_p = safe_get_plot_data(mulliken_data, c_indices, "p")
    x, h_s = safe_get_plot_data(mulliken_data, h_indices, "s")
    x, o_s = safe_get_plot_data(mulliken_data, o_indices, "s")
    x, o_p = safe_get_plot_data(mulliken_data, o_indices, "p")

    return {
        "name": calc_dir.name,
        "x": x,
        "all_data": all_data,
        "nspin": mulliken_data.get_nspin(),
        "homo": mulliken_data.get_homo(),
        # Not get_graph_xlabel(): carmm picks the label off the same `nkpts > 1` test
        # that homo_reference() no longer uses, so for a cluster it returns a bare
        # "epsilon (eV)" while the axis has in fact been shifted to the HOMO. Same fix,
        # and for the same reason, as cache_atom_resolved_dos.atom_resolved_histograms().
        "graph_xlabel": "$\\epsilon - \\epsilon_{f}$ (eV)",
        "h_present": len(h_indices) > 0,
        "o_present": len(o_indices) > 0,
        "c_s": c_s,
        "c_p": c_p,
        "h_s": h_s,
        "o_s": o_s,
        "o_p": o_p,
    }


def draw_dos_axes(ax, data, xlim, ylim, title, title_fontsize=None,
                  legend_fontsize=10):
    """
    Draw the stacked Mulliken DOS fill_between plot from data returned by
    load_dos_data() into an existing axes.

    Split out of render_dos_plot() so the side-by-side structure+DOS movie
    frames (plot_dos_structure_movie_frames.py) can put this exact plot into
    one panel of a larger figure rather than re-implementing the stacking.
    """
    from carmm.analyse.graphs import set_graph_axes_mulliken

    x = data["x"]
    c_s = data["c_s"]
    c_p = data["c_p"]
    h_s = data["h_s"]
    o_s = data["o_s"]
    o_p = data["o_p"]
    nspin = data["nspin"]

    for sp in range(nspin):
        baseline = np.zeros_like(x)

        if nspin == 1:
            spin_label = ""
        else:
            spin_label = " spin {}".format(sp + 1)

        y0 = baseline
        y1 = y0 + c_s[sp]
        ax.fill_between(
            x, y0, y1,
            lw=0,
            alpha=0.75,
            label="C s{}".format(spin_label) if sp == 0 else None,
        )

        y0 = y1
        y1 = y0 + c_p[sp]
        ax.fill_between(
            x, y0, y1,
            lw=0,
            alpha=0.75,
            label="C p{}".format(spin_label) if sp == 0 else None,
        )

        if data["h_present"]:
            y0 = y1
            y1 = y0 + h_s[sp]
            ax.fill_between(
                x, y0, y1,
                lw=0,
                alpha=0.75,
                label="H s{}".format(spin_label) if sp == 0 else None,
            )

        if data["o_present"]:
            y0 = y1
            y1 = y0 + o_s[sp]
            ax.fill_between(
                x, y0, y1,
                lw=0,
                alpha=0.75,
                label="O s{}".format(spin_label) if sp == 0 else None,
            )

            y0 = y1
            y1 = y0 + o_p[sp]
            ax.fill_between(
                x, y0, y1,
                lw=0,
                alpha=0.75,
                label="O p{}".format(spin_label) if sp == 0 else None,
            )

        ax.plot(x, y1, lw=1.0, color="black")

    try:
        # Pass 0.0, not data["homo"]. set_graph_axes_mulliken() draws its dashed
        # reference line at x = homo, which assumes the x-axis is still on the
        # raw eigenvalue scale. It is not: homo_reference() has already shifted
        # x so the HOMO sits at zero, for clusters as well as slabs. For a
        # periodic run get_homo() returns 0.0 anyway so this is a no-op, but for
        # a cluster it returns the real HOMO eigenvalue and the line lands
        # several eV away from the feature it is supposed to mark.
        set_graph_axes_mulliken(
            ax,
            x,
            data["all_data"],
            0.0,
            data["graph_xlabel"],
        )
    except Exception as exc:
        print("Warning: CARMM axis formatting failed:", exc)
        ax.set_xlabel("Energy / eV")
        ax.set_ylabel("DOS / states eV$^{-1}$")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if title is not None:
        title_kwargs = {"fontsize": title_fontsize} if title_fontsize is not None else {}
        ax.set_title(title, **title_kwargs)

    ax.legend(loc="upper right", fontsize=legend_fontsize, frameon=True)

    return ax


def render_dos_plot(data, output_file, xlim, ylim, title, title_fontsize=None):
    """
    Draw the stacked Mulliken DOS plot on its own figure and save it to
    output_file.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    draw_dos_axes(ax, data, xlim, ylim, title, title_fontsize=title_fontsize)

    fig.tight_layout()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved DOS plot:", output_file)


def plot_dos(calc_dir, output_file, carmm_path, xlim, ylim):
    data = load_dos_data(calc_dir, carmm_path)
    title = "Mulliken-projected DOS: {}".format(data["name"])
    render_dos_plot(data, output_file, xlim, ylim, title)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Mulliken-projected DOS from FHI-aims aims.out and Mulliken.out."
    )

    parser.add_argument(
        "--calc-dir",
        default="calculations/dos/c8_test",
        help="Directory containing aims.out and Mulliken.out.",
    )

    parser.add_argument(
        "--output",
        default="results/figures/dos/c8_test_mulliken_dos.png",
        help="Output figure path.",
    )

    parser.add_argument(
        "--carmm-path",
        default="~/carmm",
        help="Path to local CARMM source directory.",
    )

    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=[-15.0, 10.0],
        help="Energy x limits, e.g. --xlim -10 10.",
    )

    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        help="DOS y limits, e.g. --ylim 0 20.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    plot_dos(
        calc_dir=args.calc_dir,
        output_file=args.output,
        carmm_path=args.carmm_path,
        xlim=args.xlim,
        ylim=args.ylim,
    )


if __name__ == "__main__":
    main()