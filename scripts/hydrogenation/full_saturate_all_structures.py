#!/usr/bin/env python3
"""
Fully H-saturate internal coordination-2 defect-edge sites for all structures.

Important site-selection logic:
    - Compute C-C coordination, then count stretched bonds in
      [CC_CUTOFF, CC_CUTOFF_STRETCHED) between two carbons that are both still
      undercoordinated (heal_stretched_bonds). Without this a carbon whose third
      bond is merely long looks coordination-2, and its "missing bond" direction
      points straight at the partner carbon that is already sitting there.
    - Find all coordination-2 carbon atoms.
    - Drop the ones that are sp rather than sp2. A coordination-2 carbon is only
      a dangling bond if its third sigma orbital is empty, which needs sp2
      geometry (C-C-C angle around 120-130 degrees). As the angle opens towards
      180 the carbon becomes sp, its two remaining electrons occupy pi orbitals,
      and there is nothing for an H to cap. Sites at or above
      SP_ANGLE_THRESHOLD_DEG are therefore left bare in both periodic and
      cluster mode, labelled internal_sp_linear_coord2 or
      external_crop_sp_linear_coord2 depending on which side of the frame band
      they fall (114 carbons across the 95 structures at the 155 degree default:
      91 internal, 23 on the frame). These are the carbyne-like chains that run
      around the central tear in the most eroded frames (I08-I10, J01-J05),
      where the C-C bonds also shorten to 1.1-1.3 Å. The C-C-C angle is written
      to the site table (ccc_angle_deg) for every coordination-2 carbon, so the
      threshold can be re-tuned from the tables without rerunning.
    - Separate real nanoribbon edges from artificial TEM-frame crops.
      Each structure is a rectangular crop out of a larger TEM field of view, so
      carbons that are merely sliced by the frame are bulk graphene and must not
      be hydrogenated. A coordination-2 carbon is an artificial crop atom when it
      sits flush against a frame edge (within FRAME_BAND of the carbon bounding
      box) *and* its missing-bond direction points out of that same frame edge.
      Everything else - interior tears, necks, and edges the material has pulled
      inward from the frame - is a real defect edge.
    - Add one H atom to each real (internal) coordination-2 carbon atom, in the
      plane of the sheet. The missing-bond bisector is tried first; where it
      clashes (two coordination-2 carbons facing each other across a narrow tear
      both point into the gap) the H is rotated about the sheet normal until it
      clears, taking the smallest tilt that works (place_h_in_plane). Rotation
      about z preserves the z coordinate, so the H stay coplanar with the carbon.
      A site is only left bare if no in-plane direction clears at all.

One class of coordination-1 carbon *is* hydrogenated, in both modes: the terminus
of an sp chain. Where the sp exclusion above leaves a carbyne-like chain bare, the
carbon on the *end* of that chain is not the same object - it has one sigma bond
and nothing else, which is a genuine dangling-bond radical rather than a
closed-shell chain atom, and a worse defect than the ones the exclusion protects.
It is capped with a single H continuing the chain axis, giving ...C#C-H, and is
labelled internal_sp_terminus_coord1 or external_crop_sp_terminus_coord1. It is
identified by its *neighbour's* C-C-C angle against the same
SP_ANGLE_THRESHOLD_DEG (sp_chain_terminus_angle_deg), and that angle is recorded
per site in terminus_neighbour_angle_deg. Across the 95 structures there are 12,
all internal, concentrated in the eroded I/J frames. Note this is one H and not
the two that a coordination-1 carbon gets in cluster mode: =CH2 needs a pi bond
from the neighbour and an sp chain carbon has none spare. See
coord1_axial_direction.

Every *other* carbon with 0 or 1 C neighbours is NOT hydrogenated
(TARGET_H_COORDINATION is 2) even though they carry 3-4 missing bonds each. They
will still contribute dangling-bond states to a DOS calculation.

They are reported as two separate classes rather than one lumped
"coord0_or_coord1" bucket, because they are chemically different objects and want
different follow-up:
    - coord0_inspect: no C neighbour at all. Either a genuinely detached atom or,
      more often, a TEM-fit artefact - a carbon placed too far from the sheet for
      any bond to be found. Saturating one would need 4 H and a direction that
      the missing-bond bisector cannot supply, so these are candidates for
      deletion or re-fitting, not for hydrogenation.
    - coord1_inspect: a single C neighbour with an sp2 neighbour, i.e. a real
      chain terminus / adatom stub with 3 missing bonds. A well-defined bond axis
      exists, and the chemistry wants =CH2 (two H at +/-120 degrees, as cluster
      mode already places), but adding those here costs more coordination-2 sites
      than it fixes at the current clash floors, so it is left for a future
      change. 219 across the 95 structures.
Summary columns are n_coord0_inspect and n_coord1_inspect, with
n_coord0_or_coord1_inspect kept as their total for continuity with earlier runs.
n_coord1_inspect excludes the sp termini, which are counted in
n_sp_terminus_coord1 and are no longer awaiting anything.

Cluster mode (--cluster)
------------------------
Everything above describes the periodic slab treatment, which is the default and
is unchanged. It is however the wrong model for these structures. Each frame is a
crop that fills its cell almost exactly - A01 spans 44.1 Å of a 45.41 Å cell -
so under the minimum image the crop rim bonds to its own periodic image at an
arbitrary registry, producing 13-24 bonds per frame that exist only through the
PBC and 6-13 four-coordinate sp2 carbons that are not in the sheet. Worse for a
study that compares frames, the shape changes from frame to frame while the cell
does not, so the artefact varies along the A->J series and contaminates exactly
the trend being measured.

--cluster treats each frame as an isolated molecule instead:
    - The input lattice is read but discarded, so every coordination, bond
      direction and clash test uses plain Cartesian distances. The single branch
      that does this is in minimum_image_displacement.
    - The crop-frame rim is hydrogenated too. There is no material on the far
      side of the frame any more, so a bare rim carbon is a dangling bond that
      plants spurious mid-gap states in the DOS. The site_class labels still
      distinguish external_crop_boundary_coord2 from
      internal_defect_edge_coord2, because scripts/dos/define_dos_regions.py
      needs that distinction to project the DOS onto the real edges and ignore
      the terminated rim - the rim H are a computational termination, not
      physics, and the analysis has to be able to exclude them.
    - Coordination-1 carbons with an sp2 neighbour get two H at +/-120 degrees
      from their single C-C bond (coord1_h_directions). Three sigma bonds means
      sp2, so this terminus is genuinely planar and valid in both the planar and
      free relaxations. Those with an sp neighbour instead get the single axial
      H described above, in cluster mode exactly as in periodic mode.
    - Coordination-0 carbons are deleted rather than hydrogenated. They are
      bonded to nothing, so they are TEM-fit artefacts rather than chemistry.
    - Carbons whose *only* bond is a stretched one are deleted for the same
      reason. These have no neighbour inside CC_CUTOFF at all and hang off the
      sheet at 1.87-2.05 Å, so the stretched-bond rescue above is the only thing
      standing between them and coordination 0. Left in they cost twice: the stub
      is capped as a coord1_terminus =CH2 that is not there, and the fake bond
      pushes its partner to coordination 3 so a real defect-edge carbon is
      silently denied its H. See stretched_coord1_carbons. Deleting one changes a
      surviving atom's coordination, unlike the other two deletions, so the whole
      bonding analysis is recomputed with them masked out before anything is
      classified. Labelled stretched_coord1_deleted; --keep-stretched-coord1
      restores the old behaviour.

      This is on by default, and 59 carbons across 41 of the 95 structures fall
      into it (A02, A08-A10, B03, B06-B08, C08-C10, D01, D02, D06, D10, E01,
      E02, E04, E06, E07, E09, F01, F03-F07, F10, G01, H01-H08, H10, I03, I06,
      I07 - none in J). Only H02 has been regenerated since the rule was added,
      where it removed C370, C379 and C381 and handed C179, C371 and C181 the H
      they had been denied. The committed cluster output for the other 40
      predates the rule and will change the next time it is regenerated;
      n_stretched_coord1_deleted in the summary is how to tell which run a table
      came from.

      Not covered: 22 carbons repo-wide have no bond inside CC_CUTOFF but *two*
      stretched ones, so they read coordination 2 and take an edge H. Whether
      they are the same artefact is a separate judgement. None are in H02.
    - Carbon not connected to the main sheet is deleted as well. I08, J02 and
      J04 each carry a detached island in the top-right corner (17, 4 and 7
      atoms), which is TEM field-of-view debris rather than part of the
      structure; left in, they become free hydrocarbons floating beside the
      cluster. See disconnected_fragment_atoms. This is only visible
      non-periodically - under the minimum image the islands bond to the sheet
      through the cell boundary - which is one more symptom of the same problem
      described above.
    - geometry.in is written with no lattice_vector lines, which is how FHI-aims
      is told the structure is a cluster.

    - H that would land inside a ring of the sheet is rejected. The pairwise
      clash filters cannot see this, because a ring centre is a comfortable
      1.4-1.6 Å from every ring carbon. The test is topological rather than
      metric: the faces of the planar carbon graph are enumerated exactly and a
      candidate inside a small one is refused, so the tilt sweep keeps looking.
      See MAX_CAGE_RING_AREA and carbon_ring_faces. Cluster mode only - face
      enumeration needs a planar embedding, which the minimum image destroys.

Deleting atoms shifts every later index, so the site table gains
output_atom_index (the atom's index in the written geometry, -1 if deleted) and
h_atom_indices (the indices of the H it produced). Join on these, not on
atom_index, when reading a cluster geometry.

Cluster runs write to separate output paths (cluster_full_saturation,
cluster_saturation_sites, cluster_saturation_summary.csv) so the periodic results
are never overwritten.

Input layout:
    A/1/geometry.in
    A/2/geometry.in
    ...
    I/10/geometry.in
    J/1/geometry.in
    ...
    J/5/geometry.in

Example usage:

    # Test one specific structure
    python scripts/hydrogenation/full_saturate_all_structures.py --structure-id C08

    # Test first structure only
    python scripts/hydrogenation/full_saturate_all_structures.py --limit 1

    # Run all 95 structures
    python scripts/hydrogenation/full_saturate_all_structures.py

    # Non-periodic cluster treatment, all 95 structures
    python scripts/hydrogenation/full_saturate_all_structures.py --cluster

Outputs (periodic default):
    data/h_saturated/full_saturation/A01/geometry.in
    results/tables/full_saturation_summary.csv
    results/tables/full_saturation_sites/A01_saturation_sites.csv
    results/figures/full_saturation_site_maps/A01_h_site_selection.png

Outputs (--cluster):
    data/h_saturated/cluster_full_saturation/A01/geometry.in
    results/tables/cluster_saturation_summary.csv
    results/tables/cluster_saturation_sites/A01_saturation_sites.csv
    results/figures/cluster_saturation_site_maps/A01_h_site_selection.png
"""

from pathlib import Path
import argparse
import math
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

# Graphene C-C bond length is around 1.42 Å.
# Use a generous cutoff to catch distorted C-C bonds.
CC_CUTOFF = 1.85

# Second-tier cutoff for stretched C-C bonds (see heal_stretched_bonds).
# These TEM-derived backbones contain C-C bonds a little beyond CC_CUTOFF. Left
# uncounted they fake a missing bond on a carbon that is really coordination-3,
# and the "missing" direction points straight at the partner carbon, so the H
# lands on top of it. Bonds in [CC_CUTOFF, CC_CUTOFF_STRETCHED) are therefore
# also counted, but only between two carbons that are both still
# undercoordinated, so bulk carbons cannot be over-bonded.
CC_CUTOFF_STRETCHED = 2.05

# C-H bond length for initial H placement.
CH_BOND_LENGTH = 1.09

# A coordination-2 atom within this distance of a frame edge (the carbon
# bounding box) counts as flush against that edge. Roughly one C-C bond.
FRAME_BAND = 2.0

# NOTE: this test is inert. `direction` is a unit vector, so its dot product with
# the outward normal is always >= -1 and the direction half of the crop test in
# classify_h_sites always passes; the crop mask is purely positional (FRAME_BAND).
# Left as-is deliberately: restoring it (e.g. to -cos(45 degrees)) would change
# which sites are selected, which is a separate decision from H placement.
CROP_DIRECTION_THRESHOLD = -1

# Only coordination-2 atoms are selected automatically as H sites.
TARGET_H_COORDINATION = 2

# A coordination-2 carbon whose C-C-C angle reaches this is treated as sp, not
# sp2, and is NOT hydrogenated: an sp carbon's two remaining electrons are in pi
# orbitals, so there is no dangling sigma orbital for an H to cap. These are the
# carbyne-like chains around the central tear in the most eroded frames
# (I08-I10, J01-J05), where the C-C bonds also shorten to 1.1-1.3 Å against
# graphene's 1.42. Measured over all 95 structures this removes 91 of 3220
# internal sites, 58 of them in I/J and the other 33 scattered singly through
# A-H.
#
# The sp2 edge population peaks at 120-140 degrees and thins out past 150, so
# anywhere in 150-170 is defensible and the boundary is a judgement call rather
# than a sharp feature: in J05, for instance, carbons sit at 153.4, 154.4, 156.1
# and 157.4 degrees, so moving the cut across that range takes them one at a
# time. 155 is the working default. Set above 180 to disable the test entirely.
SP_ANGLE_THRESHOLD_DEG = 155.0

# Steric clash filters for H placement. These are chemistry, not just
# anti-overlap guards: a placement this close is rejected and the H is retried at
# a tilted in-plane angle (see place_h_in_plane) rather than abandoned.
MIN_HH_DISTANCE = 1.40
MIN_NONBONDED_HC_DISTANCE = 1.40

# Cluster-mode fallback thresholds, used only after the full-circle sweep at the
# normal thresholds has failed for every direction.
#
# The justification is that in cluster mode this file produces a *starting guess*
# for a GFN2-xTB relaxation, not a final geometry, and the alternative to a tight
# initial contact is a bare carbon - a dangling-bond radical that puts a spurious
# state in the middle of the gap and defeats the point of going non-periodic.
# A contact this size is comfortably inside GFN2's ability to push apart, and the
# relaxation audit checks the final distances independently, so a placement that
# does not resolve cannot pass unnoticed. Sites placed this way are recorded with
# status "accepted_tight" so they can be counted and inspected.
#
# These situations are real but rare: they occur where the TEM fit has put two
# undercoordinated carbons ~2.5 Å apart, so the hydrogens they both want occupy
# the same space (the same geometry noted for the periodic run in CLAUDE.md).
MIN_HH_DISTANCE_TIGHT = 1.00
MIN_NONBONDED_HC_DISTANCE_TIGHT = 1.10

# Cage filter: reject an H that ends up inside a ring of the carbon network
# rather than hanging off the outside of it.
#
# The pairwise distance filters above cannot see this. A point at the centre of a
# hexagon is 1.4-1.6 Å from all six ring carbons, which clears MIN_NONBONDED_HC_
# DISTANCE, so a position in the middle of the sheet looks exactly as good to
# them as one in open space. Sites landed there in the cluster run: I04 C336 at
# +175 degrees of tilt (min H-C 1.402 Å against the 1.400 Å floor, 0.13 Å from a
# ring centre) and A02 C461 at -167.5 degrees. Both are the facing-coordination-2
# geometry described above - the neighbour's H got there first and blocked every
# direction on the dangling-bond side - except that the full-circle sweep kept
# going past 90 degrees and flipped the H onto the far side of its own carbon.
#
# The test is topological, not metric: the carbon skeleton drawn with straight
# bonds is a planar straight-line graph, so its faces can be enumerated exactly
# (carbon_ring_faces), and the question is simply which face the candidate lands
# in. An angular "how enclosed is this point" heuristic was tried first and is
# not good enough - a hexagon on the rim of a tear is open on one side, so an H
# at its centre still sees a 135 degree clear sector and passes. That is how A07
# C422, H07 and I09 acquired hydrogens inside 5- and 6-rings.
#
# Not every bounded face is a cage. An H on an internal defect edge points into
# the hole it borders, which is a bounded face too and is the intended placement.
# Size separates them cleanly, and with a wide margin: across the 95 cluster
# structures every wrong placement sat in a 5- or 6-ring of 3.7-5.7 Å², while the
# legitimate case is a 16-carbon pore of 26.7 Å². Anything below the cut is a
# ring the H cannot fit inside; anything above it is a pore it can point into.
# 12 Å² sits in the gap and also catches 7- and 8-rings (7.4 and 9.9 Å² at the
# graphene bond length). Set to 0 to disable.
#
# Applied inside check_h_candidate, so a rejected direction only makes the sweep
# keep looking; a site is left bare (status "unplaceable") only when no direction
# clears.
MAX_CAGE_RING_AREA = 12.0

# In-plane angular search used when the plain bisector direction clashes. The H
# is rotated about the sheet normal (z) by +/- these angles, smallest first, and
# the first non-clashing direction wins. Rotation about z preserves the z
# coordinate exactly, so a planar input stays planar.
H_TILT_ANGLES_DEG = np.arange(0.0, 90.5, 2.5)

# Cluster mode sweeps the full circle instead of half of it. The periodic run
# could afford to give up on a site - a crop-rim carbon left bare there is
# continuous with its own image - but an isolated cluster cannot, because every
# unterminated carbon is a dangling bond radical that puts a spurious state in
# the middle of the gap. Some of these TEM backbones are locally very distorted
# (J05 has C-C pairs at 1.14-1.22 Å against graphene's 1.42), and where a
# neighbour sits that close the only directions clearing the steric filters are
# more than 90 degrees off the bisector. Smallest tilt still wins, so this only
# changes the outcome for sites that would otherwise have been left bare.
H_TILT_ANGLES_DEG_FULL = np.arange(0.0, 180.5, 2.5)

# Tilt sweep for the coordination-1 carbon that terminates an sp chain, which is
# capped with a single H along the chain axis (see coord1_axial_direction).
#
# This one is deliberately narrow rather than the half or full circle used above.
# An sp C-H is collinear with the chain by construction, so the tilt here is not
# a search over equally good directions the way it is for an sp2 bisector - every
# degree of it is a departure from the geometry being asked for. Bending the cap
# 60 or 90 degrees to dodge a contact would produce something that is no longer
# an alkyne terminus at all, which is worse than the bare carbon it replaces.
#
# The cap exists only to clear marginal contacts. Across the 95 structures 10 of
# the 12 sp termini place on the bare axis and I10 C316 clears at 2.5 degrees.
#
# The twelfth, J02 C297, is left bare, and is worth reading before widening this:
# it and J02 C296 are the ends of two different chains, 2.974 Å apart with C296's
# axis only 8.1 degrees off the line joining them, so the two chains point almost
# straight at each other. Their caps would be 0.8 Å apart. No tilt fixes that -
# the sweep would only trade a good geometry for a bad one - and neither would the
# relaxation, which holds the carbon fixed. The structure as fitted has room for
# one of the two caps, C296 takes it, and C297 is reported unplaceable. Capping
# one radical beats capping neither; which one is decided by atom index, and the
# two sites are otherwise equivalent.
H_TILT_ANGLES_DEG_SP_TERMINUS = np.arange(0.0, 15.1, 2.5)

# Set to True if you want atom labels on every coord-2 atom in the plots.
LABEL_COORD2_ATOMS = False


# ---------------------------------------------------------------------
# Geometry I/O
# ---------------------------------------------------------------------

def read_fhi_aims_geometry(path, require_lattice=True):
    """
    Read lattice vectors and atoms from a FHI-aims geometry.in-style file.

    With require_lattice=False a file carrying no lattice_vector lines is
    accepted and lattice is returned as None, which is how FHI-aims itself
    distinguishes a cluster from a periodic slab. A file with a partial set of
    lattice vectors is always an error.
    """
    lattice = []
    symbols = []
    positions = []

    with open(path, "r") as handle:
        for line in handle:
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()

            if parts[0] == "lattice_vector":
                lattice.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif parts[0] == "atom":
                x, y, z = map(float, parts[1:4])
                symbol = parts[4]

                positions.append([x, y, z])
                symbols.append(symbol)

    n_lattice = len(lattice)
    positions = np.array(positions, dtype=float)

    if n_lattice == 3:
        lattice = np.array(lattice, dtype=float)
    elif n_lattice == 0 and not require_lattice:
        lattice = None
    else:
        raise ValueError(
            "Expected {} lattice vectors in {}, found {}.".format(
                3 if require_lattice else "3 or 0", path, n_lattice
            )
        )

    if len(symbols) == 0:
        raise ValueError("No atoms found in {}.".format(path))

    return lattice, symbols, positions


def write_fhi_aims_geometry(path, lattice, symbols, positions):
    """
    Write FHI-aims geometry.in format.

    lattice=None writes no lattice_vector lines at all, which is how FHI-aims is
    told the structure is a cluster rather than a periodic slab.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as handle:
        for vec in (lattice if lattice is not None else []):
            handle.write(
                "lattice_vector   {:16.8f} {:16.8f} {:16.8f}\n".format(
                    vec[0], vec[1], vec[2]
                )
            )

        for symbol, pos in zip(symbols, positions):
            handle.write(
                "atom             {:16.8f} {:16.8f} {:16.8f} {}\n".format(
                    pos[0], pos[1], pos[2], symbol
                )
            )


# ---------------------------------------------------------------------
# Structure ordering and selection
# ---------------------------------------------------------------------

def structure_id_from_path(geom_file):
    """
    Convert A/1/geometry.in -> A01, C/8/geometry.in -> C08.
    """
    letter = geom_file.parents[1].name
    number = geom_file.parent.name

    try:
        number = "{:02d}".format(int(number))
    except ValueError:
        pass

    return "{}{}".format(letter, number)


def numeric_folder_sort_key(path):
    """
    Sort A/1, A/2, ..., A/10 in numerical order.
    """
    name = path.parent.name

    if name.isdigit():
        return int(name)

    return name


def find_all_geometry_files(root):
    """
    Find all A-J geometry.in files.
    """
    geometry_files = []

    for letter in list("ABCDEFGHIJ"):
        letter_dir = root / letter

        if not letter_dir.exists():
            print("Warning: missing directory {}".format(letter_dir))
            continue

        files = list(letter_dir.glob("*/geometry.in"))
        files = sorted(files, key=numeric_folder_sort_key)

        geometry_files.extend(files)

    return geometry_files


def filter_geometry_files_by_structure_id(geometry_files, structure_id):
    """
    Keep only one structure, e.g. A01, C08, J05.

    Matching is case-insensitive.
    """
    structure_id = structure_id.upper()

    matching_files = []

    for geom_file in geometry_files:
        this_id = structure_id_from_path(geom_file).upper()

        if this_id == structure_id:
            matching_files.append(geom_file)

    if len(matching_files) == 0:
        available_ids = [structure_id_from_path(p) for p in geometry_files]

        raise ValueError(
            "Could not find structure_id '{}'. Available examples: {}".format(
                structure_id,
                ", ".join(available_ids[:20]),
            )
        )

    return matching_files


# ---------------------------------------------------------------------
# Periodic geometry helpers
# ---------------------------------------------------------------------

def minimum_image_displacement(ri, rj, lattice, inv_lattice_t):
    """
    Return minimum-image displacement from ri to rj.

    The lattice is stored as rows:
        lattice[0] = a
        lattice[1] = b
        lattice[2] = c

    The inverse lattice is passed in so it is not recomputed repeatedly.

    lattice=None means cluster mode: the plain Cartesian difference is returned
    and no image is considered. This one branch is what makes the whole
    bonding analysis non-periodic, because every coordination, bond-direction
    and clash test in this module goes through here. It matters: these crops
    fill their cell almost exactly (A01 spans 44.1 Å of a 45.41 Å cell), so
    under the minimum image the crop rim bonds to its own periodic image and
    produces 4-coordinate sp2 carbon that does not exist in the sheet.
    """
    delta_cart = rj - ri

    if lattice is None:
        return delta_cart

    delta_frac = inv_lattice_t @ delta_cart
    delta_frac -= np.round(delta_frac)

    return lattice.T @ delta_frac


def minimum_image_distance(ri, rj, lattice, inv_lattice_t):
    """Return minimum-image distance between two Cartesian positions."""
    disp = minimum_image_displacement(ri, rj, lattice, inv_lattice_t)
    return float(np.linalg.norm(disp))


# ---------------------------------------------------------------------
# Coordination calculation
# ---------------------------------------------------------------------

def compute_coordination(
    symbols, positions, lattice, inv_lattice_t, cutoff=CC_CUTOFF, exclude=None
):
    """
    Compute C-C coordination numbers using minimum-image PBC distances.

    All A-J structures are genuine periodic DFT slab supercells (real
    lattice vectors, used with k_grid 2 2 1), so bonds across the periodic
    boundary must be counted or boundary atoms are undercoordinated.

    exclude is a set of atom indices to treat as absent. They are given
    coordination 0 and no neighbours, and contribute no bond to anyone else, so
    the analysis reads as if they had been deleted from the structure. Used to
    re-run the bonding analysis after a prune whose removal changes a surviving
    atom's coordination - see stretched_coord1_carbons.
    """
    n_atoms = len(symbols)

    exclude = set() if exclude is None else set(exclude)

    coordination = np.zeros(n_atoms, dtype=int)
    neighbour_indices = [[] for _ in range(n_atoms)]

    cutoff2 = cutoff * cutoff

    for i in range(n_atoms):
        if symbols[i] != "C" or i in exclude:
            continue

        ri = positions[i]

        for j in range(i + 1, n_atoms):
            if symbols[j] != "C" or j in exclude:
                continue

            rj = positions[j]
            disp = minimum_image_displacement(ri, rj, lattice, inv_lattice_t)
            d2 = float(np.dot(disp, disp))

            if d2 < cutoff2:
                coordination[i] += 1
                coordination[j] += 1
                neighbour_indices[i].append(j)
                neighbour_indices[j].append(i)

    return coordination, neighbour_indices


def heal_stretched_bonds(
    symbols,
    positions,
    coordination,
    neighbour_indices,
    lattice,
    inv_lattice_t,
    cutoff=CC_CUTOFF,
    stretched_cutoff=CC_CUTOFF_STRETCHED,
    exclude=None,
):
    """
    Count stretched C-C bonds that sit just beyond the primary cutoff.

    A carbon pair separated by [cutoff, stretched_cutoff) is bonded only if both
    atoms are still undercoordinated (fewer than 3 C neighbours). Bulk carbons
    already at coordination 3 are never given a fourth bond, so this cannot
    over-bond the sheet.

    Pairs are considered shortest-first and coordination is updated as bonds are
    added, so a carbon healed to coordination 3 stops accepting further bonds and
    the reclassification cascades correctly.

    Without this, a coordination-2 carbon whose third bond is merely stretched
    looks undercoordinated, and its "missing bond" direction points straight at
    the partner carbon that is already there.

    exclude is a set of atom indices to treat as absent, with the same meaning as
    in compute_coordination: they neither gain nor donate a stretched bond.

    Mutates coordination and neighbour_indices in place. Returns the number of
    bonds healed.
    """
    candidates = []

    n_atoms = len(symbols)

    exclude = set() if exclude is None else set(exclude)

    for i in range(n_atoms):
        if symbols[i] != "C" or i in exclude:
            continue

        for j in range(i + 1, n_atoms):
            if symbols[j] != "C" or j in exclude:
                continue

            distance = minimum_image_distance(
                positions[i], positions[j], lattice, inv_lattice_t
            )

            if cutoff <= distance < stretched_cutoff:
                candidates.append((distance, i, j))

    candidates.sort()

    n_healed = 0

    for _, i, j in candidates:
        if coordination[i] >= 3 or coordination[j] >= 3:
            continue

        coordination[i] += 1
        coordination[j] += 1
        neighbour_indices[i].append(j)
        neighbour_indices[j].append(i)

        n_healed += 1

    return n_healed


def carbon_connected_components(symbols, neighbour_indices):
    """
    Group the carbons into connected components of the C-C bond graph.

    Walks neighbour_indices rather than recomputing distances, so the components
    inherit both the primary cutoff from compute_coordination and the extra bonds
    added by heal_stretched_bonds, and they follow whichever periodicity built
    that adjacency. Call it after healing: a stretched bond is a real bond, and a
    piece hanging off the sheet by one 1.9 Å link is not a separate fragment.

    Returns a list of index lists, largest component first. Coordination-0
    carbons come back as their own single-atom components.
    """
    seen = set()
    components = []

    for start in range(len(symbols)):
        if symbols[start] != "C" or start in seen:
            continue

        seen.add(start)
        queue = [start]
        member = []

        while queue:
            index = queue.pop()
            member.append(index)

            for neighbour in neighbour_indices[index]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)

        components.append(sorted(member))

    components.sort(key=len, reverse=True)

    return components


def disconnected_fragment_atoms(symbols, neighbour_indices):
    """
    Return the set of carbons that are not part of the main sheet.

    Three of the 95 crops (I08, J02, J04) carry a detached island of carbon in
    the top-right corner - 17, 4 and 7 atoms, standing 2.3, 6.0 and 4.8 Å clear
    of the sheet. They are TEM field-of-view debris rather than part of the
    structure under study, and left in place they get hydrogenated into free
    hydrocarbon molecules floating beside the cluster.

    Everything outside the largest component is dropped. In the other 92
    structures the only non-largest components are single coordination-0 carbons,
    which cluster mode already deletes, so this changes nothing there.

    Under periodic minimum-image bonding these islands bond to the sheet through
    the cell boundary and no fragment is found at all, which is one more reason
    the crops are the wrong thing to treat periodically (see the module
    docstring). Pruning is therefore only wired up in cluster mode.
    """
    components = carbon_connected_components(symbols, neighbour_indices)

    if len(components) <= 1:
        return set()

    return set(index for component in components[1:] for index in component)


def stretched_coord1_carbons(symbols, coordination, hard_coordination):
    """
    Return the carbons attached to the sheet by a stretched bond and nothing else.

    These have *no* neighbour inside CC_CUTOFF and exactly one bond added by
    heal_stretched_bonds, so they hang off the sheet at 1.87-2.05 Å against
    graphene's 1.42 and a longest real C-C in these backbones of about 1.6. That
    is not a bond, it is the same TEM-fit artefact that coord0_deleted already
    removes - a carbon placed too far from the sheet for the fit to close - and
    the stretched-bond rescue is the only reason it does not read as
    coordination 0. The rescue is right for the case it was written for, a
    genuine coordination-2 carbon whose *third* bond is merely long, and wrong
    here.

    Leaving them in is wrong twice over. The stub is hydrogenated as a
    coord1_terminus and acquires a =CH2 that is not there, and its partner is
    pushed to coordination 3 by the same fake bond, so a real defect-edge carbon
    is silently denied its H. In H02 the three stubs C370, C379 and C381 cost six
    spurious hydrogens and suppressed three real sites at C179, C371 and C181.

    Pass hard_coordination, the coordination taken *before* healing, alongside
    the healed coordination. Deleting these atoms changes their partner's
    coordination, so unlike coord0_deleted and disconnected_fragment the caller
    has to recompute the bonding analysis with them excluded rather than just
    flag them - see process_one_structure.

    One pass is always enough and cannot cascade: membership requires zero bonds
    inside CC_CUTOFF, and removing atoms never creates a short bond, so no
    surviving carbon can join the set as a result of the prune.

    Cluster mode only, for the same reason as disconnected_fragment_atoms: the
    periodic tree is a finished record that is not being regenerated.
    """
    return set(
        index
        for index, symbol in enumerate(symbols)
        if symbol == "C"
        and int(hard_coordination[index]) == 0
        and int(coordination[index]) == 1
    )


# ---------------------------------------------------------------------
# Missing-bond direction
# ---------------------------------------------------------------------

def unit_vector(vec):
    """
    Return unit vector, or None if vector is too small.
    """
    norm = float(np.linalg.norm(vec))

    if norm < 1.0e-12:
        return None

    return vec / norm


def missing_bond_direction(
    carbon_index, positions, neighbour_indices, lattice, inv_lattice_t
):
    """
    Return the unit outward direction of the missing bond on a coord-2 carbon.

        missing direction = -sum(unit vectors to the two C neighbours)

    Neighbour directions use minimum-image displacement, since a
    coordination-2 carbon near a periodic boundary may have a neighbour
    whose stored Cartesian position is a full lattice vector away.

    Where the two neighbours are collinear with the carbon the bisector cancels
    and this direction is undefined. The missing bond is then perpendicular to
    the C-C-C axis, still in the sheet plane, so the in-plane normal is returned
    instead. Its sense is arbitrary, but place_h_in_plane searches both senses
    anyway.

    Returns None only if the atom is not coordination-2.
    """
    neighbours = neighbour_indices[carbon_index]

    if len(neighbours) != 2:
        return None

    parent_pos = positions[carbon_index]

    directions = []

    for neighbour_index in neighbours:
        disp = minimum_image_displacement(
            parent_pos, positions[neighbour_index], lattice, inv_lattice_t
        )
        direction = unit_vector(disp)

        if direction is None:
            return None

        directions.append(direction)

    bisector = unit_vector(-1.0 * (directions[0] + directions[1]))

    if bisector is not None:
        return bisector

    # Collinear neighbours: take the in-plane perpendicular to the C-C-C axis.
    axis = directions[0]

    return unit_vector(np.array([-axis[1], axis[0], 0.0], dtype=float))


def coord2_bond_angle_deg(
    carbon_index, positions, neighbour_indices, lattice, inv_lattice_t
):
    """
    Return the C-C-C angle in degrees at a coordination-2 carbon.

    This is the hybridisation discriminator. Around 120-130 degrees the carbon is
    sp2 and its third sigma orbital is a genuine dangling bond wanting an H;
    approaching 180 degrees it is sp, the two remaining electrons sit in pi
    orbitals, and there is nothing for an H to cap (see SP_ANGLE_THRESHOLD_DEG).

    Uses the same minimum-image displacements as missing_bond_direction, so the
    cluster mode (lattice=None) branch is inherited unchanged.

    Returns NaN if the atom is not coordination-2 or a bond vector is degenerate.
    """
    neighbours = neighbour_indices[carbon_index]

    if len(neighbours) != 2:
        return float("nan")

    parent_pos = positions[carbon_index]

    directions = []

    for neighbour_index in neighbours:
        disp = minimum_image_displacement(
            parent_pos, positions[neighbour_index], lattice, inv_lattice_t
        )
        direction = unit_vector(disp)

        if direction is None:
            return float("nan")

        directions.append(direction)

    cos_angle = float(np.clip(np.dot(directions[0], directions[1]), -1.0, 1.0))

    return float(np.degrees(np.arccos(cos_angle)))


def sp_chain_terminus_angle_deg(
    carbon_index, positions, neighbour_indices, lattice, inv_lattice_t
):
    """
    Return the C-C-C angle at the single neighbour of a coordination-1 carbon.

    This is the hybridisation test of coord2_bond_angle_deg applied one bond
    further out. A coordination-1 carbon has no angle of its own, so what decides
    how it should be capped is the carbon it hangs off:

        neighbour near-linear (sp)  -> the pair is the end of a carbyne chain,
                                       so this carbon is an alkyne terminus and
                                       takes one H along the chain axis.
        neighbour bent or coord-3   -> the terminus is sp2 and wants two H at
                                       +/-120 degrees (coord1_h_directions).

    Returns NaN when the atom is not coordination-1, when its neighbour is not
    coordination-2 (so it has no angle), or when a bond vector is degenerate.
    Callers compare against SP_ANGLE_THRESHOLD_DEG, and NaN fails that comparison,
    which is the intended "not an sp terminus" answer in every one of those cases.
    """
    neighbours = neighbour_indices[carbon_index]

    if len(neighbours) != 1:
        return float("nan")

    return coord2_bond_angle_deg(
        carbon_index=neighbours[0],
        positions=positions,
        neighbour_indices=neighbour_indices,
        lattice=lattice,
        inv_lattice_t=inv_lattice_t,
    )


# ---------------------------------------------------------------------
# H-site classification
# ---------------------------------------------------------------------

def crop_frame_side(x, y, frame_bounds, direction):
    """
    Return the crop-frame edge label an atom is flush against, or "" for none.

    frame_bounds is (x_min, x_max, y_min, y_max) of the carbon bounding box. An
    atom is flush against an edge when it lies within FRAME_BAND of it *and* its
    missing-bond direction points out of that same edge, with component at least
    CROP_DIRECTION_THRESHOLD. The direction half of that test is currently inert
    by construction - see the note on CROP_DIRECTION_THRESHOLD.

    Direction None means the caller could not determine a missing-bond direction,
    which leaves the crop test undefined; "" is returned so the atom keeps its
    default classification rather than being labelled a crop artefact on the
    strength of position alone.
    """
    if direction is None:
        return ""

    x_min, x_max, y_min, y_max = frame_bounds

    # (label, distance from that frame edge, outward normal of that edge)
    frame_edges = [
        ("TOP", y_max - float(y), np.array([0.0, 1.0])),
        ("BOT", float(y) - y_min, np.array([0.0, -1.0])),
        ("LEFT", float(x) - x_min, np.array([-1.0, 0.0])),
        ("RIGHT", x_max - float(x), np.array([1.0, 0.0])),
    ]

    for label, distance_to_edge, outward_normal in frame_edges:
        if distance_to_edge > FRAME_BAND:
            continue

        if float(np.dot(direction[:2], outward_normal)) >= CROP_DIRECTION_THRESHOLD:
            return label

    return ""


def classify_h_sites(
    symbols, positions, coordination, neighbour_indices, lattice, inv_lattice_t,
    cluster=False, sp_angle_deg=SP_ANGLE_THRESHOLD_DEG, fragment_atoms=None,
    stretched_coord1_atoms=None,
):
    """
    Classify all carbon atoms and select real coordination-2 H sites.

    Each structure is a rectangular crop out of a larger TEM field of view. The
    frame edges (top, bottom, left, right of the carbon bounding box) slice
    through continuous graphene, and the coordination-2 atoms they create are
    artefacts of the crop, not real edges. The pull/crop axis is vertical
    (Cartesian y) and identical in all 95 frames, but the material necks and
    shifts, so the frame box is recomputed per structure.

    A coordination-2 atom is an artificial crop atom when both hold:
        - it lies within FRAME_BAND of a frame edge, and
        - its missing-bond direction points out of that same frame edge, with
          component >= CROP_DIRECTION_THRESHOLD.

    Both tests are needed. Direction alone would reject interior tear-rim atoms
    that happen to point downwards; position alone would reject atoms sitting
    low in the frame that point up into a real hole.

    A second, independent test overrides hydrogenation on either side of that
    split: a coordination-2 carbon whose C-C-C angle is at least sp_angle_deg is
    sp rather than sp2, so it has no dangling sigma orbital and gets no H. These
    are the carbyne-like chains running around the central tear in the most
    eroded frames. Such a carbon keeps its crop/internal identity in the label -
    internal_sp_linear_coord2 or external_crop_sp_linear_coord2 - because
    scripts/dos/define_dos_regions.py needs that distinction, and a near-linear
    carbon can sit on either side of the frame band (across the 95 structures,
    91 are internal and 23 are on the frame). The angle is recorded in
    ccc_angle_deg for *every* coordination-2 carbon, not just the excluded ones,
    so the threshold can be re-tuned from the site tables without a rerun.

    Everything else at coordination 2 is a real internal/torn edge and is
    selected for hydrogenation.

    The *end* of an sp chain is a separate case, and is selected in both modes. A
    coordination-1 carbon whose single neighbour is one of those near-linear
    carbons is the terminus of a carbyne chain, and unlike the chain itself it is
    not closed-shell: one sigma bond and no H is a dangling-bond radical, a worse
    defect than any of the sp carbons the exclusion above protects. It takes one H
    along the chain axis, giving ...C#C-H, and is labelled
    internal_sp_terminus_coord1 or external_crop_sp_terminus_coord1. The deciding
    angle is the neighbour's, so it is recorded in its own column,
    terminus_neighbour_angle_deg, rather than in ccc_angle_deg, which means "this
    atom's own angle" everywhere else. See sp_chain_terminus_angle_deg.

    cluster=True changes *what is selected*, not how it is labelled. In a
    periodic slab the crop-frame carbons are continuous with their own image, so
    leaving them bare is correct. As an isolated cluster there is nothing on the
    far side of the frame and a bare rim carbon is a dangling bond that plants
    spurious mid-gap states in the DOS, so every coordination-2 carbon is
    hydrogenated regardless of which side of the frame band it sits on. All
    coordination-1 carbons become candidates too - two H at +/-120 degrees
    (coord1_h_directions), or the single axial H above where the terminus is sp -
    and coordination-0 carbons are marked for deletion as TEM-fit artefacts.

    The sp exclusion applies in cluster mode as well. That is consistent with the
    reason cluster mode terminates the rim at all: a bare *sp2* carbon is a
    dangling-bond radical that plants a spurious mid-gap state, whereas a bare sp
    carbon is a closed-shell chain atom and is what the chemistry actually is.

    The site_class labels are unchanged either way, including the
    external_crop_boundary_coord2 / internal_defect_edge_coord2 distinction,
    because scripts/dos/define_dos_regions.py reads that column to tell a real
    torn edge from a crop artefact when it builds the DOS projection regions.
    Under cluster=True both are hydrogenated but they remain distinguishable.

    fragment_atoms is the set of carbons that are not part of the main sheet (see
    disconnected_fragment_atoms). They are marked disconnected_fragment and
    flagged for deletion, and they are excluded from the frame box: a detached
    island lying past the edge of the sheet would otherwise stretch the bounding
    box away from the material and shift the crop band with it. It moves the box
    by up to 0.76 Å in I08/J02/J04 and by nothing at all in the other 92.

    stretched_coord1_atoms is the set of carbons attached to the sheet by a
    stretched bond and nothing else (see stretched_coord1_carbons). They are
    marked stretched_coord1_deleted, flagged for deletion, and excluded from the
    frame box for the same reason as the fragments. They arrive here already
    excluded from the bonding analysis, so their coordination reads 0 and the
    label has to be applied after the coord0_deleted line to win. It leaves the
    box where it was in H02, the structure this was written for.

    Returns (df, frame_box) where frame_box is the closed 5-point frame
    rectangle for plotting.
    """
    rows = []

    for i, (symbol, position, coord) in enumerate(zip(symbols, positions, coordination)):
        if symbol != "C":
            continue

        rows.append(
            {
                "atom_index": i,
                "element": symbol,
                "coordination": int(coord),
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "neighbours": " ".join(str(j) for j in neighbour_indices[i]),
            }
        )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError("No carbon atoms found for H-site classification.")

    fragment_atoms = set() if fragment_atoms is None else set(int(i) for i in fragment_atoms)

    stretched_coord1_atoms = (
        set()
        if stretched_coord1_atoms is None
        else set(int(i) for i in stretched_coord1_atoms)
    )

    is_fragment = df["atom_index"].isin(fragment_atoms)
    is_stretched_coord1 = df["atom_index"].isin(stretched_coord1_atoms)

    sheet = df[~(is_fragment | is_stretched_coord1)]

    if len(sheet) == 0:
        raise ValueError("Every carbon was classed as a disconnected fragment.")

    x_min = float(sheet["x"].min())
    x_max = float(sheet["x"].max())
    y_min = float(sheet["y"].min())
    y_max = float(sheet["y"].max())

    frame_box = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min],
        ],
        dtype=float,
    )

    frame_bounds = (x_min, x_max, y_min, y_max)

    df["site_class"] = "bulk_or_other"
    df["crop_side"] = ""
    df["ccc_angle_deg"] = np.nan
    df["terminus_neighbour_angle_deg"] = np.nan
    df["is_candidate_h_site"] = False
    df["n_h_target"] = 0
    df["delete_atom"] = False

    # Coordination-3 atoms are treated as bulk-like.
    df.loc[df["coordination"] == 3, "site_class"] = "bulk_coord3"

    # Coordination 0/1 atoms are unusual. They are kept apart from each other: a
    # coordination-1 carbon is a real chain terminus with a well-defined bond
    # axis, whereas a coordination-0 carbon has no bond direction at all and is
    # usually a TEM-fit artefact (see module docstring).
    if cluster:
        # A coordination-1 carbon is sp2 with three sigma bonds: one C-C and two
        # C-H at +/-120 degrees, an ethylene-like terminus that is genuinely
        # planar. A coordination-0 carbon is unbonded to anything and cannot be
        # rescued by hydrogenation, so it is dropped rather than turned into a
        # stray methane floating in the middle of the cluster.
        df.loc[df["coordination"] == 0, "site_class"] = "coord0_deleted"
        df.loc[df["coordination"] == 0, "delete_atom"] = True

        df.loc[df["coordination"] == 1, "site_class"] = "coord1_terminus"
        df.loc[df["coordination"] == 1, "is_candidate_h_site"] = True
        df.loc[df["coordination"] == 1, "n_h_target"] = 2
    else:
        df.loc[df["coordination"] == 0, "site_class"] = "coord0_inspect"
        df.loc[df["coordination"] == 1, "site_class"] = "coord1_inspect"

    # A carbon in a detached island is dropped whatever its coordination. The
    # label is only applied above coordination 0, because an unbonded carbon is
    # its own single-atom component and would otherwise stop being counted as
    # coord0_deleted, which is the more specific description of what it is.
    df.loc[is_fragment & (df["coordination"] > 0), "site_class"] = "disconnected_fragment"
    df.loc[is_fragment, "delete_atom"] = True
    df.loc[is_fragment, "is_candidate_h_site"] = False
    df.loc[is_fragment, "n_h_target"] = 0

    # A carbon held to the sheet only by a stretched bond is the same TEM-fit
    # artefact as a coordination-0 one and is dropped alongside it. It was
    # excluded from the bonding analysis before this point, so it reads
    # coordination 0 here and the coord0_deleted line above has already claimed
    # it; this relabels it with the more specific description of why it went.
    df.loc[is_stretched_coord1, "site_class"] = "stretched_coord1_deleted"
    df.loc[is_stretched_coord1, "delete_atom"] = True
    df.loc[is_stretched_coord1, "is_candidate_h_site"] = False
    df.loc[is_stretched_coord1, "n_h_target"] = 0

    for row_index, row in df.iterrows():
        if int(row["coordination"]) not in (1, TARGET_H_COORDINATION):
            continue

        # Deleted atoms are not sites. Skipping them here also keeps them out of
        # the crop-band test, which is meaningless for a fragment sitting outside
        # the frame box that the rest of the sheet defines.
        if int(row["atom_index"]) in fragment_atoms:
            continue

        if int(row["coordination"]) == 1:
            # The angle that decides this is the neighbour's, not this atom's.
            terminus_angle = sp_chain_terminus_angle_deg(
                carbon_index=int(row["atom_index"]),
                positions=positions,
                neighbour_indices=neighbour_indices,
                lattice=lattice,
                inv_lattice_t=inv_lattice_t,
            )

            df.at[row_index, "terminus_neighbour_angle_deg"] = terminus_angle

            if not (np.isfinite(terminus_angle) and terminus_angle >= sp_angle_deg):
                # An sp2 terminus. Keeps whichever default this mode already set:
                # coord1_inspect and no H when periodic, coord1_terminus and two H
                # at +/-120 degrees when clustered.
                continue

            axial = coord1_axial_direction(
                carbon_index=int(row["atom_index"]),
                positions=positions,
                neighbour_indices=neighbour_indices,
                lattice=lattice,
                inv_lattice_t=inv_lattice_t,
            )

            crop_side = crop_frame_side(
                row["x"], row["y"], frame_bounds,
                None if axial is None else axial[0],
            )

            if crop_side:
                df.at[row_index, "crop_side"] = crop_side
                df.at[row_index, "site_class"] = "external_crop_sp_terminus_coord1"
                # Same rule as the coord-2 crop rim: continuous with its own image
                # in a periodic slab, a real dangling bond in an isolated cluster.
                df.at[row_index, "is_candidate_h_site"] = bool(cluster)
                df.at[row_index, "n_h_target"] = 1 if cluster else 0
            else:
                df.at[row_index, "site_class"] = "internal_sp_terminus_coord1"
                df.at[row_index, "is_candidate_h_site"] = True
                df.at[row_index, "n_h_target"] = 1

            continue

        direction = missing_bond_direction(
            carbon_index=int(row["atom_index"]),
            positions=positions,
            neighbour_indices=neighbour_indices,
            lattice=lattice,
            inv_lattice_t=inv_lattice_t,
        )

        angle = coord2_bond_angle_deg(
            carbon_index=int(row["atom_index"]),
            positions=positions,
            neighbour_indices=neighbour_indices,
            lattice=lattice,
            inv_lattice_t=inv_lattice_t,
        )

        df.at[row_index, "ccc_angle_deg"] = angle

        crop_side = crop_frame_side(row["x"], row["y"], frame_bounds, direction)

        # sp rather than sp2: no dangling sigma orbital, so no H, in either mode.
        # The crop/internal distinction is still recorded, because that is what
        # scripts/dos/define_dos_regions.py uses to tell a real torn edge from a
        # frame artefact, and a near-linear carbon can sit on either. 91 of the
        # 114 sp carbons across the 95 structures are internal, 23 on the frame.
        is_sp = bool(np.isfinite(angle) and angle >= sp_angle_deg)

        if crop_side:
            df.at[row_index, "crop_side"] = crop_side
            df.at[row_index, "site_class"] = (
                "external_crop_sp_linear_coord2" if is_sp
                else "external_crop_boundary_coord2"
            )
            # Periodic: continuous with its own image, leave bare.
            # Cluster: nothing on the far side, so terminate it - unless it is
            # sp, which has nothing to terminate.
            df.at[row_index, "is_candidate_h_site"] = bool(cluster) and not is_sp
            df.at[row_index, "n_h_target"] = 1 if (cluster and not is_sp) else 0
        elif is_sp:
            df.at[row_index, "site_class"] = "internal_sp_linear_coord2"
            df.at[row_index, "is_candidate_h_site"] = False
            df.at[row_index, "n_h_target"] = 0
        else:
            df.at[row_index, "site_class"] = "internal_defect_edge_coord2"
            df.at[row_index, "is_candidate_h_site"] = True
            df.at[row_index, "n_h_target"] = 1

    return df, frame_box


# ---------------------------------------------------------------------
# H placement
# ---------------------------------------------------------------------

def rotate_about_z(vec, angle_deg):
    """
    Rotate a vector about the sheet normal (z).

    The z component is carried through untouched, so an H placed from a direction
    produced here stays in exactly the plane of the carbon it hangs off.
    """
    angle = np.radians(angle_deg)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    return np.array(
        [
            cos_a * vec[0] - sin_a * vec[1],
            sin_a * vec[0] + cos_a * vec[1],
            vec[2],
        ],
        dtype=float,
    )


def place_h_in_plane(
    carbon_index,
    missing_direction,
    symbols,
    positions,
    new_h_positions,
    lattice,
    inv_lattice_t,
    tilt_angles=None,
    min_hc=MIN_NONBONDED_HC_DISTANCE,
    min_hh=MIN_HH_DISTANCE,
    ring_polygons=None,
):
    """
    Find an in-plane H position for one coordination-2 carbon.

    Tries the missing-bond bisector first, then rotates away from it about z in
    increasing steps, taking the smallest tilt that clears the steric filters.
    Both rotation senses are tried at each angle.

    A tilt is needed where two coordination-2 carbons face each other across a
    narrow (~2.5 Å) tear: both bisectors point into the gap, so the un-tilted H
    would land on top of each other. Tilting them apart in-plane resolves this
    without breaking planarity.

    Returns (h_pos, tilt_deg, reason). h_pos is None if no in-plane direction
    works, in which case reason explains the closest failure.
    """
    parent_pos = positions[carbon_index]

    last_reason = "no in-plane direction clears the steric filters"

    if tilt_angles is None:
        tilt_angles = H_TILT_ANGLES_DEG

    for angle in tilt_angles:
        senses = (0.0,) if angle == 0.0 else (angle, -angle)

        for signed_angle in senses:
            direction = rotate_about_z(missing_direction, signed_angle)

            candidate_pos = parent_pos + CH_BOND_LENGTH * direction

            accepted, reason = check_h_candidate(
                candidate_pos=candidate_pos,
                parent_c_index=carbon_index,
                symbols=symbols,
                positions=positions,
                new_h_positions=new_h_positions,
                lattice=lattice,
                inv_lattice_t=inv_lattice_t,
                min_hc=min_hc,
                min_hh=min_hh,
                ring_polygons=ring_polygons,
            )

            if accepted:
                return candidate_pos, float(signed_angle), ""

            last_reason = reason

    return None, np.nan, last_reason


def coord1_h_directions(carbon_index, positions, neighbour_indices, lattice, inv_lattice_t):
    """
    Return the two in-plane H directions for a coordination-1 carbon.

    Such a carbon is a chain terminus with one C-C bond. Adding two hydrogens
    makes three sigma bonds, i.e. sp2, so all three lie in a plane at 120
    degrees: the two H directions are the C-C bond direction rotated by +120 and
    -120 degrees about the sheet normal. This is an ethylene-like terminus and is
    genuinely planar, so it is valid in both the planar and the free relaxation.

    Returns None if the single neighbour direction is degenerate.
    """
    neighbours = neighbour_indices[carbon_index]

    if len(neighbours) != 1:
        return None

    disp = minimum_image_displacement(
        positions[carbon_index], positions[neighbours[0]], lattice, inv_lattice_t
    )
    bond_direction = unit_vector(disp)

    if bond_direction is None:
        return None

    return [
        rotate_about_z(bond_direction, 120.0),
        rotate_about_z(bond_direction, -120.0),
    ]


def coord1_axial_direction(carbon_index, positions, neighbour_indices, lattice, inv_lattice_t):
    """
    Return the single H direction for a coordination-1 carbon ending an sp chain.

    Where coord1_h_directions gives the sp2 answer (two H at +/-120 degrees), this
    is the sp one: a single H continuing the chain axis, so the terminus reads
    ...C(sp)#C-H. The direction is therefore the C-C bond direction reversed.

    The two are not interchangeable. An sp2 =CH2 terminus needs a pi bond from its
    neighbour to be closed-shell, and an sp chain carbon has both of its pi
    systems committed along the chain, so it has none to give. Putting two H there
    would also bend a linear terminus by 60 degrees. Which one applies is decided
    by sp_chain_terminus_angle_deg, not by coordination alone.

    Returned as a one-element list so callers can treat it exactly like
    coord1_h_directions. Returns None if the bond vector is degenerate.
    """
    neighbours = neighbour_indices[carbon_index]

    if len(neighbours) != 1:
        return None

    disp = minimum_image_displacement(
        positions[carbon_index], positions[neighbours[0]], lattice, inv_lattice_t
    )
    bond_direction = unit_vector(disp)

    if bond_direction is None:
        return None

    return [-1.0 * bond_direction]


def polygon_area(polygon):
    """Absolute area of a simple polygon given as a list of (x, y)."""
    total = 0.0

    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1

    return abs(0.5 * total)


def point_in_polygon(polygon, x, y):
    """Ray-crossing test for a point against a simple polygon."""
    inside = False
    previous = len(polygon) - 1

    for current in range(len(polygon)):
        x1, y1 = polygon[current]
        x2, y2 = polygon[previous]

        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside

        previous = current

    return inside


def carbon_ring_cycles(symbols, positions, neighbour_indices, exclude=None):
    """
    Enumerate every bounded ring face of the carbon skeleton.

    The carbon skeleton with straight bonds is a planar straight-line graph, so
    its faces are enumerated exactly: walk the directed edges, and at each vertex
    always leave along the neighbour immediately clockwise from the one you
    arrived on. Every directed edge belongs to exactly one face, so the walk
    partitions them. The face of largest area is the unbounded one and is
    dropped.

    exclude drops atoms from the graph entirely, which is how disconnected
    fragments are kept out - otherwise the island's own outer boundary would come
    back as a bounded face of the whole picture.

    Returns [(cycle, polygon, area), ...] where cycle is the list of carbon
    indices in traversal order, polygon the matching list of (x, y), and area the
    absolute polygon area. The atom indices are what makes this usable for ring
    tables and plots; carbon_ring_faces() throws them away for speed.
    """
    exclude = set() if exclude is None else set(int(i) for i in exclude)

    carbons = [
        index
        for index, symbol in enumerate(symbols)
        if symbol == "C" and index not in exclude
    ]
    carbon_set = set(carbons)

    # Neighbours of each vertex sorted by bearing, so "next clockwise" is just
    # the previous entry in the list.
    order = {}

    for vertex in carbons:
        neighbours = [w for w in neighbour_indices[vertex] if w in carbon_set]
        neighbours.sort(
            key=lambda w: math.atan2(
                positions[w][1] - positions[vertex][1],
                positions[w][0] - positions[vertex][0],
            )
        )
        order[vertex] = neighbours

    unused = set(
        (vertex, neighbour) for vertex in carbons for neighbour in order[vertex]
    )

    cycles = []

    while unused:
        edge = next(iter(unused))
        cycle = []

        while edge in unused:
            unused.discard(edge)
            cycle.append(edge[0])

            tail, head = edge
            ring = order[head]
            following = ring[(ring.index(tail) - 1) % len(ring)]
            edge = (head, following)

        # A pendant bond walks out and straight back, giving a 2-cycle with no
        # interior. Nothing below a triangle can contain a point.
        if len(cycle) >= 3:
            cycles.append(cycle)

    if not cycles:
        return []

    polygons = [[(positions[i][0], positions[i][1]) for i in cycle] for cycle in cycles]
    areas = [polygon_area(polygon) for polygon in polygons]

    outer = int(np.argmax(areas))

    return [
        (cycles[index], polygons[index], areas[index])
        for index in range(len(cycles))
        if index != outer
    ]


def carbon_ring_faces(
    symbols, positions, neighbour_indices, exclude=None, max_area=MAX_CAGE_RING_AREA,
):
    """
    Map each carbon to the small rings it sits on the boundary of.

    Faces come from carbon_ring_cycles(); of them, only those below max_area are
    cages - see MAX_CAGE_RING_AREA for why a large bounded face is a pore rather
    than a cage and must be kept.

    Keyed by carbon so the placement test is cheap: a candidate H is 1.09 Å from
    its parent, so the only faces that can possibly contain it are the two or
    three that parent borders. That turns the test into a couple of
    point-in-polygon calls per direction instead of a scan over every ring.

    exclude has the same meaning as in carbon_ring_cycles().

    Returns {carbon index: [polygon, ...]} where each polygon is a list of (x, y)
    and only carbons bordering at least one small ring appear.
    """
    faces_by_carbon = {}

    for cycle, polygon, area in carbon_ring_cycles(
        symbols, positions, neighbour_indices, exclude=exclude
    ):
        if area >= max_area:
            continue

        for carbon_index in cycle:
            faces_by_carbon.setdefault(carbon_index, []).append(polygon)

    return faces_by_carbon


def check_h_candidate(
    candidate_pos, parent_c_index, symbols, positions, new_h_positions, lattice, inv_lattice_t,
    min_hc=MIN_NONBONDED_HC_DISTANCE, min_hh=MIN_HH_DISTANCE,
    ring_polygons=None,
):
    """
    Reject severe H-H or non-bonded H-C clashes, using minimum-image
    distances so clashes across a periodic boundary are detected, and reject
    positions inside a ring of the carbon network (see carbon_ring_faces).

    ring_polygons is the list of small rings the parent carbon borders, which is
    the complete set of rings this candidate could possibly fall inside.
    """
    for atom_index, (symbol, pos) in enumerate(zip(symbols, positions)):
        if atom_index == parent_c_index:
            continue

        distance = minimum_image_distance(candidate_pos, pos, lattice, inv_lattice_t)

        if symbol == "C" and distance < min_hc:
            return False, "H-C clash {:.3f} Å with atom {}".format(distance, atom_index)

        if symbol == "H" and distance < min_hh:
            return False, "H-H clash {:.3f} Å with existing H atom {}".format(
                distance, atom_index
            )

    for h_index, h_pos in enumerate(new_h_positions):
        distance = minimum_image_distance(candidate_pos, h_pos, lattice, inv_lattice_t)

        if distance < min_hh:
            return False, "H-H clash {:.3f} Å with newly added H {}".format(
                distance, h_index
            )

    for polygon in ring_polygons or []:
        if point_in_polygon(polygon, float(candidate_pos[0]), float(candidate_pos[1])):
            return False, "inside a {}-carbon ring of area {:.1f} Å²".format(
                len(polygon), polygon_area(polygon)
            )

    return True, ""


def add_h_to_candidate_sites(
    symbols, positions, site_df, neighbour_indices, lattice, inv_lattice_t,
    reject_clashes=True, cluster=False, ring_faces=None,
    min_hh=MIN_HH_DISTANCE, min_hc=MIN_NONBONDED_HC_DISTANCE,
):
    """
    Add H to every selected candidate site.

    Coordination-2 carbons take one H on the missing-bond bisector. An sp chain
    terminus takes one along the chain axis (coord1_axial_direction), in both
    modes. In cluster mode every other coordination-1 carbon takes two, at
    +/-120 degrees from its single C-C bond (coord1_h_directions). Every H is
    placed in the plane of the sheet (see place_h_in_plane); the two H of a
    coordination-1 terminus are placed one after the other, so the second sees
    the first in its clash test and tilts away from it if it has to.

    Sites are not placed in atom-index order. Every H already down is an obstacle
    to the next one, so the sites with the least freedom go first - see
    placement_order below.

    A site is only left bare when no in-plane direction at all clears the steric
    filters, which is recorded as status "unplaceable". Where a site takes two H
    and only one can be placed the status is "partial".

    min_hh and min_hc are the steric floors of the normal placement pass and
    default to the production constants. They are parameters rather than globals
    so the placement can be re-run at other floors without editing the module
    (see scripts/sensitivity/h_placement_threshold_sensitivity.py). They do not
    touch the tight retry below, which keeps its own MIN_*_TIGHT constants.

    In cluster mode, carbons flagged delete_atom are dropped from the returned
    structure: coordination-0 carbons, and any carbon in a disconnected fragment
    (see disconnected_fragment_atoms). Neither is bonded to the sheet, so
    removing them changes no other atom's bonding. Because that shifts the
    indices of every later atom, an output_atom_index column is written giving
    each atom's index in the returned structure (-1 for deleted), so the site
    table can still be joined against the written geometry.

    ring_faces maps a carbon to the small rings it borders (carbon_ring_faces).
    A candidate H is 1.09 Å from its parent, so those are the only rings it could
    land inside, and the cage test is a couple of point-in-polygon calls.
    """
    new_symbols = list(symbols)
    new_positions = [np.array(pos, dtype=float) for pos in positions]

    added_h_positions = []
    updated_rows = []

    # H index in the output structure, recorded per site so a site can be traced
    # to the hydrogens it produced. tight_output_indices records the subset that
    # needed the relaxed thresholds, so the audit and the site map can point at
    # the individual compromised hydrogen rather than its whole site.
    h_output_indices = {}
    tight_output_indices = {}

    # Sites are placed sp termini first, then in atom-index order as before.
    #
    # Order matters because each H is a hard obstacle for every H placed after
    # it, and the sites are not equally free to move: an sp2 site has a 90 (or
    # 180) degree sweep to fall back on, while an sp terminus has one admissible
    # direction and a 15 degree cap, because for it the axis *is* the answer.
    # Letting the free site claim the space first and then declaring the
    # constrained one unplaceable gets the priority exactly backwards - and it is
    # what happened: I10 C316 and C306 were both blocked by an H on a
    # coordination-2 neighbour that had a whole half-circle of equally good
    # alternatives it never had to use.
    #
    # This is deliberately narrow. Ordering *all* of coordination-1 first is a
    # bad trade - those sites want two H each and cost more coordination-2 sites
    # than they save - so only the 12 sp termini are promoted.
    #
    # The output row order is unchanged; this reorders placement only.
    site_classes = [str(v) for v in site_df["site_class"]]
    atom_indices = [int(v) for v in site_df["atom_index"]]
    placement_order = sorted(
        range(len(site_df)),
        key=lambda k: (
            not site_classes[k].endswith("_sp_terminus_coord1"),
            atom_indices[k],
        ),
    )

    for row in (site_df.iloc[k] for k in placement_order):
        row_dict = row.to_dict()

        carbon_index = int(row["atom_index"])

        row_dict["n_h_proposed"] = 0
        row_dict["n_h_added"] = 0

        row_dict["h_x"] = np.nan
        row_dict["h_y"] = np.nan
        row_dict["h_z"] = np.nan
        row_dict["h_tilt_deg"] = np.nan

        row_dict["h_x_2"] = np.nan
        row_dict["h_y_2"] = np.nan
        row_dict["h_z_2"] = np.nan
        row_dict["h_tilt_deg_2"] = np.nan

        row_dict["addition_status"] = "not_candidate"
        row_dict["addition_reason"] = ""
        row_dict["n_h_tight"] = 0

        if cluster and bool(row.get("delete_atom", False)):
            if str(row.get("site_class", "")) == "disconnected_fragment":
                row_dict["addition_status"] = "deleted_fragment"
                row_dict["addition_reason"] = (
                    "carbon in a disconnected fragment, removed as TEM debris"
                )
            elif str(row.get("site_class", "")) == "stretched_coord1_deleted":
                row_dict["addition_status"] = "deleted_stretched_coord1"
                row_dict["addition_reason"] = (
                    "carbon attached only by a stretched bond, removed as a "
                    "TEM-fit artefact"
                )
            else:
                row_dict["addition_status"] = "deleted_coord0"
                row_dict["addition_reason"] = (
                    "coordination-0 carbon removed as a TEM-fit artefact"
                )
            updated_rows.append(row_dict)
            continue

        if not bool(row["is_candidate_h_site"]):
            updated_rows.append(row_dict)
            continue

        coordination = int(row["coordination"])
        site_class = str(row.get("site_class", ""))

        # An sp chain terminus is dispatched on site_class rather than
        # coordination: it is coordination-1 like the sp2 termini, but it is the
        # neighbour's hybridisation that decides the geometry, and classify_h_sites
        # has already made that call.
        is_sp_terminus = site_class.endswith("_sp_terminus_coord1")

        if is_sp_terminus:
            directions = coord1_axial_direction(
                carbon_index=carbon_index,
                positions=positions,
                neighbour_indices=neighbour_indices,
                lattice=lattice,
                inv_lattice_t=inv_lattice_t,
            )
            direction_label = "sp chain axis"
        elif coordination == 1:
            directions = coord1_h_directions(
                carbon_index=carbon_index,
                positions=positions,
                neighbour_indices=neighbour_indices,
                lattice=lattice,
                inv_lattice_t=inv_lattice_t,
            )
            direction_label = "120 deg sp2 terminus"
        else:
            single = missing_bond_direction(
                carbon_index=carbon_index,
                positions=positions,
                neighbour_indices=neighbour_indices,
                lattice=lattice,
                inv_lattice_t=inv_lattice_t,
            )
            directions = None if single is None else [single]
            direction_label = "bisector"

        if directions is None:
            row_dict["n_h_proposed"] = int(row.get("n_h_target", 1))
            row_dict["addition_status"] = "skipped"
            row_dict["addition_reason"] = "Could not determine missing-bond direction"
            updated_rows.append(row_dict)
            continue

        row_dict["n_h_proposed"] = len(directions)

        parent_rings = (ring_faces or {}).get(carbon_index, [])

        n_added = 0
        n_tight = 0
        reasons = []

        if is_sp_terminus:
            # Narrow by design, and the same in both modes: the axis *is* the
            # answer here, so a wide sweep would only trade a correct geometry
            # for an incorrect one. See H_TILT_ANGLES_DEG_SP_TERMINUS.
            tilt_angles = H_TILT_ANGLES_DEG_SP_TERMINUS
        else:
            tilt_angles = H_TILT_ANGLES_DEG_FULL if cluster else H_TILT_ANGLES_DEG

        for slot, base_direction in enumerate(directions):
            suffix = "" if slot == 0 else "_2"

            h_pos, tilt_deg, reason = place_h_in_plane(
                carbon_index=carbon_index,
                missing_direction=base_direction,
                symbols=new_symbols,
                positions=np.array(new_positions),
                new_h_positions=added_h_positions,
                lattice=lattice,
                inv_lattice_t=inv_lattice_t,
                tilt_angles=tilt_angles,
                min_hc=min_hc,
                min_hh=min_hh,
                ring_polygons=parent_rings,
            )

            placed_tight = False

            # An sp terminus is deliberately excluded from the tight retry. That
            # pass buys room by relaxing the clash floors *and* sweeping the full
            # circle; with the tilt capped at 15 degrees there is nowhere else for
            # this H to go, so a failure here is real and should be reported
            # rather than resolved by moving the H somewhere it does not belong.
            if h_pos is None and cluster and not is_sp_terminus:
                # Second pass at the tight thresholds. A bare carbon here is a
                # dangling-bond radical in an isolated cluster, which is worse
                # than a close initial contact that the GFN2 relaxation can open
                # up. See MIN_HH_DISTANCE_TIGHT.
                h_pos, tilt_deg, tight_reason = place_h_in_plane(
                    carbon_index=carbon_index,
                    missing_direction=base_direction,
                    symbols=new_symbols,
                    positions=np.array(new_positions),
                    new_h_positions=added_h_positions,
                    lattice=lattice,
                    inv_lattice_t=inv_lattice_t,
                    tilt_angles=H_TILT_ANGLES_DEG_FULL,
                    min_hc=MIN_NONBONDED_HC_DISTANCE_TIGHT,
                    min_hh=MIN_HH_DISTANCE_TIGHT,
                    # The tight pass relaxes how close an H may sit, not where it
                    # is allowed to sit. A caged H is wrong at any threshold, and
                    # the xtb relaxation cannot push it out of a ring it is
                    # already inside. Leaving the cage test out of this pass is
                    # what put hydrogens inside 5- and 6-rings in A07, H07 and
                    # I09: those sites failed the normal pass, fell through to
                    # here, and the full-circle sweep flipped them into a ring.
                    ring_polygons=parent_rings,
                )

                if h_pos is None:
                    reason = tight_reason
                else:
                    placed_tight = True

            if h_pos is None and not reject_clashes:
                # --force-clashes: fall back to the un-tilted direction regardless.
                h_pos = positions[carbon_index] + CH_BOND_LENGTH * base_direction
                tilt_deg = 0.0
                reasons.append("forced despite clash: {}".format(reason))

            elif h_pos is None:
                reasons.append(reason)
                continue

            else:
                detail = (
                    "accepted on {}".format(direction_label)
                    if tilt_deg == 0.0
                    else "accepted with {:+.1f} deg in-plane tilt".format(tilt_deg)
                )

                if placed_tight:
                    detail += (
                        " at tight thresholds (H-C {:.2f} / H-H {:.2f} Å);"
                        " relies on the xtb relaxation to open the contact".format(
                            MIN_NONBONDED_HC_DISTANCE_TIGHT, MIN_HH_DISTANCE_TIGHT
                        )
                    )
                    n_tight += 1

                reasons.append(detail)

            h_output_indices.setdefault(carbon_index, []).append(len(new_symbols))

            if placed_tight:
                tight_output_indices.setdefault(carbon_index, []).append(len(new_symbols))

            new_symbols.append("H")
            new_positions.append(np.array(h_pos, dtype=float))
            added_h_positions.append(np.array(h_pos, dtype=float))

            row_dict["h_x" + suffix] = h_pos[0]
            row_dict["h_y" + suffix] = h_pos[1]
            row_dict["h_z" + suffix] = h_pos[2]
            row_dict["h_tilt_deg" + suffix] = tilt_deg

            n_added += 1

        row_dict["n_h_added"] = n_added
        row_dict["n_h_tight"] = n_tight
        row_dict["addition_reason"] = "; ".join(reasons)

        if n_added == 0:
            row_dict["addition_status"] = "unplaceable"
        elif n_added < len(directions):
            row_dict["addition_status"] = "partial"
        elif any(reason.startswith("forced") for reason in reasons):
            row_dict["addition_status"] = "forced"
        elif n_tight:
            row_dict["addition_status"] = "accepted_tight"
        else:
            row_dict["addition_status"] = "accepted"

        updated_rows.append(row_dict)

    # Back to atom-index order: placement_order above is a placement schedule, not
    # a change to the site table, and drop_deleted_atoms below indexes rows
    # against the atom order.
    updated_df = (
        pd.DataFrame(updated_rows)
        .sort_values("atom_index", kind="stable")
        .reset_index(drop=True)
    )

    new_positions = np.array(new_positions, dtype=float)

    if cluster:
        new_symbols, new_positions, output_index_map = drop_deleted_atoms(
            new_symbols, new_positions, updated_df
        )
    else:
        output_index_map = {i: i for i in range(len(new_symbols))}

    updated_df["output_atom_index"] = [
        output_index_map.get(int(idx), -1) for idx in updated_df["atom_index"]
    ]
    updated_df["h_atom_indices"] = [
        " ".join(
            str(output_index_map[h_index])
            for h_index in h_output_indices.get(int(idx), [])
        )
        for idx in updated_df["atom_index"]
    ]
    updated_df["h_tight_indices"] = [
        " ".join(
            str(output_index_map[h_index])
            for h_index in tight_output_indices.get(int(idx), [])
        )
        for idx in updated_df["atom_index"]
    ]

    return new_symbols, new_positions, updated_df


def drop_deleted_atoms(symbols, positions, site_df):
    """
    Remove atoms flagged delete_atom from a structure.

    Returns (symbols, positions, output_index_map) where output_index_map maps an
    index in the input structure to its index in the returned one. Deleted atoms
    are absent from the map.
    """
    deleted = set(
        int(row["atom_index"])
        for _, row in site_df.iterrows()
        if bool(row.get("delete_atom", False))
    )

    kept_symbols = []
    kept_positions = []
    output_index_map = {}

    for index, (symbol, pos) in enumerate(zip(symbols, positions)):
        if index in deleted:
            continue

        output_index_map[index] = len(kept_symbols)
        kept_symbols.append(symbol)
        kept_positions.append(pos)

    return kept_symbols, np.array(kept_positions, dtype=float), output_index_map


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def save_selection_plot(df, frame_box, output_figure, title, cluster=False):
    """
    Plot selected and excluded H-site candidates, with the TEM frame box.

    Every subset is selected by site_class, never by is_candidate_h_site. The two
    are not interchangeable in cluster mode, where crop-rim coordination-2 atoms
    and coordination-1 termini are candidates too: selecting on the flag pulled
    all three groups into one "selected internal H sites" colour, so a
    coordination-1 terminus (which takes two H) was indistinguishable from a
    coordination-2 site (which takes one), and the crop-boundary markers were
    painted over by the candidate markers drawn above them - the legend counted
    40 of them while none were visible. Both cluster site_class spellings are
    matched here so the coordination-0/1 groups are not silently empty.
    """
    output_figure.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    bulk = df[df["site_class"] == "bulk_coord3"]
    external = df[df["site_class"] == "external_crop_boundary_coord2"]
    sp_linear = df[
        df["site_class"].isin(
            ["internal_sp_linear_coord2", "external_crop_sp_linear_coord2"]
        )
    ]
    internal = df[df["site_class"] == "internal_defect_edge_coord2"]
    sp_terminus = df[
        df["site_class"].isin(
            ["internal_sp_terminus_coord1", "external_crop_sp_terminus_coord1"]
        )
    ]
    coord0 = df[df["site_class"].isin(["coord0_inspect", "coord0_deleted"])]
    coord1 = df[df["site_class"].isin(["coord1_inspect", "coord1_terminus"])]
    other = df[df["site_class"] == "bulk_or_other"]
    fragment = df[df["site_class"] == "disconnected_fragment"]
    stretched_coord1 = df[df["site_class"] == "stretched_coord1_deleted"]

    if len(bulk) > 0:
        ax.scatter(
            bulk["x"],
            bulk["y"],
            s=16,
            c="lightgrey",
            edgecolors="black",
            linewidths=0.4,
            label="coordination 3 / bulk-like ({})".format(len(bulk)),
            zorder=1,
        )

    if len(other) > 0:
        ax.scatter(
            other["x"],
            other["y"],
            s=20,
            c="silver",
            edgecolors="black",
            linewidths=0.4,
            label="other coordination ({})".format(len(other)),
            zorder=1,
        )

    if len(external) > 0:
        ax.scatter(
            external["x"],
            external["y"],
            s=45,
            c="deepskyblue",
            edgecolors="black",
            linewidths=0.6,
            label="coordination 2 crop boundary - {} ({})".format(
                "1 H each" if cluster else "excluded", len(external)
            ),
            zorder=2,
        )

    if len(sp_linear) > 0:
        ax.scatter(
            sp_linear["x"],
            sp_linear["y"],
            s=70,
            c="mediumpurple",
            edgecolors="black",
            linewidths=0.8,
            marker="D",
            label="coordination 2 excluded sp / near-linear ({})".format(
                len(sp_linear)
            ),
            zorder=4,
        )

    if len(internal) > 0:
        ax.scatter(
            internal["x"],
            internal["y"],
            s=80,
            c="limegreen",
            edgecolors="black",
            linewidths=0.8,
            label="internal defect-edge H sites, 1 H each ({})".format(len(internal)),
            zorder=4,
        )

    if len(sp_terminus) > 0:
        # Same purple as the sp chain it caps, but filled and marked as a site:
        # these are the one atom in that chain that does take an H.
        ax.scatter(
            sp_terminus["x"],
            sp_terminus["y"],
            s=90,
            c="mediumpurple",
            edgecolors="limegreen",
            linewidths=1.6,
            marker="P",
            label="sp chain terminus, 1 axial H each ({})".format(len(sp_terminus)),
            zorder=5,
        )

    if len(coord1) > 0:
        ax.scatter(
            coord1["x"],
            coord1["y"],
            s=100,
            c="orange",
            edgecolors="black",
            linewidths=0.8,
            marker="s",
            label="coordination 1 - {} ({})".format(
                "terminus, 2 H each" if cluster else "inspect, not hydrogenated",
                len(coord1),
            ),
            zorder=5,
        )

    if len(coord0) > 0:
        ax.scatter(
            coord0["x"],
            coord0["y"],
            s=130,
            c="crimson",
            edgecolors="black",
            linewidths=0.8,
            marker="X",
            label="coordination 0 - {} ({})".format(
                "deleted" if cluster else "inspect, not hydrogenated", len(coord0)
            ),
            zorder=6,
        )

    if len(fragment) > 0:
        # Drawn above the frame box, because the point of the marker is to show
        # that these atoms sit outside it.
        ax.scatter(
            fragment["x"],
            fragment["y"],
            s=60,
            c="none",
            edgecolors="crimson",
            linewidths=1.2,
            marker="s",
            label="disconnected fragment - deleted ({})".format(len(fragment)),
            zorder=5,
        )

    if len(stretched_coord1) > 0:
        # Same crimson as the other two deletions, with its own marker: these
        # atoms sit inside the frame and next to the sheet, so position alone
        # does not say why they went.
        ax.scatter(
            stretched_coord1["x"],
            stretched_coord1["y"],
            s=130,
            c="none",
            edgecolors="crimson",
            linewidths=1.4,
            marker="D",
            label="stretched bond only - deleted ({})".format(len(stretched_coord1)),
            zorder=6,
        )

    ax.plot(
        frame_box[:, 0],
        frame_box[:, 1],
        c="black",
        linewidth=1.2,
        linestyle="--",
        label="TEM frame box",
        zorder=3,
    )

    if LABEL_COORD2_ATOMS:
        label_df = df[df["coordination"] == TARGET_H_COORDINATION]

        for _, row in label_df.iterrows():
            ax.text(
                row["x"] + 0.20,
                row["y"] + 0.20,
                str(int(row["atom_index"])),
                fontsize=7,
                color="black",
                zorder=6,
            )

    # Always label coordination 0/1 atoms for inspection. The label carries the
    # coordination so a bare atom index in the figure is unambiguous.
    for _, row in pd.concat([coord0, coord1]).iterrows():
        ax.text(
            row["x"] + 0.25,
            row["y"] + 0.25,
            "{} (c{})".format(int(row["atom_index"]), int(row["coordination"])),
            fontsize=8,
            color="crimson" if int(row["coordination"]) == 0 else "black",
            zorder=7,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / Å", fontsize=16)
    ax.set_ylabel("y / Å", fontsize=16)
    ax.set_title(title, fontsize=18, pad=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="best", frameon=True, fontsize=9)

    pad = 3.0
    ax.set_xlim(df["x"].min() - pad, df["x"].max() + pad)
    ax.set_ylim(df["y"].min() - pad, df["y"].max() + pad)

    fig.tight_layout()
    fig.savefig(output_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------

def process_one_structure(
    geom_file,
    root,
    output_root,
    site_table_root,
    figure_root,
    reject_clashes=True,
    make_plots=True,
    cluster=False,
    sp_angle_deg=SP_ANGLE_THRESHOLD_DEG,
    max_cage_ring_area=MAX_CAGE_RING_AREA,
    delete_stretched_coord1=True,
    min_hh=MIN_HH_DISTANCE,
    min_hc=MIN_NONBONDED_HC_DISTANCE,
):
    """
    Process a single A01/C08/etc. structure.
    """
    structure_id = structure_id_from_path(geom_file)

    print()
    print("=" * 80)
    print("Processing {}: {}".format(structure_id, geom_file.relative_to(root)))
    print("=" * 80)

    file_lattice, symbols, positions = read_fhi_aims_geometry(
        geom_file, require_lattice=not cluster
    )

    # In cluster mode the input lattice is read but deliberately not used: it is
    # discarded here so every downstream distance is a plain Cartesian one, and
    # the output geometry carries no lattice_vector lines at all.
    lattice = None if cluster else file_lattice
    inv_lattice_t = None if lattice is None else np.linalg.inv(lattice.T)

    coordination, neighbour_indices = compute_coordination(
        symbols=symbols,
        positions=positions,
        lattice=lattice,
        inv_lattice_t=inv_lattice_t,
        cutoff=CC_CUTOFF,
    )

    # Keep the pre-healing coordination: it is what tells a carbon whose third
    # bond is merely long from one that has no short bond at all.
    hard_coordination = coordination.copy()

    n_healed = heal_stretched_bonds(
        symbols=symbols,
        positions=positions,
        coordination=coordination,
        neighbour_indices=neighbour_indices,
        lattice=lattice,
        inv_lattice_t=inv_lattice_t,
        cutoff=CC_CUTOFF,
        stretched_cutoff=CC_CUTOFF_STRETCHED,
    )

    # Cluster mode only, as below. Unlike every other deletion in this script,
    # dropping these carbons changes a *surviving* atom's coordination - the
    # partner that the stretched bond had pushed to 3 falls back to 2 and becomes
    # a real H site - so the bonding analysis has to be redone with them masked
    # out rather than simply flagged after the fact.
    stretched_coord1_atoms = (
        stretched_coord1_carbons(symbols, coordination, hard_coordination)
        if cluster and delete_stretched_coord1
        else set()
    )

    if stretched_coord1_atoms:
        coordination, neighbour_indices = compute_coordination(
            symbols=symbols,
            positions=positions,
            lattice=lattice,
            inv_lattice_t=inv_lattice_t,
            cutoff=CC_CUTOFF,
            exclude=stretched_coord1_atoms,
        )

        n_healed = heal_stretched_bonds(
            symbols=symbols,
            positions=positions,
            coordination=coordination,
            neighbour_indices=neighbour_indices,
            lattice=lattice,
            inv_lattice_t=inv_lattice_t,
            cutoff=CC_CUTOFF,
            stretched_cutoff=CC_CUTOFF_STRETCHED,
            exclude=stretched_coord1_atoms,
        )

    # Only cluster mode prunes. Under the periodic minimum image these islands
    # bond to the sheet through the cell boundary, so no fragment is detectable
    # there, and the periodic tree is a finished record that must not move.
    fragment_atoms = (
        disconnected_fragment_atoms(symbols, neighbour_indices) if cluster else set()
    )

    # A carbon already pruned above is not a fragment as well: it has no bonds
    # left, so it is its own component, and stretched_coord1_deleted is the more
    # specific description of why it went.
    fragment_atoms -= stretched_coord1_atoms

    site_df, frame_box = classify_h_sites(
        symbols=symbols,
        positions=positions,
        coordination=coordination,
        neighbour_indices=neighbour_indices,
        lattice=lattice,
        inv_lattice_t=inv_lattice_t,
        cluster=cluster,
        sp_angle_deg=sp_angle_deg,
        fragment_atoms=fragment_atoms,
        stretched_coord1_atoms=stretched_coord1_atoms,
    )

    # Face enumeration needs a planar straight-line embedding, which only holds
    # non-periodically: under the minimum image a bond that wraps the cell is
    # drawn as a long segment across the whole picture and the faces are
    # meaningless. Cluster mode is the mode that has the embedding, and the
    # periodic tree is a finished record that is not being regenerated, so the
    # cage test is wired up there only.
    ring_faces = (
        carbon_ring_faces(
            symbols=symbols,
            positions=positions,
            neighbour_indices=neighbour_indices,
            exclude=fragment_atoms | stretched_coord1_atoms,
            max_area=max_cage_ring_area,
        )
        if cluster and max_cage_ring_area > 0.0
        else {}
    )

    new_symbols, new_positions, updated_site_df = add_h_to_candidate_sites(
        symbols=symbols,
        positions=positions,
        site_df=site_df,
        neighbour_indices=neighbour_indices,
        lattice=lattice,
        inv_lattice_t=inv_lattice_t,
        reject_clashes=reject_clashes,
        cluster=cluster,
        ring_faces=ring_faces,
        min_hh=min_hh,
        min_hc=min_hc,
    )

    output_geom = output_root / structure_id / "geometry.in"
    site_table = site_table_root / "{}_saturation_sites.csv".format(structure_id)
    output_figure = figure_root / "{}_h_site_selection.png".format(structure_id)

    write_fhi_aims_geometry(output_geom, lattice, new_symbols, new_positions)

    site_table.parent.mkdir(parents=True, exist_ok=True)
    updated_site_df.to_csv(site_table, index=False)

    if make_plots:
        save_selection_plot(
            df=updated_site_df,
            frame_box=frame_box,
            output_figure=output_figure,
            title="{} {} H-site selection".format(
                structure_id, "cluster" if cluster else "internal"
            ),
            cluster=cluster,
        )

    n_c = sum(1 for sym in symbols if sym == "C")
    n_c_final = sum(1 for sym in new_symbols if sym == "C")
    n_h_initial = sum(1 for sym in symbols if sym == "H")
    n_h_final = sum(1 for sym in new_symbols if sym == "H")
    n_h_added = n_h_final - n_h_initial
    n_c_deleted = n_c - n_c_final

    n_coord0 = int((updated_site_df["coordination"] == 0).sum())
    n_coord1 = int((updated_site_df["coordination"] == 1).sum())
    n_coord2 = int((updated_site_df["coordination"] == 2).sum())
    n_coord3 = int((updated_site_df["coordination"] == 3).sum())
    n_coord_gt3 = int((updated_site_df["coordination"] > 3).sum())

    n_external_coord2 = int(
        (updated_site_df["site_class"] == "external_crop_boundary_coord2").sum()
    )
    is_sp_site = updated_site_df["site_class"].isin(
        ["internal_sp_linear_coord2", "external_crop_sp_linear_coord2"]
    )
    n_sp_linear_coord2 = int(
        (updated_site_df["site_class"] == "internal_sp_linear_coord2").sum()
    )
    n_sp_linear_crop_coord2 = int(
        (updated_site_df["site_class"] == "external_crop_sp_linear_coord2").sum()
    )
    sp_angles = updated_site_df.loc[is_sp_site, "ccc_angle_deg"]
    max_ccc_angle = float(updated_site_df["ccc_angle_deg"].max())
    n_internal_candidates = int(updated_site_df["is_candidate_h_site"].sum())
    n_accepted = int(
        updated_site_df["addition_status"].isin(["accepted", "accepted_tight"]).sum()
    )
    n_rejected = int((updated_site_df["addition_status"] == "rejected").sum())
    n_forced = int((updated_site_df["addition_status"] == "forced").sum())
    n_skipped = int((updated_site_df["addition_status"] == "skipped").sum())
    n_unplaceable = int((updated_site_df["addition_status"] == "unplaceable").sum())
    n_coord0_inspect = int((updated_site_df["site_class"] == "coord0_inspect").sum())
    # Excludes the sp termini by construction: classify_h_sites relabels those, so
    # a carbon that is now capped no longer counts as awaiting inspection.
    n_coord1_inspect = int((updated_site_df["site_class"] == "coord1_inspect").sum())
    n_inspect = n_coord0_inspect + n_coord1_inspect

    is_sp_terminus_site = updated_site_df["site_class"].isin(
        ["internal_sp_terminus_coord1", "external_crop_sp_terminus_coord1"]
    )
    n_sp_terminus_coord1 = int(
        (updated_site_df["site_class"] == "internal_sp_terminus_coord1").sum()
    )
    n_sp_terminus_crop_coord1 = int(
        (updated_site_df["site_class"] == "external_crop_sp_terminus_coord1").sum()
    )
    n_sp_terminus_h_added = int(
        updated_site_df.loc[is_sp_terminus_site, "n_h_added"].fillna(0).sum()
    )
    terminus_angles = updated_site_df.loc[
        is_sp_terminus_site, "terminus_neighbour_angle_deg"
    ].dropna()

    n_partial = int((updated_site_df["addition_status"] == "partial").sum())
    n_tight_sites = int((updated_site_df["addition_status"] == "accepted_tight").sum())
    n_tight_h = int(updated_site_df["n_h_tight"].fillna(0).sum())
    n_deleted_coord0 = int(
        (updated_site_df["addition_status"] == "deleted_coord0").sum()
    )
    n_deleted_fragment = int(
        (updated_site_df["addition_status"] == "deleted_fragment").sum()
    )
    n_deleted_stretched_coord1 = int(
        (updated_site_df["addition_status"] == "deleted_stretched_coord1").sum()
    )
    # Sites whose *final* failure was the cage test. A site that was merely
    # pushed onto a different tilt by it does not appear here, because
    # place_h_in_plane only reports a reason when every direction failed - which
    # is the useful reading anyway: these are the sites the test left bare.
    n_caged_sites = int(
        updated_site_df["addition_reason"]
        .fillna("")
        .str.contains("inside a .*-carbon ring", regex=True)
        .sum()
    )

    # Verification: re-test every H that was actually placed against the rings of
    # its own parent. Must be 0 - this is the property the whole filter exists to
    # guarantee, checked on the output rather than assumed from the input.
    n_h_in_ring = 0

    for _, placed_row in updated_site_df.iterrows():
        rings = ring_faces.get(int(placed_row["atom_index"]), [])

        if not rings:
            continue

        for suffix in ("", "_2"):
            hx = placed_row.get("h_x" + suffix)
            hy = placed_row.get("h_y" + suffix)

            if hx is None or not np.isfinite(hx):
                continue

            if any(point_in_polygon(polygon, float(hx), float(hy)) for polygon in rings):
                n_h_in_ring += 1

    # The point of cluster mode: after saturation no carbon should be left with a
    # dangling bond. Anything undercoordinated that did not get its full
    # complement of H is counted here so a whole run can be checked from the
    # summary CSV alone.
    n_bare_dangling_c = int(
        (
            (updated_site_df["coordination"] < 3)
            & (~updated_site_df.get("delete_atom", False))
            & (updated_site_df["n_h_added"] < updated_site_df["n_h_target"])
        ).sum()
    ) if cluster else int(
        (updated_site_df["addition_status"].isin(["unplaceable", "skipped"])).sum()
    )

    tilts = pd.concat(
        [updated_site_df["h_tilt_deg"], updated_site_df["h_tilt_deg_2"]]
    ).dropna()
    n_tilted = int((tilts.abs() > 0.0).sum())
    max_tilt = float(tilts.abs().max()) if len(tilts) else 0.0

    print("C atoms:", n_c)
    print("Coordination counts:")
    print("  coord 0:", n_coord0, "(inspect, not hydrogenated: {})".format(n_coord0_inspect))
    print("  coord 1:", n_coord1, "(inspect, not hydrogenated: {})".format(n_coord1_inspect))
    print("  coord 2:", n_coord2)
    print("  coord 3:", n_coord3)
    print("  coord >3:", n_coord_gt3)
    print("Stretched C-C bonds healed:", n_healed)
    print("Mode:", "cluster (non-periodic)" if cluster else "periodic slab")

    if cluster:
        print("Crop-boundary coord-2 atoms (hydrogenated in cluster mode):", n_external_coord2)
        print("Coord-0 carbons deleted:", n_deleted_coord0)
        print("Disconnected-fragment carbons deleted:", n_deleted_fragment)
        print(
            "Stretched coord-1 carbons deleted:", n_deleted_stretched_coord1
        )
    else:
        print("Excluded crop-boundary coord-2 atoms:", n_external_coord2)

    if len(sp_angles) > 0:
        print(
            "Excluded sp / near-linear coord-2 atoms:",
            n_sp_linear_coord2 + n_sp_linear_crop_coord2,
            "({} internal, {} on crop frame; >= {:.1f} deg, range {:.1f}-{:.1f})".format(
                n_sp_linear_coord2,
                n_sp_linear_crop_coord2,
                sp_angle_deg,
                float(sp_angles.min()),
                float(sp_angles.max()),
            ),
        )
    else:
        print(
            "Excluded sp / near-linear coord-2 atoms: 0",
            "(>= {:.1f} deg, widest angle present {:.1f})".format(
                sp_angle_deg, max_ccc_angle
            ),
        )

    if len(terminus_angles) > 0:
        print(
            "sp chain termini (coord-1, capped with one axial H):",
            n_sp_terminus_coord1 + n_sp_terminus_crop_coord1,
            "({} internal, {} on crop frame; neighbour angle {:.1f}-{:.1f} deg;"
            " {} H added)".format(
                n_sp_terminus_coord1,
                n_sp_terminus_crop_coord1,
                float(terminus_angles.min()),
                float(terminus_angles.max()),
                n_sp_terminus_h_added,
            ),
        )

    print("Selected H sites:", n_internal_candidates)
    print("H added:", n_h_added)
    print("  on bisector:", n_accepted - n_tilted)
    print("  with in-plane tilt:", n_tilted, "(max {:.1f} deg)".format(max_tilt))
    print("Partially hydrogenated sites:", n_partial)
    print("Sites needing tight thresholds:", n_tight_sites, "({} H)".format(n_tight_h))
    # The same polygon object is shared by every carbon on its boundary, so
    # counting distinct objects counts distinct rings.
    n_rings = len({id(polygon) for polygons in ring_faces.values() for polygon in polygons})

    print(
        "H sites left bare to avoid caging in a ring:",
        n_caged_sites,
        "({} rings below {:.1f} Å²)".format(n_rings, max_cage_ring_area),
    )
    print("H placed inside a carbon ring (must be 0):", n_h_in_ring)
    print("Unplaceable H sites (left bare):", n_unplaceable)
    print("Carbons left with a dangling bond:", n_bare_dangling_c)
    print("Carbons with coordination > 3 (must be 0 in cluster mode):", n_coord_gt3)
    print("Output geometry:", output_geom.relative_to(root))

    return {
        "structure_id": structure_id,
        "mode": "cluster" if cluster else "periodic",
        "input_geometry": str(geom_file.relative_to(root)),
        "output_geometry": str(output_geom.relative_to(root)),
        "site_table": str(site_table.relative_to(root)),
        "site_figure": str(output_figure.relative_to(root)) if make_plots else "",
        "n_atoms_initial": len(symbols),
        "n_atoms_final": len(new_symbols),
        "n_C": n_c,
        "n_C_final": n_c_final,
        "n_C_deleted": n_c_deleted,
        "n_H_initial": n_h_initial,
        "n_H_added": n_h_added,
        "n_H_final": n_h_final,
        "n_coord0": n_coord0,
        "n_coord1": n_coord1,
        "n_coord2_total": n_coord2,
        "n_coord3": n_coord3,
        "n_coord_gt3": n_coord_gt3,
        "n_stretched_bonds_healed": n_healed,
        "n_external_crop_boundary_coord2": n_external_coord2,
        # Coordination-2 carbons left bare because they are sp, not sp2. They are
        # excluded from n_internal_coord2_H_candidates and are NOT a failure to
        # hydrogenate: an sp carbon has no dangling sigma orbital to cap. Split
        # by which side of the frame band they sit on, the same split as
        # internal_defect_edge_coord2 vs external_crop_boundary_coord2.
        "n_sp_linear_coord2": n_sp_linear_coord2,
        "n_sp_linear_crop_boundary_coord2": n_sp_linear_crop_coord2,
        # The *ends* of those sp chains, which are the opposite case: a
        # coordination-1 carbon with one sigma bond and nothing else is a real
        # dangling-bond radical, so it is capped with one H along the chain axis.
        # Counted separately from the coord-2 sites above because it is selected
        # in both modes and comes out of the coordination-1 population, and
        # n_sp_terminus_H_added is the check that they were actually placed - the
        # tilt is capped at 15 degrees, so these can fail where a coord-2 site
        # would have found another direction.
        "n_sp_terminus_coord1": n_sp_terminus_coord1,
        "n_sp_terminus_crop_boundary_coord1": n_sp_terminus_crop_coord1,
        "n_sp_terminus_H_added": n_sp_terminus_h_added,
        "sp_angle_threshold_deg": float(sp_angle_deg),
        "max_ccc_angle_deg": max_ccc_angle,
        "n_internal_coord2_H_candidates": n_internal_candidates,
        "n_H_sites_accepted": n_accepted,
        "n_H_sites_tilted": n_tilted,
        "max_H_tilt_deg": max_tilt,
        "n_H_sites_rejected": n_rejected,
        "n_H_sites_unplaceable": n_unplaceable,
        "n_H_sites_partial": n_partial,
        "n_H_sites_accepted_tight": n_tight_sites,
        "n_H_placed_tight": n_tight_h,
        "n_H_sites_forced": n_forced,
        "n_H_sites_skipped": n_skipped,
        "n_coord0_deleted": n_deleted_coord0,
        # Carbons removed because they were not connected to the main sheet.
        # Non-zero only for I08 (17), J02 (4) and J04 (7), which each carry a
        # detached island of TEM debris in the top-right corner.
        "n_disconnected_fragment_deleted": n_deleted_fragment,
        # Carbons removed because their only bond was a stretched one (see
        # stretched_coord1_carbons). 59 across the 95 structures, 3 of them in
        # H02, none in J.
        "n_stretched_coord1_deleted": n_deleted_stretched_coord1,
        # Cage filter (see MAX_CAGE_RING_AREA). n_H_sites_caged counts only the
        # sites the filter left bare; sites it merely pushed onto a different
        # tilt are not counted, and are the common case. n_H_inside_ring is the
        # verification column and must be 0 for every structure - it re-tests the
        # hydrogens that were actually written, so it fails loudly if the filter
        # is ever bypassed.
        "max_cage_ring_area": float(max_cage_ring_area),
        # Steric floors of the normal placement pass. Recorded so a site table
        # can always be traced back to the thresholds that produced it; the
        # tight retry's own floors are fixed constants and are not recorded.
        "min_hh_distance": float(min_hh),
        "min_nonbonded_hc_distance": float(min_hc),
        "n_rings_below_cage_area": n_rings,
        "n_H_sites_caged": n_caged_sites,
        "n_H_inside_ring": n_h_in_ring,
        # Verification columns: in cluster mode both must be 0 for every
        # structure, otherwise the cluster still carries dangling bonds or
        # impossible 4-coordinate sp2 carbon.
        "n_bare_dangling_C": n_bare_dangling_c,
        # Carbons with 0 or 1 C neighbours carry 3-4 missing bonds each. They are
        # never hydrogenated (TARGET_H_COORDINATION is 2), so they stay dangling
        # and will contribute mid-gap states to a DOS. Reported, not fixed. Split
        # by coordination: coord-0 carbons have no bond axis at all and are
        # usually TEM-fit artefacts, coord-1 carbons are real chain termini.
        "n_coord0_inspect": n_coord0_inspect,
        "n_coord1_inspect": n_coord1_inspect,
        "n_coord0_or_coord1_inspect": n_inspect,
        "status": "success",
        "error": "",
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fully H-saturate internal coordination-2 defect-edge sites "
            "for A-J defective graphene structures."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing A, B, C, ..., J folders.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/h_saturated/full_saturation"),
        help="Output directory for saturated structures.",
    )

    parser.add_argument(
        "--site-table-root",
        type=Path,
        default=Path("results/tables/full_saturation_sites"),
        help="Output directory for per-structure site tables.",
    )

    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("results/figures/full_saturation_site_maps"),
        help="Output directory for per-structure site-selection figures.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/tables/full_saturation_summary.csv"),
        help="Output summary CSV.",
    )

    parser.add_argument(
        "--structure-id",
        type=str,
        default=None,
        help="Optional single structure to process, e.g. A01, C08, J05.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of structures to process.",
    )

    parser.add_argument(
        "--sp-angle-deg",
        type=float,
        default=SP_ANGLE_THRESHOLD_DEG,
        help=(
            "C-C-C angle in degrees at or above which a coordination-2 carbon is "
            "treated as sp rather than sp2 and left un-hydrogenated (default "
            "{}). Such a carbon has no dangling sigma orbital for an H to cap; "
            "these are the carbyne-like chains around the central tear in the "
            "I and J frames. The angle is written to the site tables for every "
            "coordination-2 carbon regardless, so the cut can be re-tuned "
            "afterwards. Pass a value above 180 to disable the test."
        ).format(SP_ANGLE_THRESHOLD_DEG),
    )

    parser.add_argument(
        "--max-cage-ring-area",
        type=float,
        default=MAX_CAGE_RING_AREA,
        help=(
            "Area in Å² below which a bounded face of the carbon graph counts as "
            "a ring the H cannot fit inside, rather than a pore it may point "
            "into (default {}). A candidate landing inside such a ring is "
            "rejected and the tilt sweep keeps looking. Across the 95 cluster "
            "structures the wrong placements sat in 5- and 6-rings of 3.7-5.7 Å² "
            "and the legitimate pore is 26.7 Å², so the cut is not sensitive. "
            "Cluster mode only; pass 0 to disable the test."
        ).format(MAX_CAGE_RING_AREA),
    )

    parser.add_argument(
        "--min-hh",
        type=float,
        default=MIN_HH_DISTANCE,
        help=(
            "Minimum H-H separation in Å for the normal placement pass (default "
            "{}). A candidate closer than this to an existing or newly added H "
            "is rejected and the tilt sweep keeps looking. This is chemistry, "
            "not just an anti-overlap guard. It does not change the tight retry "
            "in cluster mode, which keeps its own fixed {} Å floor."
        ).format(MIN_HH_DISTANCE, MIN_HH_DISTANCE_TIGHT),
    )

    parser.add_argument(
        "--min-hc",
        type=float,
        default=MIN_NONBONDED_HC_DISTANCE,
        help=(
            "Minimum non-bonded H-C separation in Å for the normal placement "
            "pass (default {}). As --min-hh, and likewise leaves the tight "
            "retry's fixed {} Å floor alone."
        ).format(MIN_NONBONDED_HC_DISTANCE, MIN_NONBONDED_HC_DISTANCE_TIGHT),
    )

    parser.add_argument(
        "--force-clashes",
        action="store_true",
        help="Force addition of H even if clash checks fail.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write site-selection plots.",
    )

    parser.add_argument(
        "--cluster",
        action="store_true",
        help=(
            "Treat each structure as a non-periodic cluster: ignore the input "
            "lattice, hydrogenate the crop-frame rim as well as the internal "
            "defect edges, give coordination-1 carbons two H, delete "
            "coordination-0 carbons, and write geometry.in with no "
            "lattice_vector lines. Switches the default output paths to the "
            "cluster_* trees so the periodic results are not overwritten."
        ),
    )

    parser.add_argument(
        "--keep-stretched-coord1",
        action="store_true",
        help=(
            "Keep carbons whose only bond is a stretched one, i.e. that have no "
            "neighbour inside the primary C-C cutoff and are attached to the "
            "sheet solely by a bond in [{:.2f}, {:.2f}) Å. They are deleted by "
            "default in cluster mode as TEM-fit artefacts, the same treatment "
            "coordination-0 carbons already get (see stretched_coord1_carbons). "
            "Pass this to restore the older behaviour, where each is instead "
            "hydrogenated as a coord1_terminus =CH2 and its partner is held at "
            "coordination 3 by the stretched bond. Cluster mode only."
        ).format(CC_CUTOFF, CC_CUTOFF_STRETCHED),
    )

    return parser.parse_args()


# Cluster mode writes to its own output tree. The periodic results under
# data/h_saturated/full_saturation are the record of the earlier periodic study
# and are still needed for comparison, so they must never be overwritten by a
# cluster run.
CLUSTER_DEFAULT_PATHS = {
    "output_root": Path("data/h_saturated/cluster_full_saturation"),
    "site_table_root": Path("results/tables/cluster_saturation_sites"),
    "figure_root": Path("results/figures/cluster_saturation_site_maps"),
    "summary": Path("results/tables/cluster_saturation_summary.csv"),
}


def resolve_output_paths(args, parser_defaults):
    """
    Pick the cluster default output paths unless the user set them explicitly.
    """
    resolved = {}

    for name, cluster_default in CLUSTER_DEFAULT_PATHS.items():
        given = getattr(args, name)

        if args.cluster and given == parser_defaults[name]:
            resolved[name] = cluster_default
        else:
            resolved[name] = given

    return resolved


def main():
    args = parse_args()

    parser_defaults = {
        "output_root": Path("data/h_saturated/full_saturation"),
        "site_table_root": Path("results/tables/full_saturation_sites"),
        "figure_root": Path("results/figures/full_saturation_site_maps"),
        "summary": Path("results/tables/full_saturation_summary.csv"),
    }

    paths = resolve_output_paths(args, parser_defaults)

    root = args.root.resolve()

    output_root = (root / paths["output_root"]).resolve()
    site_table_root = (root / paths["site_table_root"]).resolve()
    figure_root = (root / paths["figure_root"]).resolve()
    summary_file = (root / paths["summary"]).resolve()

    reject_clashes = not bool(args.force_clashes)
    make_plots = not bool(args.no_plots)

    output_root.mkdir(parents=True, exist_ok=True)
    site_table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    geometry_files = find_all_geometry_files(root)

    if args.structure_id is not None:
        geometry_files = filter_geometry_files_by_structure_id(
            geometry_files=geometry_files,
            structure_id=args.structure_id,
        )

    if args.limit is not None:
        geometry_files = geometry_files[:args.limit]

    if len(geometry_files) == 0:
        raise RuntimeError("No A-J geometry.in files found.")

    print("Root:", root)
    print("Found {} structure(s).".format(len(geometry_files)))
    print("Mode:", "cluster (non-periodic)" if args.cluster else "periodic slab")
    print("C-C cutoff: {:.3f} Å".format(CC_CUTOFF))
    print("C-H bond length: {:.3f} Å".format(CH_BOND_LENGTH))
    print("Frame band: {:.3f} Å".format(FRAME_BAND))
    print("Crop direction threshold: {:.3f}".format(CROP_DIRECTION_THRESHOLD))
    print(
        "sp angle threshold: {:.1f} deg{}".format(
            args.sp_angle_deg,
            " (disabled)" if args.sp_angle_deg > 180.0 else "",
        )
    )
    print("Reject clashes:", reject_clashes)
    if args.cluster:
        print(
            "Delete stretched coord-1 carbons:",
            not bool(args.keep_stretched_coord1),
        )
    print("Make plots:", make_plots)
    print("Output root:", output_root)

    if args.structure_id is not None:
        print("Selected structure ID:", args.structure_id.upper())

    summary_rows = []

    for geom_file in geometry_files:
        try:
            row = process_one_structure(
                geom_file=geom_file,
                root=root,
                output_root=output_root,
                site_table_root=site_table_root,
                figure_root=figure_root,
                reject_clashes=reject_clashes,
                make_plots=make_plots,
                cluster=args.cluster,
                sp_angle_deg=args.sp_angle_deg,
                max_cage_ring_area=args.max_cage_ring_area,
                delete_stretched_coord1=not bool(args.keep_stretched_coord1),
                min_hh=args.min_hh,
                min_hc=args.min_hc,
            )
            summary_rows.append(row)

        except Exception as exc:
            structure_id = structure_id_from_path(geom_file)

            print()
            print("ERROR processing {}: {}".format(structure_id, exc))
            print(traceback.format_exc())

            summary_rows.append(
                {
                    "structure_id": structure_id,
                    "input_geometry": str(geom_file.relative_to(root)),
                    "output_geometry": "",
                    "site_table": "",
                    "site_figure": "",
                    "n_atoms_initial": None,
                    "n_atoms_final": None,
                    "n_C": None,
                    "n_H_initial": None,
                    "n_H_added": None,
                    "n_H_final": None,
                    "n_coord0": None,
                    "n_coord1": None,
                    "n_coord2_total": None,
                    "n_coord3": None,
                    "n_coord_gt3": None,
                    "n_external_crop_boundary_coord2": None,
                    "n_internal_coord2_H_candidates": None,
                    "n_H_sites_accepted": None,
                    "n_H_sites_rejected": None,
                    "n_H_sites_forced": None,
                    "n_H_sites_skipped": None,
                    "n_coord0_inspect": None,
                    "n_coord1_inspect": None,
                    "n_coord0_or_coord1_inspect": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_file, index=False)

    print()
    print("=" * 80)
    print("Done.")
    print("Summary written to:", summary_file.relative_to(root))
    print("Successful:", int((summary_df["status"] == "success").sum()))
    print("Failed:", int((summary_df["status"] == "failed").sum()))
    print("Total H added:", int(summary_df["n_H_added"].fillna(0).sum()))

    if args.cluster:
        # These are the cluster-mode acceptance criteria (see the module
        # docstring): a cluster carrying dangling bonds or 4-coordinate carbon is
        # not usable for a DOS, so say so loudly rather than leaving it in a CSV.
        total_dangling = int(summary_df["n_bare_dangling_C"].fillna(0).sum())
        total_overbonded = int(summary_df["n_coord_gt3"].fillna(0).sum())
        total_deleted = int(summary_df["n_coord0_deleted"].fillna(0).sum())
        total_fragment = int(
            summary_df["n_disconnected_fragment_deleted"].fillna(0).sum()
        )
        total_stretched_coord1 = int(
            summary_df.get(
                "n_stretched_coord1_deleted", pd.Series(dtype=float)
            ).fillna(0).sum()
        )
        total_caged = int(summary_df["n_H_sites_caged"].fillna(0).sum())
        total_in_ring = int(summary_df["n_H_inside_ring"].fillna(0).sum())

        print("Total coord-0 carbons deleted:", total_deleted)
        print("Total disconnected-fragment carbons deleted:", total_fragment)
        print("Total stretched coord-1 carbons deleted:", total_stretched_coord1)
        print("Total H sites left bare to avoid caging in a ring:", total_caged)
        print("Total H placed inside a carbon ring (must be 0):", total_in_ring)
        print("Total carbons left with a dangling bond:", total_dangling)
        print("Total carbons with coordination > 3:", total_overbonded)

        # Over-coordination is a hard failure: 4-coordinate sp2 carbon does not
        # exist and means the bonding analysis is wrong. So is an H that ended up
        # inside a ring anyway. A bare dangling bond is neither, at least not
        # when the cage filter put it there - the only alternative for those
        # sites is exactly that caged H, which is worse than the radical. They
        # are reported apart so a real problem is not lost in the expected
        # residue.
        if total_in_ring:
            print()
            print("WARNING: {} H ended up inside a carbon ring.".format(total_in_ring))
            for _, bad_row in summary_df[
                summary_df["n_H_inside_ring"].fillna(0) > 0
            ].iterrows():
                print(
                    "  {}: {} H inside a ring".format(
                        bad_row["structure_id"], int(bad_row["n_H_inside_ring"] or 0)
                    )
                )

        if total_overbonded:
            print()
            print("WARNING: over-coordinated carbon present. Structures affected:")
            for _, bad_row in summary_df[
                summary_df["n_coord_gt3"].fillna(0) > 0
            ].iterrows():
                print(
                    "  {}: {} carbons with coordination > 3".format(
                        bad_row["structure_id"], int(bad_row["n_coord_gt3"] or 0)
                    )
                )

        if total_dangling:
            print()
            print("Carbons left with a dangling bond, by structure:")
            for _, bare_row in summary_df[
                summary_df["n_bare_dangling_C"].fillna(0) > 0
            ].iterrows():
                print(
                    "  {}: {} dangling ({} site(s) would have been caged)".format(
                        bare_row["structure_id"],
                        int(bare_row["n_bare_dangling_C"] or 0),
                        int(bare_row["n_H_sites_caged"] or 0),
                    )
                )

        if not total_dangling and not total_overbonded:
            print("All clusters fully saturated with no over-coordinated carbon.")

    print("=" * 80)


if __name__ == "__main__":
    main()