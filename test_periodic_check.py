"""Tests for analysis.check_periodic_split, using tiny in-memory proteins (no files, no internet).

Run with:   python test_periodic_check.py     (or:  python -m pytest test_periodic_check.py)
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

from molecule import MolecularSystem
import analysis

SPACING = 3.8   # Angstrom between neighbouring alpha carbons


def make_system(resids, chains=None, names=None, frames=3, with_segids=True):
    """A protein with one atom per residue, laid out along the x axis.
    resids: residue number of every atom; chains: chain label of every atom;
    names: atom name of every atom (default all "CA")."""
    n = len(resids)
    chains = chains or ["A"] * n
    names = names or ["CA"] * n
    labels = sorted(set(chains))

    universe = mda.Universe.empty(
        n, n_residues=n, n_segments=len(labels), atom_resindex=np.arange(n),
        residue_segindex=[labels.index(c) for c in chains], trajectory=True)
    universe.add_TopologyAttr("names", names)
    universe.add_TopologyAttr("resids", resids)
    universe.add_TopologyAttr("resnames", ["ALA"] * n)
    if with_segids:
        universe.add_TopologyAttr("segids", labels)

    positions = np.zeros((frames, n, 3), dtype=np.float32)
    positions[:, :, 0] = np.arange(n) * SPACING
    universe.load_new(positions, format=MemoryReader, order="fac")
    return MolecularSystem(universe, None, None)


def test_normal_chain_has_no_obvious_split():
    system = make_system(list(range(1, 11)))
    answer, details = analysis.check_periodic_split(system)
    assert answer == analysis.NO_SPLIT, details


def test_large_jump_is_reported_with_its_frame_and_residues():
    system = make_system(list(range(1, 11)))
    system.coordinates[1, 6:, 0] += 50.0   # in frame 1, residues 7-10 jump across the box
    answer, details = analysis.check_periodic_split(system)
    assert answer == analysis.SPLIT
    assert "residues 6 and 7" in details and "frame 1" in details and "1 of 3 frames" in details


def test_two_chains_with_restarting_numbers_are_not_compared():
    system = make_system([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], chains=["A"] * 5 + ["B"] * 5)
    system.coordinates[:, 5:, 0] += 200.0   # chain B sits far away from chain A
    answer, details = analysis.check_periodic_split(system)
    assert answer == analysis.NO_SPLIT, details


def test_two_chains_with_continuous_numbers_are_not_compared():
    system = make_system(list(range(1, 11)), chains=["A"] * 5 + ["B"] * 5)
    system.coordinates[:, 5:, 0] += 200.0   # residue 5 -> 6 crosses from chain A to chain B
    answer, details = analysis.check_periodic_split(system)
    assert answer == analysis.NO_SPLIT, details


def test_jump_inside_one_chain_of_two_is_still_found():
    system = make_system([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], chains=["A"] * 5 + ["B"] * 5)
    system.coordinates[0, 8:, 1] += 80.0    # chain B is cut between its residues 3 and 4
    answer, details = analysis.check_periodic_split(system)
    assert answer == analysis.SPLIT, details


def test_numbering_gap_is_not_a_split():
    system = make_system([1, 2, 3, 4, 5, 8, 9, 10, 11, 12])   # residues 6 and 7 are missing
    system.coordinates[:, 5:, 0] += 100.0
    answer, details = analysis.check_periodic_split(system)
    assert answer == analysis.NO_SPLIT, details


def test_residue_without_ca_is_skipped_not_compared_across():
    names = ["CA"] * 10
    names[4] = "N"                           # residue 5 has no alpha carbon
    system = make_system(list(range(1, 11)), names=names)
    system.coordinates[:, 5:, 0] += 100.0    # a big gap, but residues 4 and 6 are not neighbours
    answer, details = analysis.check_periodic_split(system)
    assert answer == analysis.NO_SPLIT, details


def test_no_alpha_carbons_cannot_be_assessed():
    system = make_system(list(range(1, 11)), names=["N"] * 10)
    answer, details = analysis.check_periodic_split(system)
    assert answer == analysis.UNKNOWN, details


def test_no_neighbouring_residues_cannot_be_assessed():
    system = make_system([1, 4, 7, 10])      # numbering gaps everywhere
    answer, details = analysis.check_periodic_split(system)
    assert answer == analysis.UNKNOWN, details


def test_one_residue_per_chain_cannot_be_assessed():
    system = make_system([1, 1, 1], chains=["A", "B", "C"])
    answer, details = analysis.check_periodic_split(system)
    assert answer == analysis.UNKNOWN, details


def test_non_finite_coordinates_cannot_be_assessed():
    system = make_system(list(range(1, 11)))
    system.coordinates[:] = np.nan
    answer, details = analysis.check_periodic_split(system)
    assert answer == analysis.UNKNOWN, details


def test_one_nan_atom_does_not_hide_a_real_split_or_invent_one():
    system = make_system(list(range(1, 11)))
    system.coordinates[0, 2, :] = np.nan
    assert analysis.check_periodic_split(system)[0] == analysis.NO_SPLIT
    system.coordinates[2, 6:, 0] += 50.0
    assert analysis.check_periodic_split(system)[0] == analysis.SPLIT


def test_missing_box_and_chain_labels_do_not_decide_the_answer():
    # these systems have no box dimensions and no chain labels at all
    whole = make_system(list(range(1, 11)), with_segids=False)
    assert whole.segids.tolist() == [""] * 10
    assert analysis.check_periodic_split(whole)[0] == analysis.NO_SPLIT   # never "safe", only "no obvious split"
    cut = make_system(list(range(1, 11)), with_segids=False)
    cut.coordinates[:, 5:, 0] += 50.0
    assert analysis.check_periodic_split(cut)[0] == analysis.SPLIT


def test_threshold_can_be_changed():
    system = make_system(list(range(1, 11)))   # neighbours are 3.8 A apart
    assert analysis.check_periodic_split(system, max_distance=3.0)[0] == analysis.SPLIT
    assert analysis.check_periodic_split(system, max_distance=6.0)[0] == analysis.NO_SPLIT


if __name__ == "__main__":
    tests = [(name, function) for name, function in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, function in tests:
        try:
            function()
            print("PASS", name)
        except Exception as error:
            failed += 1
            print("FAIL", name, type(error).__name__, error)
    print("%d tests, %d failed" % (len(tests), failed))
    raise SystemExit(1 if failed else 0)
