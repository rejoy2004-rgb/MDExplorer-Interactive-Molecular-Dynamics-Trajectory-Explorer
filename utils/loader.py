"""Helpers for loading MD files into the common representation."""

import mdtraj as md

from molnumpy import load_molecule


def load_simulation(top_path, traj_path=None):
    """Read a topology (+ trajectory) and return the common MolecularSystem.

    The original file paths are kept on the system so that format-specific
    readers (e.g. mdtraj) can still access the raw files if needed.
    """
    system = load_molecule(top_path, traj_path)
    system.top_path = top_path
    system.traj_path = traj_path
    return system


def load_mdtraj(system):
    """Load the original files with mdtraj, which reads the same formats."""
    if system.traj_path:
        return md.load(system.traj_path, top=system.top_path)
    return md.load(system.top_path)
