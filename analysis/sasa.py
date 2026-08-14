"""Solvent accessible surface area (SASA) of the protein."""

import mdtraj as md

from utils.loader import load_mdtraj


def calculate_sasa(system, selection="protein"):
    """Total SASA (Å²) of the selection for every frame.

    Uses mdtraj's Shrake-Rupley implementation because this version of
    MDAnalysis no longer ships the sasa module.
    """
    traj = load_mdtraj(system)
    protein = traj.atom_slice(traj.top.select(selection))
    sasa = md.shrake_rupley(protein, mode="atom")
    return sasa.sum(axis=1) * 100  # mdtraj reports nm², convert to Å²
