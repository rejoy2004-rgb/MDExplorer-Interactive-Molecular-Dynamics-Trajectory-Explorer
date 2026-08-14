"""Per-residue root mean square fluctuation (RMSF)."""

from MDAnalysis.analysis.rms import RMSF


def calculate_rmsf(system, selection="name CA"):
    """RMSF (Å) of each atom in the selection across the trajectory.

    Alpha carbons are the default selection, giving one value per residue.
    """
    atoms = system.universe.select_atoms(selection)
    return RMSF(atoms).run().results.rmsf
