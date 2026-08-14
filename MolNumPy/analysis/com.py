"""Center of mass of the protein."""

import numpy as np


def center_of_mass(system, selection="protein"):
    """Center of mass (x, y, z) of the selection for every frame."""
    protein = system.universe.select_atoms(selection)

    com = []
    for ts in system.universe.trajectory:
        com.append(protein.center_of_mass())
    return np.array(com)
