"""Radius of gyration of the protein."""

import numpy as np


def calculate_rg(system, selection="protein"):
    """Radius of gyration (Å) of the selection for every frame."""
    protein = system.universe.select_atoms(selection)

    rg = []
    for ts in system.universe.trajectory:
        rg.append(protein.radius_of_gyration())
    return np.array(rg)
