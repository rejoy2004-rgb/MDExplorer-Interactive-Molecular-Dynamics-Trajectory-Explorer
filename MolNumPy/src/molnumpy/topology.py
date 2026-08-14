from dataclasses import dataclass, field
import numpy as np


@dataclass
class Topology:
    """Common topology representation. All data is stored as NumPy arrays.

    Attributes:
        n_atoms: number of atoms.
        atoms: structured array with one row per atom containing
            index, name, type, element, resid, resname, segid.
        residues: structured array of unique residues (resid, resname).
        segments: array of unique segment ids.
        masses: 1D array of shape (n_atoms,).
        charges: 1D array of shape (n_atoms,).
        bonds: integer array of shape (n_bonds, 2) with 0-based atom indices.
        angles: integer array of shape (n_angles, 3) with 0-based atom indices.
        dihedrals: integer array of shape (n_dihedrals, 4).
    """
    n_atoms: int

    atoms: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=[
        ('index', 'i4'),
        ('name', 'U10'),
        ('type', 'U10'),
        ('element', 'U5'),
        ('resid', 'i4'),
        ('resname', 'U10'),
        ('segid', 'U10')
    ]))

    residues: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=[
        ('resid', 'i4'),
        ('resname', 'U10')
    ]))

    segments: np.ndarray = field(default_factory=lambda: np.array([], dtype='U10'))

    masses: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    charges: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    bonds: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    angles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    dihedrals: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=int))

    def __post_init__(self):
        if len(self.masses) == 0:
            self.masses = np.zeros(self.n_atoms, dtype=float)
        elif len(self.masses) != self.n_atoms:
            raise ValueError(f"Masses length ({len(self.masses)}) must match n_atoms ({self.n_atoms})")

        if len(self.charges) == 0:
            self.charges = np.zeros(self.n_atoms, dtype=float)
        elif len(self.charges) != self.n_atoms:
            raise ValueError(f"Charges length ({len(self.charges)}) must match n_atoms ({self.n_atoms})")

    @property
    def n_residues(self) -> int:
        """Number of unique residues."""
        return len(self.residues)
