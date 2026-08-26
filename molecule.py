"""Reads any supported simulation file into one common MolecularSystem."""

import os

import numpy as np
import MDAnalysis as mda

# the file formats the program accepts
TOPOLOGY_FORMATS = [".pdb", ".psf", ".gro", ".tpr", ".prmtop", ".parm7"]
TRAJECTORY_FORMATS = [".dcd", ".xtc", ".trr", ".nc"]

# water and ion residue names, left out of every calculation
SOLVENT = ["HOH", "WAT", "SOL", "TIP3", "SPC", "NA", "CL", "K", "MG",
           "ZN", "SOD", "CLA", "POT", "ION"]


class MolecularSystem:
    """One simulation, stored as plain NumPy arrays: atom info + per-frame coordinates."""

    def __init__(self, universe, top_path, traj_path):
        self.n_atoms = len(universe.atoms)
        self.n_frames = len(universe.trajectory)

        # information about the atoms, which is the same in every frame
        self.names = np.array(universe.atoms.names)
        self.resids = np.array(universe.atoms.resids)
        self.resnames = np.array(universe.atoms.resnames)
        self.elements = np.array([guess_element(name) for name in self.names])

        # a few file formats simply do not have mass information at all
        if hasattr(universe.atoms, "masses"):
            self.masses = np.array(universe.atoms.masses, dtype=float)
        else:
            self.masses = np.zeros(self.n_atoms)

        # if no format gave us masses, treat every atom as equal weight
        if self.masses.sum() == 0:
            self.masses = np.ones(self.n_atoms)

        # float32 is accurate enough here and halves the memory of float64
        self.coordinates = np.zeros((self.n_frames, self.n_atoms, 3), dtype=np.float32)
        self.time = np.zeros(self.n_frames)

        for i, frame in enumerate(universe.trajectory):
            self.coordinates[i] = universe.atoms.positions
            self.time[i] = frame.time

        # protein residues only, otherwise water would inflate this count
        self.n_residues = len(np.unique(self.resids[self.protein_mask()]))

        # kept so SASA/DSSP can reopen the files with mdtraj; caches the result
        self.top_path = top_path
        self.traj_path = traj_path
        self.mdtraj_protein = None

    def protein_mask(self):
        """True for protein atoms, False for water and ions."""
        return ~np.isin(self.resnames, SOLVENT)

    def select(self, atom_names):
        """Indices of protein atoms named in atom_names, e.g. select(["CA"])."""
        return np.where(self.protein_mask() & np.isin(self.names, atom_names))[0]


def guess_element(name):
    """Guess the element from an atom name, e.g. "CA" -> C, "1HB" -> H."""
    letters = name.lstrip("0123456789")
    if letters == "":
        return "X"
    return letters[0]


def check_files(top_path, traj_path):
    """Check the files before opening them, so we can show a clear error."""
    if top_path is None:
        raise ValueError("Please upload a topology file (PDB, PSF, GRO, TPR or PRMTOP).")

    top_ext = os.path.splitext(top_path)[1].lower()
    if top_ext not in TOPOLOGY_FORMATS:
        raise ValueError("The topology format " + top_ext + " is not supported.")

    # only PDB contains coordinates by itself; other formats need a trajectory file
    if top_ext != ".pdb" and traj_path is None:
        raise ValueError("A " + top_ext + " file does not contain any coordinates. "
                         "Please also upload a trajectory file (DCD, XTC, TRR or NC).")

    if traj_path is not None:
        traj_ext = os.path.splitext(traj_path)[1].lower()
        if traj_ext not in TRAJECTORY_FORMATS:
            raise ValueError("The trajectory format " + traj_ext + " is not supported.")


def load_simulation(top_path, traj_path=None):
    """Read the files and return one MolecularSystem."""
    check_files(top_path, traj_path)

    if traj_path is None:
        universe = mda.Universe(top_path)
    else:
        universe = mda.Universe(top_path, traj_path)

    return MolecularSystem(universe, top_path, traj_path)
