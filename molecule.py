"""Reads any supported simulation file into one common MolecularSystem."""

import os

import numpy as np
import MDAnalysis as mda

# the file formats the program accepts
TOPOLOGY_FORMATS = [".pdb", ".psf", ".gro", ".tpr", ".prmtop", ".parm7"]
TRAJECTORY_FORMATS = [".dcd", ".xtc", ".trr", ".nc"]

# water and ion residue names, left out of every calculation
SOLVENT = ["HOH", "WAT", "SOL", "TIP3", "SPC", "NA", "CL", "K", "MG",
           "ZN", "SOD", "CLA", "POT", "ION", "NA+", "CL-"]


class MolecularSystem:
    """One simulation, stored as plain NumPy arrays: atom info + per-frame coordinates."""

    def __init__(self, universe, stride=1):
        self.n_atoms = len(universe.atoms)

        # a long trajectory can be read every stride-th frame, e.g. stride 10 keeps frames 0, 10, 20, ...
        self.stride = stride
        self.frame_numbers = np.arange(0, len(universe.trajectory), stride)
        self.n_frames = len(self.frame_numbers)

        # information about the atoms, which is the same in every frame
        self.names = np.array(universe.atoms.names)
        self.resids = np.array(universe.atoms.resids)
        self.resnames = np.array(universe.atoms.resnames)

        # chain / segment label of each atom; a few formats have none, then every atom shares one label
        if hasattr(universe.atoms, "segids"):
            self.segids = np.array(universe.atoms.segids)
        else:
            self.segids = np.full(self.n_atoms, "")

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

        for i, frame in enumerate(universe.trajectory[::stride]):
            self.coordinates[i] = universe.atoms.positions
            self.time[i] = frame.time

        # protein residues only, otherwise water would inflate this count. A new residue starts wherever
        # the chain label, residue number or name changes, so chains that reuse numbers count separately.
        protein = self.protein_mask()
        chain, number, name = self.segids[protein], self.resids[protein], self.resnames[protein]
        starts = (chain[1:] != chain[:-1]) | (number[1:] != number[:-1]) | (name[1:] != name[:-1])
        self.n_residues = 0 if len(number) == 0 else 1 + int(starts.sum())

        # SASA and DSSP are done by mdtraj; analysis.mdtraj_protein() fills this in once and reuses it
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


def check_files(top_path, traj_path, stride=1):
    """Check the files before opening them, so we can show a clear error."""
    if stride < 1:
        raise ValueError("Use every Nth frame: N must be 1 or more.")

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


def load_simulation(top_path, traj_path=None, stride=1):
    """Read the files and return one MolecularSystem (every stride-th frame)."""
    check_files(top_path, traj_path, stride)

    if traj_path is None:
        universe = mda.Universe(top_path)
    else:
        universe = mda.Universe(top_path, traj_path)

    system = MolecularSystem(universe, stride)
    if system.n_frames == 0:
        raise ValueError("These files contain no frames, so there is nothing to show.")
    if not system.protein_mask().any():
        raise ValueError("No protein atoms were found, only water and ions.")
    return system
