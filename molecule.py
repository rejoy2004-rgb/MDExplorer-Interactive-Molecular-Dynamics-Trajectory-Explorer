"""Reading a simulation and holding it in one common format.

MD simulations are saved in many different file formats, and each one stores
the same information in its own way. Instead of writing separate code for every
format, we read the file once with MDAnalysis and copy everything into plain
NumPy arrays inside a MolecularSystem object. After that the rest of the
program never has to know which format we started from.
"""

import os

import numpy as np
import MDAnalysis as mda

# the file formats the program accepts
TOPOLOGY_FORMATS = [".pdb", ".psf", ".gro", ".tpr", ".prmtop", ".parm7"]
TRAJECTORY_FORMATS = [".dcd", ".xtc", ".trr", ".nc"]

# Residue names for water and simple ions. These are part of the simulation box
# but not part of the protein, so we leave them out of the calculations.
SOLVENT = ["HOH", "WAT", "SOL", "TIP3", "SPC", "NA", "CL", "K", "MG",
           "ZN", "SOD", "CLA", "POT", "ION"]


class MolecularSystem:
    """One simulation, stored as plain NumPy arrays.

    The arrays that describe the atoms have one entry per atom and never
    change. The coordinates do change, so they have an extra dimension at the
    front for the frame number.
    """

    def __init__(self, universe, top_path, traj_path):
        self.n_atoms = len(universe.atoms)
        self.n_frames = len(universe.trajectory)

        # information about the atoms, which is the same in every frame
        self.names = np.array(universe.atoms.names)
        self.resids = np.array(universe.atoms.resids)
        self.resnames = np.array(universe.atoms.resnames)
        self.elements = np.array([guess_element(name) for name in self.names])

        try:
            self.masses = np.array(universe.atoms.masses, dtype=float)
        except Exception:
            self.masses = np.zeros(self.n_atoms)

        # a few formats do not store the masses at all, then we give every atom
        # the same weight so the centre of mass and Rg still make sense
        if self.masses.sum() == 0:
            self.masses = np.ones(self.n_atoms)

        # the coordinates, which are different in every frame.
        # float32 is accurate enough here and uses half the memory of float64.
        self.coordinates = np.zeros((self.n_frames, self.n_atoms, 3), dtype=np.float32)
        self.time = np.zeros(self.n_frames)

        for i, frame in enumerate(universe.trajectory):
            self.coordinates[i] = universe.atoms.positions
            self.time[i] = frame.time

        # count the protein residues only, otherwise a simulation with a big
        # water box would report tens of thousands of "residues"
        self.n_residues = len(np.unique(self.resids[self.protein_mask()]))

        # SASA and DSSP open the files a second time with mdtraj, so we keep
        # the file names here. mdtraj_protein caches what they read.
        self.top_path = top_path
        self.traj_path = traj_path
        self.mdtraj_protein = None

    def protein_mask(self):
        """True for every atom that belongs to the protein, False for the rest.

        np.isin checks each residue name against the SOLVENT list, and the ~
        flips the answer, so we keep the atoms that are NOT water or ions.
        """
        return ~np.isin(self.resnames, SOLVENT)

    def select(self, atom_names):
        """Positions of the protein atoms whose name is in the given list.

        For example select(["CA"]) gives the alpha carbons, one per residue.
        """
        return np.where(self.protein_mask() & np.isin(self.names, atom_names))[0]


def guess_element(name):
    """Guess the element from the atom name, e.g. "CA" -> C and "1HB" -> H.

    Atom names sometimes start with a number, so we drop any digits first and
    then take the first letter.
    """
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

    # a PDB file already contains coordinates, the other formats only describe
    # the atoms and need a separate trajectory file
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
