import os
import numpy as np
from molnumpy import load_molecule

NAMD_TOP = "NAMD/ifabp_water.psf"
NAMD_TRAJ = "NAMD/rmsfit_ifabp_water_1.dcd"


def main():
    if not os.path.exists(NAMD_TOP) or not os.path.exists(NAMD_TRAJ):
        print("Error: sample NAMD files not found in the project directory.")
        return

    # 1. Load any format through the reader factory
    print(f"Loading: {NAMD_TOP} + {NAMD_TRAJ}")
    system = load_molecule(NAMD_TOP, NAMD_TRAJ)

    # 2. Everything after the reader works with the common MolecularSystem
    print(f"System: {system}")
    print(f"Number of atoms:  {system.n_atoms}")
    print(f"Number of frames: {system.n_frames}")
    print(f"Number of residues: {system.n_residues}")
    print(f"coordinates.shape = {system.coordinates.shape}  # (frames, atoms, 3)")

    # 3. Coordinates are plain NumPy arrays
    coords = system.coordinates[0]
    print(f"\nCoordinates of atom 0, frame 0: {np.asarray(coords[0])}")

    # 4. The same code works for a PDB (single frame)
    pdb = load_molecule("NAMD/ifabp_water_0.pdb")
    print(f"\nPDB loaded: {pdb} -> coordinates.shape = {pdb.coordinates.shape}")


if __name__ == "__main__":
    main()
