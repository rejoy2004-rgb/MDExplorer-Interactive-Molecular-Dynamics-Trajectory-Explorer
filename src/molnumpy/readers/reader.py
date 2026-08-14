"""Low-level reading helpers: turn MDAnalysis data into our common format."""

import numpy as np
import MDAnalysis as mda

from molnumpy.topology import Topology
from molnumpy.trajectory import Trajectory
from molnumpy.system import MolecularSystem


def universe_to_system(u: mda.Universe) -> MolecularSystem:
    """Convert an MDAnalysis Universe into a MolecularSystem.

    All fields (topology, coordinates, box, time, ...) are extracted into
    plain NumPy arrays. Missing data (charges, velocities, ...) becomes zeros
    or None instead of raising an error.
    """
    n_atoms = len(u.atoms)

    # ---------- Topology ----------
    atoms_arr = np.empty(n_atoms, dtype=[
        ('index', 'i4'),
        ('name', 'U10'),
        ('type', 'U10'),
        ('element', 'U5'),
        ('resid', 'i4'),
        ('resname', 'U10'),
        ('segid', 'U10')
    ])
    atoms_arr['index'] = u.atoms.indices
    atoms_arr['name'] = u.atoms.names
    atoms_arr['resid'] = u.atoms.resids
    atoms_arr['resname'] = u.atoms.resnames
    atoms_arr['segid'] = u.atoms.segids

    # 'type' and 'element' may not exist in every format, so fall back
    try:
        atoms_arr['type'] = u.atoms.types
    except (AttributeError, ValueError):
        atoms_arr['type'] = u.atoms.names

    try:
        atoms_arr['element'] = u.atoms.elements
    except (AttributeError, ValueError):
        atoms_arr['element'] = [name[0] if name else 'X' for name in u.atoms.names]

    # Unique residues
    unique_resids, first_idx = np.unique(u.atoms.resids, return_index=True)
    residues_arr = np.empty(len(unique_resids), dtype=[('resid', 'i4'), ('resname', 'U10')])
    residues_arr['resid'] = unique_resids
    residues_arr['resname'] = u.atoms.resnames[first_idx]

    # Masses, charges and connectivity (may be missing in some formats)
    try:
        masses = np.array(u.atoms.masses, dtype=float)
    except (AttributeError, ValueError):
        masses = np.zeros(n_atoms, dtype=float)

    try:
        charges = np.array(u.atoms.charges, dtype=float)
    except (AttributeError, ValueError):
        charges = np.zeros(n_atoms, dtype=float)

    def safe_bonds():
        try:
            return u.atoms.bonds.to_indices()
        except Exception:
            return np.empty((0, 2), dtype=int)

    bonds = safe_bonds()

    topology = Topology(
        n_atoms=n_atoms,
        atoms=atoms_arr,
        residues=residues_arr,
        segments=np.unique(u.atoms.segids),
        masses=masses,
        charges=charges,
        bonds=bonds,
    )

    # ---------- Trajectory ----------
    n_frames = len(u.trajectory)
    coordinates = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
    time = np.empty(n_frames, dtype=float)
    box = np.zeros((n_frames, 6), dtype=np.float32)

    has_velocities = False
    try:
        has_velocities = u.atoms.velocities is not None
    except Exception:
        pass
    velocities = np.empty((n_frames, n_atoms, 3), dtype=np.float32) if has_velocities else None

    for frame_idx, ts in enumerate(u.trajectory):
        coordinates[frame_idx] = ts.positions
        time[frame_idx] = ts.time
        if ts.dimensions is not None:
            box[frame_idx] = ts.dimensions
        if has_velocities:
            velocities[frame_idx] = ts.velocities

    trajectory = Trajectory(
        n_frames=n_frames,
        n_atoms=n_atoms,
        coordinates=coordinates,
        velocities=velocities,
        box=box,
        time=time,
    )

    system = MolecularSystem(topology=topology, trajectory=trajectory)
    system._mda_universe = u
    return system
