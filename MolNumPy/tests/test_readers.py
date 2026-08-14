import os
import pytest
import numpy as np

from molnumpy import load_molecule, MolecularSystem
from molnumpy.visualization.viewer import get_pdb_string

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NAMD_TOP = os.path.join(BASE, "NAMD", "ifabp_water.psf")
NAMD_TRAJ = os.path.join(BASE, "NAMD", "rmsfit_ifabp_water_1.dcd")
NAMD_PDB = os.path.join(BASE, "NAMD", "ifabp_water_0.pdb")

GROMACS_TOP = os.path.join(BASE, "GROMACS", "memb_pept.tpr")
GROMACS_TRAJ = os.path.join(BASE, "GROMACS", "memb_pept.xtc")


def test_pdb_loading():
    system = load_molecule(NAMD_PDB)
    assert isinstance(system, MolecularSystem)
    assert system.coordinates.shape == (1, 12445, 3)
    assert system.n_frames == 1
    assert system.n_atoms == 12445


def test_namd_psf_dcd():
    system = load_molecule(NAMD_TOP, NAMD_TRAJ)
    assert system.coordinates.shape == (500, 12445, 3)
    assert len(system.time) == 500
    assert len(system.masses) == 12445
    assert len(system.charges) == 12445
    assert system.topology.bonds.ndim == 2 and system.topology.bonds.shape[1] == 2


def test_gromacs_tpr_xtc():
    system = load_molecule(GROMACS_TOP, GROMACS_TRAJ)
    assert system.coordinates.shape == (1001, 18727, 3)


def test_coordinate_shape_is_frames_atoms_3():
    system = load_molecule(NAMD_TOP, NAMD_TRAJ)
    n_frames, n_atoms, _ = system.coordinates.shape
    assert system.coordinates.shape == (system.n_frames, system.n_atoms, 3)
    assert n_frames == 500 and n_atoms == 12445


def test_frame_switching_changes_coordinates():
    system = load_molecule(NAMD_TOP, NAMD_TRAJ)
    assert not np.allclose(system.coordinates[0], system.coordinates[1])

    # The PDB string used by the 3D viewer must change with the frame too
    pdb0 = get_pdb_string(system, 0)
    pdb1 = get_pdb_string(system, 1)
    assert pdb0 != pdb1


def test_viewer_pdb_string():
    system = load_molecule(NAMD_PDB)
    pdb_str = get_pdb_string(system, 0)
    lines = [ln for ln in pdb_str.splitlines() if ln.strip()]
    assert lines[-1] == "END"
    assert any(ln.startswith(("ATOM", "HETATM")) for ln in lines)
    assert len([ln for ln in lines if ln.startswith(("ATOM", "HETATM"))]) == 12445


def test_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_molecule("something.xyz")


def test_requires_both_files():
    with pytest.raises(ValueError, match="requires both a topology file and a trajectory file"):
        load_molecule(NAMD_TOP)  # PSF without DCD


def test_mismatched_trajectory_extension_raises():
    with pytest.raises(ValueError, match="Unsupported trajectory format"):
        load_molecule(NAMD_TOP, GROMACS_TRAJ)  # PSF + XTC is not a supported pair


def test_missing_file_raises():
    with pytest.raises(ValueError, match="Could not read the molecular file"):
        load_molecule(os.path.join(BASE, "does_not_exist.pdb"))
