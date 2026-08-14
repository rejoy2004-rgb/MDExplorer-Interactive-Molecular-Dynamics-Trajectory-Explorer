"""Simple reader factory.

The factory maps a file extension to the correct reader and always returns the
same common format: a MolecularSystem.
"""

import os
import MDAnalysis as mda

from molnumpy.readers.reader import universe_to_system
from molnumpy.system import MolecularSystem

# What we support: {topology extension: {name, hint, requires_trajectory, allowed_trajectory_exts}}
FORMATS = {
    '.pdb': {
        'name': 'PDB',
        'hint': 'PDB contains the structure and coordinates (a single frame). No extra file needed.',
        'requires_trajectory': False,
        'trajectory_exts': ['.dcd', '.xtc', '.trr', '.nc'],
    },
    '.psf': {
        'name': 'PSF + DCD',
        'hint': 'PSF is the topology. Please also upload a DCD file as the trajectory.',
        'requires_trajectory': True,
        'trajectory_exts': ['.dcd'],
    },
    '.gro': {
        'name': 'GRO + XTC/TRR',
        'hint': 'GRO is the topology. Please also upload an XTC or TRR file as the trajectory.',
        'requires_trajectory': True,
        'trajectory_exts': ['.xtc', '.trr'],
    },
    '.tpr': {
        'name': 'TPR + XTC/TRR',
        'hint': 'TPR is the topology. Please also upload an XTC or TRR file as the trajectory.',
        'requires_trajectory': True,
        'trajectory_exts': ['.xtc', '.trr'],
    },
    '.prmtop': {
        'name': 'PRMTOP + NC',
        'hint': 'PRMTOP is the topology. Please also upload an NC file as the trajectory.',
        'requires_trajectory': True,
        'trajectory_exts': ['.nc'],
    },
    '.parm7': {
        'name': 'PRMTOP + NC',
        'hint': 'PARM7 is the topology. Please also upload an NC file as the trajectory.',
        'requires_trajectory': True,
        'trajectory_exts': ['.nc'],
    },
}

SUPPORTED_EXTENSIONS = '.pdb .psf .gro .tpr .prmtop .parm7 .dcd .xtc .trr .nc'


def detect_format(top_path: str, traj_path: str = None) -> dict:
    """Identify the format of the given files.

    Returns the FORMATS entry for the files, or raises a ValueError with a
    clear message if the format is not supported.
    """
    if top_path is None:
        raise ValueError("Please provide a topology file (PDB, PSF, GRO, TPR or PRMTOP).")

    top_ext = os.path.splitext(top_path)[1].lower()
    info = FORMATS.get(top_ext)

    if info is None:
        raise ValueError(
            f"Unsupported file format: '{top_ext or os.path.basename(top_path)}'. "
            f"Supported extensions: {SUPPORTED_EXTENSIONS}"
        )

    if traj_path is None:
        if info['requires_trajectory']:
            raise ValueError(
                f"This format requires both a topology file and a trajectory file. "
                f"{info['hint']}"
            )
        return info

    traj_ext = os.path.splitext(traj_path)[1].lower()
    if traj_ext not in info['trajectory_exts']:
        raise ValueError(
            f"Unsupported trajectory format '{traj_ext}' for {info['name']}. "
            f"Expected one of: {', '.join(info['trajectory_exts'])}"
        )

    return info


def load_molecule(top_path: str, traj_path: str = None) -> MolecularSystem:
    """Read any supported input and return a common MolecularSystem.

    Examples:
        load_molecule("protein.pdb")
        load_molecule("protein.psf", "traj.dcd")
        load_molecule("conf.gro", "traj.xtc")
    """
    detect_format(top_path, traj_path)

    try:
        if traj_path is not None:
            u = mda.Universe(top_path, traj_path)
        else:
            u = mda.Universe(top_path)
    except Exception as e:
        raise ValueError(
            "Could not read the molecular file. Please check that the file is valid.\n"
            f"Details: {e}"
        )

    return universe_to_system(u)
