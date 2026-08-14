"""Simple 3D visualization using py3Dmol + stmol inside Streamlit."""

import py3Dmol
from stmol import showmol

from molnumpy.system import MolecularSystem

# Residue names that represent water (shown differently from the protein)
_WATER_NAMES = {'HOH', 'WAT', 'SOL', 'TIP3', 'SPC'}


def get_pdb_string(system: MolecularSystem, frame_idx: int) -> str:
    """Build a PDB string for one trajectory frame from the common NumPy data.

    This lets the 3D viewer work with any input format, because everything has
    already been converted into the same MolecularSystem.
    """
    coords = system.coordinates[frame_idx]
    atoms = system.topology.atoms

    lines = []
    for i, atom in enumerate(atoms):
        pos = coords[i]
        name = str(atom['name'])
        resname = str(atom['resname'])
        resid = int(atom['resid'])

        # Waters/ions become HETATM so py3Dmol can style them separately
        record = 'HETATM' if resname in _WATER_NAMES else 'ATOM'

        # PDB uses 4 columns for the atom name
        atom_name = f" {name:>3}" if len(name) < 4 else name[:4]

        element = str(atom['element']).strip()
        if not element:
            element = name[0] if name else 'X'

        lines.append(
            f"{record:<6}{i + 1:5d} {atom_name} {resname[:3]:>3} A{resid:4d}    "
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}"
        )
    lines.append("END")
    return "\n".join(lines)


def render_molecular_viewer(system: MolecularSystem, frame_idx: int, width: int = 900, height: int = 520):
    """Show one frame of the system in a py3Dmol viewer.

    Protein-like atoms are drawn as a cartoon, the rest (water, ions, small
    molecules) as sticks so they stay visible.
    """
    pdb_str = get_pdb_string(system, frame_idx)

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_str, 'pdb')

    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.setStyle({'hetflag': True}, {'stick': {'colorscheme': 'Jmol', 'radius': 0.2}})

    view.zoomTo()
    showmol(view, height=height, width=width)
