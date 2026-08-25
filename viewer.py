"""Showing one frame of the simulation in 3D.

py3Dmol draws the molecule in the browser, but it only understands text file
formats. So we write one frame of our NumPy coordinates back out as PDB text
and hand that over. Because the text is built from our own arrays, the 3D view
works for every input format.
"""

import py3Dmol
from stmol import showmol

from molecule import SOLVENT


def make_pdb_text(system, frame):
    """Write one frame as PDB text."""
    lines = []

    for i in range(system.n_atoms):
        x, y, z = system.coordinates[frame][i]
        name = str(system.names[i])
        resname = str(system.resnames[i])

        # water and ions are written as HETATM so the viewer can draw them in a
        # different style from the protein
        if resname in SOLVENT:
            record = "HETATM"
        else:
            record = "ATOM  "

        # a PDB file has fixed column positions, so the padding below matters.
        # the atom name goes in columns 14-16, with one space in front of it.
        if len(name) < 4:
            name = " " + name.ljust(3)

        line = record + str(i + 1).rjust(5) + " " + name[:4] + " " + resname[:3].rjust(3)
        line += " A" + str(int(system.resids[i])).rjust(4) + "    "
        line += "%8.3f%8.3f%8.3f  1.00  0.00          %2s" % (x, y, z, system.elements[i])
        lines.append(line)

    lines.append("END")
    return "\n".join(lines)


def show_structure(system, frame, width=900, height=520):
    """Draw the protein as a cartoon and the water and ions as sticks."""
    view = py3Dmol.view(width=width, height=height)
    view.addModel(make_pdb_text(system, frame), "pdb")

    # colour the chain from blue at the start to red at the end
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.setStyle({"hetflag": True}, {"stick": {"radius": 0.2}})

    view.zoomTo()
    showmol(view, height=height, width=width)
