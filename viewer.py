"""Showing one frame of the simulation in 3D.

py3Dmol draws the molecule in the browser, but it only understands text file
formats. So we write one frame of our NumPy coordinates back out as PDB text
and hand that over. Because the text is built from our own arrays, the 3D view
works for every input format.

Water and ions are left out of the PDB text completely, not just drawn
differently. py3Dmol's zoomTo() fits the camera to every atom it is given, so
if the water was included the camera would zoom out to fit the whole solvent
box and the protein itself would look tiny in the middle of it.
"""

import numpy as np
import py3Dmol
from stmol import showmol


def make_pdb_text(system, frame):
    """Write the protein atoms of one frame as PDB text."""
    lines = []

    for i in np.where(system.protein_mask())[0]:
        x, y, z = system.coordinates[frame][i]
        name = str(system.names[i])
        resname = str(system.resnames[i])

        # a PDB file has fixed column positions, so the padding below matters.
        # the atom name goes in columns 14-16, with one space in front of it.
        if len(name) < 4:
            name = " " + name.ljust(3)

        line = "ATOM  " + str(i + 1).rjust(5) + " " + name[:4] + " " + resname[:3].rjust(3)
        line += " A" + str(int(system.resids[i])).rjust(4) + "    "
        line += "%8.3f%8.3f%8.3f  1.00  0.00          %2s" % (x, y, z, system.elements[i])
        lines.append(line)

    lines.append("END")
    return "\n".join(lines)


def show_structure(system, frame, width=900, height=520):
    """Draw the protein as a rainbow coloured cartoon."""
    view = py3Dmol.view(width=width, height=height)
    view.addModel(make_pdb_text(system, frame), "pdb")

    # colour the chain from blue at the start to red at the end
    view.setStyle({"cartoon": {"color": "spectrum"}})

    view.zoomTo()
    showmol(view, height=height, width=width)
