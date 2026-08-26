"""Draws one frame in 3D by writing it out as PDB text for py3Dmol.

Water and ions are excluded, or the camera would zoom out to fit them.
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

        # PDB uses fixed columns; pad short names with a leading space
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
