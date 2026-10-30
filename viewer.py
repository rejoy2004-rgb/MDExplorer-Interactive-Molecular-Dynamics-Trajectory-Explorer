"""Draws one frame in 3D by writing it out as PDB text and rendering it with PyMOL.

Water and ions are excluded, or the camera would zoom out to fit them.
"""

import threading

import numpy as np
import pymol2

# The classic `pymol.finish_launching()` API expects to run on the process's
# main thread (it sets up a windowing/GL context), but Streamlit runs the
# app script on a worker thread, so it hangs forever there. pymol2's PyMOL()
# is the embeddable session API built for exactly this: no GUI, no
# main-thread requirement.
_session = pymol2.PyMOL()
_session.start()
cmd = _session.cmd

# The session is still one shared object. Streamlit reruns each interaction
# (e.g. every slider tick) on its own thread, so without this lock two
# overlapping renders can corrupt PyMOL's state and crash the whole server,
# dropping every connected browser.
_pymol_lock = threading.Lock()


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


def render_structure(system, frame, width=900, height=520, high_quality=False):
    """Render the protein as a rainbow coloured cartoon and return PNG bytes."""
    with _pymol_lock:
        cmd.reinitialize()
        cmd.read_pdbstr(make_pdb_text(system, frame), "frame")

        cmd.hide("everything")
        cmd.show("cartoon")
        cmd.spectrum("count", "rainbow", "name CA")
        cmd.bg_color("white")
        cmd.orient()

        # cmd.png(..., ray=1) already ray-traces internally, so a separate cmd.ray()
        # call before it would ray-trace the same image twice for no benefit.
        return cmd.png(None, width=width, height=height, dpi=150, ray=1 if high_quality else 0)
