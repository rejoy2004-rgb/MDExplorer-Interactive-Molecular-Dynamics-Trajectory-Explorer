"""Calculates SASA using the freesasa library instead of mdtraj.

Why this file exists: mdtraj's shrake_rupley has a bug where calling it with many
frames at once makes every frame after the first come out about 0.1% too large,
so an earlier version of analysis.py called it one frame at a time instead - that
was correct, but slow (about 3 seconds per frame for a big protein). freesasa is
a different library with its own code, so it does not have that particular bug,
and is generally fast.

This file IS used by the app: analysis.calculate_sasa() calls
calculate_sasa_freesasa() below directly, and no longer uses mdtraj for SASA at
all.

It is in requirements.txt:
    freesasa>=2.1.0

Note: as of this writing, PyPI does not publish prebuilt wheels for freesasa for
Python 3.12+ (the newest officially listed version is 3.11), and building it from
source requires a C compiler (MSVC Build Tools on Windows). If `import freesasa`
fails with ModuleNotFoundError, that is almost always why - either install the
build tools, or run the app on an environment with Python 3.11 or older until
freesasa publishes newer wheels.

Freesasa's own docs (this code is based on them):
https://freesasa.github.io/python/intro.html
https://freesasa.github.io/python/functions.html
"""

import freesasa
import numpy as np
from multiprocessing import Pool, cpu_count


_COORDS = None
_RADII = None
_PARAMETERS = None


def _make_parameters(n_points):
    parameters = freesasa.Parameters()
    parameters.setAlgorithm(freesasa.ShrakeRupley)
    parameters.setNPoints(n_points)
    parameters.setNThreads(1)  # the Pool already uses every core; freesasa threads on top slow it down
    return parameters


def _init_worker(coords, radii, n_points):
    global _COORDS, _RADII, _PARAMETERS
    _COORDS = coords
    _RADII = radii
    _PARAMETERS = _make_parameters(n_points)


def _compute_frame(frame):
    # calcCoord needs only the coordinates (flat x,y,z list) and one radius per atom
    xyz = _COORDS[frame].ravel().tolist()
    return freesasa.calcCoord(xyz, _RADII, _PARAMETERS).totalArea()


def _protein_radii(system, protein_atoms):
    """Atom names and residues do not change between frames, so look the radii up
    once, from frame 0, instead of rebuilding a freesasa Structure every frame."""
    structure = freesasa.Structure(options={"hydrogen": True})
    for i in protein_atoms:
        x, y, z = system.coordinates[0][i]
        chain = str(system.segids[i])
        if chain == "":
            chain = "A"
        else:
            chain = chain[0]
        structure.addAtom(
            str(system.names[i]),
            str(system.resnames[i]),
            str(int(system.resids[i])),
            chain,
            float(x), float(y), float(z),
        )
    return [structure.radius(k) for k in range(structure.nAtoms())]


def calculate_sasa_freesasa(system, n_points=960):
    """Total SASA (A^2) per frame. n_points is the number of sphere points per atom:
    960 is the original setting; 100 (freesasa's default) is about 5x faster but
    changes the area by roughly 0.5%."""
    protein_atoms = list(system.protein_mask().nonzero()[0])
    if len(protein_atoms) == 0:
        raise ValueError("No protein atoms were found in this file.")

    radii = _protein_radii(system, protein_atoms)
    coords = np.asarray(system.coordinates)[:, protein_atoms]

    # measured on a 12-core PC: 6 workers was fastest, 12 was slower than 4
    workers = min(cpu_count() or 1, 6, system.n_frames)
    if workers < 2:
        _init_worker(coords, radii, n_points)
        return [_compute_frame(frame) for frame in range(system.n_frames)]

    with Pool(processes=workers, initializer=_init_worker, initargs=(coords, radii, n_points)) as pool:
        return list(pool.map(_compute_frame, range(system.n_frames)))
