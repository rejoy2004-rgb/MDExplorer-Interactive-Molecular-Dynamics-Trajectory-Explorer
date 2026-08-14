"""Intramolecular protein hydrogen bonds."""

import numpy as np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis


def calculate_hbonds(system, selection="protein"):
    """Number of protein hydrogen bonds for every frame."""
    hbonds_analysis = HydrogenBondAnalysis(
        system.universe,
        donors_sel=selection,
        acceptors_sel=selection,
        hydrogens_sel="element H or name H* or name HN*",  # avoids needing charge information or element metadata
    )
    hbonds_analysis.run()

    hbonds = hbonds_analysis.results.hbonds  # rows: [frame, donor, H, acceptor, dist, angle]
    counts = np.zeros(system.n_frames)
    if len(hbonds) > 0:
        frame_ids = hbonds[:, 0].astype(int)
        for frame in range(system.n_frames):
            counts[frame] = np.count_nonzero(frame_ids == frame)
    return counts
