"""Backbone RMSD relative to the first frame."""

from MDAnalysis.analysis.rms import RMSD


def calculate_rmsd(system, selection="backbone"):
    """RMSD (Å) of the selection vs the first frame, one value per frame.

    Each frame is superimposed onto the first frame before the RMSD is
    computed, so rotation of the molecule does not affect the result.
    """
    rmsd_analysis = RMSD(system.universe, select=selection).run()
    # results.rmsd columns: [frame, time, rmsd_after_superposition, rmsd_no_superposition]
    return rmsd_analysis.results.rmsd[:, 2]
