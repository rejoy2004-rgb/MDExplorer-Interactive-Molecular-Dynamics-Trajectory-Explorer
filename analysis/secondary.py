"""Per-frame secondary structure using mdtraj's DSSP implementation."""

import mdtraj as md

from utils.loader import load_mdtraj

# DSSP codes grouped into simple classes
HELIX = {"H", "G", "I"}
SHEET = {"E", "B"}


def calculate_secondary_structure(system):
    """DSSP code for every protein residue and frame.

    Returns a (n_frames, n_residues) array of single-character codes
    ('H' helix, 'E' strand, 'C' coil, 'NA' unassigned, ...).
    """
    traj = load_mdtraj(system)
    # keep only the protein residues, otherwise water molecules
    # would dominate the per-frame percentages
    protein = traj.atom_slice(traj.top.select("protein"))
    return md.compute_dssp(protein)


def summarize_secondary_structure(codes):
    """Turn one frame of DSSP codes into (helix %, sheet %, coil %)."""
    # first/last residues often cannot be assigned and are reported as "NA"
    assigned = [c for c in codes if c != "NA"]
    n = len(assigned)
    if n == 0:
        return 0.0, 0.0, 0.0
    helix = sum(c in HELIX for c in assigned) / n * 100
    sheet = sum(c in SHEET for c in assigned) / n * 100
    coil = sum(c == "C" for c in assigned) / n * 100
    return helix, sheet, coil
