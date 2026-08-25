"""The measurements we take from a trajectory.

Each function takes a MolecularSystem and gives back a NumPy array with one
value per frame. RMSF is the exception: it gives one value per residue.

Most of these are worked out directly from the NumPy coordinates. Only SASA and
DSSP come from a library, because those two algorithms are far bigger than the
rest. Water and ions are left out everywhere, otherwise the solvent box around
the protein would dominate the results.
"""

import numpy as np
import mdtraj as md
from scipy.spatial.distance import cdist

# the four atoms that make up the protein backbone
BACKBONE = ["N", "CA", "C", "O"]

# Angstrom, a typical N-H or O-H covalent bond length (used to work out which
# heavy atom a hydrogen belongs to)
BOND_LENGTH = 1.3

# the Baker-Hubbard geometric criteria for a hydrogen bond
HBOND_DIST = 2.5    # Angstrom, hydrogen to acceptor
HBOND_ANGLE = 120    # degrees, donor-H...acceptor

# DSSP uses eight letters, we group them into the three shapes people talk about
HELIX = ["H", "G", "I"]
SHEET = ["E", "B"]


def superimpose(mobile, reference):
    """Rotate 'mobile' so it sits on top of 'reference' as well as possible.

    This is the Kabsch algorithm. We need it because a protein also floats and
    spins around inside the water box during a simulation. Without this step
    the RMSD would mostly measure that spinning instead of the change in shape.

    Both inputs are (n_atoms, 3) arrays already moved to the origin.
    """
    # how the two sets of points vary together
    covariance = np.dot(mobile.T, reference)

    # the SVD of that matrix gives us the best rotation between the two sets
    u, _, vt = np.linalg.svd(covariance)

    # if the determinant is negative the SVD has given a mirror image instead
    # of a rotation, and flipping the last column fixes it
    d = np.sign(np.linalg.det(np.dot(vt.T, u.T)))
    correction = np.diag([1.0, 1.0, d])

    rotation = np.dot(np.dot(vt.T, correction), u.T)
    return np.dot(mobile, rotation.T)


def calculate_rmsd(system):
    """RMSD of the protein backbone in Angstrom, one value per frame.

    RMSD is Root Mean Square Deviation: how far the backbone atoms have moved
    compared to the first frame. A flat curve means the protein is stable, a
    rising curve means it is still changing shape.
    """
    atoms = system.select(BACKBONE)
    if len(atoms) == 0:
        raise ValueError("No protein backbone atoms were found in this file.")

    # the first frame is the reference that everything is compared against
    reference = system.coordinates[0][atoms]
    reference = reference - reference.mean(axis=0)      # move it to the origin

    rmsd = np.zeros(system.n_frames)

    for i in range(system.n_frames):
        frame = system.coordinates[i][atoms]
        frame = frame - frame.mean(axis=0)
        frame = superimpose(frame, reference)

        # square the distance between the two structures atom by atom,
        # take the average, then the square root
        difference = frame - reference
        rmsd[i] = np.sqrt((difference ** 2).sum() / len(atoms))

    return rmsd


def calculate_rmsf(system):
    """RMSF of every residue in Angstrom, measured at its alpha carbon.

    RMSD gives one number for the whole protein at each moment in time. RMSF is
    the other way round: one number per residue, averaged over the whole
    simulation. The peaks are the flexible loops and the flat parts are the
    rigid helices and sheets.
    """
    # one alpha carbon per residue, so we get one value per residue
    atoms = system.select(["CA"])
    if len(atoms) == 0:
        raise ValueError("No alpha carbon (CA) atoms were found in this file.")

    # the frames have to be aligned first, exactly like in the RMSD, otherwise
    # the movement of the whole protein would look like fluctuation
    reference = system.coordinates[0][atoms]
    reference = reference - reference.mean(axis=0)

    aligned = np.zeros((system.n_frames, len(atoms), 3))
    for i in range(system.n_frames):
        frame = system.coordinates[i][atoms]
        frame = frame - frame.mean(axis=0)
        aligned[i] = superimpose(frame, reference)

    # the average position of each residue over the whole simulation
    average = aligned.mean(axis=0)

    # how far each residue is from its own average position, on average
    difference = aligned - average
    squared = (difference ** 2).sum(axis=2)     # squared distance in each frame
    return np.sqrt(squared.mean(axis=0))        # averaged over the frames


def center_of_mass(system):
    """Centre of mass (x, y, z) of the protein, one row per frame.

    This is the balance point of the molecule. It shows whether the protein is
    drifting through the box, and the radius of gyration needs it as well.
    """
    atoms = np.where(system.protein_mask())[0]
    masses = system.masses[atoms]

    com = np.zeros((system.n_frames, 3))

    for i in range(system.n_frames):
        # heavy atoms count more than light ones, so every coordinate is
        # multiplied by the mass of its atom before we take the average
        weighted = system.coordinates[i][atoms] * masses[:, np.newaxis]
        com[i] = weighted.sum(axis=0) / masses.sum()

    return com


def calculate_rg(system):
    """Radius of gyration of the protein in Angstrom, one value per frame.

    This is the average distance of the atoms from the centre of mass. A small
    value means a compact, folded protein and a large value means an open one.
    If Rg suddenly grows during a simulation, the protein is unfolding.
    """
    atoms = np.where(system.protein_mask())[0]
    masses = system.masses[atoms]
    centres = center_of_mass(system)

    rg = np.zeros(system.n_frames)

    for i in range(system.n_frames):
        # distance of every atom from the centre of mass of that same frame
        distance = system.coordinates[i][atoms] - centres[i]
        squared = (distance ** 2).sum(axis=1)

        # the mass weighted average of those distances, then the square root
        rg[i] = np.sqrt((masses * squared).sum() / masses.sum())

    return rg


def calculate_hbonds(system):
    """Number of hydrogen bonds inside the protein, one value per frame.

    Hydrogen bonds are what hold a folded protein together, so counting them
    frame by frame shows whether the structure is staying stable.

    A hydrogen bond is a donor N-H or O-H pointing at an acceptor N or O on a
    different part of the chain. We use the Baker-Hubbard geometric definition
    (Baker & Hubbard, 1984): the distance from the hydrogen to the acceptor
    must be under 2.5 A, and the donor-H...acceptor angle must be over 120
    degrees. This needs the actual hydrogen atoms, which is why it only works
    on files that include them (most MD trajectories do, X-ray PDB files often
    do not).
    """
    protein = system.protein_mask()
    heavy = np.where(protein & np.isin(system.elements, ["N", "O"]))[0]
    hydrogens = np.where(protein & (system.elements == "H"))[0]
    if len(heavy) == 0 or len(hydrogens) == 0:
        raise ValueError("No N-H or O-H donors were found in this file.")

    # Work out which heavy atom each hydrogen is bonded to. We do this once,
    # from the first frame, by finding its nearest N or O: an N-H or O-H bond
    # is only about 1 A long, much shorter than any hydrogen bond.
    first_frame = system.coordinates[0]
    distance_to_heavy = cdist(first_frame[hydrogens], first_frame[heavy])
    nearest_heavy = distance_to_heavy.argmin(axis=1)
    is_bonded = distance_to_heavy.min(axis=1) < BOND_LENGTH

    donor_h = hydrogens[is_bonded]
    donor_heavy = heavy[nearest_heavy[is_bonded]]
    if len(donor_h) == 0:
        raise ValueError("Could not find any bonded N-H or O-H pairs.")

    counts = np.zeros(system.n_frames)

    for i in range(system.n_frames):
        h_pos = system.coordinates[i][donor_h]
        donor_pos = system.coordinates[i][donor_heavy]
        acceptor_pos = system.coordinates[i][heavy]

        h_to_acceptor = cdist(h_pos, acceptor_pos)

        # the angle at the hydrogen, between its own bond and the line to
        # each candidate acceptor
        donor_to_h = donor_pos - h_pos
        donor_to_h = donor_to_h / np.linalg.norm(donor_to_h, axis=1, keepdims=True)
        h_to_a_vec = acceptor_pos[np.newaxis, :, :] - h_pos[:, np.newaxis, :]
        h_to_a_vec = h_to_a_vec / np.linalg.norm(h_to_a_vec, axis=2, keepdims=True)
        cos_angle = np.einsum("ij,ikj->ik", donor_to_h, h_to_a_vec)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        # a hydrogen cannot form a bond back to its own heavy atom
        own_donor = donor_heavy[:, np.newaxis] == heavy[np.newaxis, :]

        is_hbond = (h_to_acceptor < HBOND_DIST) & (angle > HBOND_ANGLE) & (~own_donor)
        counts[i] = np.count_nonzero(is_hbond)

    return counts


def mdtraj_protein(system):
    """Open the files again with mdtraj and keep only the protein.

    SASA and DSSP both need this, and reading a large trajectory twice is slow,
    so the result is stored on the system the first time and reused after that.
    """
    if system.mdtraj_protein is None:
        if system.traj_path is None:
            traj = md.load(system.top_path)
        else:
            traj = md.load(system.traj_path, top=system.top_path)
        system.mdtraj_protein = traj.atom_slice(traj.top.select("protein"))

    return system.mdtraj_protein


def calculate_sasa(system):
    """Total SASA of the protein in square Angstrom, one value per frame.

    SASA is the area of the surface that water can reach. A folded protein
    hides most of its atoms inside so its SASA is small, and when it unfolds
    more is exposed and the SASA goes up.

    The standard method is the Shrake-Rupley algorithm, which rolls a water
    sized ball over the surface. That is a lot to write from scratch, so this
    is one of the two measurements we take from a library.
    """
    # one area value per atom per frame
    sasa = md.shrake_rupley(mdtraj_protein(system))

    # add up the atoms to get the area of the whole protein. mdtraj works in
    # nanometres and 1 nm2 is 100 A2, so we convert at the same time.
    return sasa.sum(axis=1) * 100


def calculate_secondary_structure(system):
    """DSSP letter for every residue in every frame.

    DSSP is the standard way of deciding which shape each residue is in. It
    looks at the hydrogen bonding pattern of the backbone and gives every
    residue a letter. The result has one row per frame and one column per
    residue. Like SASA, this comes from mdtraj.
    """
    return md.compute_dssp(mdtraj_protein(system))


def summarize(codes):
    """Turn the DSSP letters of one frame into helix %, sheet % and coil %."""
    # the residues at the two ends of a chain often cannot be assigned,
    # DSSP marks those as "NA" and we leave them out
    assigned = [code for code in codes if code != "NA"]
    if len(assigned) == 0:
        return 0.0, 0.0, 0.0

    helix = 0
    sheet = 0
    coil = 0

    for code in assigned:
        if code in HELIX:
            helix += 1
        elif code in SHEET:
            sheet += 1
        else:
            coil += 1

    total = len(assigned)
    return helix / total * 100, sheet / total * 100, coil / total * 100
