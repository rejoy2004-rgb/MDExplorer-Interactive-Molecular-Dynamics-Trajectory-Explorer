"""Trajectory measurements: one value per frame (RMSF: per residue).

Plain lists and loops throughout, except superimpose() which still uses
NumPy's SVD once (real linear algebra, not reimplemented) and SASA/DSSP
which come from freesasa and mdtraj (large, established algorithms). Water and
ions are excluded everywhere.
"""

import csv
import io

import mdtraj as md
import numpy as np
from scipy.spatial import cKDTree

# the four atoms that make up the protein backbone
BACKBONE = ["N", "CA", "C", "O"]

# Angstrom; a covalent N-H or O-H bond is about this long
BOND_LENGTH = 1.3

# the Baker-Hubbard geometric criteria for a hydrogen bond
HBOND_DIST = 2.5    # Angstrom, hydrogen to acceptor
HBOND_ANGLE = 120   # degrees, donor-H...acceptor

# mdtraj's compute_dssp is used with its default simplified=True, so it already
# gives only three letters: H (helix), E (sheet) and C (coil), plus "NA" for
# residues it cannot assign. Anything that is not H or E counts as coil.
HELIX = ["H"]
SHEET = ["E"]


# Angstrom; two neighbouring alpha carbons are about 3.8 A apart (at most about
# 4.1 A in the real trajectory we tested). A protein cut by the edge of the
# simulation box jumps by tens of A, so 6 A is far above any real bond and far
# below a box-sized jump.
CA_JUMP_LIMIT = 6.0

# the three possible answers of check_periodic_split
SPLIT = "Potential split detected"
NO_SPLIT = "No obvious split detected"
UNKNOWN = "Unable to assess reliably"


def check_periodic_split(system, max_distance=CA_JUMP_LIMIT):
    """Warn if the protein may be cut by the edge of the simulation box.

    MD programs wrap atoms back into the box, so one molecule can end up as
    pieces on opposite sides. We look for that by measuring the distance
    between neighbouring alpha carbons (residue n and n+1 of the same chain);
    those must always be about 3.8 A apart. Returns (answer, details), where
    answer is SPLIT, NO_SPLIT or UNKNOWN.

    This is only a hint, not a proof. It cannot see a split that happens
    between chains, or inside a stretch with missing residues, and it never
    uses the box size. NO_SPLIT does NOT prove the trajectory is whole."""
    ca = system.select(["CA"])

    # only compare residues that are neighbours in the sequence and in the same
    # chain; a numbering gap or a chain change is a real break, not a split
    first = []
    second = []
    for k in range(len(ca) - 1):
        a = ca[k]
        b = ca[k + 1]
        if system.segids[a] == system.segids[b] and system.resids[b] - system.resids[a] == 1:
            first.append(a)
            second.append(b)

    if len(first) == 0:
        return UNKNOWN, ("There are no pairs of neighbouring alpha carbons to compare "
                         "(no CA atoms, one residue per chain, or gaps in the numbering).")

    # distance of every neighbouring pair in every frame, shape (frames, pairs)
    jumps = np.linalg.norm(system.coordinates[:, second] - system.coordinates[:, first], axis=2)
    if not np.isfinite(jumps).any():
        return UNKNOWN, "The alpha carbon coordinates are not finite numbers."
    jumps = np.where(np.isfinite(jumps), jumps, 0.0)   # a pair with no position cannot show a split
    too_far = jumps > max_distance
    frames_split = int(too_far.any(axis=1).sum())

    if frames_split == 0:
        return NO_SPLIT, ("The largest distance between neighbouring alpha carbons is %.1f A "
                          "(%d pairs checked in every frame)." % (jumps.max(), len(first)))

    worst = np.unravel_index(jumps.argmax(), jumps.shape)
    return SPLIT, ("%d of %d frames have neighbouring alpha carbons more than %.1f A apart; "
                   "the largest gap is %.1f A, between residues %d and %d in frame %d."
                   % (frames_split, system.n_frames, max_distance, jumps.max(),
                      system.resids[first[worst[1]]], system.resids[second[worst[1]]], worst[0]))


def frame_positions(system, frame_index, atom_indices):
    """Plain [x, y, z] positions of the given atoms at one frame."""
    points = []
    for idx in atom_indices:
        x, y, z = system.coordinates[frame_index][idx]
        points.append([float(x), float(y), float(z)])
    return points


def transpose3(m):
    """Flip a 3x3 matrix's rows and columns."""
    result = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            result[j][i] = m[i][j]
    return result


def matmul3(a, b):
    """Multiply two 3x3 matrices."""
    result = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            total = 0.0
            for k in range(3):
                total += a[i][k] * b[k][j]
            result[i][j] = total
    return result


def determinant3(m):
    """Determinant of a 3x3 matrix, using the standard formula."""
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def superimpose(mobile, reference):
    """Rotate 'mobile' onto 'reference' (Kabsch algorithm), so RMSD measures
    shape change, not tumbling. Both are point lists centred on the origin."""
    n = len(mobile)

    # how the two sets of points vary together (a 3x3 matrix)
    covariance = [[0.0, 0.0, 0.0] for _ in range(3)]
    for a in range(3):
        for b in range(3):
            total = 0.0
            for k in range(n):
                total += mobile[k][a] * reference[k][b]
            covariance[a][b] = total

    # SVD gives the best rotation; the one place we still need NumPy
    u, _, vt = np.linalg.svd(covariance)
    u, vt = u.tolist(), vt.tolist()

    # a negative determinant means a mirror flip; fix it via the last column
    vt_t = transpose3(vt)
    u_t = transpose3(u)
    d = 1.0 if determinant3(matmul3(vt_t, u_t)) >= 0 else -1.0
    correction = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, d]]

    rotation = matmul3(matmul3(vt_t, correction), u_t)

    rotated = []
    for point in mobile:
        new_point = [0.0, 0.0, 0.0]
        for j in range(3):
            total = 0.0
            for i in range(3):
                total += point[i] * rotation[j][i]
            new_point[j] = total
        rotated.append(new_point)
    return rotated


def calculate_rmsd(system, reference_frame=0):
    """RMSD (Root Mean Square Deviation) of the backbone per frame, vs one reference frame
    (the first frame unless told otherwise)."""
    atoms = system.select(BACKBONE).tolist()
    if len(atoms) == 0:
        raise ValueError("No protein backbone atoms were found in this file.")
    if not 0 <= reference_frame < system.n_frames:
        raise ValueError("The reference frame must be between 0 and %d." % (system.n_frames - 1))

    coords = system.coordinates[:, atoms, :].astype(float)   # (F, A, 3)
    n_frames = coords.shape[0]
    n_atoms = coords.shape[1]

    reference = coords[reference_frame]                      # (A, 3)
    mobile_c = coords - coords.mean(axis=1, keepdims=True)  # (F, A, 3)
    ref_c = reference - reference.mean(axis=0, keepdims=True)  # (A, 3)

    # per-frame 3x3 covariance
    cov = np.einsum("fai,aj->fij", mobile_c, ref_c)         # (F, 3, 3)

    # Kabsch: SVD per frame, then determinant correction
    u, _, vt = np.linalg.svd(cov)                           # (F,3,3),(F,3,3)
    det_val = np.linalg.det(np.einsum("fij,fjk->fik", vt.transpose(0, 2, 1), u.transpose(0, 2, 1)))
    d = np.where(det_val >= 0, 1.0, -1.0)

    correction = np.tile(np.eye(3)[None, :, :], (n_frames, 1, 1))
    correction[:, 2, 2] = d

    rot = vt.transpose(0, 2, 1) @ correction @ u.transpose(0, 2, 1)   # (F,3,3)

    rotated = np.einsum("fij,faj->fai", rot, mobile_c)      # (F, A, 3)
    diff = rotated - ref_c[None, :, :]
    rmsd = np.sqrt((diff * diff).sum(axis=(1, 2)) / n_atoms)
    return rmsd.tolist()


def calculate_rmsf(system):
    """RMSF (fluctuation) of each residue's alpha carbon, over the whole simulation."""
    atoms = system.select(["CA"]).tolist()
    if len(atoms) == 0:
        raise ValueError("No alpha carbon (CA) atoms were found in this file.")

    coords = system.coordinates[:, atoms, :].astype(float)   # (F, A, 3)
    n_frames = coords.shape[0]

    reference = coords[0]
    mobile_c = coords - coords.mean(axis=1, keepdims=True)
    ref_c = reference - reference.mean(axis=0, keepdims=True)

    cov = np.einsum("fai,aj->fij", mobile_c, ref_c)
    u, _, vt = np.linalg.svd(cov)
    det_val = np.linalg.det(np.einsum("fij,fjk->fik", vt.transpose(0, 2, 1), u.transpose(0, 2, 1)))
    d = np.where(det_val >= 0, 1.0, -1.0)
    correction = np.tile(np.eye(3)[None, :, :], (n_frames, 1, 1))
    correction[:, 2, 2] = d
    rot = vt.transpose(0, 2, 1) @ correction @ u.transpose(0, 2, 1)

    aligned = np.einsum("fij,faj->fai", rot, mobile_c)      # (F, A, 3)

    average = aligned.mean(axis=0)                          # (A, 3)
    diff = aligned - average[None, :, :]
    rmsf = np.sqrt((diff * diff).sum(axis=2).mean(axis=0))
    return rmsf.tolist()


def center_of_mass(system):
    """Mass-weighted centre of mass of the protein, one [x, y, z] per frame."""
    mask = system.protein_mask()
    atoms = np.where(mask)[0]
    if len(atoms) == 0:
        raise ValueError("No protein atoms were found in this file.")
    masses = system.masses[atoms].astype(float)
    total_mass = masses.sum()

    coords = system.coordinates[:, atoms, :].astype(float)    # (F, A, 3)
    # mass-weighted sum per frame
    weighted = coords * masses[None, :, None]               # (F, A, 3)
    com = weighted.sum(axis=1) / total_mass                 # (F, 3)
    return com.tolist()


def calculate_rg(system):
    """Radius of gyration (spread from the centre of mass) per frame."""
    mask = system.protein_mask()
    atoms = np.where(mask)[0]
    if len(atoms) == 0:
        raise ValueError("No protein atoms were found in this file.")
    masses = system.masses[atoms].astype(float)
    total_mass = masses.sum()

    coords = system.coordinates[:, atoms, :].astype(float)    # (F, A, 3)
    com = center_of_mass(system)
    com = np.array(com)                                     # (F, 3)
    diff = coords - com[:, None, :]                         # (F, A, 3)
    sq = (diff * diff).sum(axis=2)                          # (F, A)
    rg = np.sqrt((masses[None, :] * sq).sum(axis=1) / total_mass)
    return rg.tolist()


def calculate_hbonds(system):
    """Hydrogen bond count per frame: the Baker-Hubbard rule (H-acceptor
    under 2.5 A, donor-H-acceptor angle over 120 deg). Needs real H atoms."""
    mask = system.protein_mask()
    heavy = [i for i in range(system.n_atoms) if mask[i] and system.elements[i] in ("N", "O")]
    hydrogens = [i for i in range(system.n_atoms) if mask[i] and system.elements[i] == "H"]
    if len(heavy) == 0 or len(hydrogens) == 0:
        raise ValueError("No N-H or O-H donors were found in this file.")

    # find each hydrogen's own heavy atom once, from frame 0
    hydrogen_points = np.array(frame_positions(system, 0, hydrogens))
    heavy_points = np.array(frame_positions(system, 0, heavy))

    # vectorized nearest-heavy search
    diff = hydrogen_points[:, None, :] - heavy_points[None, :, :]   # (Hh, H, 3)
    distances = np.linalg.norm(diff, axis=2)                       # (Hh, H)
    nearest = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(hydrogens)), nearest]

    donor_h = []
    donor_heavy = []
    for h in range(len(hydrogens)):
        if nearest_distance[h] < BOND_LENGTH:
            donor_h.append(hydrogens[h])
            donor_heavy.append(heavy[nearest[h]])

    if len(donor_h) == 0:
        raise ValueError("Could not find any bonded N-H or O-H pairs.")

    heavy_index_of = {atom: k for k, atom in enumerate(heavy)}
    own_heavy = np.array([heavy_index_of[atom] for atom in donor_heavy])  # per donor H

    coords = system.coordinates.astype(float)  # (F, N, 3)
    donor_h = np.array(donor_h)
    donor_heavy = np.array(donor_heavy)
    heavy = np.array(heavy)

    counts = []
    for i in range(system.n_frames):
        h_pos = coords[i, donor_h, :]          # (D, 3)
        donor_pos = coords[i, donor_heavy, :]   # (D, 3)
        heavy_pos = coords[i, heavy, :]         # (H, 3)

        # only the few (H, acceptor) pairs closer than HBOND_DIST need an angle check
        pairs = cKDTree(h_pos).query_ball_tree(cKDTree(heavy_pos), HBOND_DIST)
        d_index = np.repeat(np.arange(len(pairs)), [len(p) for p in pairs])
        a_index = np.array([k for p in pairs for k in p], dtype=int)
        keep = a_index != own_heavy[d_index]   # an H cannot bond to its own donor
        d_index, a_index = d_index[keep], a_index[keep]

        to_donor = donor_pos[d_index] - h_pos[d_index]
        to_acceptor = heavy_pos[a_index] - h_pos[d_index]
        norms = np.linalg.norm(to_donor, axis=1) * np.linalg.norm(to_acceptor, axis=1)
        cos_angle = (to_donor * to_acceptor).sum(axis=1) / np.where(norms == 0, 1.0, norms)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        counts.append(int((angle > HBOND_ANGLE).sum()))

    return counts


# C-terminal oxygens are named differently by each force field (OT1/OT2, OC1/OC2, O1/O2);
# mdtraj's own file readers rename them to O/OXT, and DSSP needs an atom called O
TERMINAL_OXYGEN = {"OT1": "O", "OC1": "O", "O1": "O", "OT2": "OXT", "OC2": "OXT", "O2": "OXT"}


def mdtraj_protein(system):
    """The protein as an mdtraj trajectory (cached), built from the system's own arrays.

    DSSP comes from mdtraj, which only needs atom names, residues, elements and
    coordinates. Building it from our arrays (not from the files) means it works for every
    file format, including .tpr, which mdtraj cannot read."""
    if system.mdtraj_protein is not None:
        return system.mdtraj_protein

    atoms = np.where(system.protein_mask())[0]
    topology = md.Topology()
    chain = None
    previous_segid = None
    previous_resid = None

    start = 0
    while start < len(atoms):
        first = atoms[start]

        # one residue = the run of atoms with the same chain label, residue number and name
        stop = start
        while (stop < len(atoms) and system.segids[atoms[stop]] == system.segids[first]
               and system.resids[atoms[stop]] == system.resids[first]
               and system.resnames[atoms[stop]] == system.resnames[first]):
            stop += 1
        group = atoms[start:stop]

        # a new chain starts when the chain label changes or the residue number does not go up
        if chain is None or system.segids[first] != previous_segid or system.resids[first] <= previous_resid:
            chain = topology.add_chain()
        residue = topology.add_residue(str(system.resnames[first]), chain, resSeq=int(system.resids[first]))

        names = [str(system.names[a]) for a in group]
        for a, name in zip(group, names):
            if "O" not in names:
                name = TERMINAL_OXYGEN.get(name, name)
            try:
                element = md.element.get_by_symbol(system.elements[a])
            except KeyError:
                raise ValueError("Cannot tell which element atom %s of residue %d is." % (name, system.resids[a]))
            topology.add_atom(name, element, residue)

        previous_segid = system.segids[first]
        previous_resid = system.resids[first]
        start = stop

    # mdtraj works in nm, we store Angstrom
    system.mdtraj_protein = md.Trajectory(system.coordinates[:, atoms, :] / 10.0, topology)
    return system.mdtraj_protein


def calculate_sasa(system, n_points=960):
    """Total solvent accessible surface area per frame, using freesasa (Shrake-Rupley,
    hydrogens included). n_points = sphere points per atom: 960 is finest, 100 is ~5x faster."""
    import sasa_freesasa
    return sasa_freesasa.calculate_sasa_freesasa(system, n_points)


def calculate_secondary_structure(system):
    """DSSP letter (H, E, C or NA) per residue per frame (mdtraj, simplified 3-class scheme)."""
    return md.compute_dssp(mdtraj_protein(system)).tolist()


def summarize(codes):
    """Turn the DSSP letters of one frame into helix %, sheet % and coil %."""
    # DSSP marks unassignable ends as "NA"; skip them
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


def per_frame_csv(system, results):
    """The per-frame results as CSV text. A measurement that is not available is left empty."""
    def value(name, i):
        if results[name] is None:
            return ""
        return round(results[name][i], 6)

    columns = ["frame", "time_ps", "rmsd_A", "rg_A", "sasa_A2", "hydrogen_bonds",
               "com_x_A", "com_y_A", "com_z_A", "helix_percent", "sheet_percent", "coil_percent"]
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(columns)

    for i in range(system.n_frames):
        if results["com"] is None:
            com = ["", "", ""]
        else:
            com = [round(x, 6) for x in results["com"][i]]
        if results["secondary"] is None:
            shapes = ["", "", ""]
        else:
            shapes = [round(x, 6) for x in summarize(results["secondary"][i])]

        # "frame" is the frame's number in the original file (they differ when frames were skipped)
        writer.writerow([system.frame_numbers[i], round(float(system.time[i]), 6), value("rmsd", i), value("rg", i),
                         value("sasa", i), value("hbonds", i), *com, *shapes])

    return text.getvalue()


def rmsf_csv(system, results):
    """The RMSF of every residue as CSV text (empty if RMSF is not available)."""
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(["position", "residue_number", "residue_name", "rmsf_A"])

    if results["rmsf"] is not None:
        for position, atom in enumerate(system.select(["CA"])):
            writer.writerow([position + 1, system.resids[atom], system.resnames[atom], round(results["rmsf"][position], 6)])

    return text.getvalue()
