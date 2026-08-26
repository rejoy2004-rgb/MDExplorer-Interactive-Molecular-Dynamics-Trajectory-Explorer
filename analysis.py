"""Trajectory measurements: one value per frame (RMSF: per residue).

Plain lists and loops throughout, except superimpose() which still uses
NumPy's SVD once (real linear algebra, not reimplemented) and SASA/DSSP
which still come from mdtraj (large, established algorithms). Water and
ions are excluded everywhere.
"""

import math
import mdtraj as md
import numpy as np

# the four atoms that make up the protein backbone
BACKBONE = ["N", "CA", "C", "O"]

# Angstrom; a covalent N-H or O-H bond is about this long
BOND_LENGTH = 1.3

# the Baker-Hubbard geometric criteria for a hydrogen bond
HBOND_DIST = 2.5    # Angstrom, hydrogen to acceptor
HBOND_ANGLE = 120   # degrees, donor-H...acceptor

# DSSP uses eight letters, we group them into the three shapes people talk about
HELIX = ["H", "G", "I"]
SHEET = ["E", "B"]


def frame_positions(system, frame_index, atom_indices):
    """Plain [x, y, z] positions of the given atoms at one frame."""
    points = []
    for idx in atom_indices:
        x, y, z = system.coordinates[frame_index][idx]
        points.append([float(x), float(y), float(z)])
    return points


def mean_center(points):
    """Move a list of points so their average position sits at the origin."""
    n = len(points)
    total = [0.0, 0.0, 0.0]
    for p in points:
        total[0] += p[0]
        total[1] += p[1]
        total[2] += p[2]
    average = [total[0] / n, total[1] / n, total[2] / n]
    return [[p[0] - average[0], p[1] - average[1], p[2] - average[2]] for p in points]


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


def calculate_rmsd(system):
    """RMSD (Root Mean Square Deviation) of the backbone per frame, vs frame 0."""
    atoms = system.select(BACKBONE).tolist()
    if len(atoms) == 0:
        raise ValueError("No protein backbone atoms were found in this file.")

    # the first frame is the reference that everything is compared against
    reference = mean_center(frame_positions(system, 0, atoms))

    rmsd = []
    for i in range(system.n_frames):
        frame = mean_center(frame_positions(system, i, atoms))
        frame = superimpose(frame, reference)

        # root mean square of the per-atom distance
        total = 0.0
        for a in range(len(atoms)):
            dx = frame[a][0] - reference[a][0]
            dy = frame[a][1] - reference[a][1]
            dz = frame[a][2] - reference[a][2]
            total += dx * dx + dy * dy + dz * dz
        rmsd.append(math.sqrt(total / len(atoms)))

    return rmsd


def calculate_rmsf(system):
    """RMSF (fluctuation) of each residue's alpha carbon, over the whole simulation."""
    atoms = system.select(["CA"]).tolist()
    if len(atoms) == 0:
        raise ValueError("No alpha carbon (CA) atoms were found in this file.")

    # align first, like RMSD, so drift isn't mistaken for fluctuation
    reference = mean_center(frame_positions(system, 0, atoms))

    aligned = []
    for i in range(system.n_frames):
        frame = mean_center(frame_positions(system, i, atoms))
        aligned.append(superimpose(frame, reference))

    n_atoms = len(atoms)
    n_frames = system.n_frames

    # the average position of each residue over the whole simulation
    average = []
    for a in range(n_atoms):
        total = [0.0, 0.0, 0.0]
        for i in range(n_frames):
            total[0] += aligned[i][a][0]
            total[1] += aligned[i][a][1]
            total[2] += aligned[i][a][2]
        average.append([total[0] / n_frames, total[1] / n_frames, total[2] / n_frames])

    # how far each residue is from its own average position, on average
    rmsf = []
    for a in range(n_atoms):
        total = 0.0
        for i in range(n_frames):
            dx = aligned[i][a][0] - average[a][0]
            dy = aligned[i][a][1] - average[a][1]
            dz = aligned[i][a][2] - average[a][2]
            total += dx * dx + dy * dy + dz * dz
        rmsf.append(math.sqrt(total / n_frames))

    return rmsf


def center_of_mass(system):
    """Mass-weighted centre of mass of the protein, one [x, y, z] per frame."""
    mask = system.protein_mask()
    atoms = [i for i in range(system.n_atoms) if mask[i]]
    masses = [float(system.masses[i]) for i in atoms]
    total_mass = sum(masses)

    com = []
    for i in range(system.n_frames):
        points = frame_positions(system, i, atoms)

        # weight each atom's position by its mass
        total = [0.0, 0.0, 0.0]
        for a in range(len(atoms)):
            total[0] += points[a][0] * masses[a]
            total[1] += points[a][1] * masses[a]
            total[2] += points[a][2] * masses[a]
        com.append([total[0] / total_mass, total[1] / total_mass, total[2] / total_mass])

    return com


def calculate_rg(system):
    """Radius of gyration (spread from the centre of mass) per frame."""
    mask = system.protein_mask()
    atoms = [i for i in range(system.n_atoms) if mask[i]]
    masses = [float(system.masses[i]) for i in atoms]
    total_mass = sum(masses)
    centres = center_of_mass(system)

    rg = []
    for i in range(system.n_frames):
        points = frame_positions(system, i, atoms)

        # mass-weighted root-mean-square distance from the centre
        total = 0.0
        for a in range(len(atoms)):
            dx = points[a][0] - centres[i][0]
            dy = points[a][1] - centres[i][1]
            dz = points[a][2] - centres[i][2]
            total += masses[a] * (dx * dx + dy * dy + dz * dz)
        rg.append(math.sqrt(total / total_mass))

    return rg


def calculate_hbonds(system):
    """Hydrogen bond count per frame: the Baker-Hubbard rule (H-acceptor
    under 2.5 A, donor-H-acceptor angle over 120 deg). Needs real H atoms."""
    mask = system.protein_mask()
    heavy = [i for i in range(system.n_atoms) if mask[i] and system.elements[i] in ("N", "O")]
    hydrogens = [i for i in range(system.n_atoms) if mask[i] and system.elements[i] == "H"]
    if len(heavy) == 0 or len(hydrogens) == 0:
        raise ValueError("No N-H or O-H donors were found in this file.")

    # find each hydrogen's own heavy atom once, from frame 0
    hydrogen_points = frame_positions(system, 0, hydrogens)
    heavy_points = frame_positions(system, 0, heavy)

    donor_h = []
    donor_heavy = []
    for h in range(len(hydrogens)):
        nearest_distance = None
        nearest_heavy = None
        for a in range(len(heavy)):
            dx = hydrogen_points[h][0] - heavy_points[a][0]
            dy = hydrogen_points[h][1] - heavy_points[a][1]
            dz = hydrogen_points[h][2] - heavy_points[a][2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_heavy = heavy[a]

        if nearest_distance < BOND_LENGTH:
            donor_h.append(hydrogens[h])
            donor_heavy.append(nearest_heavy)

    if len(donor_h) == 0:
        raise ValueError("Could not find any bonded N-H or O-H pairs.")

    counts = []
    for i in range(system.n_frames):
        h_points = frame_positions(system, i, donor_h)
        donor_points = frame_positions(system, i, donor_heavy)
        acceptor_points = frame_positions(system, i, heavy)

        count = 0
        for d in range(len(donor_h)):
            # direction from H to its donor, normalised (must point away from H)
            bx = donor_points[d][0] - h_points[d][0]
            by = donor_points[d][1] - h_points[d][1]
            bz = donor_points[d][2] - h_points[d][2]
            bond_length = math.sqrt(bx * bx + by * by + bz * bz)
            bx, by, bz = bx / bond_length, by / bond_length, bz / bond_length

            for a in range(len(heavy)):
                # a hydrogen cannot form a bond back to its own heavy atom
                if heavy[a] == donor_heavy[d]:
                    continue

                ax = acceptor_points[a][0] - h_points[d][0]
                ay = acceptor_points[a][1] - h_points[d][1]
                az = acceptor_points[a][2] - h_points[d][2]
                distance = math.sqrt(ax * ax + ay * ay + az * az)
                if distance >= HBOND_DIST:
                    continue

                # angle at H between its own bond and the line to this acceptor
                ax, ay, az = ax / distance, ay / distance, az / distance
                cos_angle = bx * ax + by * ay + bz * az
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle = math.degrees(math.acos(cos_angle))

                if angle > HBOND_ANGLE:
                    count += 1

        counts.append(count)

    return counts


def mdtraj_protein(system):
    """Reopen the file with mdtraj (cached), keeping only the protein."""
    if system.mdtraj_protein is None:
        if system.traj_path is None:
            traj = md.load(system.top_path)
        else:
            traj = md.load(system.traj_path, top=system.top_path)
        system.mdtraj_protein = traj.atom_slice(traj.top.select("protein"))

    return system.mdtraj_protein


def calculate_sasa(system):
    """Total solvent accessible surface area per frame (mdtraj's Shrake-Rupley)."""
    # per-atom area, converted to a plain list
    per_atom = md.shrake_rupley(mdtraj_protein(system)).tolist()

    # sum per atom; convert mdtraj's nm2 to A2
    return [sum(frame) * 100 for frame in per_atom]


def calculate_secondary_structure(system):
    """DSSP letter per residue per frame (mdtraj)."""
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
