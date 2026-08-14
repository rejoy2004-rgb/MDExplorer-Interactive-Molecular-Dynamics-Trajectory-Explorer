from molnumpy.topology import Topology
from molnumpy.trajectory import Trajectory


class MolecularSystem:
    """The common format: a Topology plus a Trajectory.

    Every input format is converted into this same object, so the rest of the
    application never has to care where the data came from.
    """
    def __init__(self, topology: Topology, trajectory: Trajectory):
        self._topology = topology
        self._trajectory = trajectory
        # Optional: the source MDAnalysis Universe (kept for fast frame access).
        self._mda_universe = None

    @property
    def topology(self) -> Topology:
        return self._topology

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def coordinates(self):
        return self._trajectory.coordinates

    @property
    def time(self):
        return self._trajectory.time

    @property
    def box(self):
        return self._trajectory.box

    @property
    def n_atoms(self) -> int:
        return self._topology.n_atoms

    @property
    def n_frames(self) -> int:
        return self._trajectory.n_frames

    @property
    def n_residues(self) -> int:
        return self._topology.n_residues

    @property
    def masses(self):
        return self._topology.masses

    @property
    def charges(self):
        return self._topology.charges

    def __repr__(self) -> str:
        return f"<MolecularSystem {self.n_atoms} atoms, {self.n_frames} frames>"
