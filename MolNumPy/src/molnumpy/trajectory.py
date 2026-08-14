from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Trajectory:
    """Common trajectory representation. All data is stored as NumPy arrays.

    Attributes:
        n_frames: number of frames in the trajectory.
        n_atoms: number of atoms per frame.
        coordinates: 3D float array of shape (n_frames, n_atoms, 3).
        velocities: optional 3D float array of shape (n_frames, n_atoms, 3).
        box: optional float array of shape (n_frames, 6) with box dimensions
            (a, b, c, alpha, beta, gamma) per frame.
        time: 1D float array of shape (n_frames,) with the frame time.
    """
    n_frames: int
    n_atoms: int
    coordinates: np.ndarray

    velocities: Optional[np.ndarray] = None
    box: Optional[np.ndarray] = None
    time: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def __post_init__(self):
        expected_coord_shape = (self.n_frames, self.n_atoms, 3)
        if self.coordinates.shape != expected_coord_shape:
            raise ValueError(
                f"Coordinates shape {self.coordinates.shape} does not match "
                f"expected (n_frames={self.n_frames}, n_atoms={self.n_atoms}, 3)"
            )

        if self.velocities is not None and self.velocities.shape != expected_coord_shape:
            raise ValueError(
                f"Velocities shape {self.velocities.shape} does not match "
                f"expected {expected_coord_shape}"
            )

        if self.box is not None and self.box.shape != (self.n_frames, 6):
            raise ValueError(
                f"Box shape {self.box.shape} does not match expected {(self.n_frames, 6)}"
            )

        if len(self.time) == 0:
            self.time = np.arange(self.n_frames, dtype=float)
        elif len(self.time) != self.n_frames:
            raise ValueError(f"Time array length ({len(self.time)}) must match n_frames ({self.n_frames})")
