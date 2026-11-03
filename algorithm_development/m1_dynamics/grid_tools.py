"""Bu-inspired grid primitives, not a reproduction of the WCM estimator.

All delays here are in seconds. A cell uses its LEFT endpoint and a unitless
fraction in [0, 1]. This deliberately resolves the nearest-point/nonnegative
offset ambiguity in the paper; it does not copy both conventions at once.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DelayGrid:
    edges_sec: np.ndarray

    def __post_init__(self):
        edges = np.array(self.edges_sec, dtype=float, copy=True)
        if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
            raise ValueError("Delay edges must be finite, nonnegative and strictly increasing")
        edges.setflags(write=False)
        object.__setattr__(self, "edges_sec", edges)

    @property
    def nodes_sec(self):
        return self.edges_sec[:-1]

    @property
    def widths_sec(self):
        return np.diff(self.edges_sec)

    def encode(self, delays_sec):
        delays = np.asarray(delays_sec, dtype=float)
        if (not np.isfinite(delays).all() or np.any(delays < self.edges_sec[0])
                or np.any(delays > self.edges_sec[-1])):
            raise ValueError("Delay lies outside the declared physical interval")
        cells = np.minimum(np.searchsorted(self.edges_sec, delays, side="right") - 1,
                           len(self.nodes_sec) - 1)
        fractions = (delays - self.nodes_sec[cells]) / self.widths_sec[cells]
        return cells, fractions

    def decode(self, cells, fractions):
        raw_cells = np.asarray(cells)
        frac = np.asarray(fractions, dtype=float)
        if (not np.issubdtype(raw_cells.dtype, np.integer) or np.any(raw_cells < 0)
                or np.any(raw_cells >= len(self.nodes_sec))):
            raise ValueError("Cell indexes must be integers in the grid")
        if not np.isfinite(frac).all() or np.any(frac < 0) or np.any(frac > 1):
            raise ValueError("Within-cell fractions must lie in [0, 1]")
        return self.nodes_sec[raw_cells] + self.widths_sec[raw_cells] * frac


def bu_style_delay_grid(n_cells, maximum_delay_sec, concentration_end_sec,
                        dense_fraction=0.75):
    """Two-piece grid following Bu Eq. (17)-(18), expressed in seconds.

    concentration_end_sec is mu_delay + rms_delay_spread in that paper. The
    caller must establish that these are available ABSOLUTE-delay statistics.
    Cyclic-delay map powers alone do not establish that information.

    maximum_delay_sec is an explicit model-support bound. We do not silently
    use the CP length: the existing receiver also permits ISI beyond the CP.
    """
    if isinstance(n_cells, bool) or int(n_cells) != n_cells or n_cells < 2:
        raise ValueError("At least two integer delay cells are required")
    n_cells = int(n_cells)
    if not (np.isfinite(maximum_delay_sec) and np.isfinite(concentration_end_sec)
            and 0 < concentration_end_sec < maximum_delay_sec
            and 0 < dense_fraction < 1):
        raise ValueError("Require 0 < concentration endpoint < maximum delay and 0 < fraction < 1")
    n_dense = int(np.ceil(dense_fraction * n_cells))
    if n_dense == n_cells:
        raise ValueError("Both delay regions must contain at least one cell")
    edges = np.r_[np.linspace(0, concentration_end_sec, n_dense + 1)[:-1],
                  np.linspace(concentration_end_sec, maximum_delay_sec,
                              n_cells - n_dense + 1)]
    return DelayGrid(edges)


def centered_dft(n):
    """Unitary positive-sign ANALYSIS matrix in Bu Eq. (13)-(14).

    Rows have normalized spatial frequencies (2*a-1)/(2*n)-1/2, a=1..n.
    A physical array needs the correct element ordering and coordinate frame.
    A UPA transform is the appropriate Kronecker product after that ordering
    has been established; this function does not guess it from array size.
    """
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("DFT size must be a positive integer")
    n = int(n)
    frequencies = (2 * np.arange(1, n + 1) - 1) / (2 * n) - 0.5
    return np.exp(2j * np.pi * frequencies[:, None] * np.arange(n)[None, :]) / np.sqrt(n)


def angular_analysis(channel, receive_transform, transmit_transform):
    """H[rx, tx] -> A; transmit side uses transpose, consistent with M1."""
    return receive_transform @ channel @ transmit_transform.T


def angular_synthesis(coefficients, receive_transform, transmit_transform):
    """Inverse of angular_analysis for unitary transforms."""
    return receive_transform.conj().T @ coefficients @ transmit_transform.conj()
