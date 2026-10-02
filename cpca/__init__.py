"""Complex PCA of spike phase."""

__version__ = "0.1.0"

from cpca.decode import readout
from cpca.decompose import decompose, lag_matrix
from cpca.encode import encode
from cpca.mixture import direction_split_null, split_by_direction
from cpca.stats import coactivation_null, split_half, timing_centered, timing_excess

__all__ = [
    "coactivation_null",
    "decompose",
    "direction_split_null",
    "encode",
    "lag_matrix",
    "readout",
    "split_by_direction",
    "split_half",
    "timing_centered",
    "timing_excess",
]
