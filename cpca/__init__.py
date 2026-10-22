"""Complex PCA of spike phase: assemblies with internal timing from reactivation events."""

__version__ = "0.2.0"

from cpca.decode import readout
from cpca.decompose import decompose, lag_matrix, marchenko_pastur_bound, n_above_null, n_significant, prepare, shuffle_null
from cpca.encode import encode, event_scale
from cpca.events import event_spike_times, population_bursts
from cpca.ica import unmix
from cpca.pipeline import fit
from cpca.stats import coactivation_null, timing_centered, timing_excess

__all__ = [
    "coactivation_null",
    "decompose",
    "encode",
    "event_scale",
    "event_spike_times",
    "fit",
    "lag_matrix",
    "marchenko_pastur_bound",
    "n_above_null",
    "n_significant",
    "population_bursts",
    "prepare",
    "readout",
    "shuffle_null",
    "timing_centered",
    "timing_excess",
    "unmix",
]
