"""
io
==
Data ingestion, simulation dataset loaders, and experiment serialization.
"""

from __future__ import annotations

from pde_slam.io.experiment import (
    SlamExperimentData,
    load_experiment,
    save_experiment,
)
from pde_slam.io.simulation import (
    SimulationDataset,
    generate_ic_anchors,
    load_simulation_dataset,
    match_field_name,
    sample_simulation_field,
)

__all__ = [
    "SlamExperimentData",
    "save_experiment",
    "load_experiment",
    "SimulationDataset",
    "load_simulation_dataset",
    "sample_simulation_field",
    "generate_ic_anchors",
    "match_field_name",
]
