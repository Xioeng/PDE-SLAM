"""
pde_slam
========
Aquatic SLAM constrained by continuous Physics-Informed Neural Networks
and Rao-Blackwellized Particle Filtering.

Public API
----------
"""

from importlib.metadata import PackageNotFoundError, version

from pde_slam.config import (
    GridConfig,
    IcAnchorsConfig,
    OutputConfig,
    PinnMapConfig,
    PipelineConfig,
    PlumeConfig,
    RbpfExperimentConfig,
    RbpfFilterConfig,
    RobotConfig,
    SimulationConfig,
    load_config,
    load_rbpf_experiment_config,
)
from pde_slam.coords import ENUFrame
from pde_slam.interpolators import (
    FieldInterpolator,
    GaussianProcessField,
    SpatialGrid,
    SpatiotemporalInterpolator,
)
from pde_slam.io import (
    SimulationDataset,
    SlamExperimentData,
    generate_ic_anchors,
    load_experiment,
    load_simulation_dataset,
    sample_simulation_field,
    save_experiment,
)
from pde_slam.kinematics import (
    BaseKinematics,
    DiffDriveKinematics,
)
from pde_slam.pinn import (
    PinnConfig,
    PinnFieldMap,
    PinnParams,
    pinn_forward,
    pinn_forward_mlp,
    pinn_forward_modified_mlp,
    pinn_loss_fn,
    pinn_pde_residual,
)
from pde_slam.slam import RBPFSLAM, RbpfSlam, RbpfState
from pde_slam.viz import (
    PLOT_RC_PARAMS,
    TRAJECTORY_COLORS,
    InteractiveWaypointPicker,
    LiveSlamVisualizer,
    compose_evolution_grid,
    compose_residuals_grid,
    ensure_closed_polygon,
    fetch_satellite_enu_backdrop,
    fetch_satellite_image,
    get_feature_cmap,
    mask_field_grid,
    pick_waypoints_gui,
    plot_saved_experiment,
    render_field_panel,
    render_residual_panel,
    render_tracking_error_panel,
    render_trajectories_panel,
    setup_spatial_axes,
)

try:
    __version__: str = version("pde-slam")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

__all__: list[str] = [
    "__version__",
    "BaseKinematics",
    "DiffDriveKinematics",
    "ENUFrame",
    "FieldInterpolator",
    "GaussianProcessField",
    "InteractiveWaypointPicker",
    "LiveSlamVisualizer",
    "PinnConfig",
    "PinnFieldMap",
    "PinnParams",
    "RBPFSLAM",
    "RbpfSlam",
    "RbpfState",
    "SpatialGrid",
    "SpatiotemporalInterpolator",
    "SimulationDataset",
    "SlamExperimentData",
    "save_experiment",
    "load_experiment",
    "load_simulation_dataset",
    "sample_simulation_field",
    "generate_ic_anchors",
    "GridConfig",
    "PlumeConfig",
    "PipelineConfig",
    "SimulationConfig",
    "IcAnchorsConfig",
    "RobotConfig",
    "RbpfFilterConfig",
    "PinnMapConfig",
    "OutputConfig",
    "RbpfExperimentConfig",
    "load_config",
    "load_rbpf_experiment_config",
    "pinn_forward",
    "pinn_forward_mlp",
    "pinn_forward_modified_mlp",
    "pinn_loss_fn",
    "pinn_pde_residual",
    "PLOT_RC_PARAMS",
    "TRAJECTORY_COLORS",
    "ensure_closed_polygon",
    "mask_field_grid",
    "setup_spatial_axes",
    "pick_waypoints_gui",
    "get_feature_cmap",
    "fetch_satellite_image",
    "fetch_satellite_enu_backdrop",
    "render_field_panel",
    "render_residual_panel",
    "render_trajectories_panel",
    "render_tracking_error_panel",
    "compose_evolution_grid",
    "compose_residuals_grid",
    "plot_saved_experiment",
]
