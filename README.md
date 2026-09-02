# PDE-SLAM

> **Aquatic SLAM constrained by an advection-diffusion PDE**  
> A modular JAX/diffrax framework for solving, mapping, and visualising 2-D
> spatio-temporal advection-diffusion PDEs over water-feature scalar fields.

---

## Repository Layout

```
PDE-SLAM/
├── pde_slam/
│   ├── coords.py            # Geodetic (WGS-84) ↔ local ENU Cartesian transformations
│   ├── kinematics/          # Differential drive kinematics & dead reckoning
│   │   ├── base.py          # Abstract base kinematics class
│   │   ├── diff_drive.py    # Differential drive model & trajectory integration
│   │   └── dead_reckoning.py# (Experimental) Analytical/autodiff DR covariance estimator
│   ├── interpolators/       # Spatio-temporal and spatial field evaluators
│   │   ├── grid.py          # SpatialGrid domain mesh representations
│   │   ├── field.py         # 2D bilinear field lookup-table interpolator
│   │   ├── spatiotemporal.py# 3D (t, x, y) spatio-temporal lookup-table interpolator
│   │   ├── gp.py            # Gaussian Process regression field interpolator
│   │   └── water_features.py# Synthetic Gaussian plume feature generators
│   ├── pinn.py              # Physics-Informed Neural Network (PINN) mapping & PDE loss
│   ├── slam/                # Online State Estimation
│   │   └── rbpf.py          # Multi-field Rao-Blackwellized Particle Filter (RBPF) SLAM
│   ├── io/                  # Data Ingestion & Serialization
│   │   ├── experiment.py    # SlamExperimentData serialization (.pkl)
│   │   ├── simulation.py    # Multi-field NPZ hydrodynamic simulation dataset loader
│   │   └── survey.py        # CSV survey loader & kinematic differential-drive interpolation
│   ├── config.py            # Typed YAML experiment configuration dataclasses & parsers
│   └── viz/                 # Visualisation panels, grids, and satellite backdrops
├── configs/                 # YAML experiment configurations (Biscayne, Miami Canal, Survey)
├── examples/                # Core SLAM application entrypoints
│   ├── rbpf_slam.py         # Flagship multi-field RBPF-SLAM on simulation datasets
│   ├── rbpf_slam_survey.py  # Field survey CSV trajectory playback RBPF-SLAM
│   └── rbpf_slam_toy.py     # Fast analytical toy plume SLAM sanity check
├── test_scripts/            # Diagnostic, visualization & testing scripts
│   ├── plot_saved_experiment.py     # Publication-grade batch figure visualizer
│   ├── plot_simulation_dataset.py   # Raw simulation dataset inspector
│   ├── diff_drive_waypoints.py      # Differential drive waypoint controller test
│   └── spatiotemporal_interpolator.py # Classical interpolation baseline
├── tests/                   # Pytest test suite (95 tests)
├── pyproject.toml
└── README.md
```

---

## Core Library Modules (`pde_slam/`)

### 1. Kinematics (`pde_slam/kinematics/`)
* **`DiffDriveKinematics`** (`diff_drive.py`): Nonholonomic differential-drive forward kinematics, state stepping, vectorised trajectory integration (`integrate_trajectory`), and waypoint-guidance control.
* **`DeadReckoningEstimator`** (`dead_reckoning.py`) — *(Experimental)*: First-order error propagation (EKF covariance propagation) tracking positional uncertainty and 2D spatial confidence ellipses using analytical and JAX automatic differentiation Jacobians. Note that standard survey playback in `examples/rbpf_slam_survey.py` simulates noise directly without running covariance estimations.

### 2. Field Interpolators (`pde_slam/interpolators/`)
* **`SpatiotemporalInterpolator`** (`spatiotemporal.py`): Continuous 3D $(t, x, y)$ spatio-temporal lookup-table interpolator for querying external hydrodynamic simulation ground truth.
* **`GaussianProcessField`** (`gp.py`): Classical multi-field Gaussian Process spatial regression with RBF / Matérn kernels as a baseline benchmark against PINN representations.
* **`FieldInterpolator`** (`field.py`) & **`SpatialGrid`** (`grid.py`): 2D spatial grid containers and bilinear interpolation routines.

### 3. Coordinate Systems (`pde_slam/coords.py`)
* **`ENUFrame`**: Anchors a local Cartesian East-North-Up (ENU) metric frame to an explicit geodetic reference origin $(\phi_0, \lambda_0)$ using an equirectangular flat-earth projection accurate to $< 0.1\text{ m}$ over $10\text{ km}$.

### 4. Physics-Informed Neural Network (PINN) Mapping (`pde_slam/pinn.py`)
* **`PinnFieldMap`**: Continuous, meshless representation $u_\theta(t, x, y)$ for multi-feature water quality fields (Salinity, Temperature, ODO, Chlorophyll). Supports standard MLPs and Wang & Perdikaris Modified MLPs with Fourier feature embeddings.
* Evaluates exact PDE physical residuals via JAX reverse-mode autodiff:
  $$\mathcal{R}_{\text{PDE}}(t, x, y) = \frac{\partial u}{\partial t} + \mathbf{v}_{\text{flow}} \cdot \nabla u - D \nabla^2 u$$

### 5. State Estimation (`pde_slam/slam/`)
* **`RbpfSlam`** (`slam/rbpf.py`): Rao-Blackwellized Particle Filter maintaining joint posterior distributions over vehicle trajectory particles and linear observation parameters via Kalman updates with map uncertainty inflation and adaptive systematic resampling.

### 6. Data Ingestion & Serialization (`pde_slam/io/`)
* **`load_survey_csv`** (`survey.py`): Ingests field survey CSVs, cleans GPS glitches/dropouts `(0.0, 0.0)`, deduplicates timestamps, and interpolates paths into kinematically compliant differential-drive trajectories capped to a specified duration (e.g. $500\text{ s}$).
* **`load_simulation_dataset`** (`simulation.py`): Ingests multi-field NPZ hydrodynamic simulation datasets (e.g. Biscayne Bay, Miami Canal).
* **`save_experiment` / `load_experiment`** (`experiment.py`): Serializes full experiment runs to `.pkl` files with complete metadata and checkpoints.

---

## Quick Start

### 1. Environment Setup

Install project dependencies using `uv`:

```bash
uv sync --all-extras
source .venv/bin/activate
```

### 2. Run Tests

```bash
uv run pytest -v
```

---

## Examples (`examples/`)

### 1. Field Survey Trajectory RBPF-SLAM (`examples/rbpf_slam_survey.py`)
Replays a real boat field survey CSV (Latitude, Longitude, Date/Time, and sensor readings), interpolates the trajectory via differential-drive kinematics capped to the first 500 seconds, simulates drifting dead reckoning corrupted by process noise ($\sigma_v, \sigma_\omega$), and executes online RBPF-SLAM with online PINN mapping.

```bash
# Run with live visualizer animation
python examples/rbpf_slam_survey.py --config configs/biscayne_survey_rbpf.yaml

# Run headless (non-interactive)
python examples/rbpf_slam_survey.py --config configs/biscayne_survey_rbpf.yaml --no-show

# Override simulation dataset directory or CSV file
python examples/rbpf_slam_survey.py \
    --config configs/biscayne_survey_rbpf.yaml \
    --sim-dir data/adv_diff_simulations/biscayne_simulation \
    --csv-file data/csv/data.csv \
    --no-show

# Use real in-situ CSV sensor observations instead of simulation field sampling
python examples/rbpf_slam_survey.py \
    --config configs/biscayne_survey_rbpf.yaml \
    --use-csv-measurements \
    --no-show
```

---

### 2. Flagship Multi-Field Simulation SLAM (`examples/rbpf_slam.py`)
Runs RBPF-SLAM with online PINN mapping on hydrodynamic simulation datasets (e.g. Biscayne Bay or Miami Canal) with user-selected or automated waypoints and Initial Condition (IC) measurement points.

```bash
# Interactive GUI waypoint selection on Biscayne Bay
python examples/rbpf_slam.py --config configs/biscayne_rbpf_simulation.yaml

# Run on Miami Canal simulation
python examples/rbpf_slam.py --config configs/miami_canal_rbpf_simulation.yaml

# Headless mode with explicit waypoints
python examples/rbpf_slam.py \
    --config configs/biscayne_rbpf_simulation.yaml \
    --waypoints "-200,50; 100,80; 300,-30" \
    --no-show
```

---

### 3. Fast Analytical Toy SLAM (`examples/rbpf_slam_toy.py`)
Instantaneous ($< 1\text{ s}$) regression sanity check running RBPF-SLAM on an exact analytical advection-diffusion Gaussian plume without requiring external dataset files.

```bash
python examples/rbpf_slam_toy.py
```

---

## Diagnostic & Visualization Scripts (`test_scripts/`)

* **`test_scripts/plot_saved_experiment.py`**: Batch-renders publication-grade figures (individual and combined trajectory paths, RMSE tracking error curves, multi-stage evolution grids, and space-time PDE residual grids) from saved `.pkl` experiment datasets with equal axes scaling.
* **`test_scripts/plot_simulation_dataset.py`**: Inspects and visualizes raw advection-diffusion `.npz` hydrodynamic simulation fields across multiple time steps with satellite imagery backdrops.
* **`test_scripts/diff_drive_waypoints.py`**: Visualizes differential drive path-following kinematics, $(v, \omega)$ velocity profiles, and waypoint switching.
* **`test_scripts/spatiotemporal_interpolator.py`**: Demonstrates classical space-time field lookup and spline/GP interpolation baselines.

---

## Experiment Configuration (YAML Schema)

Experiment runs are configured via YAML files in [`configs/`](file:///home/xioeng/Documents/Python/PDE-SLAM/configs) which deserialize directly into the typed `RbpfExperimentConfig` container.

```yaml
# =============================================================================
# 1. SIMULATION: Hydrodynamic Dataset & Physical Field Ingestion
# =============================================================================
simulation:
  sim_dir: "data/adv_diff_simulations/biscayne_simulation"  # Path to folder containing .npz simulation files
  fields:                                                  # Physical fields to track simultaneously
    - "Salinity"
    - "Temperature"
    - "ODO"
  mask_outside: true                                       # Mask ocean regions outside domain polygon with NaNs

# =============================================================================
# 2. SURVEY (Optional): Field Survey Trajectory CSV Configuration
# =============================================================================
survey:
  csv_path: "data/csv/data.csv"   # Path to survey CSV file
  t_max: 500.0                    # Trajectory duration cap [s]
  dt: 1.0                         # Uniform resampling time step [s]
  use_csv_measurements: false     # True to use CSV probe readings; false to sample sim_dir

# =============================================================================
# 3. IC_ANCHORS: Initial Condition (t = 0) Sensor Pre-Seeding
# =============================================================================
ic_anchors:
  mode: "auto"     # "auto" (uniform sample inside polygon) or "interactive"
  n_points: 100    # Number of static spatial anchor points at t=0
  seed: 42         # Random seed for anchor point distribution
  epochs: 30       # Warm-up pre-training gradient steps on the t=0 IC buffer

# =============================================================================
# 4. ROBOT: Differential Drive Kinematics & Motion Model
# =============================================================================
robot:
  kinematics: "diff_drive"  # Kinematic model
  nominal_speed: 0.7        # Commanded cruise velocity [m/s]
  dt: 1.0                   # Integration & sampling time step [s]
  v_noise_std: 0.01         # Linear velocity actuation noise std [m/s]
  omega_noise_std: 0.005    # Angular velocity turning noise std [rad/s]
  acceptance_radius: 2.0    # Distance threshold to advance to next waypoint [m]

# =============================================================================
# 5. RBPF: Rao-Blackwellized Particle Filter SLAM Settings
# =============================================================================
rbpf:
  n_particles: 200              # Number of particles in the filter
  pos_init_std: 0.2             # Initial particle cloud position std [m]
  heading_init_std: 0.02        # Initial particle cloud orientation std [rad]
  measurement_noise_std: 0.001  # Normalized sensor measurement noise std
  lin_process_noise: 0.0001     # Kalman linear state process noise covariance Q_lin
  p0_lin: 0.0025                # Initial linear state estimation covariance P_0
  resample_threshold: 0.5       # Effective sample size ratio threshold (N_eff < 0.5*N)
  seed: 43                      # JAX PRNG key seed

# =============================================================================
# 6. PINN: Physics-Informed Neural Network Map Hyperparameters
# =============================================================================
pinn:
  arch: "mlp"          # Network architecture: "mlp" or "modified_mlp"
  hidden_dim: 64       # Neurons per hidden layer
  num_layers: 3        # Number of hidden layers
  learning_rate: 0.01  # Optax Adam optimizer learning rate
  num_steps: 10        # Gradient update steps per online observation
  num_colloc: 100      # Domain collocation points per gradient step
  margin: 10.0         # Spatial domain collocation padding margin [m]
  w_pde: 0.05          # Loss weighting for autodiff PDE residual term

# =============================================================================
# 7. OUTPUT: Experiment Serialization & Checkpointing
# =============================================================================
output:
  sim_name: "biscayne_survey_experiment"  # Unique experiment name identifier
  results_dir: "output"                   # Results in output/results/, graphs in output/graphs/
  checkpoints: [0, 25, 50, 75, 100]       # Percentage milestones for checkpoint evaluation
  satellite_zoom: 20                      # Web Mercator tile zoom level for map backdrops
  save_experiment: true                   # Whether to serialize SlamExperimentData (.pkl)
  save_grids: false                       # Whether to generate composite evolution grids
```

---

## Key Dependencies

| Package   | Role                                     |
|-----------|------------------------------------------|
| `jax`     | Autodiff, JIT compilation, vectorisation |
| `optax`   | Gradient-based optimizers for PINN maps  |
| `scipy`   | Interpolation & spatial algorithms       |
| `numpy`   | Array & tensor computations              |
| `pandas`  | Field survey CSV ingestion & cleaning    |
| `matplotlib` | Visualisation & interactive GUIs      |
| `pyyaml`  | Experiment configuration files           |
| `pillow`  | Satellite imagery tile processing        |
