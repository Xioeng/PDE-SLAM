# PDE-SLAM

> **Aquatic SLAM constrained by an advection-diffusion PDE**  
> A modular JAX/diffrax framework for solving and visualising 2-D
> advection-diffusion PDEs over water-feature scalar fields.

---

## Repository Layout

```
PDE-SLAM/
├── pde_slam/
│   ├── coords/              # Geodetic / ENU spatial reference frames
│   ├── kinematics/          # Differential drive kinematics & controllers
│   ├── pinn.py              # Physics-Informed Neural Network field representations
│   ├── slam.py              # Rao-Blackwellized Particle Filter (RBPF) SLAM
│   ├── io/                  # Simulation loaders, anchor generators, serialization
│   └── viz/                 # Visualisation panels, grids, and satellite backdrops
├── configs/                 # YAML experiment configurations (Biscayne, Miami Canal)
├── examples/                # Core SLAM application entrypoints
│   ├── rbpf_slam.py         # Flagship multi-field RBPF-SLAM on simulation datasets
│   └── rbpf_slam_toy.py     # Fast analytical toy plume SLAM sanity check
├── test_scripts/            # Diagnostic, visualization & testing scripts
│   ├── plot_saved_experiment.py     # Publication-grade batch figure visualizer
│   ├── plot_simulation_dataset.py   # Raw simulation dataset inspector
│   ├── diff_drive_waypoints.py      # Differential drive waypoint controller test
│   └── spatiotemporal_interpolator.py # Classical interpolation baseline
├── tests/                   # Pytest test suite
├── pyproject.toml
└── README.md
```

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

Executable application entrypoints demonstrating the PDE-SLAM algorithm:

### 1. Flagship Multi-Field RBPF-SLAM (`examples/rbpf_slam.py`)
Runs full Rao-Blackwellized Particle Filter SLAM with online Physics-Informed Neural Network (PINN) mapping on hydrodynamic simulation datasets (e.g. Biscayne Bay or Miami Canal).

```bash
# Interactive GUI waypoint selection on Biscayne Bay
python examples/rbpf_slam.py --config configs/biscayne_rbpf_simulation.yaml

# Run on Miami Canal simulation
python examples/rbpf_slam.py --config configs/miami_canal_rbpf_simulation.yaml

# Headless mode with explicit waypoints and customized particle count
python examples/rbpf_slam.py \
    --config configs/biscayne_rbpf_simulation.yaml \
    --waypoints "-200,50; 100,80; 300,-30" \
    --num-particles 150 \
    --no-show
```

---

### 2. Fast Analytical Toy SLAM (`examples/rbpf_slam_toy.py`)
Instantaneous ($< 1\text{ s}$) regression sanity check running RBPF-SLAM on an exact analytical advection-diffusion Gaussian plume without requiring external dataset files.

```bash
# Run standalone toy simulation (saves diagnostic plot to output/graphs/demo_rbpf_slam_toy.png)
python examples/rbpf_slam_toy.py
```

---

## Diagnostic & Visualization Scripts (`test_scripts/`)

Dedicated scripts for inspecting simulation datasets, verifying kinematics controllers, testing interpolators, and rendering publication figures from serialized experiments:

### 1. Publication Figure Visualizer (`test_scripts/plot_saved_experiment.py`)
Batch-renders publication-grade figures (individual and combined trajectory paths, RMSE tracking error curves, multi-stage evolution grids, and space-time PDE residual grids) from saved `.pkl` experiment datasets with equal axes scaling.

```bash
# Auto-detects latest saved experiment in output/results/
python test_scripts/plot_saved_experiment.py

# Specify explicit experiment file and satellite zoom
python test_scripts/plot_saved_experiment.py \
    --file output/results/biscayne_simulation_rbpf_slam_experiment.pkl \
    --zoom 18

# Generate figures headless without opening interactive plot windows
python test_scripts/plot_saved_experiment.py \
    --file output/results/biscayne_simulation_rbpf_slam_experiment.pkl \
    --no-show
```

---

### 2. Simulation Dataset Inspector (`test_scripts/plot_simulation_dataset.py`)
Inspects and visualizes raw advection-diffusion `.npz` hydrodynamic simulation fields across multiple time steps with satellite imagery backdrops.

```bash
# Inspect Biscayne Bay dataset
python test_scripts/plot_simulation_dataset.py \
    --sim-dir data/adv_diff_simulations/biscayne_simulation --no-show

# Inspect Miami Canal dataset with explicit timestamps
python test_scripts/plot_simulation_dataset.py \
    --sim-dir data/adv_diff_simulations/miami_canal \
    --timestamps 0 25 50 75 100 \
    --no-show
```

---

### 3. Differential Drive Waypoint Controller (`test_scripts/diff_drive_waypoints.py`)
Visualizes differential drive path-following kinematics, $(v, \omega)$ velocity profile dynamics, and acceptance radius switching.

```bash
# Run controller analysis with custom speed and acceptance radius
python test_scripts/diff_drive_waypoints.py \
    --speed-mps 2.0 \
    --acceptance-radius 3.0 \
    --dt 0.5
```

---

### 4. Classical Spatiotemporal Interpolation (`test_scripts/spatiotemporal_interpolator.py`)
Demonstrates classical space-time field lookup and spline/GP interpolation from scattered virtual sensor observations along a robot path.

```bash
# Run spatiotemporal interpolation benchmark
python test_scripts/spatiotemporal_interpolator.py --no-show
```



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
# 2. IC_ANCHORS: Initial Condition (t = 0) Sensor Pre-Seeding
# =============================================================================
ic_anchors:
  mode: "auto"     # "auto" (uniform sample inside polygon) or "interactive"
  n_points: 100    # Number of static spatial anchor points at t=0
  seed: 42         # Random seed for anchor point distribution
  epochs: 30       # Warm-up pre-training gradient steps on the t=0 IC buffer

# =============================================================================
# 3. ROBOT: Differential Drive Kinematics & Motion Model
# =============================================================================
robot:
  kinematics: "diff_drive"  # Kinematic model
  nominal_speed: 0.5        # Commanded cruise velocity [m/s]
  dt: 1.0                   # Integration & sampling time step [s]
  v_noise_std: 0.01         # Linear velocity actuation noise std [m/s]
  omega_noise_std: 0.005    # Angular velocity turning noise std [rad/s]
  acceptance_radius: 5.0    # Distance threshold to advance to next waypoint [m]
  waypoints_mode: "auto"    # "auto" or "interactive"

# =============================================================================
# 4. RBPF: Rao-Blackwellized Particle Filter SLAM Settings
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
# 5. PINN: Physics-Informed Neural Network Map Hyperparameters
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
# 6. OUTPUT: Experiment Serialization & Checkpointing
# =============================================================================
output:
  sim_name: "biscayne_simulation"    # Unique experiment name identifier
  results_dir: "output"              # Root folder: results (.pkl) in output/results/, graphs in output/graphs/
  checkpoints: [0, 25, 50, 75, 100]  # Percentage milestones for checkpoint evaluation
  satellite_zoom: 20                 # Web Mercator tile zoom level for map backdrops
  save_experiment: true              # Whether to serialize SlamExperimentData (.pkl)
  save_grids: false                  # Whether to generate composite evolution grids
```

---

## Key Dependencies

| Package   | Role                                    |
|-----------|-----------------------------------------|
| `jax`     | Autodiff, JIT compilation, vectorisation|
| `optax`   | Gradient-based optimizers for PINN maps |
| `scipy`   | Interpolation & spatial algorithms      |
| `numpy`   | Array & tensor computations             |
| `matplotlib` | Visualisation & interactive GUIs     |
| `pyyaml`  | Experiment configuration files          |
| `pillow`  | Satellite imagery tile processing       |


