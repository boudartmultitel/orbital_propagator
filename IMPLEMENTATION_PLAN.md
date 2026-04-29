# Orbital Propagator Implementation Plan

## Context

This plan is based on the proposal in `TReC26_Proposal_1.pdf` and on current library documentation reviewed on April 23, 2026.

Target capability:

- propagate a spacecraft state around Earth first, then around other central bodies by changing body parameters
- return trajectory and acceleration histories
- include force toggles for:
  - central gravity
  - J2
  - atmospheric drag
  - third-body gravity from Sun and Moon
  - solar radiation pressure
- generate datasets for hybrid analytical + learned residual models
- expose the simulation through a web UI with interactive 3D visualization

## Workspace And Deployment Constraints

The project is split into two top-level workspaces:

- `orbital_propagator` for phases 0 to 3
- `visualization_interface` for phases 4 to 5

Containerization requirement:

- each workspace should have its own `Dockerfile`
- the repository root should provide a `docker-compose.yml`
- simulation outputs should be written to a shared Docker volume
- the visualization app should read results from that same shared volume

Recommended shared artifact path inside containers:

- `/shared/results`

Practical implication:

- `orbital_propagator` is the producer of run artifacts during phases 0 to 3
- `visualization_interface` is the consumer of those artifacts during phases 4 to 5
- later ML jobs can write to the same artifact volume using the same schema

## Recommended Technology Stack

### 1. Propagator core

Recommended default stack:

- `numpy` for vector math
- `scipy.integrate.solve_ivp` for numerical integration
- `astropy` for constants, units, time handling, coordinate utilities, and Sun/Moon ephemerides
- `pymsis` for higher-fidelity Earth atmospheric density when drag is enabled in LEO

Why this should be the baseline:

- it is transparent and easy to validate force-by-force
- it keeps the physics code under your control, which matters for residual learning and interpretability
- changing the central body mass or radius is trivial once the force models are written explicitly
- it avoids tying the first prototype to a large external propagation framework

### 2. Reference / high-fidelity simulator

Recommended comparison stack:

- `tudatpy`

Use `tudatpy` for:

- cross-checking your custom force models
- generating higher-fidelity synthetic trajectories for training data
- later extensions beyond the first Earth-only prototype

Reason:

- the current docs explicitly support spherical harmonic gravity, aerodynamic acceleration, point-mass third-body gravity, and radiation pressure in one propagation setup
- it is stronger than `poliastro` for a full perturbation environment

### 3. Lightweight astrodynamics utilities

Useful but not the main engine:

- `poliastro`

Use `poliastro` for:

- sanity checks on two-body and simple perturbation formulas
- quick notebooks and educational demonstrations

Do not make it the backbone of the project:

- its perturbation functions are useful, but for your use case the custom solver plus `tudatpy` is a cleaner architecture

### 4. ML stack

Recommended:

- `torch`
- `pykan`
- optional `lightning` only if you want standardized training orchestration

Guidance:

- implement the baseline MLP directly in PyTorch
- use `pykan` for the KAN experiments
- start from residual acceleration learning, not end-to-end state rollout learning

### 5. PINN / operator-learning baseline

Recommended:

- `DeepXDE`

Use it only as a research baseline, not as the main simulation engine.

### 6. Web app and visualization

Recommended fast path:

- `Dash`
- `plotly`

Reason:

- fastest path to a usable parameter form + run button + animated 3D orbit plot
- fully Python-based
- easy to connect to simulation outputs and diagnostics

Recommended upgrade path for more visually impressive 3D:

- `pyvista` + `trame`

Use this only after the physics pipeline works. It can produce a stronger 3D experience, but it is a heavier integration path than Dash/Plotly.

## Library Evaluation Summary

### `scipy`

Best fit for:

- your first custom propagator
- explicit force-model development
- controlled ablation studies

Main advantage:

- you own the ODE right-hand side completely

Main limitation:

- you must implement all perturbations yourself

### `astropy`

Best fit for:

- time systems
- constants
- ephemerides for Sun and Moon
- unit-safe validation scripts

Main advantage:

- reduces coordinate/time mistakes early

Main limitation:

- it is not a full orbit propagator

### `pymsis`

Best fit for:

- Earth drag studies in LEO

Main advantage:

- much better than a constant-density or crude exponential model once you want realistic drag variation

Main limitation:

- Earth-specific, so it should sit behind a density-model interface

### `tudatpy`

Best fit for:

- reference truth generation
- validation
- later higher-fidelity mission scenarios

Main advantage:

- already supports the perturbations you care about in a unified propagation framework

Main limitation:

- heavier than a custom SciPy propagator, and less transparent for initial educational debugging

### `poliastro`

Best fit for:

- simple orbit utilities and notebook-level demonstrations

Main advantage:

- accessible API and built-in perturbation functions

Main limitation:

- not the best anchor for a research workflow that will grow toward high-fidelity residual learning

### `pykan`

Best fit for:

- the interpretable residual model in the proposal

Main advantage:

- direct alignment with your scientific objective

Main limitation:

- the official repo itself warns about efficiency and recommends explicit speed configuration before training

### `DeepXDE`

Best fit for:

- PINN or DeepONet research baselines

Main advantage:

- gives you a standard scientific-ML baseline instead of building PINNs from scratch

Main limitation:

- should not be on the critical path for the first propagator milestone

### `Dash` + `plotly`

Best fit for:

- first web interface and demo

Main advantage:

- rapid prototype entirely in Python

Main limitation:

- visually good, but less cinematic than a VTK-based 3D front end

### `pyvista` + `trame`

Best fit for:

- polished 3D scientific visualization

Main advantage:

- stronger 3D rendering and scene composition

Main limitation:

- more integration effort than needed for the first usable app

## Recommended System Architecture

Build the project in four layers.

### Layer A. Physics kernel

Core module responsibilities:

- state definition: `r`, `v`, mass, ballistic properties
- body definition: `mu`, radius, `J2`, rotation rate, optional atmosphere model
- force model interface
- acceleration breakdown by source
- ODE right-hand side
- integrator wrappers

Suggested force-model API:

```python
class ForceModel(Protocol):
    name: str
    def acceleration(
        self,
        t: float,
        state: np.ndarray,
        context: PropagationContext,
    ) -> np.ndarray: ...
```

Suggested outputs per run:

- `times`
- `states`
- `accelerations_total`
- `accelerations_by_force`
- metadata about enabled models and tolerances

Recommended additional fields for app compatibility:

- `run_id`
- `producer` such as `simulation`, `reference`, or `ml`
- `central_body`
- `initial_conditions`
- `parameters`
- optional derived series such as orbital elements

### Layer B. Data-generation pipeline

Responsibilities:

- sample initial conditions
- run many propagations
- compute total and residual accelerations
- derive features
- split train/validation/test sets by orbital regime

Preferred target for ML:

- residual perturbation acceleration

That means:

```text
a_residual = a_total - a_central_gravity
```

This is closer to your proposal and easier to interpret than learning the whole acceleration vector.

### Layer C. ML training and evaluation

Responsibilities:

- train MLP baseline
- train KAN residual model
- optional PINN / DeepONet baseline
- evaluate pointwise acceleration error
- evaluate rollout stability when coupled back into the propagator

Important recommendation:

- keep the ML code completely separate from the numerical propagator module
- the propagator should be usable without PyTorch installed

### Layer D. Web application

Responsibilities:

- read saved run outputs from disk
- expose run metadata and visualization controls
- optionally submit a propagation job later
- display time series and 3D trajectory
- show acceleration contributions by force
- compare outputs from different producers

Producer examples:

- physics simulation output
- validation/reference output
- future ML pipeline output

Important design choice:

- the web app should consume a stable run artifact format
- the producer of the artifact should not matter to the UI

## Force Models To Implement

Implement them in this order.

### Milestone 1. Two-body gravity

Equation:

- central gravity only with configurable `mu`

Why first:

- establishes the integrator, units, state conventions, and regression tests

Validation:

- energy approximately conserved over long runs
- orbital elements remain stable within numerical tolerance

### Milestone 2. J2

Add:

- zonal oblateness perturbation with configurable `J2` and equatorial radius

Validation:

- compare secular RAAN / argument-of-perigee drift trends against known expectations

### Milestone 3. Third-body gravity

Add:

- Sun and Moon point-mass perturbations

Implementation note:

- use `astropy` ephemerides first
- keep an ephemeris provider abstraction in case you later switch to SPICE or `tudatpy`

### Milestone 4. Solar radiation pressure

Add:

- cannonball SRP model with `C_r` and area-to-mass ratio
- optional eclipse handling later

Validation:

- first without eclipse logic
- then decide whether eclipse is necessary for your workshop scope

### Milestone 5. Atmospheric drag

Add in two steps:

1. simple exponential atmosphere for debugging
2. `pymsis` for Earth LEO realism

Implementation note:

- drag needs the atmosphere-relative velocity, so include planet rotation in the model
- keep a `DensityModel` interface because drag is inherently body-dependent

## Recommended Repository Structure

```text
docker-compose.yml
orbital_propagator/
  Dockerfile
  requirements.txt
  src/
    orbital_propagator/
      config/
      bodies/
      forces/
        gravity.py
        j2.py
        drag.py
        third_body.py
        srp.py
      ephemerides/
      propagation/
        dynamics.py
        integrators.py
        runner.py
      io/
      validation/
  tests/
  notebooks/
visualization_interface/
  Dockerfile
  requirements.txt
  app/
    loaders/
    plots/
    components/
    assets/
docs/
```

Shared storage convention:

- both containers mount the same Docker volume at `/shared/results`
- all run artifacts should be saved there using the agreed schema
- the visualization layer should never read directly from the propagator source tree

## Proposed Implementation Sequence

### Phase 0. Project scaffolding [V]

Deliverables:

- split workspace into `orbital_propagator` and `visualization_interface`
- Python package layout
- configuration system
- unit convention decided and documented
- reproducible container setup with `Dockerfile`s and `docker-compose.yml`

Recommendation:

- use SI internally
- only convert for display or export
- define the shared results volume and artifact path before any simulation code

### Phase 1. Deterministic propagator MVP [V]

Deliverables:

- custom SciPy propagator
- central gravity + J2
- trajectory and acceleration outputs
- tests against simple known cases

This is the first non-negotiable milestone.

### Phase 2. Full Earth perturbation set [V]

Deliverables:

- third-body Sun/Moon
- SRP
- drag with simple density model
- optional `pymsis` density integration

Exit criterion:

- one configurable Earth orbit scenario can be run with any subset of perturbations enabled

### Phase 2.5. Adaptation of the UI and perturbations [V]

- implement initial parameters for eccentric orbits
- implement rotating atmosphere
- plot missing omega orbital element
- add Sun and Moon direction changes for long propagation


### Phase 3. Validation framework

Deliverables:

- comparison notebooks against `tudatpy`
- force-by-force residual plots
- regression tests for acceleration magnitudes and long-horizon behavior
- consider using a symplectic method for conservative case only (two-body, J2) to avoid numerical errors to be learned by the KAN
- verify the effect of numerical errors for non-conservative cases (drag, third-body, SRP)

Exit criterion:

- custom propagator agrees with reference simulations to an acceptable tolerance for chosen scenarios

### Phase 4. Visualization-first web interface

Deliverables:

- [X] stable run artifact schema for saved outputs
- [X] loader utilities for reading trajectory/result files
- [X] Dash app for inspecting saved runs
- [X] 3D trajectory view
- [X] Earth's 3D spherical and ellipsoid views
- [X] acceleration and orbital-element plots
- [ ] comparison mode for two or more runs

Exit criterion:

- the app can load and visualize a saved output without rerunning the simulation
- the same app structure can later read ML-produced outputs with no UI redesign

Implementation note:

- build the UI around file-backed results, not around direct in-memory solver calls
- this keeps debugging simple and decouples the app from both the propagator and the future ML pipeline
- read run artifacts only from the shared Docker volume path, not from ad hoc local file paths

### Phase 5. Demo-grade visualization

Optional upgrade:

- [ ] move selected views to `pyvista` + `trame`
- [X] render planet sphere, atmosphere shell, Sun direction, and orbit trails with stronger visual quality

### Phase 6. Dataset pipeline

Deliverables:

- scenario sampler across LEO/MEO/GEO
- dataset exporter to parquet / numpy / zarr
- train/val/test splits by regime

Recommendation:

- store both Cartesian states and derived features
- always save per-force acceleration components

### Phase 7. ML baselines

Deliverables:

- MLP residual model
- KAN residual model
- optional DeepXDE PINN / DeepONet baseline

Evaluation:

- acceleration MAE/MSE
- rollout divergence over time
- generalization across altitudes and eccentricities

### Phase 8. Hybrid propagator

Deliverables:

- analytical central gravity + learned residual term
- switchable analytical / learned / hybrid modes
- rollout comparison tooling

Important:

- do not train directly in the web app path
- the web app should consume saved models and simulation outputs

## Suggested Web UI Scope

The first web demo should focus on loading and inspecting run artifacts. Running new simulations from the UI can be added later.

Controls:

- run selector
- producer selector
- comparison run selector
- visible series toggles
- force-component toggles
- camera / animation controls
- optional later: simulation parameter form

Views:

- 3D trajectory
- animation scrubber
- acceleration norm by source
- orbital elements over time
- summary cards: min altitude, max altitude, mean speed, final state
- metadata panel: producer, body, enabled forces, solver settings, physical parameters

Recommended first release:

- Dash sidebar for loading and filtering runs
- Plotly 3D orbit scene
- Plotly time-series panels for position, velocity, and acceleration norms
- simple comparison overlay between two saved runs

## Validation Strategy

You should validate at three levels.

### Level 1. Unit tests

- sign and magnitude checks for each force model
- dimension and shape checks
- invariants in two-body propagation

### Level 2. Scenario tests

- circular LEO with no perturbations
- J2-only precession case
- drag-dominated LEO decay case
- SRP-enabled higher-altitude case

### Level 3. Reference tests

- compare selected runs against `tudatpy`
- save numerical baselines and rerun them in CI

## Risks And Mitigations

### Risk 1. Unit inconsistency

Mitigation:

- use SI everywhere internally
- centralize all constants in one module

### Risk 2. Drag model complexity derails the schedule

Mitigation:

- start with exponential density
- add `pymsis` only after the drag pipeline is already working

### Risk 3. KAN training becomes unstable or slow

Mitigation:

- baseline with an MLP first
- train on residual accelerations, not states
- keep feature scaling and regime-specific splits explicit

### Risk 4. Web UI couples too early to heavy simulations

Mitigation:

- define a stable run artifact schema before building callbacks
- store structured results that can be serialized cleanly
- make the UI a reader first and a launcher second

## Concrete 2-Week Execution Plan

### Days 1-2

- scaffold `orbital_propagator` and `visualization_interface`
- add `Dockerfile`s and root `docker-compose.yml`
- implement two-body propagator
- add tests
- define result schema

### Days 3-4

- implement J2
- implement acceleration breakdown logging
- build first notebooks for validation

### Days 5-6

- implement Sun/Moon third-body gravity via `astropy`
- implement SRP
- validate with ablation runs

### Days 7-8

- implement drag with exponential atmosphere
- add `pymsis` adapter for Earth
- compare drag-on / drag-off scenarios

### Days 9-10

- define run artifact schema
- write result loaders and serializers
- build first Dash app on top of saved simulation outputs

### Days 11-12

- add comparison views and richer diagnostics
- connect validation outputs to the app
- harden the visualization workflow for debugging

### Days 13-14

- polish Dash demo
- add animated 3D orbit view
- optionally prototype one stronger `pyvista` / `trame` scene
- finalize documentation and presentation material

## Final Recommendation

Use this stack for the first implementation:

- custom propagator: `numpy` + `scipy` + `astropy`
- Earth drag realism: `pymsis`
- reference truth / validation: `tudatpy`
- ML: `torch` + `pykan`
- optional PINN baseline: `DeepXDE`
- web app: `Dash` + `plotly`

This is the best balance between:

- scientific transparency
- implementation speed
- extensibility toward hybrid ML models
- a demo-ready web visualization

## Sources Reviewed

- SciPy `solve_ivp`: https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.integrate.solve_ivp.html
- Astropy solar-system ephemerides: https://docs.astropy.org/en/stable/coordinates/solarsystem.html
- Astropy `get_body_barycentric`: https://docs.astropy.org/en/latest/api/astropy.coordinates.get_body_barycentric.html
- poliastro perturbations: https://docs.poliastro.space/en/stable/autoapi/poliastro/core/perturbations/index.html
- TudatPy quickstart: https://docs.tudat.space/en/latest/getting-started/quickstart.html
- TudatPy perturbed satellite example: https://docs.tudat.space/en/latest/examples/tudatpy-examples/propagation/perturbed_satellite_orbit.html
- Tudat third-body accelerations: https://docs.tudat.space/en/stable/user-guide/state-propagation/propagation-setup/translational/third-body-acceleration.html
- pymsis docs: https://swxtrec.github.io/pymsis/
- pyKAN repository: https://github.com/KindXiaoming/pykan
- DeepXDE docs: https://deepxde.readthedocs.io/
- Dash graph docs: https://dash.plotly.com/dash-core-components/graph
- Dash callbacks: https://dash.plotly.com/basic-callbacks
- Plotly 3D scatter docs: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter3d.html
- Trame docs: https://trame.readthedocs.io/
- PyVista docs: https://docs.pyvista.org/index.html
