# Orbital Propagator And Visualization Interface

This repository contains two connected parts:

- `orbital_propagator`: a Python orbital propagation package for Earth-centered runs with configurable perturbations.
- `visualization_interface`: a Dash web app for launching runs, browsing saved results, and visualizing trajectories and diagnostics.

Saved run artifacts are written as JSON files into [results](./results), which is bind-mounted into both containers at `/shared/results`.

## What's Inside

### `orbital_propagator`

Main areas:

- `bodies/`: central-body definitions such as Earth constants
- `forces/`: force models
- `ephemerides/`: Sun and Moon position providers
- `propagation/`: dynamics, integrators, orbital elements, and simulation runner
- `io/`: JSON artifact generation and persistence
- `cli.py`: command-line entrypoint

Current force models:

- central gravity
- `J2`, Earth's oblateness
- atmospheric drag with either `piecewise_exponential` or `pymsis`
- Sun third-body gravity
- Moon third-body gravity
- solar radiation pressure

### `visualization_interface`

Main areas:

- `app.py`: Dash app entrypoint and callbacks
- `loaders/`: loading saved run artifacts
- `launchers/`: launching propagations from the UI
- `assets/`: CSS styling

Current UI features:

- launch a new propagation from the browser
- load saved JSON run artifacts
- 3D trajectory view
- exaggerated 3D trajectory view
- acceleration, altitude, speed, energy, and orbital-element plots
- clear saved run artifacts from the app

## Run With Docker

Build once:

```bash
docker compose build
```

### Start The Web App

```bash
docker compose up visualization_interface
```

Then open `http://localhost:8050`.

### Run A Propagation From The CLI

Example:

```bash
docker compose run --rm orbital_propagator \
  --output /shared/results/j2_drag_demo.json \
  --run-name j2_drag_demo \
  --start-epoch-utc 2026-04-23T12:00:00Z \
  --altitude-km 500 \
  --inclination-deg 51.6 \
  --duration-s 5400 \
  --sample-count 721 \
  --enable-j2 \
  --enable-drag \
  --atmosphere-model piecewise_exponential
```

The result will appear in [results](./results).

### Run A Propagation From The Web Interface

1. Start the app with `docker compose up visualization_interface`
2. Open `http://localhost:8050`
3. Fill in the run parameters in the sidebar
4. Click `Run Propagation`
5. Inspect the generated run and plots

The generated JSON artifact is also saved in [results](./results).

## Notes

- The app and the CLI read and write the same artifact format.
- The `results/` folder is now part of the repository workspace, so you can inspect or delete JSON outputs directly.
- If you use `pymsis`, the container must have the required dependency data available at runtime.

## Interface Preview

<img src="./utils/layout_interface_v3.png" alt="Visualization interface layout" width="100%" />
