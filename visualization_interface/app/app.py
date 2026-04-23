from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from plotly.subplots import make_subplots

from launchers.simulation import launch_simulation_from_ui
from loaders.runs import list_run_files, load_run_file


RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/shared/results"))
PANEL_BG = "#fffaf2"
PAPER_BG = "#f6f1e8"

app = Dash(__name__)
app.title = "Orbital Run Viewer"


def build_run_options() -> list[dict[str, str]]:
    return [{"label": path.name, "value": str(path)} for path in list_run_files(RESULTS_DIR)]


def clear_run_files(results_dir: Path) -> int:
    deleted_count = 0
    for path in list_run_files(results_dir):
        path.unlink(missing_ok=True)
        deleted_count += 1
    return deleted_count


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        title=message,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANEL_BG,
        margin={"l": 60, "r": 20, "t": 56, "b": 48},
    )
    return figure


def vector_norms(vectors: list[list[float]]) -> list[float]:
    return [math.sqrt(sum(component * component for component in row)) for row in vectors]


def compute_fallback_derived_series(artifact: dict[str, Any]) -> dict[str, list[float]]:
    states = artifact.get("states_m_s", [])
    positions = [row[:3] for row in states]
    velocities = [row[3:] for row in states]
    central_body_parameters = artifact.get("parameters", {}).get("central_body", {})
    mu_m3_s2 = central_body_parameters.get("mu_m3_s2")
    body_radius_m = central_body_parameters.get("radius_m")

    radius_m = vector_norms(positions)
    speed_m_s = vector_norms(velocities)
    altitude_m = []
    if body_radius_m is not None:
        altitude_m = [radius - body_radius_m for radius in radius_m]

    specific_energy_j_kg: list[float] = []
    semimajor_axis_m: list[float] = []
    if mu_m3_s2 is not None:
        for radius, speed in zip(radius_m, speed_m_s, strict=False):
            energy = 0.5 * speed * speed - mu_m3_s2 / radius
            specific_energy_j_kg.append(energy)
            semimajor_axis_m.append(-mu_m3_s2 / (2.0 * energy))

    return {
        "radius_m": radius_m,
        "altitude_m": altitude_m,
        "speed_m_s": speed_m_s,
        "specific_energy_j_kg": specific_energy_j_kg,
        "semimajor_axis_m": semimajor_axis_m,
    }


def line_figure(
    times: list[float],
    values: list[float],
    *,
    title: str,
    yaxis_title: str,
    color: str,
) -> go.Figure:
    if not times or not values:
        return empty_figure(f"{title}: no data")

    figure = go.Figure(
        data=[
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                line={"width": 3, "color": color},
                name=title,
            )
        ]
    )
    figure.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Time [s]",
        yaxis_title=yaxis_title,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANEL_BG,
        margin={"l": 60, "r": 20, "t": 56, "b": 48},
    )
    return figure


def trajectory_figure(
    artifact_path: Path,
    states: list[list[float]],
    reference_vectors_m: dict[str, list[float]] | None = None,
) -> go.Figure:
    if not states:
        return empty_figure("3D Trajectory: no data")

    x = [row[0] for row in states]
    y = [row[1] for row in states]
    z = [row[2] for row in states]
    orbit_scale = max(
        math.sqrt(x_i * x_i + y_i * y_i + z_i * z_i)
        for x_i, y_i, z_i in zip(x, y, z, strict=False)
    )

    traces = [
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line={"width": 6, "color": "#0f766e"},
            name="trajectory",
        ),
        go.Scatter3d(
            x=[x[0]],
            y=[y[0]],
            z=[z[0]],
            mode="markers",
            marker={"size": 7, "color": "#b45309"},
            name="start",
        ),
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            marker={"size": 10, "color": "#1f2937"},
            name="central body",
        ),
    ]

    if reference_vectors_m:
        body_colors = {"sun": "#f59e0b", "moon": "#2563eb"}
        for body_name, vector in reference_vectors_m.items():
            norm = math.sqrt(sum(component * component for component in vector))
            if norm == 0.0:
                continue
            scaled = [component / norm * orbit_scale * 1.2 for component in vector]
            traces.append(
                go.Scatter3d(
                    x=[0.0, scaled[0]],
                    y=[0.0, scaled[1]],
                    z=[0.0, scaled[2]],
                    mode="lines+markers",
                    line={"width": 4, "color": body_colors.get(body_name, "#6b7280")},
                    marker={"size": 4, "color": body_colors.get(body_name, "#6b7280")},
                    name=f"{body_name} direction",
                )
            )

    figure = go.Figure(data=traces)
    figure.update_layout(
        template="plotly_white",
        title=f"3D Trajectory: {artifact_path.name}",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANEL_BG,
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
            "aspectmode": "data",
            "bgcolor": PANEL_BG,
        },
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
    )
    return figure


def exaggerated_trajectory_figure(
    artifact_path: Path,
    states: list[list[float]],
    times: list[float],
    radius_m: list[float],
    exaggeration_factor: float,
    reference_vectors_m: dict[str, list[float]] | None = None,
) -> go.Figure:
    if not states or not radius_m or not times:
        return empty_figure("3D Exaggerated Trajectory: no data")

    if len(states) != len(radius_m) or len(states) != len(times):
        return empty_figure("3D Exaggerated Trajectory: inconsistent data")

    base_radius_m = float(radius_m[0])
    residuals_m = [radius - base_radius_m for radius in radius_m]
    exaggeration_factor = max(float(exaggeration_factor), 0.0)

    deformed_positions: list[tuple[float, float, float]] = []
    for state, radius, residual in zip(states, radius_m, residuals_m, strict=False):
        if radius <= 0.0:
            deformed_positions.append((0.0, 0.0, 0.0))
            continue
        unit_vector = [component / radius for component in state[:3]]
        deformed_radius_m = base_radius_m + exaggeration_factor * residual
        deformed_positions.append(
            tuple(component * deformed_radius_m for component in unit_vector)
        )

    x = [position[0] for position in deformed_positions]
    y = [position[1] for position in deformed_positions]
    z = [position[2] for position in deformed_positions]
    orbit_scale = max(
        math.sqrt(x_i * x_i + y_i * y_i + z_i * z_i)
        for x_i, y_i, z_i in zip(x, y, z, strict=False)
    )

    traces = [
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line={"width": 5, "color": "rgba(15, 118, 110, 0.35)"},
            name="deformed path",
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={
                "size": 3.5,
                "color": times,
                "colorscale": "Viridis",
                "colorbar": {"title": "Time [s]"},
            },
            customdata=[[time_s] for time_s in times],
            hovertemplate=(
                "x=%{x:.3e} m<br>"
                "y=%{y:.3e} m<br>"
                "z=%{z:.3e} m<br>"
                "time=%{customdata[0]:.3f} s<extra></extra>"
            ),
            name="time-colored trajectory",
        ),
        go.Scatter3d(
            x=[x[0]],
            y=[y[0]],
            z=[z[0]],
            mode="markers",
            marker={"size": 7, "color": "#b45309"},
            name="start",
        ),
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            marker={"size": 10, "color": "#1f2937"},
            name="central body",
        ),
    ]

    if reference_vectors_m:
        body_colors = {"sun": "#f59e0b", "moon": "#2563eb"}
        for body_name, vector in reference_vectors_m.items():
            norm = math.sqrt(sum(component * component for component in vector))
            if norm == 0.0:
                continue
            scaled = [component / norm * orbit_scale * 1.2 for component in vector]
            traces.append(
                go.Scatter3d(
                    x=[0.0, scaled[0]],
                    y=[0.0, scaled[1]],
                    z=[0.0, scaled[2]],
                    mode="lines+markers",
                    line={"width": 4, "color": body_colors.get(body_name, "#6b7280")},
                    marker={"size": 4, "color": body_colors.get(body_name, "#6b7280")},
                    name=f"{body_name} direction",
                )
            )

    figure = go.Figure(data=traces)
    figure.update_layout(
        template="plotly_white",
        title=(
            f"3D Exaggerated Trajectory: {artifact_path.name}"
            f" | radial exaggeration x{exaggeration_factor:.1f}"
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANEL_BG,
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
            "aspectmode": "data",
            "bgcolor": PANEL_BG,
        },
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
    )
    return figure


def force_components_figure(
    times: list[float],
    accelerations_by_force: dict[str, list[list[float]]],
) -> go.Figure:
    if not accelerations_by_force:
        return empty_figure("Acceleration By Force: no data")

    perturbation_only = {
        force_name: vectors
        for force_name, vectors in accelerations_by_force.items()
        if force_name != "central_gravity"
    }
    if perturbation_only:
        accelerations_to_plot = perturbation_only
        title = "Perturbation Accelerations"
    else:
        accelerations_to_plot = accelerations_by_force
        title = "Acceleration By Force"

    figure = go.Figure()
    palette = {
        "central_gravity": "#0f766e",
        "j2": "#b45309",
        "drag": "#dc2626",
        "third_body_sun": "#f59e0b",
        "third_body_moon": "#2563eb",
        "solar_radiation_pressure": "#7c3aed",
    }

    for force_name, vectors in accelerations_to_plot.items():
        norms = [max(value, 1e-30) for value in vector_norms(vectors)]
        figure.add_trace(
            go.Scatter(
                x=times,
                y=norms,
                mode="lines",
                line={"width": 3, "color": palette.get(force_name, "#1f2937")},
                name=force_name,
            )
        )

    figure.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Time [s]",
        yaxis_title="Acceleration [m/s^2]",
        yaxis_type="log",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANEL_BG,
        margin={"l": 60, "r": 20, "t": 56, "b": 48},
    )
    return figure


def orbital_elements_figure(
    times: list[float],
    derived_series: dict[str, list[float]],
) -> go.Figure:
    subplot_titles = (
        "Semi-Major Axis [m]",
        "Eccentricity",
        "Inclination [deg]",
        "RAAN [deg]",
    )
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=subplot_titles,
    )

    traces = [
        ("semimajor_axis_m", "#0f766e"),
        ("eccentricity", "#7c3aed"),
        ("inclination_deg", "#b45309"),
        ("raan_deg", "#1f2937"),
    ]

    has_data = False
    for row_index, (series_name, color) in enumerate(traces, start=1):
        values = derived_series.get(series_name, [])
        if not values:
            continue
        has_data = True
        figure.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                line={"width": 2.5, "color": color},
                name=series_name,
                showlegend=False,
            ),
            row=row_index,
            col=1,
        )

    if not has_data:
        return empty_figure("Orbital Elements: no data")

    figure.update_layout(
        template="plotly_white",
        title="Orbital Elements",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANEL_BG,
        height=900,
        margin={"l": 60, "r": 20, "t": 72, "b": 48},
    )
    figure.update_xaxes(title_text="Time [s]", row=4, col=1)
    return figure


def refresh_run_selection(
    current_value: str | None,
) -> tuple[list[dict[str, str]], str | None, str]:
    options = build_run_options()
    values = {option["value"] for option in options}

    if current_value in values:
        selected_value = current_value
    elif options:
        selected_value = options[-1]["value"]
    else:
        selected_value = None

    status = (
        f"{len(options)} run artifact(s) available in the Docker volume mounted at "
        f"{RESULTS_DIR}"
    )
    return options, selected_value, status


app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="sidebar",
            children=[
                html.H1("Orbital Viewer"),
                html.P(
                    "Inspect saved runs or launch a new propagation directly from the dashboard.",
                    className="intro",
                ),
                html.Div(
                    className="sidebar-section",
                    children=[
                        html.Button(
                            "Refresh Runs",
                            id="refresh-button",
                            className="refresh-button",
                        ),
                        html.Button(
                            "Clear Saved Runs",
                            id="clear-runs-button",
                            className="clear-button",
                        ),
                        html.Div(id="results-status", className="results-status"),
                        dcc.Dropdown(
                            id="run-selector",
                            options=build_run_options(),
                            placeholder="Select a saved run artifact",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    className="sidebar-section",
                    children=[
                        html.H2("Launch Run"),
                        html.Div(
                            className="control-group",
                            children=[
                                html.Label("Run Name", className="control-label"),
                                dcc.Input(
                                    id="run-name-input",
                                    type="text",
                                    value="ui_run",
                                    className="control-input",
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-grid",
                            children=[
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Altitude [km]", className="control-label"),
                                        dcc.Input(
                                            id="altitude-input",
                                            type="number",
                                            value=500.0,
                                            className="control-input",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label(
                                            "Inclination [deg]",
                                            className="control-label",
                                        ),
                                        dcc.Input(
                                            id="inclination-input",
                                            type="number",
                                            value=28.5,
                                            className="control-input",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("RAAN [deg]", className="control-label"),
                                        dcc.Input(
                                            id="raan-input",
                                            type="number",
                                            value=0.0,
                                            className="control-input",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label(
                                            "True Anomaly [deg]",
                                            className="control-label",
                                        ),
                                        dcc.Input(
                                            id="true-anomaly-input",
                                            type="number",
                                            value=0.0,
                                            className="control-input",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label(
                                            "Start Epoch [UTC]",
                                            className="control-label",
                                        ),
                                        dcc.Input(
                                            id="start-epoch-input",
                                            type="text",
                                            value="2026-01-01T00:00:00Z",
                                            className="control-input",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Duration [s]", className="control-label"),
                                        dcc.Input(
                                            id="duration-input",
                                            type="number",
                                            value=5400.0,
                                            className="control-input",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Samples", className="control-label"),
                                        dcc.Input(
                                            id="sample-count-input",
                                            type="number",
                                            value=721,
                                            className="control-input",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Mass [kg]", className="control-label"),
                                        dcc.Input(
                                            id="mass-input",
                                            type="number",
                                            value=1000.0,
                                            className="control-input",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label(
                                            "Area [m^2]",
                                            className="control-label",
                                        ),
                                        dcc.Input(
                                            id="area-input",
                                            type="number",
                                            value=10.0,
                                            className="control-input",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Cd", className="control-label"),
                                        dcc.Input(
                                            id="drag-coefficient-input",
                                            type="number",
                                            value=2.2,
                                            className="control-input",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Cr", className="control-label"),
                                        dcc.Input(
                                            id="reflectivity-input",
                                            type="number",
                                            value=1.2,
                                            className="control-input",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-group",
                            children=[
                                html.Label("Integrator", className="control-label"),
                                dcc.Dropdown(
                                    id="integrator-backend-input",
                                    options=[
                                        {"label": "Auto", "value": "auto"},
                                        {"label": "SciPy", "value": "scipy"},
                                        {"label": "RK4", "value": "rk4"},
                                    ],
                                    value="auto",
                                    clearable=False,
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-group",
                            children=[
                                html.Label("Atmosphere Model", className="control-label"),
                                dcc.Dropdown(
                                    id="atmosphere-model-input",
                                    options=[
                                        {
                                            "label": "Piecewise Exponential",
                                            "value": "piecewise_exponential",
                                        },
                                        {"label": "pymsis", "value": "pymsis"},
                                    ],
                                    value="piecewise_exponential",
                                    clearable=False,
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-group",
                            children=[
                                html.Label("Force Models", className="control-label"),
                                dcc.Checklist(
                                    id="force-selection-input",
                                    options=[
                                        {"label": "Central Gravity", "value": "central_gravity", "disabled": True},
                                        {"label": "J2", "value": "j2"},
                                        {"label": "Drag", "value": "drag"},
                                        {"label": "SRP", "value": "solar_radiation_pressure"},
                                        {"label": "Sun Third-Body", "value": "third_body_sun"},
                                        {"label": "Moon Third-Body", "value": "third_body_moon"},
                                    ],
                                    value=["central_gravity"],
                                    className="force-checklist",
                                ),
                            ],
                        ),
                        html.Button(
                            "Run Propagation",
                            id="launch-button",
                            className="launch-button",
                        ),
                        html.Div(id="launch-status", className="launch-status"),
                    ],
                ),
                html.Div(id="run-summary", className="summary-grid"),
            ],
        ),
        html.Div(
            className="content",
            children=[
                html.Div(
                    className="orbit-views",
                    children=[
                        dcc.Graph(
                            id="trajectory-graph",
                            figure=empty_figure("No run selected"),
                        ),
                        html.Div(
                            className="orbit-view-panel",
                            children=[
                                html.Div(
                                    className="orbit-view-toolbar",
                                    children=[
                                        html.Label(
                                            "3D Exaggeration",
                                            className="control-label orbit-view-label",
                                            htmlFor="exaggeration-factor-input",
                                        ),
                                        dcc.Input(
                                            id="exaggeration-factor-input",
                                            type="number",
                                            value=5000.0,
                                            min=0.0,
                                            step=1.0,
                                            className="control-input orbit-view-input",
                                        ),
                                    ],
                                ),
                                dcc.Graph(
                                    id="residual-trajectory-graph",
                                    figure=empty_figure("No run selected"),
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="diagnostics-grid",
                    children=[
                        dcc.Graph(
                            id="acceleration-graph",
                            figure=empty_figure("No run selected"),
                        ),
                        dcc.Graph(
                            id="force-components-graph",
                            figure=empty_figure("No run selected"),
                        ),
                        dcc.Graph(id="altitude-graph", figure=empty_figure("No run selected")),
                        dcc.Graph(id="speed-graph", figure=empty_figure("No run selected")),
                        dcc.Graph(id="energy-graph", figure=empty_figure("No run selected")),
                        dcc.Graph(
                            id="orbital-elements-graph",
                            figure=empty_figure("No run selected"),
                        ),
                    ],
                ),
                html.H2("Metadata"),
                html.Pre(id="metadata-panel", className="metadata-panel"),
            ],
        ),
    ],
)


@app.callback(
    Output("run-selector", "options"),
    Output("run-selector", "value"),
    Output("results-status", "children"),
    Output("launch-status", "children"),
    Input("refresh-button", "n_clicks"),
    Input("clear-runs-button", "n_clicks"),
    Input("launch-button", "n_clicks"),
    State("run-selector", "value"),
    State("run-name-input", "value"),
    State("altitude-input", "value"),
    State("inclination-input", "value"),
    State("raan-input", "value"),
    State("true-anomaly-input", "value"),
    State("start-epoch-input", "value"),
    State("duration-input", "value"),
    State("sample-count-input", "value"),
    State("mass-input", "value"),
    State("area-input", "value"),
    State("drag-coefficient-input", "value"),
    State("reflectivity-input", "value"),
    State("integrator-backend-input", "value"),
    State("atmosphere-model-input", "value"),
    State("force-selection-input", "value"),
)
def manage_runs(
    _refresh_clicks: int | None,
    _clear_clicks: int | None,
    _launch_clicks: int | None,
    current_value: str | None,
    run_name: str | None,
    altitude_km: float | None,
    inclination_deg: float | None,
    raan_deg: float | None,
    true_anomaly_deg: float | None,
    start_epoch_utc: str | None,
    duration_s: float | None,
    sample_count: int | None,
    mass_kg: float | None,
    area_m2: float | None,
    drag_coefficient: float | None,
    reflectivity_coefficient: float | None,
    integrator_backend: str | None,
    atmosphere_model: str | None,
    selected_forces: list[str] | None,
) -> tuple[list[dict[str, str]], str | None, str, str]:
    trigger = ctx.triggered_id
    launch_message = "Ready to run new propagations from the dashboard."

    if trigger == "clear-runs-button":
        deleted_count = clear_run_files(RESULTS_DIR)
        current_value = None
        options, selected_value, status = refresh_run_selection(current_value)
        launch_message = f"Removed {deleted_count} saved run artifact(s) from /shared/results."
        return options, selected_value, status, launch_message

    if trigger == "launch-button":
        try:
            if altitude_km is None or duration_s is None or sample_count is None:
                raise ValueError("Altitude, duration, and sample count are required.")
            output_path = launch_simulation_from_ui(
                results_dir=RESULTS_DIR,
                run_name=run_name or "ui_run",
                altitude_km=float(altitude_km),
                inclination_deg=float(inclination_deg or 0.0),
                raan_deg=float(raan_deg or 0.0),
                true_anomaly_deg=float(true_anomaly_deg or 0.0),
                start_epoch_utc=start_epoch_utc or "2026-01-01T00:00:00Z",
                duration_s=float(duration_s),
                sample_count=int(sample_count),
                integrator_backend=integrator_backend or "auto",
                mass_kg=float(mass_kg or 1000.0),
                cross_section_area_m2=float(area_m2 or 10.0),
                drag_coefficient=float(drag_coefficient or 2.2),
                reflectivity_coefficient=float(reflectivity_coefficient or 1.2),
                atmosphere_model=atmosphere_model or "piecewise_exponential",
                enable_j2=bool(selected_forces and "j2" in selected_forces),
                enable_drag=bool(selected_forces and "drag" in selected_forces),
                enable_solar_radiation_pressure=bool(
                    selected_forces and "solar_radiation_pressure" in selected_forces
                ),
                enable_third_body_sun=bool(
                    selected_forces and "third_body_sun" in selected_forces
                ),
                enable_third_body_moon=bool(
                    selected_forces and "third_body_moon" in selected_forces
                ),
            )
            current_value = str(output_path)
            launch_message = f"Launched propagation and wrote {output_path.name}"
        except Exception as exc:  # pragma: no cover - runtime UI path
            options, selected_value, status = refresh_run_selection(current_value)
            return options, selected_value, status, f"Launch failed: {exc}"

    options, selected_value, status = refresh_run_selection(current_value)
    return options, selected_value, status, launch_message


@app.callback(
    Output("trajectory-graph", "figure"),
    Output("residual-trajectory-graph", "figure"),
    Output("acceleration-graph", "figure"),
    Output("force-components-graph", "figure"),
    Output("altitude-graph", "figure"),
    Output("speed-graph", "figure"),
    Output("energy-graph", "figure"),
    Output("orbital-elements-graph", "figure"),
    Output("metadata-panel", "children"),
    Output("run-summary", "children"),
    Input("run-selector", "value"),
    Input("exaggeration-factor-input", "value"),
    State("run-selector", "options"),
)
def update_view(
    selected_run: str | None,
    exaggeration_factor: float | None,
    run_options: list[dict[str, str]] | None,
):
    if not selected_run:
        if run_options:
            hint = "Select one of the discovered artifacts from the sidebar."
        else:
            hint = "Run a simulation first, then click Refresh Runs."
        return (
            empty_figure("No run selected"),
            empty_figure("No run selected"),
            empty_figure("No run selected"),
            empty_figure("No run selected"),
            empty_figure("No run selected"),
            empty_figure("No run selected"),
            empty_figure("No run selected"),
            empty_figure("No run selected"),
            hint,
            html.Div("No artifact loaded.", className="summary-item"),
        )

    artifact_path = Path(selected_run)
    artifact = load_run_file(artifact_path)
    states = artifact["states_m_s"]
    times = artifact["times_s"]
    accelerations = artifact["accelerations_total_m_s2"]
    acceleration_norms = vector_norms(accelerations)

    derived_series = compute_fallback_derived_series(artifact)
    derived_series.update(artifact.get("derived_series", {}))

    altitude_m = derived_series.get("altitude_m", [])
    radius_m = derived_series.get("radius_m", [])
    speed_m_s = derived_series.get("speed_m_s", [])
    specific_energy_j_kg = derived_series.get("specific_energy_j_kg", [])
    raan_deg = derived_series.get("raan_deg", [])
    eccentricity = derived_series.get("eccentricity", [])
    reference_vectors_m = artifact.get("metadata", {}).get("reference_vectors_m", {})

    trajectory = trajectory_figure(artifact_path, states, reference_vectors_m)
    exaggerated_trajectory = exaggerated_trajectory_figure(
        artifact_path,
        states,
        times,
        radius_m,
        exaggeration_factor if exaggeration_factor is not None else 5000.0,
        reference_vectors_m,
    )
    acceleration = line_figure(
        times,
        acceleration_norms,
        title="Acceleration Norm",
        yaxis_title="Acceleration [m/s^2]",
        color="#7c3aed",
    )
    force_components = force_components_figure(
        times,
        artifact.get("accelerations_by_force_m_s2", {}),
    )
    altitude = line_figure(
        times,
        altitude_m,
        title="Altitude",
        yaxis_title="Altitude [m]",
        color="#0f766e",
    )
    speed = line_figure(
        times,
        speed_m_s,
        title="Speed",
        yaxis_title="Speed [m/s]",
        color="#b45309",
    )
    energy = line_figure(
        times,
        specific_energy_j_kg,
        title="Specific Energy",
        yaxis_title="Energy [J/kg]",
        color="#1f2937",
    )
    orbital_elements = orbital_elements_figure(times, derived_series)

    summary_data = artifact.get("summary", {})
    summary = [
        html.Div(
            [
                html.Div("Run", className="summary-label"),
                html.Div(artifact.get("run_name", artifact_path.stem), className="summary-value"),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Producer", className="summary-label"),
                html.Div(artifact.get("producer", "unknown"), className="summary-value"),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Central Body", className="summary-label"),
                html.Div(artifact.get("central_body", "unknown"), className="summary-value"),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Epoch [UTC]", className="summary-label"),
                html.Div(
                    artifact.get("parameters", {})
                    .get("propagation", {})
                    .get("start_epoch_utc", "unknown"),
                    className="summary-value",
                ),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Forces", className="summary-label"),
                html.Div(", ".join(artifact.get("enabled_forces", [])), className="summary-value"),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Samples", className="summary-label"),
                html.Div(str(len(states)), className="summary-value"),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Duration [s]", className="summary-label"),
                html.Div(
                    f"{summary_data.get('duration_s', times[-1] if times else 0.0):.3f}",
                    className="summary-value",
                ),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Min Altitude [m]", className="summary-label"),
                html.Div(
                    f"{summary_data.get('min_altitude_m', min(altitude_m) if altitude_m else 0.0):.3f}",
                    className="summary-value",
                ),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Max Altitude [m]", className="summary-label"),
                html.Div(
                    f"{summary_data.get('max_altitude_m', max(altitude_m) if altitude_m else 0.0):.3f}",
                    className="summary-value",
                ),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Energy Span [J/kg]", className="summary-label"),
                html.Div(
                    f"{summary_data.get('specific_energy_span_j_kg', 0.0):.6f}",
                    className="summary-value",
                ),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Final RAAN [deg]", className="summary-label"),
                html.Div(
                    f"{raan_deg[-1] if raan_deg else 0.0:.6f}",
                    className="summary-value",
                ),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Ephemeris", className="summary-label"),
                html.Div(
                    artifact.get("metadata", {}).get("ephemeris_source", "unknown"),
                    className="summary-value",
                ),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Max Eccentricity", className="summary-label"),
                html.Div(
                    f"{max(eccentricity) if eccentricity else 0.0:.8f}",
                    className="summary-value",
                ),
            ],
            className="summary-item",
        ),
    ]

    return (
        trajectory,
        exaggerated_trajectory,
        acceleration,
        force_components,
        altitude,
        speed,
        energy,
        orbital_elements,
        json.dumps(artifact, indent=2),
        summary,
    )


if __name__ == "__main__":
    host = os.getenv("DASH_HOST", "0.0.0.0")
    port = int(os.getenv("DASH_PORT", "8050"))
    app.run(host=host, port=port, debug=False)
