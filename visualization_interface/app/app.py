from __future__ import annotations

import json
import math
import os
from bisect import bisect_left
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html
from launchers.simulation import estimate_sampling_parameters, launch_simulation_from_ui
from loaders.runs import list_run_files, load_run_file
from plotly.subplots import make_subplots

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/shared/results"))
PANEL_BG = "#fffaf2"
PAPER_BG = "#f6f1e8"
TITLE_FONT = {"family": "Space Grotesk, Segoe UI, sans-serif", "size": 22, "color": "#2b2118"}

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
        title={"text": f"<b>{message}</b>", "x": 0.02, "font": TITLE_FONT},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANEL_BG,
        margin={"l": 60, "r": 20, "t": 56, "b": 48},
    )
    return figure


def vector_norms(vectors: list[list[float]]) -> list[float]:
    return [math.sqrt(sum(component * component for component in row)) for row in vectors]


def build_start_epoch_utc(start_date_utc: str | None, start_time_utc: str | None) -> str:
    if not start_date_utc:
        raise ValueError("A UTC start date is required.")

    date_value = start_date_utc.strip()
    time_value = (start_time_utc or "00:00:00").strip() or "00:00:00"
    time_parts = time_value.split(":")
    if len(time_parts) != 3 or any(not part.isdigit() for part in time_parts):
        raise ValueError("UTC time must use the HH:MM:SS format.")

    hours, minutes, seconds = (int(part) for part in time_parts)
    if hours not in range(24) or minutes not in range(60) or seconds not in range(60):
        raise ValueError("UTC time must be a valid 24-hour clock value.")

    return f"{date_value}T{hours:02d}:{minutes:02d}:{seconds:02d}Z"


def describe_moon_geometry(start_epoch_utc: str) -> str:
    try:
        from orbital_propagator.ephemerides.provider import body_position_m
    except ModuleNotFoundError:
        return "Moon geometry hint unavailable: orbital_propagator is not importable."

    sun_position_m = body_position_m("sun", start_epoch_utc, 0.0)
    moon_position_m = body_position_m("moon", start_epoch_utc, 0.0)

    sun_norm = float(math.sqrt(float(sun_position_m @ sun_position_m)))
    moon_norm = float(math.sqrt(float(moon_position_m @ moon_position_m)))
    if sun_norm == 0.0 or moon_norm == 0.0:
        return "Moon geometry hint unavailable for the selected epoch."

    cosine = float((sun_position_m @ moon_position_m) / (sun_norm * moon_norm))
    separation_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    illumination_fraction = 0.5 * (1.0 - math.cos(math.radians(separation_deg)))

    if separation_deg < 30.0:
        phase_name = "Near new moon"
        geometry_note = "Moon roughly aligned with the Sun."
    elif separation_deg < 75.0:
        phase_name = "Crescent geometry"
        geometry_note = "Moon partly offset from the Sun direction."
    elif separation_deg < 120.0:
        phase_name = "Quarter geometry"
        geometry_note = "Moon roughly perpendicular to the Sun direction."
    elif separation_deg < 165.0:
        phase_name = "Gibbous geometry"
        geometry_note = "Moon trending toward opposition with the Sun."
    else:
        phase_name = "Near full moon"
        geometry_note = "Moon nearly opposite the Sun."

    return (
        f"{phase_name} | Sun-Moon separation {separation_deg:.1f} deg | "
        f"illuminated fraction {illumination_fraction * 100.0:.0f}% | {geometry_note}"
    )


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
        title={"text": f"<b>{title}</b>", "x": 0.02, "font": TITLE_FONT},
        xaxis_title="Time [s]",
        yaxis_title=yaxis_title,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANEL_BG,
        margin={"l": 60, "r": 20, "t": 56, "b": 48},
    )
    return figure


def animation_sample_indices(sample_count: int, max_frames: int = 240) -> list[int]:
    if sample_count <= 0:
        return []
    if sample_count <= max_frames:
        return list(range(sample_count))

    step = max(1, math.ceil((sample_count - 1) / (max_frames - 1)))
    indices = list(range(0, sample_count, step))
    if indices[-1] != sample_count - 1:
        indices.append(sample_count - 1)
    return indices


def visual_sample_indices(sample_count: int, max_points: int = 2_000) -> list[int]:
    """Downsample dense trajectories for browser-side 3D rendering.

    This does not change the simulation data, slider sampling, diagnostics,
    or readouts. It only limits the number of points sent to the Plotly 3D
    line/marker traces.
    """
    if sample_count <= 0:
        return []
    if sample_count <= max_points:
        return list(range(sample_count))

    step = max(1, math.ceil((sample_count - 1) / (max_points - 1)))
    indices = list(range(0, sample_count, step))
    if indices[-1] != sample_count - 1:
        indices.append(sample_count - 1)
    return indices


def format_elapsed_time(seconds: float) -> str:
    total_seconds = max(float(seconds), 0.0)
    if total_seconds < 3600.0:
        return f"{total_seconds / 60.0:.1f} min"
    if total_seconds < 86_400.0:
        return f"{total_seconds / 3600.0:.2f} h"
    return f"{total_seconds / 86_400.0:.2f} d"


def nice_step(raw_step: float) -> float:
    if raw_step <= 0.0:
        return 1.0

    exponent = math.floor(math.log10(raw_step))
    scale = 10.0**exponent
    fraction = raw_step / scale

    for nice_fraction in (1.0, 1.5, 2.0, 2.5, 5.0, 10.0):
        if fraction <= nice_fraction:
            return nice_fraction * scale

    return 10.0 * scale


def nice_tick_values(
    max_value: float,
    *,
    max_labels: int = 7,
    include_zero: bool = True,
) -> list[float]:
    if max_value <= 0.0:
        return [0.0] if include_zero else []

    raw_step = max_value / max(max_labels - 1, 1)
    step = nice_step(raw_step)
    start = 0.0 if include_zero else step

    values: list[float] = []
    current = start
    tolerance = step * 1e-9
    while current <= max_value + tolerance:
        values.append(round(current, 10))
        current += step

    return values


def format_tick_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    if abs(value * 2.0 - round(value * 2.0)) < 1e-9:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def closest_slider_frame_for_elapsed_time(
    elapsed_time_s: float,
    times: list[float],
    sampled_indices: list[int],
) -> int:
    sampled_times = [times[index] for index in sampled_indices]
    insertion_index = bisect_left(sampled_times, elapsed_time_s)

    if insertion_index <= 0:
        return 0
    if insertion_index >= len(sampled_times):
        return len(sampled_times) - 1

    before = sampled_times[insertion_index - 1]
    after = sampled_times[insertion_index]
    if abs(elapsed_time_s - before) <= abs(after - elapsed_time_s):
        return insertion_index - 1
    return insertion_index


def elapsed_time_unit(total_duration_s: float) -> tuple[float, str]:
    if total_duration_s < 2.0 * 3600.0:
        return 60.0, "min"
    if total_duration_s < 2.0 * 86_400.0:
        return 3600.0, "h"
    return 86_400.0, "d"


def slider_elapsed_time_marks_for_indices(
    times: list[float],
    sampled_indices: list[int],
    max_labels: int = 7,
) -> dict[int, str]:
    if not sampled_indices:
        return {}

    total_duration_s = times[sampled_indices[-1]]
    unit_seconds, unit_label = elapsed_time_unit(total_duration_s)
    total_duration_units = total_duration_s / unit_seconds

    marks: dict[int, str] = {}
    for tick_value in nice_tick_values(total_duration_units, max_labels=max_labels):
        elapsed_time_s = tick_value * unit_seconds
        frame_index = closest_slider_frame_for_elapsed_time(
            elapsed_time_s,
            times,
            sampled_indices,
        )
        marks[frame_index] = f"{format_tick_number(tick_value)} {unit_label}"

    return marks


def orbit_scale_marks_for_indices(
    times: list[float],
    sampled_indices: list[int],
    orbit_period_s: float | None,
) -> list[html.Div]:
    if not sampled_indices or not orbit_period_s or orbit_period_s <= 0.0:
        return []

    total_duration_s = times[sampled_indices[-1]]
    total_orbits = total_duration_s / orbit_period_s
    max_frame_index = max(len(sampled_indices) - 1, 1)

    labels: list[html.Div] = []

    # Show 0, 1, 2, 3, ... up to the last completed orbit.
    for orbit_number in range(0, math.floor(total_orbits) + 1):
        elapsed_time_s = orbit_number * orbit_period_s

        frame_index = closest_slider_frame_for_elapsed_time(
            elapsed_time_s,
            times,
            sampled_indices,
        )

        percent = 100.0 * frame_index / max_frame_index

        labels.append(
            html.Div(
                str(orbit_number) + " orb",
                className="trajectory-orbit-scale-label",
                style={"left": f"{percent:.6f}%"},
            )
        )

    return labels


def estimate_initial_orbit_period_s(
    *,
    state_m_s: list[float],
    mu_m3_s2: float,
) -> float | None:
    position = np.array(state_m_s[:3], dtype=float)
    velocity = np.array(state_m_s[3:], dtype=float)
    radius = float(np.linalg.norm(position))
    speed = float(np.linalg.norm(velocity))
    if radius <= 0.0 or mu_m3_s2 <= 0.0:
        return None

    specific_energy = 0.5 * speed * speed - mu_m3_s2 / radius
    if specific_energy >= 0.0:
        return None

    semimajor_axis_m = -mu_m3_s2 / (2.0 * specific_energy)
    return 2.0 * math.pi * math.sqrt(semimajor_axis_m**3 / mu_m3_s2)


def earth_surface_trace(
    *,
    central_body_name: str,
    equatorial_radius_km: float,
    show_earth: bool,
    j2_enabled: bool,
) -> go.Surface | None:
    if not show_earth:
        return None

    if equatorial_radius_km <= 0.0:
        return None

    polar_radius_km = equatorial_radius_km
    if central_body_name.lower() == "earth" and j2_enabled:
        polar_radius_km = 6_356_752.314245 / 1_000.0

    lon = np.linspace(0.0, 2.0 * np.pi, 48)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 24)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    x = equatorial_radius_km * np.cos(lat_grid) * np.cos(lon_grid)
    y = equatorial_radius_km * np.cos(lat_grid) * np.sin(lon_grid)
    z = polar_radius_km * np.sin(lat_grid)

    return go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale=[
            [0.0, "#9ac7d8"],
            [0.45, "#7aa2b8"],
            [0.5, "#6d8f55"],
            [1.0, "#b9c6cf"],
        ],
        showscale=False,
        opacity=0.75,
        lighting={"ambient": 0.8, "diffuse": 0.4, "specular": 0.05, "roughness": 0.9},
        hoverinfo="skip",
        name=central_body_name,
    )


def earth_equator_trace(
    *,
    central_body_name: str,
    equatorial_radius_km: float,
    show_earth: bool,
) -> go.Scatter3d | None:
    if not show_earth or central_body_name.lower() != "earth":
        return None

    if equatorial_radius_km <= 0.0:
        return None

    longitude = np.linspace(0.0, 2.0 * np.pi, 181)
    x = equatorial_radius_km * np.cos(longitude)
    y = equatorial_radius_km * np.sin(longitude)
    z = np.zeros_like(longitude)

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"width": 6, "color": "#f8fafc"},
        name="Earth equator",
        hoverinfo="skip",
    )


def trajectory_figure(
    artifact_path: Path,
    states: list[list[float]],
    times: list[float],
    frame_index: int,
    exaggeration_factor: float,
    central_body_name: str,
    central_body_mu_m3_s2: float,
    central_body_radius_m: float,
    j2_enabled: bool,
    show_earth: bool,
    reference_vectors_m: dict[str, list[float]] | None = None,
) -> go.Figure:
    if not states or not times:
        return empty_figure("3D Animated Trajectory: no data")
    if len(states) != len(times):
        return empty_figure("3D Animated Trajectory: inconsistent data")

    radius_m = [math.sqrt(row[0] ** 2 + row[1] ** 2 + row[2] ** 2) for row in states]
    base_radius_m = float(radius_m[0])
    residuals_m = [radius - base_radius_m for radius in radius_m]
    safe_exaggeration = max(float(exaggeration_factor), 0.0)

    display_positions: list[tuple[float, float, float]] = []
    for state, radius, residual in zip(states, radius_m, residuals_m, strict=False):
        if radius <= 0.0:
            display_positions.append((0.0, 0.0, 0.0))
            continue
        unit_vector = [component / radius for component in state[:3]]
        display_radius_m = base_radius_m + safe_exaggeration * residual
        display_positions.append(
            tuple(component * display_radius_m for component in unit_vector)
        )

    x = [position[0] / 1_000.0 for position in display_positions]
    y = [position[1] / 1_000.0 for position in display_positions]
    z = [position[2] / 1_000.0 for position in display_positions]
    orbit_scale = max(
        math.sqrt(x_i * x_i + y_i * y_i + z_i * z_i)
        for x_i, y_i, z_i in zip(x, y, z, strict=False)
    )
    sampled_indices = animation_sample_indices(len(states))
    if not sampled_indices:
        return empty_figure("3D Animated Trajectory: no data")
    clamped_frame_index = max(0, min(int(frame_index), len(sampled_indices) - 1))
    state_index = sampled_indices[clamped_frame_index]

    visual_indices = visual_sample_indices(len(states), max_points=2_000)
    x_visual = [x[index] for index in visual_indices]
    y_visual = [y[index] for index in visual_indices]
    z_visual = [z[index] for index in visual_indices]
    times_visual = [times[index] for index in visual_indices]

    traces = [
        go.Scatter3d(
            x=x_visual,
            y=y_visual,
            z=z_visual,
            mode="lines",
            line={"width": 5, "color": "rgba(15, 118, 110, 0.35)"},
            name="trajectory",
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=x_visual,
            y=y_visual,
            z=z_visual,
            mode="markers",
            marker={
                "size": 2.8,
                "color": times_visual,
                "colorscale": "Viridis",
                "colorbar": {"title": "Time [s]"},
            },
            customdata=[[time_s] for time_s in times_visual],
            hovertemplate=(
                "x=%{x:.3e} m<br>"
                "y=%{y:.3e} m<br>"
                "z=%{z:.3e} m<br>"
                "time=%{customdata[0]:.3f} s<extra></extra>"
            ),
            name="time-colored trajectory",
        ),
        go.Scatter3d(
            x=[x[state_index]],
            y=[y[state_index]],
            z=[z[state_index]],
            mode="markers",
            marker={"size": 7, "color": "#ef4444", "symbol": "diamond"},
            name="satellite",
        ),
        go.Scatter3d(
            x=[x[0]],
            y=[y[0]],
            z=[z[0]],
            mode="markers",
            marker={"size": 6, "color": "#b45309"},
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

    surface_trace = earth_surface_trace(
        central_body_name=central_body_name,
        equatorial_radius_km=central_body_radius_m / 1_000.0,
        show_earth=show_earth,
        j2_enabled=j2_enabled,
    )
    if surface_trace is not None:
        traces.append(surface_trace)
    equator_trace = earth_equator_trace(
        central_body_name=central_body_name,
        equatorial_radius_km=central_body_radius_m / 1_000.0,
        show_earth=show_earth,
    )
    if equator_trace is not None:
        traces.append(equator_trace)

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
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANEL_BG,
        height=680,
        uirevision=artifact_path.name,
        margin={"l": 0, "r": 0, "t": 16, "b": 0},
        scene={
            "xaxis_title": "x [km]",
            "yaxis_title": "y [km]",
            "zaxis_title": "z [km]",
            "aspectmode": "data",
            "bgcolor": PANEL_BG,
            "uirevision": artifact_path.name,
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
        title={"text": f"<b>{title}</b>", "x": 0.02, "font": TITLE_FONT},
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
        title={"text": "<b>Orbital Elements</b>", "x": 0.02, "font": TITLE_FONT},
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
        dcc.Interval(id="animation-interval", interval=140, n_intervals=0, disabled=True),
        dcc.Store(id="run-artifact-store"),
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
                                        html.Label("Start Date [UTC]", className="control-label"),
                                        dcc.DatePickerSingle(
                                            id="start-date-input",
                                            date=date(2026, 1, 1).isoformat(),
                                            display_format="YYYY-MM-DD",
                                            className="control-date-picker",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Start Time [UTC]", className="control-label"),
                                        dcc.Input(
                                            id="start-time-input",
                                            type="text",
                                            value="00:00:00",
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
                                        html.Label("Samples / Orbit", className="control-label"),
                                        dcc.Input(
                                            id="samples-per-orbit-input",
                                            type="number",
                                            value=180,
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
                        html.Div(id="moon-phase-hint", className="helper-panel"),
                        #html.Div(id="sampling-estimate", className="helper-panel", style={"display": False}),
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
                        html.Div(
                            className="orbit-view-panel",
                            children=[
                                html.Div(
                                    className="orbit-panel-header",
                                    children=[
                                        html.H2("3D Orbit Animation", className="orbit-panel-title"),
                                        html.P(
                                            "Time-colored trajectory, animated satellite motion, "
                                            "and optional main-body rendering.",
                                            className="orbit-panel-subtitle",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="orbit-view-layout",
                                    children=[
                                        html.Div(
                                            className="orbit-controls-panel",
                                            children=[
                                                html.Div(
                                                    className="orbit-toolbar-group",
                                                    children=[
                                                        html.Label(
                                                            "Animation",
                                                            className="control-label orbit-view-label",
                                                        ),
                                                        html.Div(
                                                            className="orbit-button-row",
                                                            children=[
                                                                html.Button(
                                                                    "Play",
                                                                    id="play-button",
                                                                    className="orbit-action-button",
                                                                ),
                                                                html.Button(
                                                                    "Pause",
                                                                    id="pause-button",
                                                                    className="orbit-action-button orbit-action-button-active",
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="orbit-toolbar-group",
                                                    children=[
                                                        html.Label(
                                                            "Playback Speed",
                                                            className="control-label orbit-view-label",
                                                            htmlFor="animation-speed-input",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="animation-speed-input",
                                                            options=[
                                                                {"label": "0.25x", "value": 0.25},
                                                                {"label": "0.5x", "value": 0.5},
                                                                {"label": "1x", "value": 1.0},
                                                                {"label": "2x", "value": 2.0},
                                                                {"label": "5x", "value": 5.0},
                                                            ],
                                                            value=1.0,
                                                            clearable=False,
                                                            className="orbit-view-dropdown",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="orbit-toolbar-group",
                                                    children=[
                                                        html.Label(
                                                            "3D Exaggeration",
                                                            className="control-label orbit-view-label",
                                                            htmlFor="exaggeration-factor-input",
                                                        ),
                                                        dcc.Input(
                                                            id="exaggeration-factor-input",
                                                            type="number",
                                                            value=1.0,
                                                            min=0.0,
                                                            step=0.5,
                                                            className="control-input orbit-view-input",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="orbit-toolbar-group orbit-toolbar-check",
                                                    children=[
                                                        html.Label(
                                                            "Main Body",
                                                            className="control-label orbit-view-label",
                                                        ),
                                                        dcc.Checklist(
                                                            id="show-earth-input",
                                                            options=[
                                                                {"label": "Show Main Body", "value": "show_earth"},
                                                            ],
                                                            value=[],
                                                            className="orbit-view-checklist",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="orbit-readout-panel",
                                                    children=[
                                                        html.Label(
                                                            "Current Frame",
                                                            className="control-label orbit-view-label",
                                                        ),
                                                        html.Div(
                                                            id="current-frame-readout",
                                                            className="orbit-readout-grid",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="orbit-readout-panel",
                                                    children=[
                                                        html.Label(
                                                            "Current State",
                                                            className="control-label orbit-view-label",
                                                        ),
                                                        html.Div(
                                                            id="current-orbital-quantities",
                                                            className="orbit-readout-grid",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(
                                            id="trajectory-graph",
                                            figure=empty_figure("No run selected"),
                                            className="orbit-graph",
                                        ),
                                        html.Div(
                                            className="trajectory-time-slider-panel",
                                            children=[
                                                html.Div(
                                                    id="trajectory-orbit-scale",
                                                    className="trajectory-orbit-scale",
                                                    children=[],
                                                ),
                                                dcc.Slider(
                                                    id="trajectory-time-slider",
                                                    min=0,
                                                    max=0,
                                                    step=1,
                                                    value=0,
                                                    marks={},
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                    className="trajectory-time-slider",
                                                ),
                                            ],
                                        ),
                                    ],
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
    Output("moon-phase-hint", "children"),
    Input("start-date-input", "date"),
    Input("start-time-input", "value"),
)
def update_moon_phase_hint(
    start_date_utc: str | None,
    start_time_utc: str | None,
) -> str:
    try:
        start_epoch_utc = build_start_epoch_utc(start_date_utc, start_time_utc)
    except ValueError as exc:
        return f"Moon geometry hint: {exc}"

    return describe_moon_geometry(start_epoch_utc)


# @app.callback(
#     Output("sampling-estimate", "children"),
#     Input("altitude-input", "value"),
#     Input("duration-input", "value"),
#     Input("samples-per-orbit-input", "value"),
# )
# def update_sampling_estimate(
#     altitude_km: float | None,
#     duration_s: float | None,
#     samples_per_orbit: int | None,
# ) -> str:
#     if altitude_km is None or duration_s is None or samples_per_orbit is None:
#         return "Sampling estimate unavailable."

#     try:
#         sampling = estimate_sampling_parameters(
#             altitude_km=float(altitude_km),
#             duration_s=float(duration_s),
#             samples_per_orbit=int(samples_per_orbit),
#         )
#     except Exception as exc:  # pragma: no cover - runtime UI path
#         return f"Sampling estimate unavailable: {exc}"

#     return (
#         f"Estimated total samples: {int(sampling['sample_count'])} across about "
#         f"{sampling['orbit_count']:.2f} orbit(s). Initial period: "
#         f"{sampling['orbit_period_s'] / 60.0:.1f} min."
#     )


@app.callback(
    Output("animation-interval", "disabled"),
    Output("animation-interval", "interval"),
    Output("play-button", "className"),
    Output("pause-button", "className"),
    Input("play-button", "n_clicks"),
    Input("pause-button", "n_clicks"),
    Input("animation-speed-input", "value"),
    Input("run-selector", "value"),
    State("animation-interval", "disabled"),
)
def update_animation_controls(
    _play_clicks: int | None,
    _pause_clicks: int | None,
    animation_speed: float | None,
    selected_run: str | None,
    interval_disabled: bool | None,
) -> tuple[bool, int, str, str]:
    trigger = ctx.triggered_id
    safe_speed = max(float(animation_speed or 1.0), 0.1)
    interval_ms = max(20, int(round(140.0 / safe_speed)))
    disabled = bool(interval_disabled)

    if trigger == "play-button" and selected_run:
        disabled = False
    elif trigger in {"pause-button", "run-selector"}:
        disabled = True

    play_class = "orbit-action-button"
    pause_class = "orbit-action-button"
    if disabled:
        pause_class += " orbit-action-button-active"
    else:
        play_class += " orbit-action-button-active"

    return disabled, interval_ms, play_class, pause_class


@app.callback(
    Output("run-artifact-store", "data"),
    Input("run-selector", "value"),
)
def load_selected_run_artifact(selected_run: str | None) -> dict[str, Any] | None:
    if not selected_run:
        return None

    artifact = load_run_file(Path(selected_run))
    artifact["_artifact_path"] = selected_run
    return artifact


@app.callback(
    Output("trajectory-time-slider", "max"),
    Output("trajectory-time-slider", "marks"),
    Output("trajectory-time-slider", "value"),
    Output("trajectory-orbit-scale", "children"),
    Input("run-artifact-store", "data"),
)
def update_time_slider_config(
    artifact: dict[str, Any] | None,
) -> tuple[int, dict[int, str], int, list[html.Div]]:
    if not artifact:
        return 0, {}, 0, []
    states = artifact.get("states_m_s", [])
    times = artifact.get("times_s", [])
    sampled_indices = animation_sample_indices(len(states))
    if not sampled_indices or not times:
        return 0, {}, 0, []

    initial_state = artifact.get("initial_conditions", {}).get("state_vector_m_s", [])
    central_body_mu = (
        artifact.get("parameters", {}).get("central_body", {}).get("mu_m3_s2", 0.0)
    )
    orbit_period_s = None
    if len(initial_state) >= 6:
        orbit_period_s = estimate_initial_orbit_period_s(
            state_m_s=initial_state,
            mu_m3_s2=float(central_body_mu),
        )

    time_marks = slider_elapsed_time_marks_for_indices(times, sampled_indices)
    orbit_scale_children = orbit_scale_marks_for_indices(
        times,
        sampled_indices,
        orbit_period_s,
    )

    return len(sampled_indices) - 1, time_marks, 0, orbit_scale_children


@app.callback(
    Output("trajectory-time-slider", "value", allow_duplicate=True),
    Input("animation-interval", "n_intervals"),
    State("trajectory-time-slider", "value"),
    State("trajectory-time-slider", "max"),
    prevent_initial_call=True,
)
def advance_animation_frame(
    _n_intervals: int,
    current_value: int | None,
    max_value: int | None,
) -> int:
    if max_value is None or max_value <= 0:
        return 0

    next_value = int(current_value or 0) + 1
    if next_value > int(max_value):
        return 0
    return next_value


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
    State("start-date-input", "date"),
    State("start-time-input", "value"),
    State("duration-input", "value"),
    State("samples-per-orbit-input", "value"),
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
    start_date_utc: str | None,
    start_time_utc: str | None,
    duration_s: float | None,
    samples_per_orbit: int | None,
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
            if altitude_km is None or duration_s is None or samples_per_orbit is None:
                raise ValueError(
                    "Altitude, duration, and samples per orbit are required."
                )
            start_epoch_utc = build_start_epoch_utc(start_date_utc, start_time_utc)
            output_path = launch_simulation_from_ui(
                results_dir=RESULTS_DIR,
                run_name=run_name or "ui_run",
                altitude_km=float(altitude_km),
                inclination_deg=float(inclination_deg or 0.0),
                raan_deg=float(raan_deg or 0.0),
                true_anomaly_deg=float(true_anomaly_deg or 0.0),
                start_epoch_utc=start_epoch_utc,
                duration_s=float(duration_s),
                samples_per_orbit=int(samples_per_orbit),
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


def current_state_index(
    states: list[list[float]],
    frame_index: int | None,
) -> tuple[list[int], int, int]:
    sampled_indices = animation_sample_indices(len(states))
    if not sampled_indices:
        return [], 0, 0

    safe_frame_index = max(
        0,
        min(int(frame_index or 0), len(sampled_indices) - 1),
    )
    return sampled_indices, safe_frame_index, sampled_indices[safe_frame_index]


def displayed_position_km_for_state(
    states: list[list[float]],
    state_index: int,
    exaggeration_factor: float | None,
) -> tuple[float, float, float]:
    if not states:
        return 0.0, 0.0, 0.0

    safe_index = max(0, min(int(state_index), len(states) - 1))
    state = states[safe_index]
    radius = math.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
    if radius <= 0.0:
        return 0.0, 0.0, 0.0

    initial_state = states[0]
    base_radius_m = math.sqrt(
        initial_state[0] ** 2 + initial_state[1] ** 2 + initial_state[2] ** 2
    )
    safe_exaggeration = max(float(exaggeration_factor or 1.0), 0.0)
    display_radius_m = base_radius_m + safe_exaggeration * (radius - base_radius_m)
    scale = display_radius_m / radius / 1_000.0

    return state[0] * scale, state[1] * scale, state[2] * scale


def readout_item(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="orbit-readout-label"),
            html.Div(value, className="orbit-readout-value"),
        ],
        className="orbit-readout-item",
    )


@app.callback(
    Output("trajectory-graph", "figure"),
    Input("run-artifact-store", "data"),
    Input("exaggeration-factor-input", "value"),
    Input("show-earth-input", "value"),
    State("trajectory-time-slider", "value"),
)
def update_trajectory_base_figure(
    artifact: dict[str, Any] | None,
    exaggeration_factor: float | None,
    show_earth_values: list[str] | None,
    frame_index: int | None,
) -> go.Figure:
    if not artifact:
        return empty_figure("No run selected")

    artifact_path = Path(artifact.get("_artifact_path", artifact.get("run_name", "run")))
    states = artifact.get("states_m_s", [])
    times = artifact.get("times_s", [])
    central_body_parameters = artifact.get("parameters", {}).get("central_body", {})
    force_parameters = artifact.get("parameters", {}).get("forces", {})
    reference_vectors_m = artifact.get("metadata", {}).get("reference_vectors_m", {})

    return trajectory_figure(
        artifact_path,
        states,
        times,
        frame_index if frame_index is not None else 0,
        exaggeration_factor if exaggeration_factor is not None else 1.0,
        artifact.get("central_body", "Earth"),
        float(central_body_parameters.get("mu_m3_s2", 0.0)),
        float(central_body_parameters.get("radius_m", 0.0)),
        bool(force_parameters.get("j2", False)),
        bool(show_earth_values and "show_earth" in show_earth_values),
        reference_vectors_m,
    )


@app.callback(
    Output("trajectory-graph", "figure", allow_duplicate=True),
    Input("trajectory-time-slider", "value"),
    State("trajectory-graph", "figure"),
    State("run-artifact-store", "data"),
    State("exaggeration-factor-input", "value"),
    prevent_initial_call=True,
)
def update_satellite_marker_only(
    frame_index: int | None,
    current_figure: dict[str, Any] | None,
    artifact: dict[str, Any] | None,
    exaggeration_factor: float | None,
):
    if not artifact or not current_figure or len(current_figure.get("data", [])) <= 2:
        return Patch()

    states = artifact.get("states_m_s", [])
    sampled_indices, safe_frame_index, state_index = current_state_index(states, frame_index)
    if not sampled_indices:
        return Patch()

    x_km, y_km, z_km = displayed_position_km_for_state(
        states,
        state_index,
        exaggeration_factor,
    )

    patched = Patch()
    patched["data"][2]["x"] = [x_km]
    patched["data"][2]["y"] = [y_km]
    patched["data"][2]["z"] = [z_km]
    return patched


@app.callback(
    Output("acceleration-graph", "figure"),
    Output("force-components-graph", "figure"),
    Output("altitude-graph", "figure"),
    Output("speed-graph", "figure"),
    Output("energy-graph", "figure"),
    Output("orbital-elements-graph", "figure"),
    Output("metadata-panel", "children"),
    Output("run-summary", "children"),
    Input("run-artifact-store", "data"),
    State("run-selector", "options"),
)
def update_static_diagnostics(
    artifact: dict[str, Any] | None,
    run_options: list[dict[str, str]] | None,
):
    if not artifact:
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
            hint,
            html.Div("No artifact loaded.", className="summary-item"),
        )

    artifact_path = Path(artifact.get("_artifact_path", artifact.get("run_name", "run")))
    states = artifact["states_m_s"]
    times = artifact["times_s"]
    accelerations = artifact["accelerations_total_m_s2"]
    acceleration_norms = vector_norms(accelerations)

    derived_series = compute_fallback_derived_series(artifact)
    derived_series.update(artifact.get("derived_series", {}))

    altitude_m = derived_series.get("altitude_m", [])
    speed_m_s = derived_series.get("speed_m_s", [])
    specific_energy_j_kg = derived_series.get("specific_energy_j_kg", [])
    raan_deg = derived_series.get("raan_deg", [])
    eccentricity = derived_series.get("eccentricity", [])

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
        acceleration,
        force_components,
        altitude,
        speed,
        energy,
        orbital_elements,
        json.dumps(artifact, indent=2),
        summary,
    )


@app.callback(
    Output("current-frame-readout", "children"),
    Output("current-orbital-quantities", "children"),
    Input("trajectory-time-slider", "value"),
    Input("run-artifact-store", "data"),
)
def update_current_readouts(
    frame_index: int | None,
    artifact: dict[str, Any] | None,
):
    empty_readout = html.Div("No artifact loaded.", className="orbit-readout-empty")
    if not artifact:
        return empty_readout, empty_readout

    states = artifact.get("states_m_s", [])
    times = artifact.get("times_s", [])
    sampled_indices, safe_frame_index, state_index = current_state_index(states, frame_index)
    if not sampled_indices or not times:
        return empty_readout, empty_readout

    initial_state = artifact.get("initial_conditions", {}).get("state_vector_m_s", [])
    central_body_mu_for_period = (
        artifact.get("parameters", {}).get("central_body", {}).get("mu_m3_s2", 0.0)
    )
    orbit_period_s = None
    if len(initial_state) >= 6:
        orbit_period_s = estimate_initial_orbit_period_s(
            state_m_s=initial_state,
            mu_m3_s2=float(central_body_mu_for_period),
        )

    elapsed_time_s = times[state_index] if state_index < len(times) else 0.0
    current_orbit_count = (
        elapsed_time_s / orbit_period_s
        if orbit_period_s and orbit_period_s > 0.0
        else None
    )

    current_state = states[state_index]
    position = current_state[:3]
    velocity = current_state[3:]
    radius_value_m = math.sqrt(sum(component * component for component in position))
    speed_value_m_s = math.sqrt(sum(component * component for component in velocity))

    central_body_parameters = artifact.get("parameters", {}).get("central_body", {})
    body_radius_m = central_body_parameters.get("radius_m")
    mu_m3_s2 = central_body_parameters.get("mu_m3_s2")

    altitude_value_m = (
        radius_value_m - float(body_radius_m)
        if body_radius_m is not None
        else None
    )
    specific_energy_value_j_kg = (
        0.5 * speed_value_m_s * speed_value_m_s - float(mu_m3_s2) / radius_value_m
        if mu_m3_s2 is not None and radius_value_m > 0.0
        else None
    )

    current_frame_readout = [
        readout_item("Frame", f"{safe_frame_index + 1} / {len(sampled_indices)}"),
        readout_item("Sample", f"{state_index + 1} / {len(states)}"),
        readout_item("Elapsed", format_elapsed_time(elapsed_time_s)),
        readout_item(
            "Orbit",
            f"{current_orbit_count:.3f}" if current_orbit_count is not None else "n/a",
        ),
    ]

    current_orbital_quantities = [
        readout_item(
            "Altitude",
            f"{altitude_value_m / 1_000.0:.3f} km"
            if altitude_value_m is not None
            else "n/a",
        ),
        readout_item(
            "Speed",
            f"{speed_value_m_s / 1_000.0:.4f} km/s",
        ),
        readout_item(
            "Radius",
            f"{radius_value_m / 1_000.0:.3f} km",
        ),
        readout_item(
            "Energy",
            f"{specific_energy_value_j_kg:.3e} J/kg"
            if specific_energy_value_j_kg is not None
            else "n/a",
        ),
    ]

    return current_frame_readout, current_orbital_quantities


if __name__ == "__main__":
    host = os.getenv("DASH_HOST", "0.0.0.0")
    port = int(os.getenv("DASH_PORT", "8050"))
    app.run(host=host, port=port, debug=False)
