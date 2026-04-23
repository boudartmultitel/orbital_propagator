from __future__ import annotations

from collections.abc import Callable

import numpy as np

from orbital_propagator.config import IntegratorConfig

try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - exercised indirectly via fallback
    solve_ivp = None


StateDerivative = Callable[[float, np.ndarray], np.ndarray]


def integrate_states(
    derivative: StateDerivative,
    initial_state_m_s: np.ndarray,
    times_s: np.ndarray,
    config: IntegratorConfig,
) -> np.ndarray:
    backend = _resolve_backend(config.backend)
    if backend == "scipy":
        return _integrate_with_scipy(derivative, initial_state_m_s, times_s, config)
    if backend == "rk4":
        return _integrate_with_rk4(derivative, initial_state_m_s, times_s)
    raise ValueError(f"Unsupported integrator backend: {backend}")


def _resolve_backend(requested_backend: str) -> str:
    if requested_backend == "auto":
        return "scipy" if solve_ivp is not None else "rk4"
    if requested_backend == "scipy" and solve_ivp is None:
        raise RuntimeError("SciPy backend requested but scipy is not installed.")
    return requested_backend


def _integrate_with_scipy(
    derivative: StateDerivative,
    initial_state_m_s: np.ndarray,
    times_s: np.ndarray,
    config: IntegratorConfig,
) -> np.ndarray:
    assert solve_ivp is not None
    solution = solve_ivp(
        fun=derivative,
        t_span=(float(times_s[0]), float(times_s[-1])),
        y0=initial_state_m_s,
        t_eval=times_s,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    return solution.y.T


def _integrate_with_rk4(
    derivative: StateDerivative,
    initial_state_m_s: np.ndarray,
    times_s: np.ndarray,
) -> np.ndarray:
    states = np.zeros((len(times_s), len(initial_state_m_s)), dtype=float)
    states[0] = initial_state_m_s

    for index in range(1, len(times_s)):
        previous_time = float(times_s[index - 1])
        step_s = float(times_s[index] - previous_time)
        previous_state = states[index - 1]

        k1 = derivative(previous_time, previous_state)
        k2 = derivative(previous_time + 0.5 * step_s, previous_state + 0.5 * step_s * k1)
        k3 = derivative(previous_time + 0.5 * step_s, previous_state + 0.5 * step_s * k2)
        k4 = derivative(previous_time + step_s, previous_state + step_s * k3)

        states[index] = previous_state + (step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return states
