from pathlib import Path

from collections.abc import Iterator
import math

from orbital_propagator.bodies.earth import EARTH
from orbital_propagator.config import circular_orbit_state, keplerian_orbit_state, SimulationRequest, IntegratorConfig, \
    SpacecraftConfig, ForceModelConfig, PropagationConfig
from orbital_propagator.io.artifacts import build_run_artifact, save_run_artifact
from orbital_propagator.propagation.runner import run_simulation

def custom_range(custom: tuple[str, float | int, float | int, float | int]) -> Iterator[float | int]:
    """Half-open [start, stop) range supporting int or float steps."""
    kind = custom[0]
    start = custom[1]
    stop = custom[2]
    step = custom[3]
    if step == 0:
        raise ValueError("step must be non-zero")
    if kind not in ("int", "float"):
        raise ValueError(f"Unknown range type: {kind}")

    if kind == "int":
        yield from range(int(start), int(stop), int(step))
        return

    n = math.ceil((stop - start) / step)
    for i in range(max(n, 0)):
        yield start + i * step

class DataGeneration:
    #TODO starting point start_epoch_utc
    def __init__(self,
                 orbit_type: str,
                 central_body: str,
                 altitude_m: float,
                 semimajor_axis_m: float,
                 eccentricity: float,
                 inclination_deg: float = 0.0,
                 raan_deg: float = 0.0,
                 true_anomaly_deg: float = 0.0,
                 argument_of_periapsis_deg: float = 0.0,
                 run_name: str = "two_body_earth",
                 duration_s: float = 5400.0,
                 sample_count: int = 181,
                 start_epoch_utc: str = "2026-01-01T00:00:00Z",
                 integrator_backend: str = "auto",
                 integrator_method: str = "DOP853",
                 rtol: float = 1e-9,
                 atol: float = 1e-9,
                 max_step_s: float = None,
                 enable_j2: bool = False,
                 enable_drag: bool = False,
                 atmosphere_model: str = "piecewise_exponential",
                 disable_atmosphere_corotation: bool = False,
                 enable_srp: bool = False,
                 enable_third_body_sun: bool = False,
                 enable_third_body_moon: bool = False,
                 mass_kg: float = 1000.0,
                 cross_section_area_m2: float = 10.0,
                 drag_coefficient: float = 2.2,
                 reflectivity_coefficient: float = 1.2,
                 force_breakdown: bool = False,
                 output: str = "./generated_data.json"
                 ):
        self.orbit_type = orbit_type
        self.central_body = central_body
        self.altitude_m = altitude_m
        self.semimajor_axis_m = semimajor_axis_m
        self.eccentricity = eccentricity
        self.inclination_deg = inclination_deg
        self.raan_deg = raan_deg
        self.true_anomaly_deg = true_anomaly_deg
        self.argument_of_periapsis_deg = argument_of_periapsis_deg
        self.run_name = run_name
        self.duration_s = duration_s
        self.sample_count = sample_count
        self.start_epoch_utc = start_epoch_utc
        self.integrator_backend = integrator_backend
        self.integrator_method = integrator_method
        self.rtol = rtol
        self.atol = atol
        self.max_step_s = max_step_s
        self.enable_j2 = enable_j2
        self.enable_drag = enable_drag
        self.atmosphere_model = atmosphere_model
        self.disable_atmosphere_corotation = disable_atmosphere_corotation
        self.enable_srp = enable_srp
        self.enable_third_body_sun = enable_third_body_sun
        self.enable_third_body_moon = enable_third_body_moon
        self.mass_kg = mass_kg
        self.cross_section_area_m2 = cross_section_area_m2
        self.drag_coefficient = drag_coefficient
        self.reflectivity_coefficient = reflectivity_coefficient
        self.force_breakdown = force_breakdown
        self.output = Path(output)
        self.initial_state_m_s = self.get_initial_state_m_s()

    def get_initial_state_m_s(self):
        if self.central_body != "earth":
            raise ValueError("central_body must be 'earth'.")
        if not (150 <= (self.altitude_m/1_000) <= 40_000):
            raise ValueError("altitude must be between 150 and 40_000 km.")
        if not (150 <= (self.semimajor_axis_m/1_000) <= 40_000):
            raise ValueError("semimajor axis must be between 150 and 40_000 km.")
        if not (0<=self.eccentricity<1):
            raise ValueError("eccentricity must be between 0 and 1.")
        if not(0<=self.inclination_deg<=180):
            raise ValueError("inclination deg must be between 0 and 180 degrees.")
        if not(0<=self.raan_deg<360):
            raise ValueError("raan deg must be between 0 and 360 degrees.")
        if not(0<=self.true_anomaly_deg<360):
            raise ValueError("true_anomaly_deg must be between 0 and 360 degrees.")
        if not(0<=self.argument_of_periapsis_deg<360):
            raise ValueError("argument_of_periapsis_deg must be between 0 and 360 degrees.")
        if self.orbit_type == "ellipse":
            return keplerian_orbit_state(
                central_body=EARTH,
                semimajor_axis_m=self.semimajor_axis_m,
                eccentricity=self.eccentricity,
                inclination_deg=self.inclination_deg,
                raan_deg=self.raan_deg,
                argument_of_periapsis_deg=self.argument_of_periapsis_deg,
                true_anomaly_deg=self.true_anomaly_deg,
            )
        elif self.orbit_type == "circular":
            return circular_orbit_state(
                central_body=EARTH,
                altitude_m=self.altitude_m,
                inclination_deg=self.inclination_deg,
                raan_deg=self.raan_deg,
                true_anomaly_deg=self.true_anomaly_deg,
            )
        else:
            raise ValueError(f"Unknown orbit type: {self.orbit_type}")

    def get_simulation_request(self):
        request = SimulationRequest(
            run_name=self.run_name,
            producer="simulation",
            central_body=EARTH,
            initial_state_m_s=self.initial_state_m_s,
            propagation=PropagationConfig(
                duration_s=self.duration_s,
                sample_count=self.sample_count,
                start_epoch_utc=self.start_epoch_utc,
            ),
            integrator=IntegratorConfig(
                backend=self.integrator_backend,
                method=self.integrator_method,
                rtol=self.rtol,
                atol=self.atol,
                max_step_s=self.max_step_s,
            ),
            spacecraft=SpacecraftConfig(
                mass_kg=self.mass_kg,
                cross_section_area_m2=self.cross_section_area_m2,
                drag_coefficient=self.drag_coefficient,
                reflectivity_coefficient=self.reflectivity_coefficient,
            ),
            forces=ForceModelConfig(
                central_gravity=True,
                j2=self.enable_j2,
                drag=self.enable_drag,
                atmosphere_model=self.atmosphere_model,
                corotating_atmosphere=not self.disable_atmosphere_corotation,
                solar_radiation_pressure=self.enable_srp,
                third_body_sun=self.enable_third_body_sun,
                third_body_moon=self.enable_third_body_moon,
            ),
        )
        return request

    def get_simulation_response(self, request):
        result = run_simulation(request)
        return result

    def get_artifact(self):
        request = self.get_simulation_request()
        artifact = build_run_artifact(
            request,
            self.get_simulation_response(request),
            force_breakdown=self.force_breakdown,
        )
        return artifact

    def save_artifact(self, artifact):
        save_run_artifact(artifact, self.output)

    def simulate(self):
        self.save_artifact(self.get_artifact())

class BulkGenration:
    def __init__(self, orbit_type,
                 altitude_range: tuple[str, float | int, float | int, float | int],
                 semimajor_axis_range: tuple[str, float | int, float | int, float | int],
                 eccentricity: tuple[str, float | int, float | int, float | int],
                 inclination_deg: tuple[str, float | int, float | int, float | int],
                 raan_deg: tuple[str, float | int, float | int, float | int],
                 true_anomaly_deg: tuple[str, float | int, float | int, float | int],
                 argument_of_periapsis_deg: tuple[str, float | int, float | int, float | int],
                 ):
        self.orbit_type = orbit_type
        self.altitude_range = altitude_range
        self.semimajor_axis_range = semimajor_axis_range
        self.eccentricity = eccentricity
        self.inclination_deg = inclination_deg
        self.raan_deg = raan_deg
        self.true_anomaly_deg = true_anomaly_deg
        self.argument_of_periapsis_deg = argument_of_periapsis_deg

    def bulk_genrate(self):
        #TODO constraints on generation
        for i in custom_range(self.altitude_range):
            for j in custom_range(self.semimajor_axis_range):
                for k in custom_range(self.eccentricity):
                    for l in custom_range(self.inclination_deg):
                        for m in custom_range(self.raan_deg):
                            for n in custom_range(self.true_anomaly_deg):
                                for o in custom_range(self.argument_of_periapsis_deg):
                                    data = DataGeneration(orbit_type=self.orbit_type,
                                                              central_body="earth",
                                                              altitude_m=i,
                                                              semimajor_axis_m=j,
                                                              eccentricity=k,
                                                              inclination_deg=l,
                                                              raan_deg=m,
                                                              true_anomaly_deg=n,
                                                              argument_of_periapsis_deg=o,
                                                  output=f"./data/{i}_{j}_{k}_{l}_{m}_{n}_{o}.json")
                                    data.simulate()

if __name__ == "__main__":
    """data = DataGeneration(orbit_type = "circular",
                 central_body = "earth",
                 altitude_m = 200000,
                 semimajor_axis_m = 200000,
                 eccentricity = 0.99,
                 inclination_deg = 170,
                 raan_deg = 180.0,
                 true_anomaly_deg = 0.0,
                 argument_of_periapsis_deg = 0.0)
    data.simulate()"""
    bulk = BulkGenration(orbit_type="circular",
                         altitude_range=("int",200000, 200010, 3),
                         semimajor_axis_range=("int",200000, 200010, 3),
                         eccentricity=("float",0.8, 0.9, 0.1),
                         inclination_deg=("int", 170, 180, 2),
                         raan_deg=("int", 180, 190, 2),
                         true_anomaly_deg=("int", 0, 1, 1),
                         argument_of_periapsis_deg=("int", 0, 1, 1),
                         )
    bulk.bulk_genrate()