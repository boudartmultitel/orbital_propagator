from orbital_propagator.bodies.earth import EARTH
from orbital_propagator.config import circular_orbit_state, keplerian_orbit_state, SimulationRequest, IntegratorConfig, \
    SpacecraftConfig, ForceModelConfig, PropagationConfig


class DataGeneration:
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



if __name__ == "__main__":
    data = DataGeneration(orbit_type = "circular",
                 central_body = "earth",
                 altitude_m = 200000,
                 semimajor_axis_m = 200000,
                 eccentricity = 0.99,
                 inclination_deg = 170,
                 raan_deg = 180.0,
                 true_anomaly_deg = 0.0,
                 argument_of_periapsis_deg = 0.0)
    print(data.get_initial_state_m_s())
    print(data.get_simulation_request())