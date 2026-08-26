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
            run_name=args.run_name,
            producer="simulation",
            central_body=EARTH,
            initial_state_m_s=self.get,
            propagation=PropagationConfig(
                duration_s=args.duration_s,
                sample_count=args.sample_count,
                start_epoch_utc=args.start_epoch_utc,
            ),
            integrator=IntegratorConfig(
                backend=args.integrator_backend,
                method=args.integrator_method,
                rtol=args.rtol,
                atol=args.atol,
                max_step_s=args.max_step_s,
            ),
            spacecraft=SpacecraftConfig(
                mass_kg=args.mass_kg,
                cross_section_area_m2=args.cross_section_area_m2,
                drag_coefficient=args.drag_coefficient,
                reflectivity_coefficient=args.reflectivity_coefficient,
            ),
            forces=ForceModelConfig(
                central_gravity=True,
                j2=args.enable_j2,
                drag=args.enable_drag,
                atmosphere_model=args.atmosphere_model,
                corotating_atmosphere=not args.disable_atmosphere_corotation,
                solar_radiation_pressure=args.enable_srp,
                third_body_sun=args.enable_third_body_sun,
                third_body_moon=args.enable_third_body_moon,
            ),
    )



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