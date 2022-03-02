import openmm.unit as u
from typing import Tuple


class SimulationParameters:
    """
    This class store all simulation parameters and checks their validity.
    """

    def __init__(
        self,
        temperature: u.Quantity = 310 * u.kelvin,
        pressure: u.Quantity = 1 * u.bar,
        time_step: u.Quantity = 2 * u.femtoseconds,
        force_constant: u.Quantity = 100 * u.kilocalorie_per_mole / (u.angstrom ** 2),
        friction_coefficient: u.Quantity = 1 / u.picosecond,
        n_equilibration_steps: int = 500000,
        n_production_steps: int = 2500000,
        write_out_frequency: int = 5000,
    ) -> None:
        """
        Args:
            temperature (u.Quantity, optional): Temperature at which the Umbrella Simulation is run. Defaults to 310*u.kelvin.
            pressure (u.Quantity, optional): Pressure at which the simulation is run. if pressure = None, nvt is sampled instead of npt. Defaults to 1*u.bar.
            time_step (u.Quantity, optional): timestep to use in simulation. Defaults to 2*u.femtoseconds.
            force_constant (u.Quantity, optional): force_constant used for the harmonic restraint. Defaults to 100*u.kilocalorie_per_mole/(u.angstrom**2).
            friction_coefficient (u.Quantity, optional): friction coefficient used in the Langevin integrator.dataclass Defaults to 1/u.picosecond.
            n_equilibration_steps (int, optional): Number of equilibration steps per Lamba. Defaults to 500000.
            n_production_steps (int, optional): Number of production steps per Lamda. Defaults to 2500000.
        """
        self.temperature = temperature
        self.pressure = pressure
        self.time_step = time_step
        self.force_constant = force_constant
        self.friction_coefficient = friction_coefficient
        self.n_equilibration_steps = n_equilibration_steps
        self.n_production_steps = n_production_steps
        self.write_out_frequency = write_out_frequency
        self.number_of_frames = (n_production_steps, write_out_frequency)

    # Getters
    @property
    def temperature(self) -> u.Quantity:
        return self._temperature

    @property
    def pressure(self) -> u.Quantity:
        return self._pressure

    @property
    def time_step(self) -> u.Quantity:
        return self._time_step

    @property
    def force_constant(self) -> u.Quantity:
        return self._force_constant

    @property
    def friction_coefficient(self) -> u.Quantity:
        return self._friction_coefficient

    @property
    def n_equilibration_steps(self) -> int:
        return self._n_equilibration_steps

    @property
    def n_production_steps(self) -> int:
        return self._n_production_steps

    @property
    def write_out_frequency(self) -> int:
        return self._write_out_frequency

    @property
    def number_of_frames(self) -> int:
        return self._number_of_frames

    # Setters

    @temperature.setter
    def temperature(self, value: u.Quantity) -> None:
        try:
            assert value >= 0 * u.kelvin
        except (AssertionError, AttributeError):
            raise TypeError(
                "Temperature has to be in units of kelvin and greater equal 0!"
            )
        self._temperature = value.in_units_of(u.kelvin)

    @pressure.setter
    def pressure(self, value: u.Quantity) -> None:
        if value:
            try:
                assert value >= 0 * u.bar
            except (AssertionError, AttributeError):
                raise TypeError(
                    "Pressure has to be in a unit of pressure and greater equal 0!"
                )
            self._pressure = value
        else:
            self._pressure = None

    @time_step.setter
    def time_step(self, value: u.Quantity) -> None:
        try:
            self._time_step = value.in_units_of(u.femtoseconds)
        except (TypeError, AttributeError):
            raise TypeError("Time_step has to be in units ot time!")

    @force_constant.setter
    def force_constant(self, value: u.Quantity) -> None:
        try:
            self._force_constant = value.in_units_of(
                u.kilocalorie_per_mole * u.angstrom ** -2
            )
        except (TypeError, AttributeError):
            raise TypeError(
                "Force Constant has to be in units of energy per mole per length²!"
            )

    @friction_coefficient.setter
    def friction_coefficient(self, value: u.Quantity) -> None:
        try:
            self._friction_coefficient = value.in_units_of(u.picosecond ** -1)
        except (TypeError, AttributeError):
            raise TypeError("Friction Coefficient has to be in units of inverse time!")

    @n_equilibration_steps.setter
    def n_equilibration_steps(self, value: int) -> None:
        if value < 0:
            raise ValueError("Number of equiolibration steps cannot be negative!")
        try:
            self._n_equilibration_steps = int(value)
        except:
            raise TypeError("Number of equilibration steps has to be an integer!")

    @n_production_steps.setter
    def n_production_steps(self, value: int) -> None:
        assert value >= 0, "Number of production steps cannot be negative!"
        try:
            self._n_production_steps = int(value)
        except:
            raise TypeError("Number of production steps has to be an integer!")

    @write_out_frequency.setter
    def write_out_frequency(self, value: int) -> None:
        assert value > 0, "Output frequency cannot be negative!"
        try:
            if value == 0:
                self._write_out_frequency = self.n_production_steps
            else:
                self._write_out_frequency = int(value)
        except:
            raise TypeError("Write out Frequency has to be an integer!")

    @number_of_frames.setter
    def number_of_frames(self, value: Tuple[int, int]) -> None:
        np, f = value
        try:
            if f == 0:
                self._number_of_frames = 1
            else:
                self._number_of_frames = int(np / f)
        except:
            raise TypeError(
                "Write out frequency as well as number of production steps need to be non negative integers!"
            )
