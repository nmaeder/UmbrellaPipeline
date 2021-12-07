from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
)
from typing import List
import math
import numpy as np
import openmm.unit as unit


class PMFCalculator:
    def __init__(
        self,
        simulation_properties: SimulationProperties,
        simulation_system: SimulationSystem,
        number_of_bins: int,
        path: List[unit.Quantity],
        trajectory_directory: str,
    ) -> None:
        self.simulation_properties = simulation_properties
        self.system_info = simulation_system
        self.number_of_bins = number_of_bins
        self.path = path
        self.trajectory_directory = trajectory_directory
        self.coordinates: np.zeros(
            shape=(
                len(self.path),
                self.simulation_properties.number_of_rounds,
                3,
            ),
            dtype=unit.Quantity,
        )
        self.center_pmf: List[float]
        self.pmf: List[float]

    def parse_trajectories(self):
        for i in len(self.path):
            with open(f"{self.trajectory_directory}/traj_{i}.dcd") as f:
                f.read()

    def calculate_pmf(self):
        distance = []
        for i in len(self.path):
            distance.append(math.sqrt(self.path[0][0] - 3))

    def create_pdf(self, filename: str):
        pass
