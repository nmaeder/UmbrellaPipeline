from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
)
from UmbrellaPipeline.path_generation import get_centroid_coordinates
from typing import List
import math
import numpy as np
import openmm.unit as unit
import mdtraj
import torch
from FastMBAR import FastMBAR, fastmbar

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
                ),
            dtype=unit.Quantity,
        )
        self.center_pmf: List[float]
        self.pmf: List[float]
        self.distance: List[unit.Quantity] = []

    def parse_trajectories(self):
        for i in range(len(self.path)):
            trajectory = mdtraj.load_dcd(f"{self.trajectory_directory}/traj_{i}_.dcd")
            for frame in range(self.simulation_properties.number_of_rounds):
                self.coordinates[i][frame] = get_centroid_coordinates(trajectory[frame], self.system_info.psf_object.atom_list, self.system_info.ligand_indices)
                self.distance.append(
                    math.sqrt(
                        (self.coordinates[i][frame].x - self.path[i].x) ** 2
                        + (self.coordinates[i][frame].y - self.path[i].y) ** 2
                        + (self.coordinates[i][frame].z - self.path[i].z) ** 2
                    )
                )

    def calculate_pmf(self):
        A = np.zeros((len(self.path), len(self.path) * self.simulation_properties.number_of_rounds))
        kbT = (unit.BOLTZMANN_CONSTANT_kB * self.simulation_properties.temperature * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilocalorie_per_mole)
        num_conf=[]
        for window in range(len(self.path)):
            dx = abs(self.coordinates[0:-1][0] - self.path[window].x)
            dy = abs(self.coordinates[0:-1][1] - self.path[window].y)
            dz = abs(self.coordinates[0:-1][2] - self.path[window].z)
            A[window, :] = 0.5 * self.simulation_properties.force_constant*(dx**2 + dy**2 + dz**2)
            num_conf.append(self.simulation_properties.number_of_rounds)
        num_conf = np.array(num_conf).astype(np.float64)
        wham = FastMBAR(energy=A, num_conf=num_conf, cuda=torch.cuda.is_available(), verbose=True)
        

    def create_pdf(self, filename: str):
        pass
