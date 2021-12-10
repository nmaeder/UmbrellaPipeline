from UmbrellaPipeline.path_generation.path_helper import get_center_of_mass_coordinates
from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
)
from UmbrellaPipeline.path_generation import get_center_of_mass_coordinates
from typing import List
import math
import numpy as np
import openmm.unit as unit
import openmm as mm
import mdtraj
import torch
from FastMBAR import FastMBAR
import matplotlib.pyplot as plt


class PMFCalculator:
    def __init__(
        self,
        simulation_properties: SimulationProperties,
        simulation_system: SimulationSystem,
        number_of_bins: int,
        path: List[unit.Quantity],
        trajectory_directory: str,
        path_interval: unit.Quantity,
    ) -> None:
        self.simulation_properties = simulation_properties
        self.system_info = simulation_system
        self.number_of_bins = number_of_bins
        self.path = path
        self.trajectory_directory = trajectory_directory
        self.coordinates = [
            [[] for i in range(len(self.path))]
            for i in range(self.simulation_properties.number_of_rounds)
        ]
        self.center_pmf: List[float]
        self.pmf: List[float]
        self.distance: List[unit.Quantity] = []
        self.path_interval = path_interval

    def parse_trajectories(self):
        for i in range(len(self.path)):
            trajectory = mdtraj.load_dcd(
                f"{self.trajectory_directory}/traj_{i}.dcd", self.system_info.pdb_file
            )
            for frame in range(trajectory.n_frames):
                self.coordinates[i][frame] = (
                    get_center_of_mass_coordinates(
                        positions=trajectory.xyz[frame],
                        indices=self.system_info.ligand_indices,
                        masses=self.system_info.psf_object.system,
                    )
                    / unit.nanometer
                )
                self.distance.append(
                    math.sqrt(
                        (self.coordinates[i][frame].x - self.path[i].x) ** 2
                        + (self.coordinates[i][frame].y - self.path[i].y) ** 2
                        + (self.coordinates[i][frame].z - self.path[i].z) ** 2
                    )
                )

    def calculate_pmf(self):
        A = np.zeros(
            (
                len(self.path),
                len(self.path) * self.simulation_properties.number_of_rounds,
            )
        )
        kbT = (
            unit.BOLTZMANN_CONSTANT_kB
            * self.simulation_properties.temperature
            * unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(unit.kilocalorie_per_mole)
        num_conf = []
        for window in range(len(self.path)):
            dx = abs(
                self.coordinates[0:-1][0].value_in_unit(unit.nanometer)
                - self.path[window].x
            )
            dy = abs(
                self.coordinates[0:-1][1].value_in_unit(unit.nanometer)
                - self.path[window].y
            )
            dz = abs(
                self.coordinates[0:-1][2].value_in_unit(unit.nanometer)
                - self.path[window].z
            )
            A[window, :] = (
                0.5
                * self.simulation_properties.force_constant
                * (dx ** 2 + dy ** 2 + dz ** 2)
            )
            num_conf.append(self.simulation_properties.number_of_rounds)
        num_conf = np.array(num_conf).astype(np.float64)
        wham = FastMBAR(
            energy=A, num_conf=num_conf, cuda=torch.cuda.is_available(), verbose=True
        )
        width = self.path_interval.value_in_unit(unit.nanometer)
        center_pmf = np.linspace(
            0, len(self.path) * width, self.number_of_bins, endpoint=False
        )
        B = np.zeros(
            (
                self.number_of_bins,
                len(self.path) * self.simulation_properties.number_of_rounds,
            )
        )
        for i in range(self.number_of_bins):
            center = center_pmf[i]
            center_low = center - 0.5 * width
            center_high = center + 0.5 * width
            indicator = (self.distance > center_low) & (self.distance <= center_high)
            B[i, ~indicator] = np.inf
        self.pmf, _ = wham.calculate_free_energies_of_perturbed_states(B)

    def create_pdf(self, filename: str):
        fig = plt.figure(0)
        fig.clf()
        plt.plot(self.center_pmf, self.pmf, "-o")
        plt.xlim(
            0,
            len(self.path * self.path_interval.value_in_unit(self.path_interval.unit)),
        )
        plt.xlabel("Ligand distance from binding pocked [nm]")
        plt.ylabel("Relative free energy [kcal per mole]")
        plt.savefig(filename)
