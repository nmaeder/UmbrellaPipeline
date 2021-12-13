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
        self.n_windows = len(path)
        self.n_frames_tot = self.n_windows * self.simulation_properties.number_of_frames
        self.trajectory_directory = trajectory_directory
        self.coordinates = np.zeros(
            shape=(self.n_windows, self.simulation_properties.number_of_frames),
            dtype=object,
        )
        self.center_pmf = np.linspace(
            start=0,
            stop=self.n_windows * path_interval.value_in_unit(unit.nanometer),
            num=number_of_bins,
            endpoint=False,
        )
        self.pmf: List[float]
        self.distance: List[unit.Quantity] = []
        self.path_interval = path_interval

    def parse_trajectories(self):
        for window in range(self.n_windows):
            trajectory = mdtraj.load_dcd(
                f"{self.trajectory_directory}/traj_{window}.dcd",
                self.system_info.pdb_file,
            )
            for frame in range(self.simulation_properties.number_of_frames):

                self.coordinates[window][frame] = get_center_of_mass_coordinates(
                    positions=trajectory.xyz[frame],
                    indices=self.system_info.ligand_indices,
                    masses=self.system_info.psf_object.system,
                )
                self.distance.append(
                    math.sqrt(
                        (self.coordinates[window][frame].x - self.path[window].x) ** 2
                        + (self.coordinates[window][frame].y - self.path[window].y) ** 2
                        + (self.coordinates[window][frame].z - self.path[window].z) ** 2
                    )
                )

    def calculate_pmf(self):
        coordinates = np.concatenate(self.coordinates)
        A = np.zeros(
            (
                self.n_windows,
                self.n_frames_tot,
            )
        )
        kbT = (
            unit.BOLTZMANN_CONSTANT_kB
            * self.simulation_properties.temperature
            * unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(unit.kilocalorie_per_mole)
        num_conf = []
        for window in range(self.n_windows):
            for i in range(len(coordinates)):
                dx = abs(coordinates[i].x - self.path[window].x)
                dy = abs(coordinates[i].y - self.path[window].y)
                dz = abs(coordinates[i].z - self.path[window].z)
            A[window, :] = (
                0.5
                * self.simulation_properties.force_constant
                * (dx ** 2 + dy ** 2 + dz ** 2)
            ) / kbT
            num_conf.append(self.simulation_properties.number_of_frames)
        num_conf = np.array(num_conf).astype(np.float64)
        wham = FastMBAR(
            energy=A, num_conf=num_conf, cuda=torch.cuda.is_available(), verbose=True
        )
        width = self.path_interval.value_in_unit(unit.nanometer)
        B = np.zeros(
            (
                self.number_of_bins,
                self.n_windows * self.simulation_properties.number_of_frames,
            )
        )
        for i in range(self.number_of_bins):
            center = self.center_pmf[i]
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
            self.center_pmf[0],
            self.center_pmf[-1],
        )
        plt.xlabel("Ligand distance from binding pocked [nm]")
        plt.ylabel("Relative free energy [kcal per mole]")
        plt.savefig(filename)
