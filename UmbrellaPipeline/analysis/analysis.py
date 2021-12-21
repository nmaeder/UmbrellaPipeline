from typing import List
import math
import numpy as np
import openmm.unit as unit
import mdtraj
import torch
from FastMBAR import FastMBAR
import matplotlib.pyplot as plt

from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
    get_center_of_mass_coordinates,
)


class PMFCalculator:
    def __init__(
        self,
        simulation_properties: SimulationProperties,
        simulation_system: SimulationSystem,
        path: List[unit.Quantity],
        trajectory_directory: str,
        path_interval: unit.Quantity,
    ) -> None:
        self.simulation_properties = simulation_properties
        self.system_info = simulation_system
        self.number_of_bins = len(path)
        self.path = path
        self.n_windows = len(path)
        self.n_frames_tot = self.n_windows * self.simulation_properties.number_of_frames
        self.trajectory_directory = trajectory_directory
        self.coordinates: List[unit.Quantity]
        self.center_pmf = np.linspace(
            start=0,
            stop=self.n_windows * path_interval.value_in_unit(unit.nanometer),
            num=len(path),
            endpoint=False,
        )
        self.pmf: List[float] = []
        self.distance: List[unit.Quantity] = []
        self.path_interval = path_interval
        self.KBT = (
            unit.BOLTZMANN_CONSTANT_kB
            * self.simulation_properties.temperature
            * unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(unit.kilocalorie_per_mole)

    def parse_trajectories(self) -> None:
        # lazy loading so it doesnt use the memory upon object construction. (SE nerdness, i know.)
        coordinates = np.zeros(
            shape=(self.n_windows, self.simulation_properties.number_of_frames),
            dtype=object,
        )
        for window in range(self.n_windows):
            # reading in trajectory. whole trajectory is read in,
            # since i only save the com coordinates for further use
            # and delete the trajectory object as soon as its not used anymore.
            trajectory = mdtraj.load_dcd(
                filename=f"{self.trajectory_directory}/traj_{window}.dcd",
                top=self.system_info.pdb_file,
            )
            # saving com coordinates of the ligand ot self.coordinates
            for frame in range(self.simulation_properties.number_of_frames):
                coordinates[window][frame] = get_center_of_mass_coordinates(
                    positions=trajectory.xyz[frame],
                    indices=self.system_info.ligand_indices,
                    masses=self.system_info.psf_object.system,
                )
            del trajectory
        # make one list out of the data frame.
        self.coordinates = np.concatenate(coordinates).tolist()

    def calculate_pmf(self):
        # whatever that is for, they do it so i do it
        num_conf = []

        # actual science :D
        A = np.zeros((self.n_windows, self.n_frames_tot))
        for window in range(self.n_windows):

            # calculate the distances from the potential, path holds the coordinates of the potentials.
            for i in range(len(self.coordinates)):
                dx = abs(self.coordinates[i].x - self.path[window].x)
                dy = abs(self.coordinates[i].y - self.path[window].y)
                dz = abs(self.coordinates[i].z - self.path[window].z)

            # calcualate constraint energies and add them to reduced potential energy matrix A
            A[window, :] = (
                0.5
                * self.simulation_properties.force_constant
                * (dx ** 2 + dy ** 2 + dz ** 2)
            ) / self.KBT

            # again some funky stuff
            num_conf.append(self.simulation_properties.number_of_frames)
        num_conf = np.array(num_conf).astype(np.float64)

        # solving mbar equations using fastMBAR
        wham = FastMBAR(
            energy=A, num_conf=num_conf, cuda=torch.cuda.is_available(), verbose=True
        )

        # initializing perturbed reduced potential energy matrix B.
        B = np.zeros(
            (
                self.number_of_bins,
                self.n_windows * self.simulation_properties.number_of_frames,
            )
        )

        # loop over windows and check for every sample if center of mass is within the boundaries. boundaries are set to have the stepsize of the path,
        # but in all 3 dimensions. i could also always put a sample in to the nearest bin, so i dont have to define the bins, i think this would make
        # life a lot easier, pls comment on that.
        for i in range(self.n_windows):
            # for simplicity, number of bins is now set to the number of umbrella windows. theoretically i could change this,
            # but would then need to calculate new coordinates along the path so i get enough for the number of bins. in principle easily possible,
            # since i already have a function, that does this, but will postpone until this works.
            center = self.path[i]
            # calculate of every coordinate to the potential well
            for c in self.coordinates:
                self.distance.append(
                    math.sqrt(
                        (center.x - c.x) ** 2
                        + (center.z - c.z) ** 2
                        + (center.z - c.z) ** 2
                    )
                )
            # check if distance is smaller than boundaries
            indicator = self.distance < 0.5 * self.path_interval
            # do the infinity thing
            B[i, ~indicator] = np.inf
            # empty distance vector for next round
            self.distance.clear()

        # with that we have one problem, which is why i suggest to bin in a fashion so we put every coordinate into the bin next. problem depicted below.
        #
        # optimal case:
        #  _________________________________________________________
        #  |      |      |      |      |      |      |      |      |
        #  |     x|    x |      |  x   |      | x    |      |      |
        #  |      |  x   |      |      |   x  |      |      |      |
        #  -----------------------------------------------x---------      - every x is in one of the bins.
        #  |   x  |      |   x  |      |x     |    x |      |      |
        #  |      |    x |      |      |      |      | x    |  x   |
        #  |      | x    |      |      |      |      |      |      |
        #  """""""""""""""""""""""""""""""""""""""""""""""""""""""""
        #
        # samples we miss:
        #
        #           X                          X
        #  _________________________________________________________
        #  |      |      |      |      |      |      |      |      |
        #  |     x|      |      |  x   |      | x    |      |      |
        #  |      |  x   |      |      |   x  |      |      |      |
        #  -----------------------------------------------x---------      - since we define our bins by spheres (distance calculations), there can be samples missing, here denoted as X.
        #  |   x  |      |   x  |      |x     |    x |      |      |        when using the binning method by just assigning each sample to the next nearest bin, we would not miss out on these.
        #  |      |    x |      |      |      |      | x    |  x   |        I think this should be withoug drawback from the calculation point of view and should be rather easy to implement (with kd tree as in the escape room).
        #  |      | x    |      |      |      |      |      |      |
        #  """""""""""""""""""""""""""""""""""""""""""""""""""""""""
        #                       X

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
