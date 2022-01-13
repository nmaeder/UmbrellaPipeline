from typing import List
import math
import numpy as np
import openmm.unit as unit
from openmm import Vec3
import mdtraj
from FastMBAR import FastMBAR
import matplotlib.pyplot as plt

from UmbrellaPipeline.path_generation import Tree, TreeEscapeRoom, TreeNode
from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
    deserialize_for_analysis,
    get_center_of_mass_coordinates,
)


class PMFCalculator:
    def __init__(
        self,
        simulation_properties: SimulationProperties,
        simulation_system: SimulationSystem,
        trajectory_directory: str,
        original_path_stepsize: unit.Quantity,
        n_bins: int = None,
        original_path: List[unit.Quantity] = None,
        original_grid_spacing: unit.Quantity = 0.25 * unit.angstrom,
    ) -> None:
        self.simulation_properties = simulation_properties
        self.system_info = simulation_system
        self.path_interval = original_path_stepsize
        self.trajectory_directory = trajectory_directory.rstrip("/")
        self.path = original_path if original_path else self.read_path_coordinates()
        self.gs = original_grid_spacing

        self.n_windows = len(self.path)
        self.n_bins = n_bins if n_bins else self.n_windows
        self.n_frames_tot = self.n_windows * self.simulation_properties.number_of_frames
        self.bin_path = (
            self.path
            if self.n_windows == self.n_bins
            else self.create_bin_points()
        )

        self.center_pmf = np.linspace(
            start=0,
            stop=self.n_windows * original_path_stepsize.value_in_unit(unit.nanometer),
            num=n_bins,
            endpoint=False,
        )
        self.pmf: List[float] = []
        self.distance: List[unit.Quantity] = []
        self.coordinates: np.array

        self.KBT = (
            unit.BOLTZMANN_CONSTANT_kB
            * self.simulation_properties.temperature
            * unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(unit.kilocalorie_per_mole)
        self.A = np.zeros((self.n_windows, self.n_frames_tot))
        self.B = np.zeros(
            (
                n_bins,
                self.n_frames_tot,
            )
        )

    @classmethod
    def from_file(cls, file, n_bins):
        prop, syst, tr, pi, gs = deserialize_for_analysis(filename=file)
        return cls(
            simulation_properties=prop,
            simulation_system=syst,
            trajectory_directory=tr,
            n_bins=n_bins,
            original_path_stepsize=pi,
            original_grid_size=gs,
        )

    def read_path_coordinates(self) -> List[unit.Quantity]:
        """
        reads the restraint positions from the generated coordinates.dat file

        Returns:
            List[unit.Quantity]: List of coordinates, to which the ligand was restrained.
        """
        dat = np.loadtxt(
            fname=self.trajectory_directory + "/coordinates.dat",
            delimiter=",",
            skiprows=1,
        )
        path = []
        for i in range(len(dat)):
            path.append(
                unit.Quantity(
                    Vec3(
                        x=dat[i][1],
                        y=dat[i][2],
                        z=dat[i][3],
                    ),
                    unit=unit.nanometer,
                )
            )
        return path

    def parse_trajectories(self) -> np.array:
        """
        reads the trajectories and stores the center of mass cordinates of the ligand in a class attribute.

        Returns:
            np.array: Array of all center of mass positions of the ligand from all parsed trajectories
        """
        coordinates = np.zeros(
            shape=(self.n_windows, self.simulation_properties.number_of_frames),
            dtype=object,
        )
        for window in range(self.n_windows):
            trajectory = mdtraj.load_dcd(
                filename=f"{self.trajectory_directory}/traj_{window}.dcd",
                top=self.system_info.pdb_file,
            )
            for frame in range(self.simulation_properties.number_of_frames):
                coordinates[window][frame] = get_center_of_mass_coordinates(
                    positions=trajectory.openmm_positions(frame),
                    indices=self.system_info.ligand_indices,
                    masses=self.system_info.psf_object.system,
                )
            del trajectory
        self.coordinates = np.concatenate(coordinates)
        return self.coordinates

    def create_bin_points(self) -> List[unit.Quantity]:
        """
        Creates given number of binpoints along the simulated path.

        Returns:
            List[unit.Quantity]: bin points with according coordinates.
        """
        tree = Tree(self.path)
        er = TreeEscapeRoom(tree=tree, start=TreeNode)
        newp = []
        for i in self.path:
            newp.append(
                TreeNode(
                    x=i.x,
                    y=i.y,
                    z=i.z,
                    unit=i.unit,
                )
            )
        er.shortest_path = newp
        er.stepsize = self.gs
        new_stepsize = self.n_windows * self.path_interval / self.n_bins
        return er.get_path_for_sampling(stepsize=new_stepsize)

    def calculate_pmf(self, nearest_neighbour_method: bool = True):
        """
        Does the weighted histogram analysis of the umbrella sampling to get the PMF. It is strongly advised to use the nearest neighbouring method

        Args:
            nearest_neighbour_method (bool, optional): Wheter to use the nearest neighbour method, it is recommended. see documentation for further information. Defaults to True.
        """
        num_conf = []

        A = np.zeros((self.n_windows, self.n_frames_tot))
        for window in range(self.n_windows):

            # calculate the distances from the potential, path holds the coordinates of the potentials.
            dx, dy, dz = [], [], []
            for i in range(len(self.coordinates)):
                dx.append(self.coordinates[i].x - self.path[window].x)
                dy.append(self.coordinates[i].y - self.path[window].y)
                dz.append(self.coordinates[i].z - self.path[window].z)
            dx = np.array(dx)
            dy = np.array(dy)
            dz = np.array(dz)

            # calcualate constraint energies and add them to reduced potential energy matrix A
            self.A[window, :] = (
                0.5
                * self.simulation_properties.force_constant
                * (dx ** 2 + dy ** 2 + dz ** 2)
            ) / self.KBT

            # save number of conformation per window.
            num_conf.append(self.simulation_properties.number_of_frames)
        num_conf = np.array(num_conf).astype(np.float64)

        # solving mbar equations using fastMBAR
        try:
            wham = FastMBAR(
                energy=A,
                num_conf=num_conf,
                cuda=True,
                verbose=True,
                bootstrap=True,
            )
        except:
            wham = FastMBAR(
                energy=A,
                num_conf=num_conf,
                cuda=False,
                verbose=True,
                bootstrap=True,
            )
        print(wham.F_std)

        # initializing perturbed reduced potential energy matrix B.
        B = np.zeros(
            (
                self.n_bins,
                self.n_frames_tot,
            )
        )

        for i in range(self.n_bins):
            center = self.bin_path[i]
            if nearest_neighbour_method:
                # check wheter conformation is inside its umbrella window or in a neighbouring
                tree = Tree(self.bin_path)
                indicator = []
                for c in self.coordinates:
                    indicator.append(tree.get_nearest_neighbour_index(c) == i)
                indicator = np.array(indicator)
            else:
                # this method not advised, since we miss following cases:
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
                #
                for c in self.coordinates:
                    self.distance.append(
                        math.sqrt(
                            (center.x - c.x) ** 2
                            + (center.y - c.y) ** 2
                            + (center.z - c.z) ** 2
                        )
                        * unit.nanometer
                    )
                indicator = np.array(
                    [d < 0.5 * self.path_interval for d in self.distance]
                )

            self.B[i, ~indicator] = np.inf
            self.distance.clear()

        self.pmf, stderr = wham.calculate_free_energies_of_perturbed_states(B)
        print(stderr)
        return self.pmf, stderr

    def create_pdf(self, filename: str, sum: bool = False):

        fig = plt.figure(0)
        fig.clf()
        if sum:
            y = np.cumsum(self.pmf)
        else:
            y = self.pmf
        plt.plot(self.center_pmf, y, "-o")
        plt.xlim(
            self.center_pmf[0],
            self.center_pmf[-1],
        )
        plt.xlabel("Ligand distance from binding pocked [nm]")
        plt.ylabel("Relative free energy [kcal per mole]")
        plt.savefig(filename)
