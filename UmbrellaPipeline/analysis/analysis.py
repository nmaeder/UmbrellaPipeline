from typing import List, Tuple
import numpy as np
import openmmtools
from openmm import Vec3, unit
import mdtraj
from FastMBAR import FastMBAR
import matplotlib.pyplot as plt

from UmbrellaPipeline.path_generation import TreeNode, Tree, TreeEscapeRoom
from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
    get_center_of_mass_coordinates,
)


class PMFCalculator:
    """
    Holds all the analysis information
    """

    def __init__(
        self,
        simulation_properties: SimulationProperties,
        simulation_system: SimulationSystem,
        trajectory_directory: str,
        original_path_interval: unit.Quantity,
        path_coordinates: List[unit.Quantity] = [],
        n_bins: int = None,
    ) -> None:
        """
        Args:
            simulation_properties (SimulationProperties): [description]
            simulation_system (SimulationSystem): [description]
            trajectory_directory (str): [description]
            original_path_interval (unit.Quantity): [description]
            path_coordinates (List[unit.Quantity], optional): [description]. Defaults to [].
            n_bins (int, optional): [description]. Defaults to None.
        """
        self.simulation_properties = simulation_properties
        self.system_info = simulation_system
        self.trajectory_directory = trajectory_directory.rstrip("/")

        self.masses = self.system_info.psf_object.createSystem(self.system_info.params)

        self.n_windows = len(path_coordinates)
        self.n_bins = n_bins if n_bins else self.n_windows
        self.path_coordinates = path_coordinates
        self.path_interval = original_path_interval
        self.n_frames_tot = self.n_windows * self.simulation_properties.number_of_frames

        self.KBT = (
            unit.BOLTZMANN_CONSTANT_kB
            * self.simulation_properties.temperature
            * unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(unit.kilocalorie_per_mole)

        self.A: np.ndarray
        self.B: np.ndarray
        self.sampled_coordinates: np.ndarray

        self.pmf: np.ndarray
        self.pmf_error: np.ndarray

    def parse_trajectories(self) -> None:
        """
        Reads in the trajectory files, extracts the center of mass positions for the ligand in every frame and writes them into one file to save time and memory.
        """
        coordinates = []
        for window in range(self.n_windows):
            trajectory = mdtraj.load_dcd(
                filename=f"{self.trajectory_directory}/production_trajectory_window_{window}.dcd",
                top=self.system_info.psf_file,
            )
            coordinates.append(
                [
                    get_center_of_mass_coordinates(
                        positions=trajectory.openmm_positions(frame),
                        indices=self.system_info.ligand_indices,
                        masses=self.masses,
                    )
                    for frame in range(self.simulation_properties.number_of_frames)
                ]
            )
        self.sampled_coordinates = np.concatenate(coordinates).tolist()
        with open(
            self.trajectory_directory + "/sampled_coordinates.dat", mode="w"
        ) as f:
            f.write("#sampled center of mass coordinates of the ligand in nm\n")
            for i in coordinates:
                f.write(f"{i.x}, {i.y}, {i.z}\n")

    def update_class_attributes(self) -> None:
        self.n_windows = len(self.path_coordinates)
        if self.n_bins == 0:
            self.n_bins = self.n_windows
        self.n_frames_tot = self.n_windows * self.simulation_properties.number_of_frames

    def load_original_path(self, fname: str = None) -> List[unit.Quantity]:
        """
        Loads the coordinates to which the ligand was restrained to.

        Args:
            fname (str, optional): only give if you changed the coordinates.dat file name that was created by this package. Defaults to None.

        Returns:
            List[unit.Quantity]: List containing the restraint positions of the umbrella windows
        """
        file = fname if fname else self.trajectory_directory + "/coordinates.dat"
        dat = np.loadtxt(file, delimiter=",", skiprows=1)
        self.path_coordinates = [
            unit.Quantity(
                value=Vec3(
                    x=i[1],
                    y=i[2],
                    z=i[3],
                ),
                unit=unit.nanometer,
            )
            for i in dat
        ]
        self.update_class_attributes()
        return self.path_coordinates

    def load_sampled_coordinates(self, fname: str = None) -> List[unit.Quantity]:
        """
        Reads the coordinates from the file created by this package. Can only be used if parse_trajectories() was ecexuted at one point before.

        Args:
            fname (str, optional): nly give if you changed the sampled_coordinates.dat file name that was created by this package. Defaults to None.

        Returns:
            List[unit.Quantity]: List containing the sampled center of mass coordinates of the ligand.
        """
        file = (
            fname if fname else self.trajectory_directory + "/sampled_coordinates.dat"
        )
        dat = np.loadtxt(file, delimiter=",")
        self.sampled_coordinates = [
            unit.Quantity(
                value=Vec3(
                    x=i[0],
                    y=i[1],
                    z=i[2],
                ),
                unit=unit.nanometer,
            )
            for i in dat
        ]
        return self.sampled_coordinates

    def create_extra_bin_points(self, stepsize):
        """
        helper function, see usage below.
        """
        tree = Tree(self.path_coordinates)
        er = TreeEscapeRoom(tree=tree, start=TreeNode())
        newp = []
        for window in self.path_coordinates:
            newp.append(
                TreeNode(
                    x=window.x,
                    y=window.y,
                    z=window.z,
                    unit=window.unit,
                )
            )
        er.shortest_path = newp
        er.stepsize = 0.25 * unit.angstrom
        return er.get_path_for_sampling(stepsize=stepsize)

    def calculate_pmf(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Does the actual PMF calculations using the FastMBAR package.

        Args:
            bin_path (List[unit.Quantity]): The bin points along the sampled path. if number of bins equals number of simulation windows,
            this is just the restrain coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray]: the calculated forces per bin and the stdeviation estimate.
        """

        # create extra bin point coordinates if number of bins is bigger than number of windows.
        if self.n_windows == self.n_bins:
            bin_path = self.path_coordinates
        else:
            stepsize = self.n_windows * self.path_interval / self.n_bins
            bin_path = self.create_extra_bin_points(stepsize=stepsize)

        num_conf = []

        # initialize reduced potential energy matrix
        self.A = np.zeros(shape=(self.n_windows, self.n_frames_tot))

        # calculate deviations for every frame.
        for window_number, window in enumerate(self.path_coordinates):
            dx = np.array([frame.x - window.x for frame in self.sampled_coordinates])
            dy = np.array([frame.y - window.y for frame in self.sampled_coordinates])
            dz = np.array([frame.z - window.z for frame in self.sampled_coordinates])

            # calculate reduced potential energy
            self.A[window_number, :] = (
                0.5
                * self.simulation_properties.force_constant
                * (dx ** 2 + dy ** 2 + dz ** 2)
            ) / self.KBT

            # save number of samples per window. in our case same in every window.
            num_conf.append(self.simulation_properties.number_of_frames)
        num_conf = np.array(num_conf).astype(np.float64)

        # check if cuda is available.
        cuda_available = "CUDA" in [
            pf.getName() for pf in openmmtools.utils.get_available_platforms()
        ]

        # solve mbar equations.
        wham = FastMBAR(
            energy=self.A,
            num_conf=num_conf,
            cuda=cuda_available,
            verbose=True,
            bootstrap=True,
        )

        # initialize perturbed reduced potential energy matrix
        self.B = np.zeros(shape=(self.n_bins, self.n_frames_tot))

        for bin in range(self.n_bins):
            tree = Tree(bin_path, unit=unit.nanometer)
            # for every sampling window, check if the frames from that window are actually closest to that window.
            indicator = np.array(
                [
                    tree.get_nearest_neighbour_index(frame) == bin
                    for frame in self.sampled_coordinates
                ]
            )
            self.B[bin, ~indicator] = np.inf

        # calculate the free energies of the perturbed states
        self.pmf, self.pmf_error = wham.calculate_free_energies_of_perturbed_states(
            self.B
        )
        return self.pmf, self.pmf_error

    def plot(self, filename: str = None, format: str = "svg", dpi: float = 300):
        """
        plots the pmf with error bars.

        Args:
            filename (str, optional): if given, a png file is exported to the filepath.
        """
        pmf_center = np.linspace(
            start=0,
            stop=(self.n_windows * self.path_interval).value_in_unit(unit.nanometer),
            num=self.n_bins,
            endpoint=False,
        )
        fig = plt.figure()
        plt.errorbar(pmf_center, self.pmf, yerr=self.pmf_error, fmt="-o")
        plt.xlim(
            pmf_center[0],
            pmf_center[-1],
        )
        plt.xlabel("Ligand distance from binding pocked [nm]")
        plt.ylabel("Relative free energy [kcal per mole]")
        if filename:
            if not filename.endswith(f".{format}"):
                filename += f".{format}"
            fig.savefig(
                filename,
                dpi=dpi,
                format=format,
            )

    def plot_pytest(self, filename: str = None, format: str = "svg", dpi: float = 300):
        import matplotlib

        """
        plots the pmf with error bars. only use together with pytest cases in vscode. will probabely be removed later.

        Args:
            filename (str, optional): if given, a png file is exported to the filepath.
        """
        matplotlib.use("Agg")
        pmf_center = np.linspace(
            start=0,
            stop=(self.n_windows * self.path_interval).value_in_unit(unit.nanometer),
            num=self.n_bins,
            endpoint=False,
        )
        fig = plt.figure()
        plt.errorbar(pmf_center, self.pmf, yerr=self.pmf_error, fmt="-o")
        plt.xlim(
            pmf_center[0],
            pmf_center[-1],
        )
        plt.xlabel("Ligand distance from binding pocked [nm]")
        plt.ylabel("Relative free energy [kcal per mole]")
        if filename:
            if not filename.endswith(f".{format}"):
                filename += f".{format}"
            fig.savefig(
                filename,
                dpi=dpi,
                format=format,
            )
        plt.close(fig)
