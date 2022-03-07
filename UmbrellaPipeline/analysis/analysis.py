from typing import List, Literal, Tuple
import numpy as np
import openmmtools
from openmm import Vec3, unit
import mdtraj, pymbar
from FastMBAR import FastMBAR
import matplotlib as mtl
import matplotlib.pyplot as plt
from typing import Literal

from UmbrellaPipeline.path_finding import TreeNode, Tree, TreeEscapeRoom
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
        solver: Literal = "pymbar"
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

        self.use_kcal = False

        self.kT = (
            unit.BOLTZMANN_CONSTANT_kB
            * self.simulation_properties.temperature
            * unit.AVOGADRO_CONSTANT_NA
        ).in_units_of(unit.kilocalorie_per_mole)

        self.A: np.ndarray
        self.B: np.ndarray
        self.sampled_coordinates: np.ndarray

        self.pmf: np.ndarray
        self.pmf_error: np.ndarray

        if solver.lower() == "pymbar":
            self.calculate_pmf = self.calculate_pmf_pymbar
        elif solver.lower() == "fastmbar":
            self.calculate_pmf = self.calculate_pmf_fastMBAR
        else:
            raise ValueError("solver needs to be of type Literal and can either be 'pymbar' or 'fastmbar'!")

    def parse_trajectories(self) -> None:
        """
        Reads in the trajectory files, extracts the center of mass positions for the ligand in every frame and writes them into one file to save time and memory.
        """
        with open(
            self.trajectory_directory + "/sampled_coordinates.dat", mode="w"
        ) as f:
            f.write("#sampled center of mass coordinates of the ligand in nm\n")
            for window in range(self.n_windows):
                coordinates = []
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
                for i in coordinates:
                    f.write(f"{i.x}, {i.y}, {i.z}\n")
        return self.trajectory_directory + "/sampled_coordinates.dat"

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
        c = [
            Vec3(
                x=i[1],
                y=i[2],
                z=i[3],
            )
            for i in dat
        ]
        self.path_coordinates = unit.Quantity(value=c, unit=unit.nanometer)
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
        try:
            dat = np.loadtxt(file, delimiter=",")
        except FileNotFoundError:
            file = self.parse_trajectories()
            dat = np.loadtxt(file, delimiter=",")
        c = [
            Vec3(
                x=i[0],
                y=i[1],
                z=i[2],
            )
            for i in dat
        ]
        self.sampled_coordinates = unit.Quantity(value=c, unit=unit.nanometer)
        return self.sampled_coordinates

    def calculate_pmf_pymbar(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Does the actual PMF calculations using the pymbar package.

        Returns:
            Tuple[np.ndarray, np.ndarray]: the calculated forces per bin and the stdeviation estimate.
        """
        T_K = np.ones(self.n_windows, float)*self.simulation_properties.temperature._value
        dmin = 0
        dmax = self.n_windows*self.path_interval.value_in_unit(unit.nanometer)
        N_k = np.ones([self.n_windows], np.int32)*self.simulation_properties.number_of_frames
        pos0_k = np.zeros([self.n_windows,3], np.float64)
        pos_kn = np.zeros([self.n_windows,self.simulation_properties.number_of_frames,3], np.float64)
        u_kn = np.zeros([self.n_windowsK,self.simulation_properties.number_of_frames], np.float64)
        force_constant = self.simulation_properties.force_constant.value_in_unit(unit.kilocalorie_per_mole*unit.nanometer**-2)

        for it, p in enumerate(self.path_coordinates):
            pos0_k[it][0] = p.x
            pos0_k[it][1] = p.y 
            pos0_k[it][2] = p.z

        window = 0
        frame = 0
        for it, p in enumerate(self.sampled_coordinates):
            if it % self.simulation_properties.number_of_frames == 0 and it != 0:
                window += 1
            if frame % self.simulation_properties.number_of_frame == 0:
                frame = 0
            pos_kn[window][frame][0] = p.x
            pos_kn[window][frame][1] = p.y
            pos_kn[window][frame][2] = p.z
            frame += 1
        del window
        del frame

        u_kln = np.zeros([self.n_windows,self.n_windows,self.simulation_properties.number_of_frame], np.float64)

        for k in range(self.n_windows):
            for l in range(self.n_windows):
                for n in range(self.simulation_properties.number_of_frame):
                    dx = pos_kn[l][n][0] - pos0_k[k][0] 
                    dy = pos_kn[l][n][1] - pos0_k[k][1]
                    dz = pos_kn[l][n][2] - pos0_k[k][2]
                u_kln[k][l][n] = 0.5*force_constant*(dx**2 + dy**2 + dz**2) / self.kT

        mbar = pymbar.MBAR(u_kln, N_k, verbose=True)
        bins = self.path_coordinates
        nbins = self.n_windows
        tree = Tree(bins, unit=unit.nanometer)
        bin_kn = np.zeros([self.n_windows, self.simulation_properties.number_of_frames])
        for k in range(self.n_windows):
            indlow = 0 + k*self.simulation_properties.number_of_frames
            indhigh = self.simulation_properties.number_of_frames-1 + k*self.simulation_properties.number_of_frames
            indicator = np.array(
                [
                    tree.get_nearest_neighbour_index(frame)
                    for frame in self.sampled_coordinates[indlow:indhigh]
                ], dtype=np.int32
            )
            for n,bin in enumerate(indicator):
                bin_kn[k][n] = bin

        

        results = mbar.computePMF(u_kn, bin_kn, nbins, return_dict=True)
        return results["f_i"], results["df_i"]


    def calculate_pmf_fastMBAR(
        self, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Does the actual PMF calculations using the FastMBAR package.

        Args:
            use_kcal (bool): If True, everything is calculated in kcal per mole instead of kJ per mol. Defaults to False.

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
                * self.simulation_properties.force_constant.in_units_of(unit.kilocalorie_per_mole*unit.nanometer**-2)
                * (dx ** 2 + dy ** 2 + dz ** 2)
            ) / self.kT

            # save number of samples per window. in our case same in every window.
            num_conf.append(self.simulation_properties.number_of_frames)
        num_conf = np.array(num_conf).astype(np.float64)

        # check if cuda is available.
        cuda_available = "CUDA" in [
            pf.getName() for pf in openmmtools.utils.get_available_platforms()
        ]

        # solve mbar equations.
        mbar = FastMBAR(
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
        self.pmf, self.pmf_error = mbar.calculate_free_energies_of_perturbed_states(
            self.B
        )
        return self.pmf, self.pmf_error

    def plot(
        self,
        filename: str = None,
        format: str = "svg",
        dpi: float = 300,
        cumulative: bool = False,
    ):
        """
        Plots the pmf values calculated including error bars.

        Args:
            filename (str, optional): If given, plot is saved to file. Defaults to None.
            format (str, optional): Desired File Format. Defaults to "svg".
            dpi (float, optional): desired resolution. Defaults to 300.
            cumulative (bool, optional): wheter to plot the cumulative pmf values or . Defaults to False.
        """
        energy_unit = " [kcal per mole]" if self.use_kcal else " [kJ per mole]"
        if self.in_rt:
            energy_unit = ""
        pmf_center = np.linspace(
            start=0,
            stop=(self.n_windows * self.path_interval).value_in_unit(unit.nanometer),
            num=self.n_bins,
            endpoint=False,
        )
        y = self.pmf if not cumulative else np.cumsum(self.pmf)
        y_error = self.pmf_error if not cumulative else np.cumsum(self.pmf_error)
        fig = plt.figure()
        plt.plot(pmf_center, y)
        plt.errorbar(pmf_center, y, yerr=y_error, fmt="-o")
        plt.xlim(
            pmf_center[0],
            pmf_center[-1],
        )
        plt.xlabel("Ligand distance from binding pocked [nm]")
        plt.ylabel(f"Relative free energy{energy_unit}")
        if filename:
            if not filename.endswith(f".{format}"):
                filename += f".{format}"
            fig.savefig(
                filename,
                dpi=dpi,
                format=format,
            )

    def plot_pytest(self, filename: str = None, format: str = "svg", dpi: float = 300):
        """
        plots the pmf with error bars. only use together with pytest cases in vscode. will probabely be removed later.

        Args:
            filename (str, optional): if given, a png file is exported to the filepath.. Defaults to None.
            format (str, optional): format of the saved file. Defaults to "svg".
            dpi (float, optional): resolution of the saved file. Defaults to 300.
        """
        energy_unit = " [kcal per mole]" if self.use_kcal else " [kJ per mole]"
        if self.in_rt:
            energy_unit = ""
        mtl.use("Agg")
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
        plt.ylabel(f"Relative free energy{energy_unit}")
        if filename:
            if not filename.endswith(f".{format}"):
                filename += f".{format}"
            fig.savefig(
                filename,
                dpi=dpi,
                format=format,
            )
        plt.close(fig)
