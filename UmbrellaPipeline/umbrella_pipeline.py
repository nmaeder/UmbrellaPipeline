from typing import List
import openmm as mm
from openmm import unit
import logging, os

from UmbrellaPipeline.sampling import (
    UmbrellaSampling,
    SamplingCluster,
)
from UmbrellaPipeline.utils import (
    SimulationParameters,
    SystemInfo,
)
from UmbrellaPipeline.path_finding import (
    Grid,
    GridEscapeRoom,
    TreeEscapeRoom,
)
from UmbrellaPipeline.analysis import PMFCalculator

logger = logging.getLogger(__name__)


class UmbrellaPipeline:
    def __init__(
        self,
        psf_file: str,
        crd_file: str,
        toppar_stream_file: str,
        toppar_directory: str,
        ligand_residue_name: str,
        system_info: SystemInfo = None,
        simulation_parameters: SimulationParameters = SimulationParameters(),
        only_run_production: bool = False,
        window_spacing: unit.Quantity = 0.1 * unit.nanometer,
        verbosity: int = logging.INFO,
    ) -> None:
        """
        This is the main class of the Umbrella pipeline. It lets you genereta a ligand unbinding pathway,
        runs umbrella sampling along this pathway and then calculates the Potential of Mean Force.

        At the moment, only the charmm force field is supported. It can either be used to be run locally on your computer,
        or it can be used to run on a clustersupported by the sun grid engine. To use this tool, setup your
        package on CHARMM-GUI and let it createthe files for openmm.

        Args:
            psf_file (str): path to the psf file generated by charmm-gui
            crd_file (str): path to the crd file generated by charmm-gui
            toppar_stream_file (str): path to the toppar.str file inside the openmm folder generated by charmm-gui
            toppar_directory (str): path to the directory with all the toppar files.
            ligand_residue_name (str): name of the ligand residue.
            simulation_parameters (SimulationParameters, optional): Simulation parameters object that contains all the parameters to be used in the sampling. See Documentation for more info. Defaults to SimulationParameters().
            only_run_production (bool, optional): Set this to true. Defaults to False.
            verbosity (Literal, optional): give either 'info', 'debug' or 'error', depending on how many info you want to receive. Defaults to 'info'.
        """
        self.simulation_parameters = simulation_parameters
        if system_info:
            self.system_info = system_info
        else:
            self.system_info = SystemInfo(
                psf_file=psf_file,
                crd_file=crd_file,
                toppar_directory=toppar_directory,
                toppar_stream_file=toppar_stream_file,
                ligand_name=ligand_residue_name,
            )
        self.path: List[unit.Quantity]
        self.openmm_system: mm.openmm.System
        self.escape_room: GridEscapeRoom or TreeEscapeRoom
        self.equilibrate = not only_run_production
        self.state: mm.State
        logger.setLevel(verbosity)
        self.window_spacing = window_spacing

    def generate_path(
        self,
        distance_to_protein: unit.Quantity = 1.5 * unit.nanometer,
        path_interval: unit.Quantity = None,
        use_grid: bool = False,
        positions: unit.Quantity = None,
    ) -> List[unit.Quantity]:
        """
        This function is the main function of this package. It generates a ligand dissociation pathway based on the steric requirements of a protein.
        It searches for the geometrically most accesible path out of a protein cavity. Give a large enough distance_to_protein, to ensure, the path is
        really leading outside of the cavity.

        It creates a TreeEscapeRoom Object from the files and infos provided in the Umbrella Pipeline object and does then generate the path. If your system
        needs different positions than the one in the .crd file, you can give them under positions:

        Args:
            distance_to_protein (unit.Quantity, optional): Distance to protein, at which to stop. Defaults to 1.5*unit.nanometer.
            path_interval ([type], optional): space between the windows of your umbrella sampling path. Defaults to 0.1*unit.nanometer.
            use_grid (bool, optional): If you want to deploy the grid version of the escape room algorithm, set to True. DEPRECATED. Defaults to False.
            positions (unit.Quantity, optinoal): Give the positions of your system. If none are given, the positions from the crd file specified in the UmbrellaPipeline object are taken. Defaults to None.
        Returns:
            List[unit.Quantity]: path for the umbrella sampling.
        """
        if positions != None:
            pos = positions
        else:
            pos = self.system_info.crd_object.positions

        path_interval = path_interval if path_interval else self.window_spacing

        if not use_grid:
            self.escape_room = TreeEscapeRoom.from_files(
                system_info=self.system_info, positions=pos
            )
            self.escape_room.find_path(
                resolution=0.1 * unit.angstrom,
                wall_radius=1.2 * unit.angstrom,
                distance=distance_to_protein,
            )
            self.path = self.escape_room.get_path_for_sampling(stepsize=path_interval)

        else:
            grid = Grid.from_files(
                crd=self.system_info.crd_object,
                psf=self.system_info.psf_object,
                gridsize=0.02 * unit.nanometer,
            )
            start = grid.node_from_files(
                psf=self.system_info.psf_object,
                crd=self.system_info.crd_object,
                name=self.system_info.ligand_name,
            )
            self.escape_room = GridEscapeRoom(grid=grid, start=start)
            self.escape_room.escape_room(distance=distance_to_protein)
            self.path = self.escape_room.get_path_for_sampling(path_interval)
        return self.path

    def run_analysis(self, trajectory_path: str):
        calculator = PMFCalculator(
            simulation_parameters=self.simulation_parameters,
            system_info=self.system_info,
            trajectory_directory=trajectory_path,
            original_path_interval=self.window_spacing,
        )
        return calculator.calculate_pmf()

    def run_simulations_sun_grid_engine(
        self,
        conda_environment: str,
        trajectory_path: str,
        sge_working_dir: str,
        mail: str = None,
        gpu: int = 1,
        log_prefix: str = "umbrella_simulation",
        window_spacing: unit.Quantity = 0.1 * unit.nanometer,
        restrain_backbone: bool = False,
    ) -> None:
        """
        This is a nobrainer function if you want to run the package with default parameters on a cluster supported by the sun grid engine.


        Args:
            conda_environment (str): name of the conda environment you installed this package and are working on
            trajectory_path (str): the path were the trajectories are stored.
            sge_working_dir (str): the directory from which you run this.
            mail (str, optional): give your email if you want to receive status updates from the cluster. Defaults to None.
            gpu (int, optional): wheter to deploy gpus on the cluster. Defaults to 1.
            log_prefix (str, optional): prefix for the log files created. Defaults to "umbrella_simulation".
            window_spacing (unit.Quantity, optional): The spacing between each umbrella Window. Defaults to 0.1 * unit.nanometer.
            restrain_backbone (bool, optional): Wheter to restrain the backbone atoms of the protein during the simulations or not. Defaults to False.
        """
        simulation = SamplingCluster(
            simulation_parameter=self.simulation_parameters,
            system_info=self.system_info,
            traj_write_path=trajectory_path,
            mail=mail,
            log_prefix=log_prefix,
            gpu=gpu,
            conda_environment=conda_environment,
            sge_working_dir=sge_working_dir,
            restrain_backbone=restrain_backbone,
        )

        state, simulation.serialized_state_file = simulation.run_equilibration()
        logger.info("Equilibration finished.")
        self.path = self.generate_path(
            positions=state.getPositions(),
            path_interval=window_spacing,
        )
        logger.info("path for production created.")
        simulation.run_production(path=self.path, state=state)
        logger.info("production simulation started!")

    def run_simulations_local(
        self,
        trajectory_path: str = os.getcwd(),
        window_spacing: unit.Quantity = 0.1 * unit.nanometer,
        restrain_backbone: bool = False,
    ) -> None:
        """
                This is a nobrainer function for this package. It lets you run the equilbiration, path finding, production and analysis on the default parameters.
                It

                Args:
                    trajectory_path (str): output directory, where the simulation trajectories are stored. Defaults to os.getcwd().
        wind        window_spacing (unit.Quantity, optional): The spacing between each umbrella Window. Defaults to 0.1 * unit.nanometer.
                    restrain_backbone (bool, optional): Wheter to restrain the backbone atoms of the protein during the simulations or not. Defaults to False.
        """
        simulation = UmbrellaSampling(
            simulation_parameters=self.simulation_parameters,
            system_info=self.system_info,
            traj_write_path=trajectory_path,
            restrain_protein_backbone=restrain_backbone,
        )
        self.state = simulation.run_equilibration(use_membrane_barostat=True)
        logger.info("Equilibration finished.")
        self.path = self.generate_path(
            positions=self.state.getPositions(),
            path_interval=window_spacing,
        )
        logger.info("path for production created.")
        simulation.run_production(path=self.path, state=self.state)
        logger.info("production simulation started!")
        self.run_analysis(trajectory_path=trajectory_path)
