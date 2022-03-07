from typing import List
import logging
import openmm as mm
from openmm import unit

from UmbrellaPipeline.sampling import (
    UmbrellaSampling,
    SamplingSunGridEngine,
)
from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
)
from UmbrellaPipeline.path_finding import (
    Tree,
    Grid,
    GridEscapeRoom,
    TreeEscapeRoom,
)

logger = logging.getLogger(__name__)


class UmbrellaPipeline:
    """
    wrapper for the whole package. Runs the pipeline (almost) automatically.
    it creates a simulation_system object upon construction, which stores paths and crd/psf objects.
    """

    def __init__(
        self,
        psf_file: str,
        crd_file: str,
        toppar_stream_file: str,
        toppar_directory: str,
        ligand_residue_name: str,
        simulation_properties: SimulationProperties = SimulationProperties(),
        only_run_production: bool = False,
    ) -> None:
        """
        Args:
            psf_file (str): psf file provided by charmm_gui
            crd_file (str): crd file provided by charmm_gui
            toppar_stream_file (str): toppar str file provided by charmm-gui. Don't move it around beforehand.
            toppar_directory (str): toppar directory provided by charmm-gui
            ligand_residue_name (str): name of the ligand that you want to pull out.
            simulation_properties (SimulationProperties, optional): Simulation property object. refer to the README for further info. Defaults to SimulationProperties().
        """
        self.simulation_parameters = simulation_properties
        self.system_info = SimulationSystem(
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

    def generate_path(
        self,
        distance_to_protein: unit.Quantity = 1.5 * unit.nanometer,
        path_interval=0.1 * unit.nanometer,
        use_grid: bool = False,
        positions: unit.Quantity = None,
        system=None,
    ) -> List[unit.Quantity]:
        """
        Creates the path out of the protein. use_grid is not recommended.

        Args:
            distance_to_protein (unit.Quantity, optional): Distance to protein, at which to stop. Defaults to 1.5*unit.nanometer.
            path_interval ([type], optional): Stepsize of your umbrella sampling path. Defaults to 0.2*unit.nanometer.
            use_grid (bool, optional): If you want to deploy the grid version of the escape room algorithm, set to True. Not encouraged. Defaults to False.

        Returns:
            List[unit.Quantity]: path for the umbrella sampling.
        """
        if positions != None:
            pos = positions
        else:
            pos = self.system_info.crd_object.positions

        if not use_grid:
            tree = Tree.from_files(psf=self.system_info.psf_object, positions=pos)
            start = tree.node_from_coords(
                positions=pos,
                psf=self.system_info.psf_object,
                name=self.system_info.ligand_name,
                masses=system,
            )
            self.escape_room = TreeEscapeRoom(
                tree=tree, start=start, stepsize=0.05 * unit.nanometer
            )
            self.escape_room.escape_room(distance=distance_to_protein)
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

    def run_simulations_sun_grid_engine(
        self,
        conda_environment: str,
        trajectory_path: str,
        hydra_working_dir: str,
        mail: str = None,
        gpu: int = 1,
        log_prefix: str = "umbrella_simulation",
        membrane_barostat: bool = False,
        window_spacing: unit.Quantity = 1 * unit.angstrom,
        restrain_backbone: bool = False,
    ) -> None:
        """
        Prepares and runs simulations on a cluster.

        Args:
            conda_environment (str): name of the conda environment, this package and all dependencies are installed.
            trajectory_path (str): output directory where the trajectories are stored.
            hydra_working_dir (str): if working on the hydra cluster, give the working dir you are running your script from.
            mail (str, optional): if you want to receive emails, when a simulation ended, give your email-adress here. Defaults to None.
            gpu (int, optional): set to one if you want to use CUDA. Defaults to 1.
            log_prefix (str, optional): name of the log files. Defaults to "umbrella_simulation".
        """
        simulation = SamplingSunGridEngine(
            properties=self.simulation_parameters,
            info=self.system_info,
            traj_write_path=trajectory_path,
            mail=mail,
            log_prefix=log_prefix,
            gpu=gpu,
            conda_environment=conda_environment,
            hydra_working_dir=hydra_working_dir,
            restrain_backbone=restrain_backbone,
        )

        state, simulation.serialized_state_file = simulation.run_equilibration(
            use_membrane_barostat=membrane_barostat
        )
        logger.info("Equilibration finished.")
        self.path = self.generate_path(
            positions=state.getPositions(),
            system=simulation.simulation.context.getSystem(),
            path_interval=window_spacing,
        )
        logger.info("path for production created.")
        simulation.run_production(path=self.path, state=state)
        logger.info("production simulation started!")

    def run_simulations_local(
        self,
        trajectory_path: str,
        window_spacing: unit.Quantity = 1 * unit.angstrom,
        restrain_backbone: bool = False,
    ) -> None:
        """
        Prepares and runs simulations on your local machine.

        Args:
            trajectory_path (str): output directory for the trajectories.
        """
        simulation = UmbrellaSampling(
            properties=self.simulation_parameters,
            info=self.system_info,
            traj_write_path=trajectory_path,
            restrain_protein_backbone=restrain_backbone,
        )
        self.state = simulation.run_equilibration(use_membrane_barostat=True)
        logger.info("Equilibration finished.")
        self.path = self.generate_path(
            positions=self.state.getPositions(),
            system=simulation.simulation.context.getSystem(),
            path_interval=window_spacing,
        )
        logger.info("path for production created.")
        simulation.run_production(path=self.path, state=self.state)
        logger.info("production simulation started!")
