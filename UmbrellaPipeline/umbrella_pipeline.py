from typing import List
import openmm as mm
from openmm import app, unit

from UmbrellaPipeline.sampling import (
    SamplingHydra,
    UmbrellaSimulation,
)
from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
    gen_pbc_box,
)
from UmbrellaPipeline.path_generation import (
    Tree,
    Grid,
    GridEscapeRoom,
    TreeEscapeRoom,
)


class UmbrellaPipeline:
    """
    wrapper for the whole package. Runs the pipeline (almost) automatically.
    it creates a simulation_system object upon construction, which stores paths and pdb/psf objects.
    """

    def __init__(
        self,
        psf_file: str,
        pdb_file: str,
        toppar_stream_file: str,
        toppar_directory: str,
        ligand_residue_name: str,
        simulation_properties: SimulationProperties = SimulationProperties(),
    ) -> None:
        """
        Args:
            psf_file (str): psf file provided by charmm_gui
            pdb_file (str): pdb file provided by charmm_gui
            toppar_stream_file (str): toppar str file provided by charmm-gui. Don't move it around beforehand.
            toppar_directory (str): toppar directory provided by charmm-gui
            ligand_residue_name (str): name of the ligand that you want to pull out.
            simulation_properties (SimulationProperties, optional): Simulation property object. refer to the README for further info. Defaults to SimulationProperties().
        """
        self.simulation_parameters = simulation_properties
        self.system_info = SimulationSystem(
            psf_file=psf_file,
            pdb_file=pdb_file,
            toppar_directory=toppar_directory,
            toppar_stream_file=toppar_stream_file,
            ligand_name=ligand_residue_name,
        )
        self.path: List[unit.Quantity]
        self.openmm_system: mm.openmm.System

    def generate_path(
        self,
        distance_to_protein: unit.Quantity = 1.5 * unit.nanometer,
        path_interval=2 * unit.angstrom,
        use_grid: bool = False,
    ) -> List[unit.Quantity]:
        """
        Creates the path out of the protein. use_grid is not recommended.

        Args:
            distance_to_protein (unit.Quantity, optional): Distance to protein, at which to stop. Defaults to 1.5*unit.nanometer.
            path_interval ([type], optional): Stepsize of your umbrella sampling pathz. Defaults to 2*unit.angstrom.
            use_grid (bool, optional): If you want to deploy the grid version of the escape room algorithm, set to True. Not encouraged. Defaults to False.

        Returns:
            List[unit.Quantity]: path for the umbrella sampling.
        """
        if not use_grid:
            tree = Tree.from_files(
                psf=self.system_info.psf_object, pdb=self.system_info.pdb_object
            )
            start = tree.node_from_files(
                psf=self.system_info.psf_object,
                pdb=self.system_info.pdb_object,
                name=self.system_info.ligand_name,
            )
            escape_room = TreeEscapeRoom(tree=tree, start=start)
            p = escape_room.escape_room(distance=distance_to_protein)

            self.path = escape_room.get_path_for_sampling(stepsize=path_interval)

        else:
            grid = Grid.from_files(
                pdb=self.system_info.pdb_object,
                psf=self.system_info.psf_object,
                gridsize=0.2 * unit.angstrom,
            )
            start = grid.node_from_files(
                psf=self.system_info.psf_object,
                pdb=self.system_info.pdb_object,
                name=self.system_info.ligand_name,
            )
            escape_room = GridEscapeRoom(grid=grid, start=start)
            escape_room.escape_room(distance=distance_to_protein)
            self.path = escape_room.get_path_for_sampling(path_interval)

        return self.path

    def prepare_simulations(
        self,
        nonbonded_method: app.forcefield = app.PME,
        nonbonded_cutoff: unit.Quantity = 1.2 * unit.nanometer,
        rigid_water: bool = True,
        constraints: app.forcefield = app.HBonds,
    ) -> None:
        """
        This function creates the openmm_system. is called inside run_simulations/run_simulations_cluster, so no need to run it on its own.
        Args:
            nonbonded_method (app.forcefield, optional): Nonbonded method to use. PME is highly encouraged. Defaults to app.PME.
            nonbonded_cutoff (unit.Quantity, optional): nonbonded cutoff value. Defaults to 1.2*unit.nanometer.
            rigid_water (bool, optional): wheter to use rigid water. Defaults to True.
            constraints (app.forcefield, optional): constraints to use. Defaults to app.HBonds.
        """
        gen_pbc_box(psf=self.system_info.psf_object, pdb=self.system_info.pdb_object)
        params = self.system_info.params

        self.openmm_system = self.system_info.psf_object.createSystem(
            params=params,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=nonbonded_cutoff,
            constraints=constraints,
            rigidWater=rigid_water,
        )
        if self.simulation_parameters.pressure:
            barostat = mm.openmm.MonteCarloBarostat(
                self.simulation_parameters.pressure,
                self.simulation_parameters.temperature,
            )
            self.openmm_system.addForce(barostat)

    def run_simulations_cluster(
        self,
        conda_environment: str,
        trajectory_path: str,
        hydra_working_dir: str,
        mail: str = None,
        gpu: int = 1,
        log_prefix: str = "umbrella_simulation",
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
        simulation = SamplingHydra(
            properties=self.simulation_parameters,
            path=self.path,
            openmm_system=self.openmm_system,
            info=self.system_info,
            traj_write_path=trajectory_path,
            mail=mail,
            log_prefix=log_prefix,
            gpu=gpu,
            conda_environment=conda_environment,
            hydra_working_dir=hydra_working_dir,
        )
        simulation.prepare_simulations()
        simulation.run_sampling()

    def run_simulations_local(
        self,
        trajectory_path,
    ) -> None:
        """
        Prepares and runs simulations on your local machine.

        Args:
            trajectory_path ([type]): output directory for the trajectories.
        """
        simulation = UmbrellaSimulation(
            properties=self.simulation_parameters,
            path=self.path,
            openmm_system=self.openmm_system,
            info=self.system_info,
            traj_write_path=trajectory_path,
        )
        simulation.prepare_simulations()
        simulation.run_sampling()
