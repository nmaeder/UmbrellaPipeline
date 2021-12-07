from typing import List

from UmbrellaPipeline.sampling import (
    SamplingHydra,
    UmbrellaSimulation,
)
from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
)
from UmbrellaPipeline.path_generation import (
    GridEscapeRoom,
    TreeEscapeRoom,
    Tree,
    Grid,
    parse_params,
    gen_pbc_box,
)

import openmm as mm
import openmm.app as app
import openmm.unit as unit


class UmbrellaPipeline:
    def __init__(
        self,
        psf_file: str,
        pdb_file: str,
        toppar_stream_file: str,
        toppar_directory: str,
        ligand_residue_name: str,
        simulation_properties: SimulationProperties = SimulationProperties(),
    ) -> None:
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
    ):

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
            p = escape_room.escape_room(distance=distance_to_protein)
            self.path = escape_room.get_path_for_sampling(path_interval)

        return self.path

    def prepare_simulations(
        self,
        nonbonded_method: app.forcefield = app.PME,
        nonbonded_cutoff: unit.Quantity = 1.2 * unit.nanometer,
        rigid_water: bool = True,
        constraints: app.forcefield = app.HBonds,
    ) -> None:
        gen_pbc_box(psf=self.system_info.psf_object, pdb=self.system_info.pdb_object)
        params = self.system_info.params

        self.openmm_system = self.system_info.psf_object.createSystem(
            params=params,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=nonbonded_cutoff,
            constraints=constraints,
            rigidWater=rigid_water,
        )

    def run_simulations_cluster(
        self,
        conda_environment: str,
        trajectory_path: str,
        hydra_working_dir: str,
        mail: str = None,
        gpu: int = 1,
        log_prefix: str = "umbrella_simulation",
    ) -> None:
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
        simulation = UmbrellaSimulation(
            properties=self.simulation_parameters,
            path=self.path,
            openmm_system=self.openmm_system,
            info=self.system_info,
            traj_write_path=trajectory_path,
        )
        simulation.prepare_simulations()
        simulation.run_sampling()
