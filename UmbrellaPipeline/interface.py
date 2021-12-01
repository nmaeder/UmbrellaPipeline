import socket
import logging
from typing import List
from UmbrellaPipeline.sampling import (
    SamplingHydra,
    UmbrellaSimulation,
)
from UmbrellaPipeline.pathGeneration import (
    GridAStar,
    TreeAStar,
    TreeNode,
    GridNode,
    Tree,
    Grid,
    parse_params,
    gen_pbc_box,
)

logger = logging.getLogger(__name__)

import openmm as mm
import openmm.app as app
import openmm.unit as unit


class SamplingInterface:
    def __init__(
        self,
        ligand_residue_name: str,
        psf: str,
        pdb: str,
        toppar_stream_file: str,
        toppar_directory: str,
        number_eq_steps: int,
        number_prod_steps: int,
        io_frequency: int,
        temperature: unit.Quantity = 300 * unit.kelvin,
        force_constant: unit.Quantity = 100
        * unit.kilocalorie_per_mole
        / (unit.angstrom ** 2),
        time_step: unit.Quantity = 2 * unit.femtoseconds,
        pressure: unit.Quantity = 1 * unit.bar,
        friction_coeff: unit.Quantity = 1 / unit.picosecond,
        mail: str = None,
        trajectory_output_path: str = None,
        gpu: int = 1,
        use_grid: bool = False,
        path_interval: unit.Quantity = 0.5 * unit.angstrom,
        distance_to_protein: unit.Quantity = 1.5 * unit.nanometer,
        cluster: bool = False,
        log: str = None,
        conda_environment: str = None,
        hydra_working_dir: str = None,
    ) -> None:
        self.temperature = temperature
        self.pressure = pressure
        self.friction_coeff = friction_coeff
        self.force_constant = force_constant
        self.time_step = time_step
        self.ligand_residue_name = ligand_residue_name
        self.psf = psf
        self.pdb = pdb
        self.toppar_stream_file = toppar_stream_file
        self.toppar_directory = toppar_directory
        self.number_eq_steps = number_eq_steps
        self.number_prod_steps = number_prod_steps
        self.io_frequency = io_frequency
        self.method = use_grid
        self.path_interval = path_interval
        self.distance_to_protein = distance_to_protein
        self.cluster = cluster
        self.trajectory_output_path = trajectory_output_path
        self.gpu = gpu
        self.mail = mail
        self.log = log
        self.conda_environment = conda_environment
        self.hydra_working_dir = hydra_working_dir
        self.tree: Tree
        self.grid: Grid
        self.start: GridNode or TreeNode
        self.path: List[unit.Quantity]
        self.simulation: UmbrellaSimulation or SamplingHydra
        self.psf_object = app.CharmmPsfFile(self.psf)
        self.pdb_object = app.PDBFile(self.pdb)

    def generate_path(self):

        if not self.method:
            self.tree = Tree.from_files(psf=self.psf_object, pdb=self.pdb_object)
            self.start = self.tree.node_from_files(
                psf=self.psf_object, pdb=self.pdb_object, name=self.ligand_residue_name
            )
            astar = TreeAStar(tree=self.tree, start=self.start)
            p = astar.astar_3d(distance=self.distance_to_protein)
            logger.info("Shortest Path found.")
            self.path = astar.get_path_for_sampling(stepsize=self.path_interval)

        else:
            self.grid = Grid.from_files(
                pdb=self.pdb_object, psf=self.psf_object, gridsize=0.2 * unit.angstrom
            )
            self.start = self.grid.node_from_files(
                psf=self.psf_object, pdb=self.pdb_object, name=self.ligand_residue_name
            )
            astar = GridAStar(grid=self.grid, start=self.start)
            astar.astar_3d(distance=self.distance_to_protein)
            self.path = astar.get_path_for_sampling(stepsize=self.path_interval)
        return astar.get_path_for_sampling

    def prep_and_run(self):

        gen_pbc_box(psf=self.psf_object, pdb=self.pdb_object)
        system = self.psf_object.createSystem(
            params=parse_params(
                toppar_directory=self.toppar_directory,
                toppar_str_file=self.toppar_stream_file,
            ),
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.2 * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )
        if self.cluster and "hydra" in str(socket.gethostname()):
            self.simulation = SamplingHydra(
                temp=self.temperature,
                p=self.pressure,
                iofreq=self.io_frequency,
                dt=self.time_step,
                nProd=self.number_prod_steps,
                nEq=self.number_eq_steps,
                path=self.path,
                forceK=self.force_constant,
                fric=self.friction_coeff,
                system=system,
                psf=self.psf,
                pdb=self.pdb_object,
                ligand_name=self.ligand_residue_name,
                traj_write_path=self.trajectory_output_path,
                hydra_working_dir=self.hydra_working_dir,
                mail=self.mail,
                log=self.log,
                gpu=self.gpu,
                conda_environment=self.conda_environment,
            )
        else:
            self.simulation = UmbrellaSimulation(
                temp=self.temperature,
                p=self.pressure,
                iofreq=self.io_frequency,
                dt=self.time_step,
                nProd=self.number_prod_steps,
                nEq=self.number_eq_steps,
                path=self.path,
                forceK=self.force_constant,
                fric=self.friction_coeff,
                system=system,
                psf=self.psf,
                pdb=self.pdb_object,
                ligand_name=self.ligand_residue_name,
                traj_write_path=self.trajectory_output_path,
            )

        self.simulation.prepare_simulations()
        self.simulation.run_sampling()
