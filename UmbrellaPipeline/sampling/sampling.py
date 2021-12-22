import os, time, logging, stat
import openmm.unit as unit
import openmm as mm
import openmm.app as app
import openmmtools
from typing import List

from UmbrellaPipeline.sampling import (
    add_harmonic_restraint,
    update_restraint,
    serialize_system,
)
from UmbrellaPipeline.utils import (
    execute_bash_parallel,
    display_time,
    SimulationProperties,
    SimulationSystem,
)

logger = logging.getLogger(__name__)


class UmbrellaSimulation:
    """
    holds all necessary information for your simulation on your local computer.
    """

    def __init__(
        self,
        properties: SimulationProperties,
        info: SimulationSystem,
        path: List[unit.Quantity],
        openmm_system: mm.openmm.System,
        traj_write_path: str = None,
    ) -> None:
        """
        Args:
            properties (SimulationProperties): simulation properties object, holding temp, pressure, etc
            info (SimulationSystem): simulation system object holding psf, pdb objects etc.
            path (List[unit.Quantity]): path for the ligand to walk trough.
            openmm_system (mm.openmm.System): openmm System object. Defaults to None.
            traj_write_path (str, optional): output directory where the trajectories are written to. Defaults to None.
        """
        self.simulation_properties = properties
        self.system_info = info
        self.lamdas = len(path)
        self.path = path
        self.openmm_system = openmm_system
        self.simulation: app.Simulation
        self.integrator: mm.openmm.Integrator

        self.platform = openmmtools.utils.get_fastest_platform()
        if self.platform.getName() == ("CUDA" or "OpenCL"):
            self.platformProperties = {"Precision": "mixed"}
        else:
            self.platformProperties = None

        self.traj_write_path = traj_write_path
        if not traj_write_path:
            self.traj_write_path = os.getcwd()
        if self.traj_write_path.endswith("/"):
            self.traj_write_path.rstrip("/")

        self.ligand_non_bonded_parameters = []

        for force in self.openmm_system.getForces():
            if type(force).__name__ == "NonbondedForce":
                for index in self.system_info.ligand_indices:
                    self.ligand_non_bonded_parameters.append(
                        force.getParticleParameters(index)
                    )

    def prepare_simulations(self) -> None:
        """
        Adds a harmonic restraint to the residue with the name given in self.name to the first position in self.path.
        Also changes self.integrator so it can be used on self.system.

        Returns:
            mm.openmm.Integrator: Integrator for the system. it changes self.integrator.
        """
        add_harmonic_restraint(
            system=self.openmm_system,
            atom_group=self.system_info.ligand_indices,
            values=[
                self.simulation_properties.force_constant,
                self.path[0].x,
                self.path[0].y,
                self.path[0].z,
            ],
        )

        self.integrator = openmmtools.integrators.LangevinIntegrator(
            temperature=self.simulation_properties.temperature,
            collision_rate=self.simulation_properties.friction_coefficient,
            timestep=self.simulation_properties.time_step,
        )

        self.simulation = app.Simulation(
            topology=self.system_info.psf_object.topology,
            system=self.openmm_system,
            integrator=self.integrator,
            platform=self.platform,
            platformProperties=self.platformProperties,
        )

    def run_sampling(self):
        """
        Runs the actual umbrella sampling on your local machine.
        """
        orgCoords = open(file=f"{self.traj_write_path}/coordinates.dat", mode="w")
        orgCoords.write("lamda, x0, y0, z0\n")

        self.simulation.context.setPositions(self.system_info.pdb_object.positions)
        self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(
            self.simulation_properties.temperature
        )
        for window in range(self.lamdas):
            orgCoords.write(
                f"{window}, {self.simulation.context.getParameter('x0')}, {self.simulation.context.getParameter('y0')}, {self.simulation.context.getParameter('z0')}\n"
            )

            self.simulation.step(self.simulation_properties.n_equilibration_steps)
            fileHandle = open(f"{self.traj_write_path}/traj_{window}.dcd", "bw")
            dcdFile = app.dcdfile.DCDFile(
                file=fileHandle,
                topology=self.simulation.topology,
                dt=self.simulation_properties.time_step,
            )
            total_time = 0
            for i in range(self.simulation_properties.number_of_frames):
                start_time = time.time()
                self.simulation.step(self.simulation_properties.write_out_frequency)
                dcdFile.writeModel(
                    self.simulation.context.getState(getPositions=True).getPositions()
                )
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                logger.info(
                    f"Step {i+1} of {self.simulation_properties.number_of_frames} simulated. "
                    f"Elapsed Time: {display_time(elapsed_time)}. "
                    f"Elapsed total time: {display_time(total_time)}. "
                    f"Estimated time until finish: {display_time((self.simulation_properties.number_of_frames - i -1) * total_time) }."
                )
            fileHandle.close()
            try:
                update_restraint(
                    simulation=self.simulation,
                    ligand_indices=self.system_info.ligand_indices,
                    original_parameters=self.ligand_non_bonded_parameters,
                    path=self.path,
                    window=window + 1,
                )
            except IndexError:
                pass
        orgCoords.close()


class SamplingHydra(UmbrellaSimulation):
    """
    This class holds all information for running your umbrella simulation on the hydra cluster. it works entirely differnt than the UmbrellaSimulation class.
    It first writes bash scirpts which it then submits to the submission system.
    """

    def __init__(
        self,
        properties: SimulationProperties,
        path: List[unit.Quantity],
        openmm_system: mm.openmm.System,
        info: SimulationSystem,
        traj_write_path: str,
        conda_environment: str,
        mail: str = None,
        log_prefix: str = "umbrella_simulation",
        gpu: str = 1,
        hydra_working_dir: str = None,
    ) -> None:
        """
        Args:
            properties (SimulationProperties): simulation_property object.
            path (List[unit.Quantity]): path for the ligand to walk through.
            openmm_system (mm.openmm.System): openmm system of your simulation.
            info (SimulationSystem): simulation system object.
            traj_write_path (str): output directory where the trajectories will be written to.
            conda_environment (str): name of the conda environment where this package and all its dependencies are installed.
            mail (str, optional): [description]. if given, a mail will be sent to this adress if a production run has finished.
            log_prefix (str, optional): [description]. give if you want a certain path for your log file.
            gpu (str, optional): how much cpu cores to use. on hydra choose 1. Defaults to 1.
            hydra_working_dir (str, optional): the working dir from which you are runnnig this package. Defaults to None.
        """
        super().__init__(
            properties=properties,
            info=info,
            path=path,
            openmm_system=openmm_system,
            traj_write_path=traj_write_path,
        )
        self.hydra_working_dir: str = (
            hydra_working_dir if hydra_working_dir else os.getcwd()
        )
        if self.hydra_working_dir.endswith("/"):
            self.hydra_working_dir.rstrip("/")
        self.mail: str = mail
        self.log: str = log_prefix.rstrip(".log")
        self.gpu: int = gpu
        self.conda_environment: str = conda_environment
        self.serialized_system_file: str = (
            self.hydra_working_dir + "/serialized_sys.xml"
        )
        self.commands: List[str] = []
        self.simulation_output: List[str] = []

    def write_hydra_scripts(self, window: int, serializedSystem: str) -> str:
        """
        Writes shell script which is then submitted to the cluster que.

        Args:
            window (int): lambda of your simulation
            serializedSystem (str): path to the serialized system file.

        Returns:
            str: the path where the file is written to.
        """
        path = f"{self.hydra_working_dir}/run_umbrella_{window}.sh"
        c = "#$ -S /bin/bash\n#$ -m e\n"
        c += "#$ -j y\n"
        c += "#$ -cwd\n"
        c += "#$ -p -1000\n"
        if self.gpu:
            c += f"#$ -l gpu={self.gpu}\n"
        if self.log:
            c += f"#$ -o {self.log}_{window}.log\n\n"
        if self.mail:
            c += f"#$ -M {self.mail}\n"
            c += "#$ -pe smp 1\n"
        c += "hostname\n"
        c += f"conda activate {self.conda_environment}\n"
        c += f"python {os.path.abspath(os.path.dirname(__file__)+'/../scripts/simulation_hydra.py')} "
        pos = (
            f"-x {self.path[window][0].value_in_unit(self.system_info.pdb_object.positions.unit)} "
            f"-y {self.path[window][1].value_in_unit(self.system_info.pdb_object.positions.unit)} "
            f"-z {self.path[window][2].value_in_unit(self.system_info.pdb_object.positions.unit)}"
        )
        c += f" -t {self.simulation_properties.temperature.value_in_unit(unit=unit.kelvin)}"
        c += f" -dt {self.simulation_properties.time_step.value_in_unit(unit=unit.femtosecond)}"
        c += f" -fric {self.simulation_properties.friction_coefficient.value_in_unit(unit=unit.picosecond**-1)}"

        c += (
            f" -psf {self.system_info.psf_file} -pdb {self.system_info.pdb_file} -sys {serializedSystem}"
            f" {pos} -to {self.traj_write_path} -nf {self.simulation_properties.number_of_frames} -ln {self.system_info.ligand_name}"
            f" -ne {self.simulation_properties.n_equilibration_steps} -nw {window} -io {self.simulation_properties.write_out_frequency}"
        )
        logger.info(f"{path} written.")

        with open(path, "w") as f:
            f.write(c)

        return path

    def prepare_simulations(self) -> None:
        """
        Prepares simulation, by creating all necessary objects and writing the bash scripts which are then submitted to the cluster.
        """
        add_harmonic_restraint(
            system=self.openmm_system,
            atom_group=self.system_info.ligand_indices,
            values=[
                self.simulation_properties.force_constant,
                self.path[0].x,
                self.path[0].y,
                self.path[0].z,
            ],
        )

        serialize_system(system=self.openmm_system, path=self.serialized_system_file)

        for window in range(self.lamdas):
            newfile = self.write_hydra_scripts(
                window=window, serializedSystem=self.serialized_system_file
            )
            self.commands.append(f"qsub {newfile}")

    def write_path_to_file(self) -> str:
        """
        Writes the restrain coordinates to a file, so analysis is still possible without the umbrellapipeline object

        Returns:
            str: path for the coordinates file
        """
        path = f"{self.traj_write_path}/coordinates.dat"
        orgCoords = open(file=path, mode="w")
        orgCoords.write(f"lamda, x0, y0, z0, all in units of {self.path[0].unit}\n")
        for window in range(self.lamdas):
            orgCoords.write(
                f"{window},{self.path[window].x},{self.path[window].y},{self.path[window].z}\n"
            )
        return path

    def run_sampling(self) -> List[str]:
        """
        Submits the generated bash scripts to the hydra cluster

        Raises:
            FileNotFoundError: if no bash scripts are written with this pipeline, they cannot be submitted.

        Returns:
            List[str]: returns output of the simulations if no logger is used.
        """
        try:
            self.write_path_to_file()
            self.simulation_output = execute_bash_parallel(command=self.commands)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Oops, something went wrong. Make sure you are logged in on the Hydra cluster and all the paths you specified are in acceptance with the best practice manual."
            )
        return self.simulation_output
