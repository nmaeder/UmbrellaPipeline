import torch
import os

import openmm.unit as unit
import openmm as mm
import openmm.app as app
from tqdm import tqdm
from typing import (
    List,
    Tuple,
)
from UmbrellaPipeline.sampling.sampling_helper import add_harmonic_restraint
from UmbrellaPipeline.path_generation.node import Node
import openmmtools
from UmbrellaPipeline.utils import (
    execute_bash,
    execute_bash_parallel,
)
from UmbrellaPipeline.path_generation.path_helper import (
    get_residue_indices,
)
from UmbrellaPipeline.utils import SimulationProperties
from UmbrellaPipeline.utils import SimulationSystem


class UmbrellaSimulation:
    def __init__(
        self,
        properties: SimulationProperties,
        info: SimulationSystem,
        path: List[unit.Quantity],
        openmm_system: mm.openmm.System = None,
        traj_write_path: str = None,
    ) -> None:
        self.sim_props = properties
        self.sys_info = info
        self.lamdas = len(path)
        self.path = path
        self.openmm_system = openmm_system
        self.traj_write_path = traj_write_path
        self.simulation: app.Simulation
        self.integrator: mm.openmm.Integrator

        if torch.cuda.is_available():
            self.platform = mm.Platform.getPlatformByName("CUDA")
            self.platformProperties = {"Precision": "mixed"}
        else:
            self.platform = mm.Platform.getPlatformByName("CPU")
            self.platformProperties = None

    def prepare_simulations(self) -> None:
        """
        Adds a harmonic restraint to the residue with the name given in self.name to the first position in self.path.
        Also changes self.integrator so it can be used on self.system.

        Returns:
            mm.openmm.Integrator: Integrator for the system. it changes self.integrator.
        """
        add_harmonic_restraint(
            system=self.openmm_system,
            atom_group=self.sys_info.ligand_indices,
            values=[
                self.sim_props.force_constant,
                self.path[0].x,
                self.path[0].y,
                self.path[0].z,
            ],
        )

        self.integrator = openmmtools.integrators.LangevinIntegrator(
            temperature=self.sim_props.temperature,
            collision_rate=self.sim_props.friction_coefficient,
            timestep=self.sim_props.time_step,
        )

        self.simulation = app.Simulation(
            topology=self.sys_info.psf_object.topology,
            system=self.openmm_system,
            integrator=self.integrator,
            platform=self.platform,
            platformProperties=self.platformProperties,
        )

    def run_sampling(self):
        """
        runs the actual umbrella sampling.
        """
        orgCoords = open(file=f"{self.traj_write_path}coordinates.dat", mode="w")
        orgCoords.write("lamda, x0, y0, z0\n")

        self.simulation.context.setPositions(self.sys_info.pdb_object.positions)

        for window in range(self.lamdas):
            orgCoords.write(
                f"{window}, {self.simulation.context.getParameter('x0')}, {self.simulation.context.getParameter('y0')}, {self.simulation.context.getParameter('z0')}\n"
            )
            self.simulation.minimizeEnergy()
            self.simulation.context.setVelocitiesToTemperature(
                self.sim_props.temperature
            )
            self.simulation.step(self.sim_props.n_equilibration_steps)
            fileHandle = open(f"{self.traj_write_path}traj_{window}.dcd", "bw")
            dcdFile = app.dcdfile.DCDFile(
                file=fileHandle,
                topology=self.simulation.topology,
                dt=self.sim_props.time_step,
            )
            num = (
                self.sim_props.n_production_steps / self.sim_props.write_out_frequency
                if self.sim_props.write_out_frequency == 0
                else 1
            )
            num2 = (
                self.sim_props.n_production_steps
                if self.sim_props.write_out_frequency == 0
                else self.sim_props.write_out_frequency
            )
            for i in tqdm(range(int(num))):
                self.simulation.step(num2)
                dcdFile.writeModel(
                    self.simulation.context.getState(getPositions=True).getPositions()
                )
            fileHandle.close()
            try:
                self.simulation.context.setParameter(
                    "x0",
                    self.path[window + 1].x.in_units_of(self.pdb_object.positions.unit),
                )
                self.simulation.context.setParameter(
                    "y0",
                    self.path[window + 1].y.in_units_of(self.pdb_object.positions.unit),
                )
                self.simulation.context.setParameter(
                    "z0",
                    self.path[window + 1].z.in_units_of(self.pdb_object.positions.unit),
                )
            except IndexError:
                pass
        orgCoords.close()


class SamplingHydra(UmbrellaSimulation):
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
        super().__init__(
            properties=properties,
            info=info,
            path=path,
            openmm_system=openmm_system,
            traj_write_path=traj_write_path,
        )
        self.hydra_working_dir = hydra_working_dir if hydra_working_dir else os.getcwd()
        self.mail = mail
        self.log = log_prefix.rstrip(".log")
        self.gpu = gpu
        self.conda_environment = conda_environment
        self.serialized_sys = self.hydra_working_dir + "/serialized_sys.xml"
        self.serialized_int = self.hydra_working_dir + "/serialized_int.xml"

    def write_hydra_scripts(
        self, window: int, serializedSystem: str, serializedIntegrator: str
    ):
        command = "#$ -S /bin/bash\n#$ -m e\n"
        if self.mail:
            command += f"#$ -M {self.mail}\n"
            command += "#$ -pe smp 1\n"
        command += "#$ -j y\n"
        if self.gpu:
            command += f"#$ -l gpu={self.gpu}\n"
        if self.log:
            command += f"#$ -o {self.log}_{window}.log\n"
        command += "#$ -p -1000\n"
        command += "#$ -cwd\n"
        command += "\n"
        command += "hostname\n"
        command += f"conda activate {self.conda_environment}\n"
        command += f"python {os.path.abspath(os.path.dirname(__file__)+'/../scripts/simulation_hydra.py')} "
        pos = (
            f"-x {self.path[window][0].value_in_unit(self.sys_info.pdb_object.positions.unit)} "
            f"-y {self.path[window][1].value_in_unit(self.sys_info.pdb_object.positions.unit)} "
            f"-z {self.path[window][2].value_in_unit(self.sys_info.pdb_object.positions.unit)}"
        )
        command += f" -t {self.sim_props.temperature.value_in_unit(unit=unit.kelvin)}"
        command += (
            f" -dt {self.sim_props.time_step.value_in_unit(unit=unit.femtosecond)}"
        )
        command += f" -fric {self.sim_props.friction_coefficient.value_in_unit(unit=unit.picosecond**-1)}"

        command += (
            f" -psf {self.sys_info.psf_file} -pdb {self.sys_info.pdb_file} -sys {serializedSystem}"
            f" {pos} -int {serializedIntegrator} -to {self.traj_write_path}"
            f" -ne {self.sim_props.n_equilibration_steps} -np {self.sim_props.n_production_steps} -nw {window} -io {self.sim_props.write_out_frequency}"
        )

        with open(f"{self.hydra_working_dir}/run_umbrella_{window}.sh", "w") as f:
            f.write(command)
        return f"{self.hydra_working_dir}/run_umbrella_{window}.sh"

    def prepare_simulations(self) -> Tuple[str]:
        super().prepare_simulations()
        with open(file=self.serialized_sys, mode="w") as f:
            f.write(mm.openmm.XmlSerializer.serialize(self.openmm_system))
        with open(file=self.serialized_int, mode="w") as f:
            f.write(mm.openmm.XmlSerializer.serialize(self.integrator))
        for window in range(self.lamdas):
            newfile = self.write_hydra_scripts(
                window=window,
                serializedSystem=self.serialized_sys,
                serializedIntegrator=self.serialized_int,
            )
        return self.serialized_sys, self.serialized_int

    def run_sampling(self):
        command: List[str] = []
        out: List[str] = []
        for window in range(self.lamdas):
            command.append(f"qsub {self.hydra_working_dir}/run_umbrella_{window}.sh")
        try:
            out = execute_bash_parallel(command=command)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Oops, something went wrong. Make sure you are logged in on the Hydra cluster and all the paths you specified are in acceptance with the best practice manual."
            )
        return out


class SamplingLSF(UmbrellaSimulation):
    "TODO"
    pass

    def write_lsf_scripts(self):
        "TODO"
        pass
