import torch
import os, time, logging

import openmm.unit as unit
import openmm as mm
import openmm.app as app
from typing import List
import openmmtools
from UmbrellaPipeline import (
    execute_bash_parallel,
    SimulationProperties,
    SimulationSystem,
    add_harmonic_restraint,
    display_time,
)

logger = logging.getLogger(__name__)


class UmbrellaSimulation:
    def __init__(
        self,
        properties: SimulationProperties,
        info: SimulationSystem,
        path: List[unit.Quantity],
        openmm_system: mm.openmm.System = None,
        traj_write_path: str = None,
    ) -> None:
        self.simulation_properties = properties
        self.system_info = info
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
        runs the actual umbrella sampling.
        """
        orgCoords = open(file=f"{self.traj_write_path}coordinates.dat", mode="w")
        orgCoords.write("lamda, x0, y0, z0\n")

        self.simulation.context.setPositions(self.system_info.pdb_object.positions)

        for window in range(self.lamdas):
            orgCoords.write(
                f"{window}, {self.simulation.context.getParameter('x0')}, {self.simulation.context.getParameter('y0')}, {self.simulation.context.getParameter('z0')}\n"
            )
            self.simulation.minimizeEnergy()
            self.simulation.context.setVelocitiesToTemperature(
                self.simulation_properties.temperature
            )
            self.simulation.step(self.simulation_properties.n_equilibration_steps)
            fileHandle = open(f"{self.traj_write_path}traj_{window}.dcd", "bw")
            dcdFile = app.dcdfile.DCDFile(
                file=fileHandle,
                topology=self.simulation.topology,
                dt=self.simulation_properties.time_step,
            )

            num = (
                self.simulation_properties.n_production_steps
                / self.simulation_properties.write_out_frequency
            )
            ttot = 0
            for i in range(int(num)):
                st = time.time()
                self.simulation.step(self.simulation_properties.write_out_frequency)
                dcdFile.writeModel(
                    self.simulation.context.getState(getPositions=True).getPositions()
                )
                t = time.time() - st
                ttot += t
                logger.info(
                    f"Step {i+1} of {num} simulated. "
                    f"Elapsed Time: {display_time(t)}. "
                    f"Elapsed total time: {display_time(ttot)}. "
                    f"Estimated time until finish: {display_time((num - i -1) * t) }."
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

    def write_hydra_scripts(self, window: int, serializedSystem: str):
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
            f"-x {self.path[window][0].value_in_unit(self.system_info.pdb_object.positions.unit)} "
            f"-y {self.path[window][1].value_in_unit(self.system_info.pdb_object.positions.unit)} "
            f"-z {self.path[window][2].value_in_unit(self.system_info.pdb_object.positions.unit)}"
        )
        command += f" -t {self.simulation_properties.temperature.value_in_unit(unit=unit.kelvin)}"
        command += f" -dt {self.simulation_properties.time_step.value_in_unit(unit=unit.femtosecond)}"
        command += f" -fric {self.simulation_properties.friction_coefficient.value_in_unit(unit=unit.picosecond**-1)}"

        command += (
            f" -psf {self.system_info.psf_file} -pdb {self.system_info.pdb_file} -sys {serializedSystem}"
            f" {pos} -to {self.traj_write_path}"
            f" -ne {self.simulation_properties.n_equilibration_steps} -np {self.simulation_properties.n_production_steps} -nw {window} -io {self.simulation_properties.write_out_frequency}"
        )
        logger.info()

        with open(f"{self.hydra_working_dir}/run_umbrella_{window}.sh", "w") as f:
            f.write(command)
        logger.info(f"Bash script written for Simulation window {window}.")
        return f"{self.hydra_working_dir}/run_umbrella_{window}.sh"

    def prepare_simulations(self) -> str:
        super().prepare_simulations()
        with open(file=self.serialized_sys, mode="w") as f:
            f.write(mm.openmm.XmlSerializer.serialize(self.openmm_system))
        for window in range(self.lamdas):
            newfile = self.write_hydra_scripts(
                window=window,
                serializedSystem=self.serialized_sys,
            )
        return self.serialized_sys

    def run_sampling(self):
        command: List[str] = []
        out: List[str] = []
        if not self.traj_write_path.endswith("/"):
            self.traj_write_path += "/"
        f = open(f"{self.traj_write_path}coordinates.dat", "w")
        f.write("window, x0, y0, z0\n")
        for window in range(self.lamdas):
            f.write(
                f"{window}, {self.path[window][0]}, {self.path[window][1]}, {self.path[window][2]}"
            )
            command.append(f"qsub {self.hydra_working_dir}/run_umbrella_{window}.sh")
        try:
            out = execute_bash_parallel(command=command)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Oops, something went wrong. Make sure you are logged in on the Hydra cluster and all the paths you specified are in acceptance with the best practice manual."
            )
        f.close()
        return out


class SamplingLSF(UmbrellaSimulation):
    "TODO"
    pass

    def write_lsf_scripts(self):
        "TODO"
        pass
