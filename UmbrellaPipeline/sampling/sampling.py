import os, time, logging
import openmm as mm
import openmmtools
from openmm import app, unit
from typing import List, Tuple

from UmbrellaPipeline.sampling import (
    add_ligand_restraint,
    update_restraint,
    serialize_system,
    serialize_state,
    add_barostat,
    add_backbone_restraints,
    extract_nonbonded_parameters,
    write_path_to_file,
)
from UmbrellaPipeline.utils import (
    execute_bash_parallel,
    display_time,
    SimulationProperties,
    SimulationSystem,
    gen_pbc_box,
)

logger = logging.getLogger(__name__)


class UmbrellaSampling:
    """
    holds all necessary information for your simulation on your local computer.
    """

    def __init__(
        self,
        properties: SimulationProperties,
        info: SimulationSystem,
        traj_write_path: str = None,
        restrain_protein_backbone: bool = False,
    ) -> None:
        """
        Args:
            properties (SimulationProperties): simulation properties object, holding temp, pressure, etc
            info (SimulationSystem): simulation system object holding psf, crd objects etc.
            path (List[unit.Quantity]): path for the ligand to walk trough.
            openmm_system (mm.openmm.System): openmm System object. Defaults to None.
            traj_write_path (str, optional): output directory where the trajectories are written to. Defaults to None.
        """
        self.simulation_properties = properties
        self.system_info = info
        self.openmm_system: mm.openmm.System
        self.simulation: app.Simulation
        self.integrator: mm.openmm.Integrator
        self.bb_restrains = restrain_protein_backbone

        self.platform = openmmtools.utils.get_fastest_platform()
        if self.platform.getName() == ("CUDA" or "OpenCL"):
            self.platformProperties = {"Precision": "mixed"}
        else:
            self.platformProperties = None

        self.traj_write_path = traj_write_path
        if not traj_write_path:
            self.traj_write_path = os.getcwd()
        self.traj_write_path = os.path.abspath(self.traj_write_path.rstrip("/"))
        self.ligand_non_bonded_parameters = []

    def run_equilibration(
        self,
        use_membrane_barostat: bool = False,
        nonbonded_method: app.forcefield = app.PME,
        nonbonded_cutoff: unit.Quantity = 1.2 * unit.nanometer,
        switch_distance: unit.Quantity = 1 * unit.nanometer,
        rigid_water: bool = True,
        constraints: app.forcefield = app.HBonds,
    ) -> mm.State:

        if not self.system_info.psf_object.boxLengths:
            gen_pbc_box(
                psf=self.system_info.psf_object,
                pos=self.system_info.crd_object.positions,
            )
        self.openmm_system = self.system_info.psf_object.createSystem(
            params=self.system_info.params,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=nonbonded_cutoff,
            switchDistance=switch_distance,
            constraints=constraints,
            rigidWater=rigid_water,
        )

        self.openmm_system = add_barostat(
            system=self.openmm_system,
            properties=self.simulation_properties,
            membrane_barostat=use_membrane_barostat,
        )

        self.integrator = mm.LangevinIntegrator(
            self.simulation_properties.temperature,
            self.simulation_properties.friction_coefficient,
            self.simulation_properties.time_step,
        )

        self.simulation = app.Simulation(
            topology=self.system_info.psf_object.topology,
            system=self.openmm_system,
            integrator=self.integrator,
            platform=self.platform,
            platformProperties=self.platformProperties,
        )

        self.simulation.context.setPositions(self.system_info.crd_object.positions)
        self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(
            self.simulation_properties.temperature
        )
        self.simulation.reporters.append(
            app.DCDReporter(
                file=f"{self.traj_write_path}/equilibration_trajcetory.dcd",
                reportInterval=self.simulation_properties.write_out_frequency,
            )
        )
        self.simulation.reporters.append(
            app.StateDataReporter(
                file=f"{self.traj_write_path}/equilibration_state.out",
                reportInterval=self.simulation_properties.write_out_frequency,
                step=True,
                time=True,
                potentialEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
            )
        )
        self.simulation.step(self.simulation_properties.n_equilibration_steps)
        state = self.simulation.context.getState(getPositions=True, getVelocities=True)
        return state

    def run_production(
        self,
        path: unit.Quantity,
        state: mm.State,
        nonbonded_method: app.forcefield = app.PME,
        nonbonded_cutoff: unit.Quantity = 1.2 * unit.nanometer,
        switch_distance: unit.Quantity = 1 * unit.nanometer,
        rigid_water: bool = True,
        constraints: app.forcefield = app.HBonds,
    ):

        if not self.system_info.psf_object.boxLengths:
            gen_pbc_box(
                psf=self.system_info.psf_object,
                pos=self.system_info.crd_object.positions,
            )
        self.openmm_system = self.system_info.psf_object.createSystem(
            params=self.system_info.params,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=nonbonded_cutoff,
            switchDistance=switch_distance,
            constraints=constraints,
            rigidWater=rigid_water,
        )
        self.openmm_system = add_ligand_restraint(
            system=self.openmm_system,
            atom_group=self.system_info.ligand_indices,
            values=[
                self.simulation_properties.force_constant,
                path[0].x,
                path[0].y,
                path[0].z,
            ],
        )
        if self.bb_restrains:
            self.openmm_system = add_backbone_restraints(
                system=self.openmm_system,
                atom_list=self.system_info.psf_object.atom_list,
            )
        self.integrator = mm.LangevinIntegrator(
            self.simulation_properties.temperature,
            self.simulation_properties.friction_coefficient,
            self.simulation_properties.time_step,
        )

        self.simulation = app.Simulation(
            topology=self.system_info.psf_object.topology,
            system=self.openmm_system,
            integrator=self.integrator,
            platform=self.platform,
            platformProperties=self.platformProperties,
        )

        self.simulation.context.setState(state=state)
        self.simulation.minimizeEnergy()
        self.simulation.step(self.simulation_properties.n_equilibration_steps)

        self.ligand_non_bonded_parameters = extract_nonbonded_parameters(
            self.openmm_system, self.system_info.ligand_indices
        )
        total_time = 0
        for window, position in enumerate(path):
            start_time = time.time()
            self.simulation.reporters.append(
                app.DCDReporter(
                    file=f"{self.traj_write_path}/production_trajcetory_window_{window}.dcd",
                    reportInterval=self.simulation_properties.write_out_frequency,
                )
            )
            self.simulation.reporters.append(
                app.StateDataReporter(
                    file=f"{self.traj_write_path}/production_state_window_{window}.out",
                    reportInterval=self.simulation_properties.write_out_frequency,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=True,
                )
            )

            self.simulation.step(self.simulation_properties.n_production_steps)

            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            logger.info(
                f"Window {window+1} of {len(path)} simulated. "
                f"Elapsed Time: {display_time(elapsed_time)}. "
                f"Elapsed total time: {display_time(total_time)}. "
                f"Estimated time until finish: {display_time((len(path) - window + 1) * elapsed_time) }."
            )

            self.simulation.reporters.clear()
            try:
                update_restraint(
                    simulation=self.simulation,
                    ligand_indices=self.system_info.ligand_indices,
                    original_parameters=self.ligand_non_bonded_parameters,
                    position=position,
                )
                self.simulation.step(self.simulation_properties.n_equilibration_steps)
            except IndexError:
                pass
        write_path_to_file(path, self.traj_write_path)


class SamplingSunGridEngine(UmbrellaSampling):
    """
    This class holds all information for running your umbrella simulation on the hydra cluster. it works entirely differnt than the UmbrellaSampling class.
    It first writes bash scirpts which it then submits to the submission system.
    """

    def __init__(
        self,
        properties: SimulationProperties,
        info: SimulationSystem,
        traj_write_path: str,
        conda_environment: str,
        mail: str = None,
        restrain_backbone: bool = False,
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
            traj_write_path=traj_write_path,
            restrain_protein_backbone=restrain_backbone,
        )
        self.hydra_working_dir: str = (
            hydra_working_dir if hydra_working_dir else os.getcwd()
        )
        self.hydra_working_dir.rstrip("/")
        self.mail: str = mail
        self.log: str = log_prefix.rstrip(".log")
        self.gpu: int = gpu
        self.conda_environment: str = conda_environment
        self.serialized_system_file: str = (
            self.hydra_working_dir + "/serialized_sys.xml"
        )
        self.serialized_state_file: str = (
            self.hydra_working_dir + "/serialized_state.rst"
        )
        self.commands: List[str] = []
        self.simulation_output: List[str] = []

    def write_sge_scripts(self, path: unit.Quantity) -> str:
        """
        Writes shell script which is then submitted to a sun grid engine cluster queue.

        Args:
            umbrella path: lambda of your simulation

        Returns:
            str: the path where the file is written to.
        """
        for window, position in enumerate(path):
            script_path = f"{self.hydra_working_dir}/run_umbrella_window_{window}.sh"
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
            c += f"python {os.path.abspath(os.path.dirname(__file__)+'/../scripts/worker_script_sun_grid_engine.py')} "

            pos = f"-x {position.x} " f"-y {position.y} " f"-z {position.z}"
            c += f" -t {self.simulation_properties.temperature.value_in_unit(unit=unit.kelvin)}"
            c += f" -dt {self.simulation_properties.time_step.value_in_unit(unit=unit.femtosecond)}"
            c += f" -fric {self.simulation_properties.friction_coefficient.value_in_unit(unit=unit.picosecond**-1)}"
            c += (
                f" -psf {self.system_info.psf_file} -crd {self.system_info.crd_file} -sys {self.serialized_system_file} -state {self.serialized_state_file}"
                f" {pos} -to {self.traj_write_path} -np {self.simulation_properties.n_production_steps} -ln {self.system_info.ligand_name}"
                f" -ne {self.simulation_properties.n_equilibration_steps} -nw {window} -io {self.simulation_properties.write_out_frequency}"
            )
            logger.info(f"{script_path} written.")

            with open(script_path, "w") as f:
                f.write(c)

            self.commands.append(f"qsub {script_path}")

        return self.commands

    def run_equilibration(
        self,
        use_membrane_barostat: bool = False,
        nonbonded_method: app.forcefield = app.PME,
        nonbonded_cutoff: unit.Quantity = 1.2 * unit.nanometer,
        switch_distance: unit.Quantity = 1 * unit.nanometer,
        rigid_water: bool = True,
        constraints: app.forcefield = app.HBonds,
    ) -> Tuple[mm.State, str]:
        """[summary]

        Args:
            use_membrane_barostat (bool, optional): [description]. Defaults to False.
            nonbonded_method (app.forcefield, optional): [description]. Defaults to app.PME.
            nonbonded_cutoff (unit.Quantity, optional): [description]. Defaults to 1.2*unit.nanometer.
            switch_distance (unit.Quantity, optional): [description]. Defaults to 1*unit.nanometer.
            rigid_water (bool, optional): [description]. Defaults to True.
            constraints (app.forcefield, optional): [description]. Defaults to app.HBonds.

        Returns:
            Tuple[mm.State, str]: [description]
        """
        state = super().run_equilibration(
            use_membrane_barostat=use_membrane_barostat,
            nonbonded_method=nonbonded_method,
            nonbonded_cutoff=nonbonded_cutoff,
            switch_distance=switch_distance,
            rigid_water=rigid_water,
            constraints=constraints,
        )
        path = serialize_state(state=state, path=self.serialized_state_file)
        return state, path

    def run_production(
        self,
        path: unit.Quantity,
        nonbonded_method: app.forcefield = app.PME,
        nonbonded_cutoff: unit.Quantity = 1.2 * unit.nanometer,
        switch_distance: unit.Quantity = 1 * unit.nanometer,
        rigid_water: bool = True,
        constraints: app.forcefield = app.HBonds,
    ) -> List[str]:
        """
        Submits the generated bash scripts to the hydra cluster

        Raises:
            FileNotFoundError: if no bash scripts are written with this pipeline, they cannot be submitted.

        Returns:
            List[str]: returns output of the simulations if no logger is used.
        """
        try:
            write_path_to_file(path, self.traj_write_path)
            if not self.system_info.psf_object.boxLengths:
                gen_pbc_box(
                    psf=self.system_info.psf_object,
                    pos=self.system_info.crd_object.positions,
                )
            self.openmm_system = self.system_info.psf_object.createSystem(
                params=self.system_info.params,
                nonbondedMethod=nonbonded_method,
                nonbondedCutoff=nonbonded_cutoff,
                switchDistance=switch_distance,
                constraints=constraints,
                rigidWater=rigid_water,
            )
            self.openmm_system = add_ligand_restraint(
                system=self.openmm_system,
                atom_group=self.system_info.ligand_indices,
                values=[
                    self.simulation_properties.force_constant,
                    path[0].x,
                    path[0].y,
                    path[0].z,
                ],
            )
            if self.bb_restrains:
                self.openmm_system = add_backbone_restraints(
                    system=self.openmm_system,
                    atom_list=self.system_info.psf_object.atom_list,
                )
            self.serialized_system_file = serialize_system(
                self.openmm_system, self.serialized_system_file
            )
            self.commands = self.write_sge_scripts(path=path)
            self.simulation_output = execute_bash_parallel(command=self.commands)

        except FileNotFoundError:
            raise FileNotFoundError(
                "Oops, something went wrong. Make sure you are logged in on the Hydra cluster and all the paths you specified are in acceptance with the best practice manual."
            )
        return self.simulation_output
