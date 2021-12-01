import torch
import os
import logging
import openmm.unit as unit
import openmm as mm
import openmm.app as app
from tqdm import tqdm
from typing import List

from UmbrellaPipeline.sampling.samplingHelper import add_harmonic_restraint
from UmbrellaPipeline.pathGeneration.node import Node
import openmmtools
from UmbrellaPipeline.utils.bash import (
    execute_bash,
    execute_bash_parallel,
)
from UmbrellaPipeline.pathGeneration.pathHelper import (
    get_residue_indices,
)


logger = logging.getLogger(__name__)


class UmbrellaSimulation:
    """
    Class for the sampling part of the Umbrella Pipeline. Holds all information necessary for the Umbrella Sampling.
    """

    def __init__(
        self,
        psf: str or app.CharmmPsfFile,
        pdb: str or app.PDBFile,
        path: List[unit.Quantity],
        ligand_name: str,
        temp: unit.Quantity = 310 * unit.kelvin,
        p: unit.Quantity = 1 * unit.bar,
        iofreq: int = 2500,
        dt: unit.Quantity = 2 * unit.femtosecond,
        num_prod: int = 500000,
        num_eq: int = 50000,
        method: str = "pulling",
        force_const: unit.Quantity = 100
        * unit.kilocalorie_per_mole
        / (unit.angstrom ** 2),
        fric: unit.Quantity = 1 / unit.picosecond,
        system: mm.openmm.System = None,
        traj_write_path: str = None,
    ) -> None:
        """
        Args:
            temp (unit.Quantity, optional): Simulation Temperature. Defaults to 310*unit.kelvin.
            p (unit.Quantity, optional): Simulation pressure, if 0 NVT is sampled instead of NPT, not encouraged. Defaults to 1*unit.bar.
            iofreq (int, optional): Interval after which a snapshot is saved to the trajectory. Defaults to 2500.
            dt (unit.Quantity, optional): simulation timestep. Defaults to 2*unit.femtosecond.
            nWin (int, optional): number of simulation windows. Defaults to 20.
            num_prod (int, optional): Number of production steps per window. Defaults to 500000.
            num_eq (int, optional): Number of Equilibration steps per window. Defaults to 50000.
            method (str, optional): whether to "pull" the ligand or to let it appear in new position. Defaults to "pulling".
            path (List[unit.Quantity], optional): path for the ligand. Defaults to None.
            force_const (unit.Quantity, optional): force constant used in the ligand restraint.. Defaults to 100*unit.kilocalorie_per_mole/(unit.angstrom ** 2).
            fric (unit.Quantity, optional): friction coefficient for the LangevinIntegrator. Defaults to 1/unit.picosecond.
            system (mm.openmm.System, optional): openmm System. Defaults to None.
            psf (str or app.CharmmPsfFile, optional): psf objector path to psf file. path preffered. Defaults to None.
            pdb (str or app.PDBFile, optional: pdb object or path to pdb file. Defaults to None.
            ligand_name (str, optional): name of the ligand to be pulled out. Defaults to None.
            traj_write_path (str, optional): path to where the trajectories are stored. Defaults to None.
        """
        self.temp = temp
        self.p = p
        self.freq = iofreq
        self.dt = dt
        self.nWin = len(path)
        self.num_prod = num_prod
        self.num_eq = num_eq
        self.method = method
        self.path = path
        self.force_const = force_const
        self.fric = fric
        self.system = system
        self.ligName = ligand_name
        self.tOutput = traj_write_path
        self.simulation: app.Simulation
        self.integrator: mm.openmm.Integrator

        if not isinstance(self.tOutput, str):
            logger.warning(
                "No trajectory output was given. All generated trajectories are now stored in the current working directory. This can get messy! :/"
            )
            self.tOutput = os.getcwd()

        if not self.tOutput.endswith("/"):
            self.tOutput += "/"

        try:
            self.pdb = app.PDBFile(pdb)
        # except TypeError:
        #    if not isinstance(pdb, app.PDBFile):
        #        raise ValueError("pdb cannot be None!")
        except:
            # pdb = input("Enter absolute path to pdb file: ")
            self.pdb = pdb

        try:
            self.psf = app.CharmmPsfFile(psf)
        # except TypeError:
        #    if not isinstance(psf, app.CharmmPsfFile):
        #        raise ValueError("psf cannot be None!")
        except:
            # psf = input("Enter absolute path to psf file: ")
            self.psf = psf

        self.platform = (
            mm.Platform.getPlatformByName("CUDA")
            if torch.cuda.is_available()
            else mm.Platform.GetPlatformByName("CPU")
        )
        self.platformProperties = {"Precision": "Single"}

    def prepare_simulations(self) -> mm.openmm.Integrator:
        """
        Adds a harmonic restraint to the residue with the name given in self.name to the first position in self.path.
        Also changes self.integrator so it can be used on self.system.

        Returns:
            mm.openmm.Integrator: Integrator for the system. it changes self.integrator.
        """

        ligandIndices = get_residue_indices(self.psf.atom_list, self.ligName)
        add_harmonic_restraint(
            system=self.system,
            atom_group=ligandIndices,
            values=[self.force_const, self.path[0].x, self.path[0].y, self.path[0].z],
        )
        self.integrator = openmmtools.integrators.LangevinIntegrator(
            temperature=self.temp, collision_rate=self.fric, timestep=self.dt
        )

        self.simulation = app.Simulation(
            topology=self.psf.topology,
            system=self.system,
            integrator=self.integrator,
            platform=self.platform,
            platformProperties=self.platformProperties,
        )

        return self.integrator

    def run_sampling(self):
        """
        runs the actual umbrella sampling.
        """
        orgCoords = open(file=f"{self.tOutput}coordinates.dat", mode="w")
        orgCoords.write("nwin, x0, y0, z0\n")

        self.simulation.context.setPositions(self.pdb.positions)

        for window in range(self.nWin):
            orgCoords.write(
                f"{window}, {self.simulation.context.getParameter('x0')}, {self.simulation.context.getParameter('y0')}, {self.simulation.context.getParameter('z0')}\n"
            )
            self.simulation.minimizeEnergy(maxIterations=100)
            self.simulation.context.setVelocitiesToTemperature(self.temp)
            self.simulation.step(self.num_eq)
            fileHandle = open(f"{self.tOutput}traj_{window}.dcd", "bw")
            dcdFile = app.dcdfile.DCDFile(
                file=fileHandle, topology=self.simulation.topology, dt=self.dt
            )
            for i in tqdm(range(int(self.num_prod / self.freq))):
                self.simulation.step(self.freq)
                dcdFile.writeModel(
                    self.simulation.context.getState(getPositions=True).getPositions()
                )
            fileHandle.close()
            try:
                self.simulation.context.setParameter("x0", self.path[window + 1].x)
                self.simulation.context.setParameter("y0", self.path[window + 1].y)
                self.simulation.context.setParameter("z0", self.path[window + 1].z)
            except IndexError:
                pass
        orgCoords.close()


class SamplingHydra(UmbrellaSimulation):
    def __init__(
        self,
        temp: unit.Quantity = 310 * unit.kelvin,
        p: unit.Quantity = 1 * unit.bar,
        iofreq: int = 2500,
        dt: unit.Quantity = 2 * unit.femtosecond,
        num_prod: int = 500000,
        num_eq: int = 50000,
        method: str = "pulling",
        path: List[Node] = None,
        force_const: unit.Quantity = 100
        * unit.kilocalorie_per_mole
        / (unit.angstrom ** 2),
        fric: unit.Quantity = 1 / unit.picosecond,
        system: mm.openmm.System = None,
        psf: str or app.CharmmPsfFile = None,
        pdb: str or app.PDBFile = None,
        ligand_name: str = None,
        traj_write_path: str = None,
        hydra_working_dir: str = None,
        mail: str = None,
        log: str = None,
        gpu: int = 1,
        conda_environment: str = "openmm",
    ) -> None:
        super().__init__(
            temp=temp,
            p=p,
            iofreq=iofreq,
            dt=dt,
            num_prod=num_prod,
            num_eq=num_eq,
            method=method,
            path=path,
            force_const=force_const,
            fric=fric,
            system=system,
            psf=psf,
            pdb=pdb,
            ligand_name=ligand_name,
            traj_write_path=traj_write_path,
        )
        self.psfPath = psf
        self.hydra_working_dir = hydra_working_dir
        self.mail = mail
        self.log = log
        self.gpu = gpu
        self.conda_environment = conda_environment
        if not isinstance(hydra_working_dir, str):
            logger.warning(
                "No hydra directory was given. Now everything is carried out in the current working directory. Ignore this warning if you are in /cluster/projects/..."
            )
            self.tOutput = os.getcwd()

        if not self.tOutput("/"):
            self.tOutput += "/"

        if not isinstance(self.psfPath, str):
            logger.warning("For serialization purposes, the PSF file path is needed.")
            self.psfPath = input("Absolute path for PSF file that is used: ")

    def write_hydra_scripts(
        self, window: int, serializedSystem: str, serializedIntegrator: str
    ):
        if not self.hydra_working_dir.endswith("/"):
            self.hydra_working_dir += "/"

        f = open(f"{self.hydra_working_dir}run_umbrella_{window}.sh", "w")
        command = "#$ -S /bin/bash\n#$ -m e"
        if self.mail:
            command += f"#$ -M {self.mail}\n"
            command += "#$ -m e\n" + "#$ -pe smp 1\n"
        command += "#$ -j y\n"
        if self.gpu:
            command += f"#$ -l gpu={self.gpu}\n"
        if self.log:
            command += f"#$ -o {self.log}\n"
        if self.cwd:
            command += "#$ -cwd\n"
        command += "\n"
        command += f"conda activate {self.conda_environment}\n"
        command += "python UmbrellaPipeline/sampling/simulationHydra.py "
        command += f"-psf {self.psfPath} -sys {serializedSystem} -int {serializedIntegrator} -to {self.tOutput} -ne {self.num_eq} -np {self.num_prod} -nw {window} -io"
        f.write(command)
        f.close()
        return f"{self.hydra_working_dir}run_umbrella_{window}.sh"

    def prepare_simulations(self):
        super().prepareSystem()
        serializedSystem = mm.openmm.XmlSerializer.serialize(self.system)
        serializedIntegrator = mm.openmm.XmlSerializer.serialize(self.integrator)
        for window in range(self.nWin):
            newfile = self.write_hydra_scripts(
                window=window,
                serializedSystem=serializedSystem,
                serializedIntegrator=serializedIntegrator,
            )
            logger.info(f"File written: {newfile}")
        return self.integrator

    def run_sampling(self):
        command: List[str] = []
        out: List[str] = []
        for window in range(self.nWin):
            command.append(f"qsub {self.hydra_working_dir}run_umbrella_{window}.sh")
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
