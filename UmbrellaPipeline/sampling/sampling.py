from openmm.openmm import XmlSerializer
import torch
import os, time
import logging
import openmm.unit as unit
import openmm as mm
import openmm.app as app
from tqdm import tqdm
from typing import (
    List,
    Tuple,
)

from UmbrellaPipeline.sampling.samplingHelper import addHarmonicRestraint
from UmbrellaPipeline.pathGeneration.node import Node
from UmbrellaPipeline.utils.bash import executeBashCommand
from UmbrellaPipeline.pathGeneration.pathHelper import (
    getIndices,
)


logger = logging.getLogger(__name__)


class UmbrellaSimulation:
    """
    Class for the sampling part of the Umbrella Pipeline. Holds all information necessary for the Umbrella Sampling.
    """

    def __init__(
        self,
        temp: unit.Quantity = 310 * unit.kelvin,
        p: unit.Quantity = 1 * unit.bar,
        iofreq: int = 2500,
        dt: unit.Quantity = 2 * unit.femtosecond,
        nWin: int = 20,
        nProd: int = 500000,
        nEq: int = 50000,
        method: str = "pulling",
        path: List[unit.Quantity] = None,
        forceK: unit.Quantity = 100 * unit.kilocalorie_per_mole / (unit.angstrom ** 2),
        fric: unit.Quantity = 1 / unit.picosecond,
        system: mm.openmm.System = None,
        psf: str or app.CharmmPsfFile = None,
        pdb: str or app.PDBFile = None,
        ligandName: str = None,
        trajOutputPath: str = None,
    ) -> None:
        """
        Args:
            temp (unit.Quantity, optional): Simulation Temperature. Defaults to 310*unit.kelvin.
            p (unit.Quantity, optional): Simulation pressure, if 0 NVT is sampled instead of NPT, not encouraged. Defaults to 1*unit.bar.
            iofreq (int, optional): Interval after which a snapshot is saved to the trajectory. Defaults to 2500.
            dt (unit.Quantity, optional): simulation timestep. Defaults to 2*unit.femtosecond.
            nWin (int, optional): number of simulation windows. Defaults to 20.
            nProd (int, optional): Number of production steps per window. Defaults to 500000.
            nEq (int, optional): Number of Equilibration steps per window. Defaults to 50000.
            method (str, optional): whether to "pull" the ligand or to let it appear in new position. Defaults to "pulling".
            path (List[unit.Quantity], optional): path for the ligand. Defaults to None.
            forceK (unit.Quantity, optional): force constant used in the ligand restraint.. Defaults to 100*unit.kilocalorie_per_mole/(unit.angstrom ** 2).
            fric (unit.Quantity, optional): friction coefficient for the LangevinIntegrator. Defaults to 1/unit.picosecond.
            system (mm.openmm.System, optional): openmm System. Defaults to None.
            psf (str or app.CharmmPsfFile, optional): psf objector path to psf file. path preffered. Defaults to None.
            pdb (str or app.PDBFile, optional: pdb object or path to pdb file. Defaults to None.
            ligandName (str, optional): name of the ligand to be pulled out. Defaults to None.
            trajOutputPath (str, optional): path to where the trajectories are stored. Defaults to None.
        """
        self.temp = temp
        self.p = p
        self.freq = iofreq
        self.dt = dt
        self.nWin = nWin
        self.nProd = nProd
        self.nEq = nEq
        self.method = method
        self.path = path
        self.forceK = forceK
        self.fric = fric
        self.system = system
        self.ligName = ligandName
        self.tOutput = trajOutputPath
        self.simulation: app.Simulation
        self.integrator: mm.openmm.Integrator

        if not isinstance(self.tOutput, str):
            logger.warning(
                "No trajectory output was given. All generated trajectories are now stored in the current working directory. This can get messy! :/"
            )
            self.tOutput = os.getcwd()

        if not self.tOutput("/"):
            self.tOutput += "/"

        try:
            self.pdb = app.PDBFile(pdb)
        except TypeError:
            if not isinstance(pdb, app.PDBFile):
                raise ValueError("pdb cannot be None!")
        except ValueError:
            pdb = input("Enter absolute path to pdb file: ")
            self.pdb = app.PDBFile(pdb)

        try:
            self.psf = app.CharmmPsfFile(psf)
        except TypeError:
            if not isinstance(psf, app.CharmmPsfFile):
                raise ValueError("psf cannot be None!")
        except ValueError:
            psf = input("Enter absolute path to psf file: ")
            self.psf = app.CharmmPsfFile(psf)

        self.platform = (
            mm.Platform.getPlatformByName("CUDA")
            if torch.cuda.is_available()
            else mm.Platform.GetPlatformByName("CPU")
        )
        self.platformProperties = {"Precision": "Single"}

    def prepareSystem(self) -> mm.openmm.Integrator:
        """
        Adds a harmonic restraint to the residue with the name given in self.name to the first position in self.path.
        Also changes self.integrator so it can be used on self.system.

        Returns:
            mm.openmm.Integrator: Integrator for the system. it changes self.integrator.
        """

        ligandIndices = getIndices(self.psf.atom_list, self.ligName)
        addHarmonicRestraint(
            system=self.system,
            atomGroup=ligandIndices,
            values=[self.forceK, self.path[0].x, self.path[0].y, self.path[0].z],
        )
        self.integrator = mm.LangevinIntegrator(self.temp, self.fric, self.dt)

        return self.integrator

    def runUmbrellaSampling(self):
        """
        runs the actual umbrella sampling.
        """
        orgCoords = open(f"{self.tOutput}coordinates.dat")
        orgCoords.write("nwin, x0, y0, z0\n")

        self.simulation = app.Simulation(
            topology=self.psf.topology,
            system=self.system,
            integrator=self.integrator,
            platform=self.platform,
            platformProperties=self.platformProperties,
        )
        self.simulation.context.setPositions(self.pdb.positions)

        for window in range(self.nWin):
            orgCoords.write(
                f"{self.nwin}, {self.simulation.context.getParameter('x0')}, {self.simulation.context.getParameter('y0')}, {self.simulation.context.getParameter('z0')}\n"
            )
            self.simulation.minimizeEnergy()
            self.simulation.context.setVelocities(self.temp)
            self.simulation.step(self.nEq)
            fileHandle = open("f{self.tOutput}/traj_{window}.dcd")
            dcdFile = app.DCDFile(fileHandle, self.simulation.topology, dt=self.dt)
            for i in tqdm(range(int(self.nProd / self.freq))):
                self.simulation.step(self.freq)
                dcdFile.writeModel(
                    self.simulation.context.getState(getPositions=True).getPositions()
                )
            fileHandle.close()
            try:
                self.simulation.context.setParameter("x0", self.path[window + 1].x)
                self.simulation.context.setParameter("y0", self.path[window + 1].x)
                self.simulation.context.setParameter("z0", self.path[window + 1].x)
            except StopIteration:
                pass
        orgCoords.close()


class SimulationsHydra(UmbrellaSimulation):
    def __init__(
        self,
        temp: unit.Quantity = 310 * unit.kelvin,
        p: unit.Quantity = 1 * unit.bar,
        iofreq: int = 2500,
        dt: unit.Quantity = 2 * unit.femtosecond,
        nWin: int = 20,
        nProd: int = 500000,
        nEq: int = 50000,
        method: str = "pulling",
        path: List[Node] = None,
        forceK: unit.Quantity = 100 * unit.kilocalorie_per_mole / (unit.angstrom ** 2),
        fric: unit.Quantity = 1 / unit.picosecond,
        system: mm.openmm.System = None,
        psf: str or app.CharmmPsfFile = None,
        pdb: str or app.PDBFile = None,
        ligandName: str = None,
        trajOutputPath: str = None,
        hydraWorkingDirectoryPath: str = None,
        mail: str = None,
        log: str = None,
        gpu: int = 1,
        condaEnv: str = "openmm",
    ) -> None:
        super().__init__(
            temp=temp,
            p=p,
            iofreq=iofreq,
            dt=dt,
            nWin=nWin,
            nProd=nProd,
            nEq=nEq,
            method=method,
            path=path,
            forceK=forceK,
            fric=fric,
            system=system,
            psf=psf,
            pdb=pdb,
            ligandName=ligandName,
            trajOutputPath=trajOutputPath,
        )
        self.psfPath = psf
        self.hydraWorkingDirectoryPath = hydraWorkingDirectoryPath
        self.mail = mail
        self.log = log
        self.gpu = gpu
        self.condaEnv = condaEnv
        if not isinstance(hydraWorkingDirectoryPath, str):
            logger.warning(
                "No hydra directory was given. Now everything is carried out in the current working directory. Ignore this warning if you are in /cluster/projects/..."
            )
            self.tOutput = os.getcwd()

        if not self.tOutput("/"):
            self.tOutput += "/"

        if not isinstance(self.psfPath, str):
            logger.warning("For serialization purposes, the PSF file path is needed.")
            self.psfPath = input("Absolute path for PSF file that is used: ")

    def writeHydraScripts(
        self, window: int, serializedSystem: str, serializedIntegrator: str
    ):
        if not self.hydraWorkingDirectoryPath.endswith("/"):
            self.hydraWorkingDirectoryPath += "/"

        f = open(f"{self.hydraWorkingDirectoryPath}run_umbrella_{window}.sh", "w")
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
        command += f"conda activate {self.condaEnv}\n"
        command += "python UmbrellaPipeline/sampling/simulationHydra.py "
        command += f"-psf {self.psfPath} -sys {serializedSystem} -int {serializedIntegrator} -to {self.tOutput} -ne {self.nEq} -np {self.nProd} -nw {window} -io"
        f.write(command)
        f.close()
        return f"{self.hydraWorkingDirectoryPath}run_umbrella_{window}.sh"

    def prepareSimulatiojns(self):
        super().prepareSystem()
        serializedSystem = mm.openmm.XmlSerializer.serialize(self.system)
        serializedIntegrator = mm.openmm.XmlSerializer.serialize(self.integrator)
        for window in range(self.nWin):
            newfile = self.writeHydraScripts(
                window=window,
                serializedSystem=serializedSystem,
                serializedIntegrator=serializedIntegrator,
            )
            logger.info(f"File written: {newfile}")

    def runUmbrellaSampling(self):
        for window in range(self.nWin):
            try:
                executeBashCommand(
                    f"qsub {self.hydraWorkingDirectoryPath}run_umbrella_{window}.sh"
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Oops, something went wrong. Make sure you are logged in on the Hydra cluster and all the paths you specified are in acceptance with the best practice manual."
                )


class SimulationsLSF(UmbrellaSimulation):
    "TODO"
    pass

    def WriteLSFScripts(self):
        "TODO"
        pass
