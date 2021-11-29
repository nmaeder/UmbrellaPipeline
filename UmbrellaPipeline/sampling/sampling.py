import torch
import os
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
from UmbrellaPipeline.pathGeneration.pathHelper import (
    getIndices,
)


logger = logging.getLogger(__name__)


class UmbrellaSimulation:
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
    ) -> None:
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
                "No trajectory output was given. All generated trajectories are now stored in the current working directory"
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

    def prepareSystem(self) -> Tuple[mm.openmm.Integrator, app.simulation.Simulation]:
        ligandIndices = getIndices(self.psf.atom_list, self.ligName)
        addHarmonicRestraint(
            system=self.system,
            atomGroup=ligandIndices,
            values=[self.forceK, self.path[0].x, self.path[0].y, self.path[0].z],
        )

        self.integrator = mm.LangevinIntegrator(self.temp, self.fric, self.dt)
        self.simulation = app.Simulation(
            topology=self.psf.topology,
            system=self.system,
            integrator=self.integrator,
            platform=self.platform,
            platformProperties=self.platformProperties,
        )
        self.simulation.context.setPositions(self.pdb.positions)
        return (self.integrator, self.simulation)

    def runUmbrellaSampling(self):

        orgCoords = open(f"{self.tOutput}coordinates.dat")
        orgCoords.write("nwin, x0, y0, z0\n")

        for window in range(self.nWin):
            orgCoords.write(
                f"{self.nwin}, {self.simulation.context.getParameter('x0')}, {self.simulation.context.getParameter('y0')}, {self.simulation.context.getParameter('z0')}\n"
            )
            self.simulation.minimizeEnergy()
            self.simulation.context.setVelocities(self.temp)
            self.simulation.step(self.nEq)
            fileHandle = open("f{self.tOutput}/traj_{window}.dcd")
            dcdFile = app.DCDFile(fileHandle, self.psf.topology, dt=self.dt)
            for i in tqdm(range(int(self.nProd / self.freq))):
                self.simulation.step(self.freq)
                dcdFile.writeModel(
                    self.simulation.context.getState(getPositions=True).getPositions()
                )
            fileHandle.close()
            self.simulation.context.setParameter("x0", self.path[window].x)
            self.simulation.context.setParameter("y0", self.path[window].x)
            self.simulation.context.setParameter("z0", self.path[window].x)

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

    def writeHydraScripts(self):
        "TODO"
        if not self.hydraWorkingDirectoryPath.endswith("/"):
            self.hydraWorkingDirectoryPath += "/"

        f = open(f"{self.hydraWorkingDirectoryPath}run_umbrella_{self.nWin}.sh", "w")
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
        command += "python simulationHydra.py "
        command += self.execCommand
        f.write(command)
        f.close()

    def runUmbrellaSampling(self):
        "TODO"
        self.writeHydraScripts()

    def prepAndRun(self):
        super().prepareSystem()
        self.runUmbrellaSampling


class SimulationsLSF(UmbrellaSimulation):
    "TODO"
    pass

    def WriteLSFScripts(self):
        "TODO"
        pass
