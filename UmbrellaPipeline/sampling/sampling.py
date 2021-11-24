import torch
import os
import openmm.unit as unit
import openmm as mm
import openmm.app as app
from tqdm import tqdm
from typing import List
from UmbrellaPipeline.sampling.samplingHelper import addHarmonicRestraint
from UmbrellaPipeline.pathGeneration.node import Node
from UmbrellaPipeline.pathGeneration.pathHelper import (
    getCenterOfMassCoordinates,
    get_indices,
)


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
        trajOutputPath: str = None
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

    def runUmbrellaSampling(self):
        ligandIndices = get_indices(self.psf.atom_list, self.ligName)
        addHarmonicRestraint(
            system=self.system,
            atomGroup=ligandIndices,
            values=[self.forceK, self.path[0].x, self.path[0].y, self.path[0].z],
        )
        integrator = mm.LangevinIntegrator(self.temp, self.fric, self.dt)
        simulation = app.Simulation(
            topology=self.psf.topology,
            system=self.system,
            integrator=integrator,
            platform=self.platform,
            platformProperties=self.platformProperties,
        )
        simulation.context.setPositions(self.pdb.positions)
        
        if not isinstance(self.tOutput, str):
            self.tOutput = os.getcwd()

        if not self.tOutput.endswith("/"): 
            self.tOutput += "/"

        orgCoords = open(f"{self.tOutput}coordinates.dat")
        orgCoords.write("nwin, x0, y0, z0\n")
        
        for window in range(self.nWin):
            orgCoords.write(f"{self.nwin}, {simulation.context.getParameter('x0')}, {simulation.context.getParameter('y0')}, {simulation.context.getParameter('z0')}\n")
            simulation.minimizeEnergy()
            simulation.context.setVelocities(self.temp)
            simulation.step(self.nEq)
            fileHandle = open("f{self.tOutput}/traj_{window}.dcd")
            dcdFile = app.DCDFile(fileHandle, self.psf.topology, dt=self.dt)
            for i in tqdm(range(int(self.nProd/self.freq))):
                simulation.step(self.freq)
                dcdFile.writeModel(simulation.context.getState(getPositions=True).getPositions())
            fileHandle.close()
            simulation.context.setParameter('x0', self.path[window].x)
            simulation.context.setParameter('y0', self.path[window].x)
            simulation.context.setParameter('z0', self.path[window].x)



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
        )


class SimulationsLSF(UmbrellaSimulation):
    def __init__(self, temp: unit.Quantity = 310 * unit.kelvin, p: unit.Quantity = 1 * unit.bar, iofreq: int = 2500, dt: unit.Quantity = 2 * unit.femtosecond, nWin: int = 20, nProd: int = 500000, nEq: int = 50000, method: str = "pulling", path: List[Node] = None, forceK: unit.Quantity = 100 * unit.kilocalorie_per_mole / (unit.angstrom ** 2), fric: unit.Quantity = 1 / unit.picosecond, system: mm.openmm.System = None, psf: str or app.CharmmPsfFile = None, pdb: str or app.PDBFile = None, ligandName: str = None, trajOutputPath: str = None) -> None:
        super().__init__(temp=temp, p=p, iofreq=iofreq, dt=dt, nWin=nWin, nProd=nProd, nEq=nEq, method=method, path=path, forceK=forceK, fric=fric, system=system, psf=psf, pdb=pdb, ligandName=ligandName, trajOutputPath=trajOutputPath)
    
