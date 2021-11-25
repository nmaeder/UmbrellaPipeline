import openmm as mm
import openmm.unit as unit
from typing import List

from torch._C import _from_dlpack

harmonicFormula = (
    "0.5 * k * (dx^2 + dy^2 + dz^2) ; dx=abs(x1-x0) ; dy=abs(y1-y0) ; dz=abs(z1-z0)",
)
harmonicParams = ["k", "x0", "y0", "z0"]


def addHarmonicRestraint(
    system: mm.openmm.Syastem,
    atomGroup: List[int],
    values: List[float],
) -> mm.openmm.System:
    if len(values) != len(harmonicParams):
        raise IndexError("Give same number of parameters and values!")
    force = mm.CustomCentroidBondForce(1, harmonicFormula)
    for i in values:
        force.addGlobalParameter(harmonicParams[i], values[i])
    force.addGroup(atomGroup)
    force.addBond([0])
    system.addForce(force)
    return system


class ScriptWriter:
    def __init__(
        self,
        nWin: int,
        scriptPath: str,
        mail: str = None,
        gpu: int = 1,
        log: str = None,
        cwd: bool = True,
        condaEnv: str = "openmm",
        positions: unit.Quantity = None,
        temperature: unit.Quantity = 310 * unit.kelvin,
        nEq: int = 50000,
        nProd: int = 5000000,
        frec: int = 2500,
        fric: unit.Quantity = 1 / unit.picosecond,
        forceK: unit.Quantity = 100 * unit.kilocalorie_per_mole / (unit.angstrom ** 2),
        ligandIndices: List[int] = None,
        platform: mm.openmm.Platform = mm.Platform.getPlatformByName("CUDA"),
        platformProperties: dict = {"Precision": "single"},
    ) -> None:
        self.nWin = nWin
        self.scriptPath = scriptPath
        self.mail = mail
        self.gpu = gpu
        self.log = log
        self.cwd = cwd
        self.condaEnv = condaEnv
        self.positions = positions
        self.temperature = temperature
        self.nEq = nEq
        self.nProd = nProd
        self.frec = frec
        self.fric = fric
        self.forceK = forceK
        self.ligandIndices = ligandIndices
        self.platform = platform
        self.platformProperties = platformProperties
        self.execCommand += ""

    def writeHydraScripts(self):
        if not self.scriptPath.endswith("/"):
            self.scriptPath += "/"

        f = open(f"{self.scriptPath}run_umbrella_{self.nWin}.sh", "w")
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
