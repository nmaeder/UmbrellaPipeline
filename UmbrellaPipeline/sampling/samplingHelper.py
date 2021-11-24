import openmm as mm
from typing import List

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

def writeHydrascripts():pass

def writeLSFscripts():pass

