import openmm as mm
import openmm.unit as unit
from typing import List

harmonicFormula = (
    "0.5 * k * (dx^2 + dy^2 + dz^2) ; dx=abs(x1-x0) ; dy=abs(y1-y0) ; dz=abs(z1-z0)",
)
harmonicParams = ["k", "x0", "y0", "z0"]


def addHarmonicRestraint(
    system: mm.openmm.System,
    atomGroup: List[int],
    values: List[unit.Quantity],
) -> mm.openmm.System:
    """
    This function adds a harmonic to the center of mass of given atom group at given cartesian values, by adding a CustomCentroidBondForce.

    Args:
        system (mm.openmm.System): System the force should be added to.
        atomGroup (List[int]): list of atoms that are restrained
        values (unit.Quantity): position of the restraint

    Raises:
        IndexError: if not enough or to much values are given. expected 4.

    Returns:
        mm.openmm.System: system with added force.
    """
    if len(values) != len(harmonicParams):
        raise IndexError("Give same number of parameters and values!")
    force = mm.CustomCentroidBondForce(1, harmonicFormula)
    for i in values:
        force.addGlobalParameter(harmonicParams[i], i)
    force.addGroup(atomGroup)
    force.addBond([0])
    system.addForce(force)
    return system
