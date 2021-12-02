import openmm as mm
import openmm.unit as unit
from typing import List

HARMONIC_FORMULA = (
    "0.5 * k * (dx^2 + dy^2 + dz^2); dx=abs(x1-x0); dy=abs(y1-y0); dz=abs(z1-z0)"
)

HARMONIC_PARAMS = ["k", "x0", "y0", "z0"]


def add_harmonic_restraint(
    system: mm.openmm.System,
    atom_group: List[int],
    values: List[unit.Quantity],
) -> mm.openmm.System:
    """
    This function adds a harmonic to the center of mass of given atom group at given cartesian values, by adding a CustomCentroidBondForce.

    Args:
        system (mm.openmm.System): System the force should be added to.
        atom_group (List[int]): list of atoms that are restrained
        values (unit.Quantity): position of the restraint

    Raises:
        IndexError: if not enough or to much values are given. expected 4.

    Returns:
        mm.openmm.System: system with added force.
    """
    if len(values) != len(HARMONIC_PARAMS):
        raise IndexError("Give same number of parameters and values!")
    force = mm.CustomCentroidBondForce(1, HARMONIC_FORMULA)
    for i, val in enumerate(values):
        force.addGlobalParameter(HARMONIC_PARAMS[i], val)
    force.addGroup(atom_group)
    force.addBond([0])
    system.addForce(force)
    return system
