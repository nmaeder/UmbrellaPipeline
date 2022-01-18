from typing import List
import openmm as mm
from openmm import Vec3, unit


def get_centroid_coordinates(
    positions: unit.Quantity,
    indices: List[int],
) -> unit.Quantity:
    """
    Calculates centroid coordinates of a given number of atoms or molecules.

    Args:
        positions (unit.Quantity): positions of all atoms in the system
        indices (List[int]): indices of the atoms you want the centroid from.

    Returns:
        unit.Quantity: Centroid coordinates of the specified atoms
    """
    ret = [0 * positions.unit, 0 * positions.unit, 0 * positions.unit]
    for coordinate in range(3):
        for i in indices:
            ret[coordinate] += positions[i][coordinate]
        ret[coordinate] /= len(indices) * positions.unit
    return unit.Quantity(value=Vec3(x=ret[0], y=ret[1], z=ret[2]), unit=positions.unit)


def get_center_of_mass_coordinates(
    positions: unit.Quantity,
    indices: unit.Quantity,
    masses: mm.openmm.System,
    include_hydrogens: bool = True,
) -> unit.Quantity:
    """
    Calculates center of mass coordinates of a given number of atoms or molecules.

    Args:
        positions (unit.Quantity): positions of all atoms in the system
        indices (List[int]): indices of the atoms you want the centroid from.
        masses (mm.openmm.System): openmm System that contains all the atom masses.

    Returns:
        unit.Quantity: center of mass coordinates of the specified atoms
    """
    ret = [0 * unit.dalton, 0 * unit.dalton, 0 * unit.dalton]
    mass = 0 * unit.dalton
    for coordinate in range(3):
        for atomnr in indices:
            if (
                not include_hydrogens
                and masses.getParticleMass(atomnr) < 1.2 * unit.dalton
            ):
                continue
            ret[coordinate] += positions[atomnr][coordinate].value_in_unit(
                positions.unit
            ) * masses.getParticleMass(atomnr)
            mass += masses.getParticleMass(atomnr)
        ret[coordinate] /= mass
    return unit.Quantity(value=Vec3(x=ret[0], y=ret[1], z=ret[2]), unit=positions.unit)
