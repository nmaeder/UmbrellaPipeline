from typing import List
import openmm as mm
from openmm import Vec3, unit
import numpy as np


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
    ret = [0, 0, 0]
    for coordinate in range(3):
        for i in indices:
            ret[coordinate] += positions[i][coordinate].value_in_unit(unit.nanometer)
        ret[coordinate] /= len(indices)
    return unit.Quantity(value=Vec3(x=ret[0], y=ret[1], z=ret[2]), unit=unit.nanometer)


def get_center_of_mass_coordinates(
    positions: unit.Quantity,
    indices: unit.Quantity,
    masses: mm.openmm.System,
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
    mass_list = np.zeros(len(indices))
    for mass_idx, idx in enumerate(indices):
        mass_list[mass_idx] = masses.getParticleMass(idx).value_in_unit(unit.dalton)
    scaled_mass = mass_list / mass_list.sum()
    coords = np.array([positions[i].value_in_unit(unit.nanometer) for i in indices])
    x, y, z = np.matmul(coords.T, scaled_mass)
    return unit.Quantity(value=Vec3(x=x, y=y, z=z), unit=unit.nanometer)
