import openmm as mm
import openmm.unit as unit
import openmm.app as app
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

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


def scale(
    initial_value: unit.Quantity, final_value: unit.Quantity, lamda: float
) -> unit.Quantity:
    """
    Returns a scaled value between initial and final value, dependent on lamda.
    if lamda is zero, initial value is returned, if 1 is given, final value is returned.
    Args:
        initial_value (unit.Quantity): initial value.
        final_value (unit.Quantity): final value
        lamda (float): value between 0 and 1 to scale between the two values.

    Returns:
        unit.Quantity: scaled value between initial and final value
    """
    assert lamda <= 1.0 and lamda >= 0.0
    return (1 - lamda) * initial_value + lamda * final_value


def ghost_ligand(simulation: app.Simulation, ligand_indices: List[int]) -> None:
    """
    converts all ligand atoms to dummy atoms by setting VDW as well as Coulomb parameters to zero.

    """
    for force in simulation.context.getSystem().getForces():
        if type(force).__name__ == "NonbondedForce":
            for idx in ligand_indices:
                force.setParticleParameters(
                    idx,
                    0 * unit.elementary_charge,
                    0 * unit.nanometer,
                    0 * unit.kilojoule_per_mole,
                )
            force.updateParametersInContext(simulation.context)


def ramp_up_coulomb(
    lamda: float,
    simulation: app.Simulation,
    ligand_indices: List[int],
    original_parameters: List[unit.Quantity],
) -> None:
    """
    Helper function for the ghost_busters_ligand function. It updates the charge parameter in the nonbonded force of your simulation context.

    Args:
        lamda (float):
    """
    for force in simulation.context.getSystem().getForces():
        if type(force).__name__ == "NonbondedForce":
            for it, index in enumerate(ligand_indices):
                scaled_charge = scale(
                    initial_value=0 * unit.elementary_charge,
                    final_value=original_parameters[it][0],
                    lamda=lamda,
                )
                force.setParticleParameters(
                    index,
                    scaled_charge,
                    force.getParticleParameters(index)[1],
                    force.getParticleParameters(index)[2],
                )
            force.updateParametersInContext(simulation.context)


def ramp_up_vdw(
    lamda: float,
    simulation: app.Simulation,
    ligand_indices: List[int],
    original_parameters: List[unit.Quantity],
) -> None:
    """
    Helper function for the ghost_busters_ligand function. It updates the epsilon and sigma parameter in the nonbonded force of your simulation context.

    Args:
        lamda (float): [description]
    """
    for force in simulation.context.getSystem().getForces():
        if type(force).__name__ == "NonbondedForce":
            for it, index in enumerate(ligand_indices):
                scaled_sigma = scale(
                    initial_value=0 * unit.nanometer,
                    final_value=original_parameters[it][1],
                    lamda=lamda,
                )
                scaled_epsilon = scale(
                    initial_value=0 * unit.kilojoule_per_mole,
                    final_value=original_parameters[it][2],
                    lamda=lamda,
                )
                force.setParticleParameters(
                    index,
                    force.getParticleParameters(index)[0],
                    scaled_sigma,
                    scaled_epsilon,
                )
            force.updateParametersInContext(simulation.context)


def ghost_busters_ligand(
    simulation: app.Simulation,
    ligand_indices: List[int],
    original_parameters: List[unit.Quantity],
    nr_steps: int = 1000,
) -> None:
    """
    Converts ligand atoms form dummy back to thei normal parameters by first ramping up the vdw, and then ramping up the coulomb parameters.

    Args:
        nr_steps (int, optional): How many ramp up steps for vdw and coulomb you want each. Defaults to 1000.
    """
    for lamda in np.linspace(0, 1, nr_steps, endpoint=True):
        ramp_up_vdw(
            lamda=lamda,
            simulation=simulation,
            ligand_indices=ligand_indices,
            original_parameters=original_parameters,
        )
        simulation.step(1)
    for lamda in np.linspace(0, 1, nr_steps, endpoint=True):
        ramp_up_coulomb(
            lamda=lamda,
            simulation=simulation,
            ligand_indices=ligand_indices,
            original_parameters=original_parameters,
        )
        simulation.step(1)


def update_restraint(
    simulation: app.Simulation,
    ligand_indices: List[int],
    original_parameters: List[unit.Quantity],
    path: List[unit.Quantity],
    window: int,
    nr_steps: int = 1000,
) -> None:
    """Updates the constraint coordinates of the simulation

    Args:
        window (int): path window.
    """
    ghost_ligand(simulation=simulation, ligand_indices=ligand_indices)
    logger.info("Liggand turned to dummy.")
    for a, b in zip(
        ["x0", "y0", "z0"],
        [path[window].x, path[window].y, path[window].z],
    ):
        simulation.context.setParameter(a, b)
        logger.info(
            f"Restraint position updated to x={path[window].x} y={path[window].y} z={path[window].z}"
        )
    simulation.minimizeEnergy()
    simulation.step(250000)
    ghost_busters_ligand(
        simulation=simulation,
        ligand_indices=ligand_indices,
        original_parameters=original_parameters,
        nr_steps=nr_steps,
    )
    logger.info("Ligand back to full throttle. :)")
    return simulation


def serialize_system(system: mm.openmm.System, path: str) -> str:
    """
    Serializes the openmm system object

    Returns:
        str: path to serialized system xmlfile.
    """
    with open(file=path, mode="w") as f:
        f.write(mm.openmm.XmlSerializer.serialize(system))
    return path
