import openmm as mm
from openmm import (
    unit,
    app,
)
from typing import List
import numpy as np

from UmbrellaPipeline.utils import get_backbone_indices


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
    force.setUsesPeriodicBoundaryConditions(True)
    system.addForce(force)
    return system


def add_barostat(
    system: mm.openmm.System,
    pressure: unit.Quantity = 1 * unit.bar,
    temperature: unit.Quantity = 310 * unit.kelvin,
    membrane_barostat: bool = False,
    frequency: int = 25,
) -> mm.openmm.System:
    if membrane_barostat:
        add_membrane_barostat(
            system=system,
            pressure=pressure,
            temperature=temperature,
            frequency=frequency,
        )
    else:
        add_isotropic_barostat(
            system=system,
            pressure=pressure,
            temperature=temperature,
            frequency=frequency,
        )
    return system


def add_membrane_barostat(
    system: mm.openmm.System,
    pressure: unit.Quantity,
    temperature: unit.Quantity,
    frequency: int,
) -> mm.openmm.System:
    barostat = mm.MonteCarloMembraneBarostat(
        defaultPressure=pressure,
        defaultSurfaceTension=0 * unit.bar * unit.nanometer,
        defaultTemperature=temperature,
        xymode=mm.MonteCarloMembraneBarostat.XYIsotropic,
        zmode=mm.MonteCarloMembraneBarostat.ZFree,
        frequency=frequency,
    )
    system.addForce(barostat)
    return system


def add_isotropic_barostat(
    system: mm.openmm.System,
    pressure: unit.Quantity,
    temperature: unit.Quantity,
    frequency: int,
) -> mm.openmm.System:
    barostat = mm.MonteCarloBarostat(
        defaultPressure=pressure,
        defaultTemperature=temperature,
        frequency=frequency,
    )
    system.addForce(barostat)
    return system


def add_backbone_restraints(
    positions: app.Simulation,
    system: mm.openmm.System,
    atom_list: app.internal.charmm.topologyobjects.AtomList,
    force_constant: unit.Quantity = 10 * unit.kilocalorie_per_mole / unit.angstrom ** 2,
) -> mm.openmm.System:
    """Adds a CustomExternalForce object to your systems force list. it will not restrain at this point, since the force constant is set to 0.
    Use activate_backbone_restraints() to set the restraints active.

    Args:
        system (mm.openmm.System): openmm system object
        atom_list (app.internal.charmm.topologyobjects.AtomList): psf.atom_list of your whole system

    Returns:
        mm.openmm.System: [description]
    """
    indices = get_backbone_indices(atom_list=atom_list)
    force = mm.CustomExternalForce("1/2*k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k", force_constant)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    force.setName("bb_restraint_force")
    for index in indices:
        force.addParticle(index, positions[index])
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
    nr_steps: int = 10,
) -> None:
    """Updates the constraint coordinates of the simulation

    Args:
        window (int): path window.
    """
    ghost_ligand(simulation=simulation, ligand_indices=ligand_indices)
    for a, b in zip(
        ["x0", "y0", "z0"],
        [path[window].x, path[window].y, path[window].z],
    ):
        simulation.context.setParameter(a, b)
    simulation.minimizeEnergy(200)
    simulation.step(1000)
    ghost_busters_ligand(
        simulation=simulation,
        ligand_indices=ligand_indices,
        original_parameters=original_parameters,
        nr_steps=nr_steps,
    )
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


def serialize_state(state: mm.State, path: str) -> str:
    with open(file=path, mode="w") as f:
        f.write(mm.openmm.XmlSerializer.serialize(state))
    return path


def deserialize_state(path: str) -> mm.State:
    with open(file=path, mode="r") as f:
        state = mm.openmm.XmlSerializer.deserialize(f.read())
    return state
