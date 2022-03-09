import openmm as mm
from openmm import (
    unit,
    app,
)
from typing import List
import numpy as np
import logging

try:
    from typing import Literal
except:
    from typing_extensions import Literal

logger = logging.getLogger(__name__)

from UmbrellaPipeline.utils import (
    SimulationParameters,
    SystemInfo,
    get_backbone_indices,
    gen_pbc_box,
)


HARMONIC_FORMULA = (
    "0.5 * k * (dx^2 + dy^2 + dz^2); dx=abs(x1-x0); dy=abs(y1-y0); dz=abs(z1-z0)"
)

HARMONIC_PARAMS = ["k", "x0", "y0", "z0"]


def add_ligand_restraint(
    system: mm.openmm.System,
    atom_group: List[int],
    force_constant: unit.Quantity,
    positions: unit.Quantity,
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
    values = [force_constant, positions.x, positions.y, positions.z]
    if len(values) != len(HARMONIC_PARAMS):
        raise IndexError("Give same number of parameters and values!")
    force = mm.CustomCentroidBondForce(1, HARMONIC_FORMULA)
    for i, val in enumerate(values):
        force.addGlobalParameter(HARMONIC_PARAMS[i], val)
    force.addGroup(atom_group)
    force.addBond([0])
    force.setUsesPeriodicBoundaryConditions(True)
    system.addForce(force)
    logger.info(
        f"Ligand restraint added at position x={positions.x}, y={positions.x} ,z={positions.x} with a force constant of {force_constant}"
    )
    return system


def add_barostat(
    system: mm.openmm.System,
    properties: SimulationParameters,
    membrane_barostat: bool = False,
    frequency: int = 25,
) -> mm.openmm.System:
    """
    Add a monte carlo barostat to a openmm system. either an isotropic or a membrane barostat.

    Args:
        system (mm.openmm.System): openmm system to add the barostat to.
        properties (SimulationParameters): simulation properties object containint temperature and pressure.
        membrane_barostat (bool, optional): true if you want to use a membrane_barostat. Defaults to False.
        frequency (int, optional): update frequency for the barostat. Defaults to 25.

    Returns:
        mm.openmm.System: openmm system with added barostat.
    """
    if membrane_barostat:
        add_membrane_barostat(
            system=system,
            pressure=properties.pressure,
            temperature=properties.temperature,
            frequency=frequency,
        )
    else:
        add_isotropic_barostat(
            system=system,
            pressure=properties.pressure,
            temperature=properties.temperature,
            frequency=frequency,
        )
    return system


def add_membrane_barostat(
    system: mm.openmm.System,
    pressure: unit.Quantity,
    temperature: unit.Quantity,
    frequency: int,
) -> mm.openmm.System:
    """
    Implements add barostat, in case you want a membrane barostat

    Args:
        system (mm.openmm.System): openmm system object to which the barostat is added.
        pressure (unit.Quantity): pressure
        temperature (unit.Quantity): temperature
        frequency (int): update frequency
    Returns:
        mm.openmm.System: openmm system object with added barostat
    """
    barostat = mm.MonteCarloMembraneBarostat(
        pressure,
        0 * unit.bar * unit.nanometer,
        temperature,
        mm.MonteCarloMembraneBarostat.XYIsotropic,
        mm.MonteCarloMembraneBarostat.ZFree,
        frequency,
    )
    system.addForce(barostat)
    logger.info(
        f"Membrane MC Barostat added with p = {pressure}, T ={temperature} and frequency = {frequency}."
    )
    return system


def add_isotropic_barostat(
    system: mm.openmm.System,
    pressure: unit.Quantity,
    temperature: unit.Quantity,
    frequency: int,
) -> mm.openmm.System:
    """
    Implements add barostat, in case you want a isotropic barostat

    Args:
        system (mm.openmm.System): openmm system object to which the barostat is added.
        pressure (unit.Quantity): pressure
        temperature (unit.Quantity): temperature
        frequency (int): update frequency

    Returns:
        mm.openmm.System: openmm system object with added barostat
    """
    barostat = mm.MonteCarloBarostat(
        pressure,
        temperature,
        frequency,
    )
    system.addForce(barostat)
    logger.info(
        f"Isotropic MC Barostat added with p = {pressure}, T ={temperature} and frequency = {frequency}."
    )
    return system


def add_backbone_restraints(
    positions: unit.Quantity,
    system: mm.openmm.System,
    atom_list: app.internal.charmm.topologyobjects.AtomList,
    force_constant: unit.Quantity = 10 * unit.kilocalorie_per_mole / unit.angstrom**2,
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
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for index in indices:
        values = [
            force_constant,
            positions[index].x,
            positions[index].y,
            positions[index].z,
        ]
        force.addParticle(index, values)
    system.addForce(force)
    logger.info(
        f"Protein backbone restraints added with a force constant of {force_constant}."
    )
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
    assert 0 <= lamda <= 1.0
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
    position: unit.Quantity,
    nr_steps: int = 1000,
) -> None:
    """Updates the constraint coordinates of the simulation

    Args:
        window (int): path window.
    """
    ghost_ligand(simulation=simulation, ligand_indices=ligand_indices)
    for a, b in zip(
        ["x0", "y0", "z0"],
        [position.x, position.y, position.z],
    ):
        simulation.context.setParameter(a, b)
    simulation.minimizeEnergy()
    simulation.step(250000)
    ghost_busters_ligand(
        simulation=simulation,
        ligand_indices=ligand_indices,
        original_parameters=original_parameters,
        nr_steps=nr_steps,
    )
    logger.info(
        f"Restraint position updated to x={path[window].x} y={path[window].y} z={path[window].z}"
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
    logger.info(f"System serialized to {path}.")
    return path


def serialize_state(state: mm.State, path: str) -> str:
    """
    Serializes a state to an rst file. take sin velocities, positions and pbc box info.

    Args:
        state (mm.State): State object. use Simulation.context.getState(getPositions=True, getVelocities=True)
        path (str): path to the serialization file

    Returns:
        str: path of the serialization file
    """
    try:
        assert state.getPositions() is not None
        assert state.getVelocities() is not None
    except AssertionError:
        TypeError("State has to contain velocities and positions.")
    with open(file=path, mode="w") as f:
        f.write(mm.openmm.XmlSerializer.serialize(state))

    return path


def deserialize_state(path: str) -> mm.State:
    """
    Deserializes state for simulations from a file

    Args:
        path (str): path of the file that contains the serialized state.

    Returns:
        mm.State: openmm state with pbc box info, positions and velocities.
    """
    try:
        with open(file=path, mode="r") as f:
            state = mm.openmm.XmlSerializer.deserialize(f.read())
    except:
        FileNotFoundError
    logger.info(f"State deserialized from {path}.")
    return state


def write_path_to_file(path: unit.Quantity, directory: str) -> str:
    """
    Writes the restrain coordinates to a file, so analysis is still possible without the umbrellapipeline object

    Returns:
        str: path for the coordinates file
    """

    opath = "{}/coordinates.dat".format(directory.rstrip("/"))
    orgCoords = open(file=opath, mode="w")
    orgCoords.write(f"#lamda, x0, y0, z0, all in units of {path[0].unit}\n")
    for window, position in enumerate(path):
        orgCoords.write(f"{window}, {position.x}, {position.y}, {position.z}\n")
    logger.info(f"Unbinding Pathway written to {path}.")
    return path


def extract_nonbonded_parameters(
    system: mm.openmm.System, indices: List[int]
) -> List[unit.Quantity]:
    """
    Extracts lennard-Jones and coulomb parameters of given atoms.

    Args:
        system (mm.openmm.System): openmm system object.
        indices (List[int]): list of the atoms from which to extract the nb parameters

    Returns:
        List[unit.Quantity]: list of nonbonded parameters.
    """
    ret = []
    for force in system.getForces():
        if type(force).__name__ == "NonbondedForce":
            for index in indices:
                ret.append(force.getParticleParameters(index))
    return ret


def create_openmm_system(
    system_info: SystemInfo,
    simulation_parameters: SimulationParameters,
    nonbonded_method: app.forcefield = app.PME,
    nonbonded_cutoff: unit.Quantity = 1.2 * unit.nanometer,
    switch_distance: unit.Quantity = 1 * unit.nanometer,
    rigid_water: bool = True,
    constraints: app.forcefield = app.HBonds,
    barostat: Literal = None,
    ligand_restraint: bool = False,
    path: unit.Quantity = None,
    bb_restraints: bool = False,
    positions: unit.Quantity = None,
) -> mm.openmm.System:
    pos = positions if positions else system_info.crd_object.positions
    if not system_info.psf_object.boxLengths:
        gen_pbc_box(
            psf=system_info.psf_object,
            pos=system_info.crd_object.positions,
        )
    openmm_system = system_info.psf_object.createSystem(
        params=system_info.params,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=nonbonded_cutoff,
        switchDistance=switch_distance,
        constraints=constraints,
        rigidWater=rigid_water,
    )

    if barostat:
        if barostat == "membrane":
            mem = True
        elif barostat == "isotropic":
            mem = False
        else:
            raise ValueError("barostat can either be None, 'membrane' or 'isotropic'.")
        add_barostat(
            system=openmm_system,
            properties=simulation_parameters,
            membrane_barostat=mem,
        )

    if ligand_restraint:
        add_ligand_restraint(
            system=openmm_system,
            atom_group=system_info.ligand_indices,
            force_constant=simulation_parameters.force_constant,
            positions=path[0],
        )

    if bb_restraints:
        add_backbone_restraints(
            positions=pos,
            system=openmm_system,
            atom_list=system_info.psf_object.atom_list,
        )
    msg = "OpenMM system created."
    if bb_restraints:
        if msg.endswith("."):
            msg.rstrip(".")
            msg += " with "
        msg += "Backbone restraints and "
    if ligand_restraint:
        if msg.endswith("."):
            msg.rstrip(".")
            msg += " with "
        msg += "Ligand restraints and "
    if barostat:
        if msg.endswith("."):
            msg.rstrip(".")
            msg += " with "
        msg += "Barostat and "
    if msg.endswith(" "):
        msg.rstrip(" ")
        msg += "."
    logger.info(msg)
    return openmm_system
