import argparse
import openmm as mm
import openmmtools
from openmm import Vec3, unit, app
import logging

from UmbrellaPipeline.utils import (
    get_residue_indices,
)
from UmbrellaPipeline.sampling import (
    update_restraint,
    deserialize_state,
    extract_nonbonded_parameters,
)


logger = logging.getLogger(__name__)

"""
Worker script for the sampling.py script. Highly specific, not encouraged to use on its own.
"""


def main():
    parser = argparse.ArgumentParser(
        description="System Info to run Umbrella Sampling with"
    )

    parser.add_argument("-x", type=float, required=True)
    parser.add_argument("-y", type=float, required=True)
    parser.add_argument("-z", type=float, required=True)

    parser.add_argument("-t", dest="temperature", type=float, required=True)
    parser.add_argument("-fric", dest="friction_coefficient", type=float, required=True)
    parser.add_argument("-dt", dest="time_step", type=float, required=True)

    parser.add_argument("-psf", type=str, required=True)
    parser.add_argument("-crd", type=str, required=True)

    parser.add_argument("-sys", dest="serialized_system", type=str, required=True)
    parser.add_argument("-state", dest="serialized_state", type=str, required=True)

    parser.add_argument("-to", dest="output_path", type=str, required=True)
    parser.add_argument(
        "-ne", dest="number_of_equilibration_steps", type=int, required=True
    )
    parser.add_argument(
        "-np", dest="number_of_production_steps", type=int, required=True
    )
    parser.add_argument("-nw", dest="window_number", type=int, required=True)
    parser.add_argument("-io", dest="write_out_frequency", type=int, required=True)
    parser.add_argument("-ln", dest="ligand_name", type=str, required=True)

    args, _ = parser.parse_known_args()

    path_position = unit.Quantity(
        value=Vec3(
            x=args.x,
            y=args.y,
            z=args.z,
        ),
        unit=unit.nanometer,
    )

    crd = app.CharmmCrdFile(args.crd)
    crd.positions = crd.positions.in_units_of(unit.nanometer)
    crd.positions.unit = unit.nanometer
    psf = app.CharmmPsfFile(args.psf)

    platform = openmmtools.utils.get_fastest_platform()
    if platform.getName() == "CUDA" or "OpenCL":
        properties = {"Precision": "mixed"}
    else:
        properties = None

    with open(args.serialized_system, mode="r") as f:
        system = f.read()
    system = mm.openmm.XmlSerializer.deserialize(system)

    integrator = mm.LangevinIntegrator(
        args.temperature * unit.kelvin,
        args.friction_coefficient / unit.picosecond,
        args.time_step * unit.femtosecond,
    )

    simulation = app.Simulation(
        topology=psf.topology,
        system=system,
        integrator=integrator,
        platform=platform,
        platformProperties=properties,
    )

    state = deserialize_state(args.serialized_state)

    simulation.context.setState(state)

    if args.window_number > 0:
        indices = get_residue_indices(atom_list=psf.atom_list, name=args.ligand_name)
        original_parameters = extract_nonbonded_parameters(system, indices)
        update_restraint(
            simulation=simulation,
            ligand_indices=indices,
            original_parameters=original_parameters,
            position=path_position,
        )
        simulation.step(args.number_of_equilibration_steps)

    simulation.reporters.append(
        app.DCDReporter(
            file=f"{args.output_path}/production_trajcetory_window_{args.window_number}.dcd",
            reportInterval=args.write_out_frequency,
        )
    )
    simulation.reporters.append(
        app.StateDataReporter(
            file=f"{args.output_path}/production_state_window_{args.window_number}.out",
            reportInterval=args.write_out_frequency,
            step=True,
            time=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
        )
    )

    simulation.step(args.number_of_production_steps)


if __name__ == "__main__":
    main()
