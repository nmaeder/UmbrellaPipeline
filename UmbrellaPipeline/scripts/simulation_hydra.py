import argparse
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import time


"""
Worker script for the sampling.py script. Highly specific, not encouraged to use on its own.
"""


def main():
    parser = argparse.ArgumentParser(
        description="System Info to run Umbrella Sampling with"
    )

    parser.add_argument("-psf", type=str, required=True)
    parser.add_argument("-pdb", type=str, required=True)
    parser.add_argument("-sys", type=str, required=True)
    parser.add_argument("-to", type=str, required=True)
    parser.add_argument("-ne", type=int, required=True)
    parser.add_argument("-nw", type=int, required=True)
    parser.add_argument("-x", type=float, required=True)
    parser.add_argument("-y", type=float, required=True)
    parser.add_argument("-z", type=float, required=True)
    parser.add_argument("-io", type=int, required=True)
    parser.add_argument("-t", type=float, required=True)
    parser.add_argument("-fric", type=float, required=True)
    parser.add_argument("-dt", type=float, required=True)
    parser.add_argument("-nf", type=int, required=True)

    args, unkn = parser.parse_known_args()
    pdb = app.PDBFile(args.pdb)
    psf = app.CharmmPsfFile(args.psf)
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {"Precision": "mixed"}

    with open(args.sys, mode="r") as f:
        system = f.read()
    system = mm.openmm.XmlSerializer.deserialize(system)

    integrator = mm.LangevinIntegrator(
        args.t * unit.kelvin, args.fric / unit.picosecond, args.dt * unit.femtosecond
    )

    simulation = app.Simulation(
        topology=psf.topology,
        system=system,
        integrator=integrator,
        platform=platform,
        platformProperties=properties,
    )

    simulation.context.setPositions(pdb.positions)
    simulation.context.setParameter("x0", args.x)
    simulation.context.setParameter("y0", args.y)
    simulation.context.setParameter("z0", args.z)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(integrator.getTemperature())
    simulation.step(args.ne)
    fileHandle = open(f"{args.to}/traj_{args.nw}.dcd", "bw")
    dcdFile = app.DCDFile(fileHandle, simulation.topology, dt=args.dt)

    ttot = 0
    for i in range(args.nf):
        st = time.time()
        simulation.step(args.io)
        dcdFile.writeModel(
            simulation.context.getState(getPositions=True).getPositions()
        )
        t = time.time() - st
        ttot += t
        logger.info(
            f"Step {i+1} of {totruns} simulated. "
            f"Elapsed Time: {display_time(t)}. "
            f"Elapsed total time: {display_time(ttot)}. "
            f"Estimated time until finish: {display_time((totruns - i -1) * t) }."
        )

    fileHandle.close()


if __name__ == "__main__":
    main()
