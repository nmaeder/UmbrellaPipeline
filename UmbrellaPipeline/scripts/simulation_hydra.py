import argparse
from numpy import mod
import openmm as mm
import openmm.app as app
import time
import openmm.unit as unit
from typing import List

"""
Worker script for the sampling.py script. Highly specific, not encouraged to use on its own.
"""


def displayTime(seconds: float) -> str:
    ret = ""
    tot = seconds
    intervals = [
        (604800, 0),
        (86400, 0),
        (3600, 0),
        (60, 0),
        (1, 0),
    ]
    for number, count in intervals:
        while tot >= number:
            count += 1
            tot -= number
        if count:
            ret += f"{count:02d}:"
    ret = ret.rstrip(":")
    return f"00:{ret}"


def main():
    parser = argparse.ArgumentParser(
        description="System Info to run Umbrella Sampling with"
    )

    parser.add_argument("-psf", type=str, required=True)
    parser.add_argument("-pdb", type=str, required=True)
    parser.add_argument("-sys", type=str, required=True)
    parser.add_argument("-int", type=str, required=True)
    parser.add_argument("-to", type=str, required=True)
    parser.add_argument("-ne", type=int, required=True)
    parser.add_argument("-np", type=int, required=True)
    parser.add_argument("-nw", type=int, required=True)
    parser.add_argument("-x", type=float, required=True)
    parser.add_argument("-y", type=float, required=True)
    parser.add_argument("-z", type=float, required=True)
    parser.add_argument("-io", type=int, required=True)
    parser.add_argument("-t", type=float, required=True)
    parser.add_argument("-fric", type=float, required=True)
    parser.add_argument("-dt", type=float, required=True)

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

    if not args.to.endswith("/"):
        args.to += "/"

    simulation.context.setPositions(pdb.positions)
    simulation.context.setParameter("x0", args.x)
    simulation.context.setParameter("y0", args.y)
    simulation.context.setParameter("z0", args.z)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(integrator.getTemperature())
    simulation.step(args.ne)
    fileHandle = open(f"{args.to}traj_{args.nw}.dcd", "bw")
    dcdFile = app.DCDFile(fileHandle, simulation.topology, dt=args.dt)

    ttot = 0
    totruns = int(args.np / args.io)
    for i in range(totruns):
        st = time.time()
        simulation.step(args.io)
        dcdFile.writeModel(
            simulation.context.getState(getPositions=True).getPositions()
        )
        t = time.time() - st
        ttot += t

    fileHandle.close()
    orgCoords.close()


if __name__ == "__main__":
    main()
