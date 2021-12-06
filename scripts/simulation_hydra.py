import argparse
import openmm as mm
import openmm.app as app
import time
import torch
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

    parser.add_argument("-psf", "--psffile", type=str, required=True)
    parser.add_argument("-pdb", "--pdbfile", type=str, required=True)
    parser.add_argument("-sys", "--system", type=str, required=True)
    parser.add_argument("-int", "--integrator", type=str, required=True)
    parser.add_argument("-to", "--outputpath", type=str, required=True)
    parser.add_argument("-ne", "--equilibrationsteps", type=int, required=True)
    parser.add_argument("-np", "--productionsteps", type=int, required=True)
    parser.add_argument("-nw", "--window", type=int, required=True)
    parser.add_argument("-x", "--posx", type=float, required=True)
    parser.add_argument("-y", "--posy", type=float, required=True)
    parser.add_argument("-z", "--posz", type=float, required=True)
    parser.add_argument("-io", "--outputfreq", type=int, required=True)

    args, unkn = parser.parse_known_args()
    pdb = app.PDBFile(args.pdb)
    psf = app.CharmmPsfFile(args.psf)
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {"Precision": "mixed"}

    system = mm.openmm.XmlSerializer.deserialize(args.system)
    integrator = mm.openmm.XmlSerializer.deserialize(args.integrator)
    simulation = app.Simulation(
        topology=psf.topology,
        system=system,
        integrator=integrator,
        platform=platform,
        platformProperties=properties,
    )

    if not args.outputpath.endswith("/"):
        args.outputpath += "/"

    orgCoords = open(f"{args.outputpath}coordinates.dat")
    orgCoords.write("nwin, x0, y0, z0\n")

    orgCoords.write(
        f"{args.nw}, {simulation.context.getParameter('x0')}, {simulation.context.getParameter('y0')}, {simulation.context.getParameter('z0')}\n"
    )
    simulation.context.setPositions(pdb.positions)
    simulation.context.setParameter("x0", args.x)
    simulation.context.setParameter("y0", args.y)
    simulation.context.setParameter("z0", args.z)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(integrator.getTemperature())
    simulation.step(args.equilibrationsteps)
    fileHandle = open("f{args.outputpath}traj_{window}.dcd")
    dcdFile = app.DCDFile(fileHandle, simulation.topology, dt=args.dt)

    ttot = 0
    totruns = int(args.productionsteps / args.outputfreq)
    for i in range(totruns):
        st = time.time()
        args.simulation.step(args.outputfreq)
        dcdFile.writeModel(
            args.simulation.context.getState(getPositions=True).getPositions()
        )
        t = time.time() - st
        ttot += t

    fileHandle.close()
    orgCoords.close()


if __name__ == "__main__":
    main()

