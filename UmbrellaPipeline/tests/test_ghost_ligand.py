import os

import numpy as np
import openmm as mm
import openmm.app as app
from UmbrellaPipeline.path_generation import Tree, parse_params
from UmbrellaPipeline.sampling import SamplingHydra
from UmbrellaPipeline.utils.simulation_properties import SimulationProperties
from UmbrellaPipeline.utils.simulation_system import SimulationSystem
from openmm import unit

psf = "UmbrellaPipeline/data/step5_input.psf"
pdb = "UmbrellaPipeline/data/step5_input.pdb"
toppar_stream_file = "UmbrellaPipeline/data/toppar/toppar.str"
toppar_directory = "UmbrellaPipeline/data/toppar"


def read_pdb(pdb: str = pdb) -> app.PDBFile:
    return app.PDBFile(pdb)


def read_psf(psf: str = psf) -> app.CharmmPsfFile:
    return app.CharmmPsfFile(psf)


def create_system() -> mm.openmm.System:
    p = read_psf()
    pd = read_pdb()
    par = parse_params(
        toppar_directory=toppar_directory, toppar_str_file=toppar_stream_file
    )
    return p.createSystem(params=par)


def scale(value: unit.Quantity, lamb: float):
    return (1 - lamb) * value + lamb * value


def test_ghosting():

    # create simulation, system and context
    pdbo = read_pdb()
    psfo = read_psf()
    tree = Tree.from_files(pdb=pdbo, psf=psfo)
    st = tree.node_from_files(psf=psfo, pdb=pdbo, name="unl").get_coordinates()
    path = [st, st]
    properties = SimulationProperties()
    info = SimulationSystem(
        psf_file=psf,
        pdb_file=pdb,
        ligand_name="UNL",
        toppar_directory="UmbrellaPipeline/data/toppar",
        toppar_stream_file="UmbrellaPipeline/data/toppar/toppar.str",
    )

    sim = SamplingHydra(
        openmm_system=create_system(),
        properties=properties,
        info=info,
        path=path,
        traj_write_path=os.path.dirname(__file__),
        conda_environment="openmm",
        hydra_working_dir=os.path.dirname(__file__),
    )
    sim.prepare_simulations()

    # define ligand atom_idx
    ligand_idx = [0, 1, 2, 3, 4, 5, 6]
    ligand_nonbonded_parameters = []

    # first get context
    context = sim.simulation.context
    # get system
    system = context.getSystem()

    # get nonbonded parameters and save them for later
    for force in system.getForces():
        if type(force).__name__ == "NonbondedForce":
            for atom_idx in ligand_idx:
                charge, sigma, epsilon = force.getParticleParameters(atom_idx)
                ligand_nonbonded_parameters.append((charge, sigma, epsilon))

    # mutate charge to zero in 10 steps
    # get nonbonded force and set scaled parameters
    for force in system.getForces():
        if type(force).__name__ == "NonbondedForce":
            for lamb in np.linspace(1, 0, 10):
                print(lamb)
                for idx, atom_idx in enumerate(ligand_idx):
                    charge, sigma, epsilon = ligand_nonbonded_parameters[idx]
                    charge_, sigma_, epsilon_ = scale(charge, lamb), sigma, epsilon
                    force.setParticleParameters(idx, charge_, sigma_, epsilon_)
                # update parameter change that was done in system, su that context knows about it
                # and the changes can be used by the siulation object
                force.updateParametersInContext(context)
                # now we make simulation steps
                # sim.simulation.step(10) # commented out because particle positions have not been set
