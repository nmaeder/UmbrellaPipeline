import openmm.app as app
import openmm.unit as unit
import openmmtools
from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
)
from UmbrellaPipeline.path_generation import Tree
from UmbrellaPipeline.sampling import (
    ghost_ligand,
    ramp_up_coulomb,
    ramp_up_vdw,
    ghost_busters_ligand,
)

psf = "UmbrellaPipeline/data/step5_input.psf"
pdb = "UmbrellaPipeline/data/step5_input.pdb"
toppar_directory = "UmbrellaPipeline/data/toppar"
toppar_stream_file = "UmbrellaPipeline/data/toppar/toppar.str"


def test_ghosting():

    # create simulation, system and context
    info = SimulationSystem(
        psf_file=psf,
        pdb_file=pdb,
        ligand_name="UNL",
        toppar_directory=toppar_directory,
        toppar_stream_file=toppar_stream_file,
    )

    system = info.psf_object.createSystem(info.params)
    integrator = openmmtools.integrators.LangevinIntegrator()
    simulation = app.Simulation(
        topology=info.pdb_object.topology, system=system, integrator=integrator
    )

    orig_params = []

    f = simulation.context.getSystem().getForces()
    for fs in f:
        if type(fs).__name__ == "NonbondedForce":
            for index in info.ligand_indices:
                orig_params.append(fs.getParticleParameters(index))

    ghost_ligand(simulation=simulation, ligand_indices=info.ligand_indices)
    f = simulation.context.getSystem().getForces()
    for fs in f:
        if type(fs).__name__ == "NonbondedForce":
            for index in info.ligand_indices:
                assert fs.getParticleParameters(index) == [
                    0 * unit.elementary_charge,
                    0 * unit.nanometer,
                    0 * unit.kilojoule_per_mole,
                ]

    ramp_up_vdw(
        lamda=0.5,
        simulation=simulation,
        ligand_indices=info.ligand_indices,
        original_parameters=orig_params,
    )
    f = simulation.context.getSystem().getForces()
    for fs in f:
        if type(fs).__name__ == "NonbondedForce":
            for it, index in enumerate(info.ligand_indices):
                assert fs.getParticleParameters(index) == [
                    0 * unit.elementary_charge,
                    0.5 * orig_params[it][1],
                    0.5 * orig_params[it][2],
                ]

    ramp_up_coulomb(
        lamda=1,
        simulation=simulation,
        ligand_indices=info.ligand_indices,
        original_parameters=orig_params,
    )
    f = simulation.context.getSystem().getForces()
    for fs in f:
        if type(fs).__name__ == "NonbondedForce":
            for it, index in enumerate(info.ligand_indices):
                assert fs.getParticleParameters(index) == [
                    1 * orig_params[it][0],
                    0.5 * orig_params[it][1],
                    0.5 * orig_params[it][2],
                ]

    simulation.context.setPositions(info.pdb_object.positions)
    ghost_busters_ligand(
        simulation=simulation,
        ligand_indices=info.ligand_indices,
        original_parameters=orig_params,
        nr_steps=10,
    )
    for fs in f:
        if type(fs).__name__ == "NonbondedForce":
            for it, index in enumerate(info.ligand_indices):
                assert fs.getParticleParameters(index) == [
                    orig_params[it][0],
                    orig_params[it][1],
                    orig_params[it][2],
                ]
