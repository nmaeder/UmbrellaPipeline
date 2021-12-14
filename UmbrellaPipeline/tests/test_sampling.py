import os
import openmm.app as app
import openmm as mm

from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
    parse_params
)
from UmbrellaPipeline.path_generation import Tree
from UmbrellaPipeline.sampling import SamplingHydra


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


"""
def test_simulations():
    pdb = read_pdb()
    psf = read_psf()
    tree = Tree.from_files(pdb=pdb, psf=psf)
    st = tree.node_from_files(psf=psf, pdb=pdb, name="unl").get_coordinates()
    path = [st, st]
    sim = UmbrellaSimulation(
        system=create_system(),
        psf=psf,
        pdb=pdb,
        ligand_name="unl",
        path=path,
        num_eq=10,
        num_prod=10,
        iofreq=1,
        traj_write_path=os.path.dirname(__file__),
    )
    sim.prepare_simulations()
    sim.run_sampling()
    assert os.path.exists("UmbrellaPipeline/tests/coordinates.dat")
    os.remove("UmbrellaPipeline/tests/coordinates.dat")
    for i in range(len(path)):
        assert os.path.exists(f"UmbrellaPipeline/tests/traj_{i}.dcd")
        os.remove(f"UmbrellaPipeline/tests/traj_{i}.dcd")

"""


def test_script_writing():
    output = [
        "run_umbrella_0.sh",
        "run_umbrella_1.sh",
        "serialized_sys.xml",
    ]
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

    for i in output:
        p = os.path.abspath(os.path.dirname(__file__) + "/" + i)
        assert os.path.exists(p)
        os.remove(p)
