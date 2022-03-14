import os, pytest, time, math
import openmmtools
from openmm import Vec3, unit, app
import numpy as np
import warnings

from UmbrellaPipeline import UmbrellaPipeline
from UmbrellaPipeline.analysis import PMFCalculator
from UmbrellaPipeline.path_finding import (
    Tree,
    Grid,
    GridNode,
    TreeNode,
    GridEscapeRoom,
    TreeEscapeRoom,
)
from UmbrellaPipeline.sampling import (
    add_ligand_restraint,
    ghost_ligand,
    ramp_up_coulomb,
    ramp_up_vdw,
    SamplingCluster,
    create_openmm_system,
    add_barostat,
)
from UmbrellaPipeline.utils import (
    gen_pbc_box,
    get_residue_indices,
    get_center_of_mass_coordinates,
    get_centroid_coordinates,
    parse_params,
    execute_bash,
    execute_bash_parallel,
)

warnings.filterwarnings(action="ignore")

pipeline = UmbrellaPipeline(
    ligand_residue_name="unl",
    toppar_stream_file="UmbrellaPipeline/data/toppar/toppar.str",
    toppar_directory="UmbrellaPipeline/data/toppar",
    psf_file="UmbrellaPipeline/data/step5_input.psf",
    crd_file="UmbrellaPipeline/data/step5_input.crd",
)


def test_genbox():

    assert pipeline.system_info.psf_object.boxVectors == None
    minC = gen_pbc_box(
        psf=pipeline.system_info.psf_object,
        pos=pipeline.system_info.crd_object.positions,
    )
    assert minC == [
        unit.Quantity(value=-0.56125095743, unit=unit.nanometer),
        unit.Quantity(value=-0.46094509581000004, unit=unit.nanometer),
        unit.Quantity(value=-0.06344883114, unit=unit.nanometer),
    ]
    print(pipeline.system_info.psf_object.boxVectors)
    assert pipeline.system_info.psf_object.boxVectors == unit.Quantity(
        value=(
            Vec3(x=11.07094907954, y=0.0, z=0.0),
            Vec3(x=0.0, y=10.882602253800002, z=0.0),
            Vec3(x=0.0, y=0.0, z=10.20869495182),
        ),
        unit=unit.nanometer,
    )


def test_pipeline():
    path = pipeline.generate_path()


def test_add_harmonic_restraint():

    gen_pbc_box(
        psf=pipeline.system_info.psf_object,
        pos=pipeline.system_info.crd_object.positions,
    )
    system = pipeline.system_info.psf_object.createSystem(
        params=pipeline.system_info.params
    )
    ind = get_residue_indices(
        atom_list=pipeline.system_info.psf_object.atom_list, name="unl"
    )
    fc = 10 * unit.kilocalorie_per_mole / (unit.angstrom ** 2)
    pos = Vec3(
        x=1 * unit.angstrom,
        y=2 * unit.angstrom,
        z=3 * unit.angstrom,
    )
    add_ligand_restraint(
        system=system, atom_group=ind, force_constant=fc, positions=pos
    )


def test_script_writing():
    output = [
        "run_umbrella_window_0.sh",
        "run_umbrella_window_1.sh",
    ]
    tree = Tree.from_files(
        positions=pipeline.system_info.crd_object.positions,
        psf=pipeline.system_info.psf_object,
    )
    st = Vec3(1, 2, 3)
    path = [st, st]

    sim = SamplingCluster(
        simulation_parameter=pipeline.simulation_parameters,
        system_info=pipeline.system_info,
        traj_write_path=os.path.dirname(__file__),
        conda_environment="openmm",
        sge_working_dir=os.path.dirname(__file__),
    )

    sim.openmm_system = sim.system_info.psf_object.createSystem(sim.system_info.params)

    sim.write_scripts(path=path)

    for i in output:
        p = os.path.abspath(os.path.dirname(__file__) + "/" + i)
        assert os.path.exists(p)
        os.remove(p)


def test_ligand_indices():
    indices = get_residue_indices(pipeline.system_info.psf_object.atom_list, name="unl")
    goal = list(range(8478, 8514, 1))
    assert indices == goal

    indices = get_residue_indices(
        pipeline.system_info.psf_object.atom_list, name="unl", include_hydrogens=False
    )
    goal = list(range(8478, 8499, 1))
    assert indices == goal


def test_protein_indices():
    indices = get_residue_indices(pipeline.system_info.psf_object.atom_list)
    goal = list(range(0, 8478, 1))
    assert indices == goal


def test_param_parser():
    params = parse_params(
        toppar_directory="UmbrellaPipeline/data/toppar",
        toppar_str_file="UmbrellaPipeline/data/toppar/toppar.str",
    )


def test_centroid_coords():
    ind1 = get_residue_indices(
        atom_list=pipeline.system_info.psf_object.atom_list, name="unl"
    )
    ind2 = get_residue_indices(
        atom_list=pipeline.system_info.psf_object.atom_list,
        name="unl",
        include_hydrogens=False,
    )

    print(get_centroid_coordinates(pipeline.system_info.crd_object.positions, ind1))

    print(get_centroid_coordinates(pipeline.system_info.crd_object.positions, ind2))

    a = get_centroid_coordinates(pipeline.system_info.crd_object.positions, ind1)
    b = unit.Quantity(
        value=Vec3(x=4.800868342909999, y=5.1623615832338885, z=5.116963445551665),
        unit=unit.nanometer,
    )
    assert round(a.x, 5) == round(b.x, 5)
    assert round(a.y, 5) == round(b.y, 5)
    assert round(a.z, 5) == round(b.z, 5)
    a = get_centroid_coordinates(pipeline.system_info.crd_object.positions, ind2)
    b = unit.Quantity(
        value=Vec3(x=4.791905722784763, y=5.152082995253809, z=5.1381769457266655),
        unit=unit.nanometer,
    )
    assert round(a.x, 5) == round(b.x, 5)
    assert round(a.y, 5) == round(b.y, 5)
    assert round(a.z, 5) == round(b.z, 5)


def test_com_coords():
    gen_pbc_box(
        pos=pipeline.system_info.crd_object.positions,
        psf=pipeline.system_info.psf_object,
    )
    system = pipeline.system_info.psf_object.createSystem(
        params=pipeline.system_info.params,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
    )
    a = get_center_of_mass_coordinates(
        positions=pipeline.system_info.crd_object.positions,
        indices=pipeline.system_info.ligand_indices,
        masses=system,
    )
    b = unit.Quantity(
        value=Vec3(x=4.7843494194906455, y=5.141548282974986, z=5.1745056529196995),
        unit=unit.nanometer,
    )
    assert round(a.x, 5) == round(b.x, 5)
    assert round(a.y, 5) == round(b.y, 5)
    assert round(a.z, 5) == round(b.z, 5)


def test_execute_bash():
    command1 = "echo Hello World"
    command2 = ["echo", "Hello World"]
    command3 = "sleep 12"
    stderr = "testerr.log"
    stdout = "testout.log"
    ret1 = execute_bash(command=command1)
    ret2 = execute_bash(command=command2, stdout_file=stdout)
    with pytest.raises(TimeoutError):
        execute_bash(command=command3, kill_after_wait=True, stderr_file=stderr)
    assert ret1 == "Hello World\n"
    assert ret2 == "Hello World\n"
    os.remove(stderr)
    os.remove(stdout)


def test_parallel_bash():
    commands = ["sleep 3", "sleep 3", "sleep 3", "echo World"]
    start = time.time()
    o = execute_bash_parallel(command=commands)
    end = time.time() - start
    assert end < 5
    assert o[3] == "World\n"


def test_ghosting():

    # create simulation, system and context
    platform = openmmtools.utils.get_fastest_platform()
    if platform.getName() == ("CUDA" or "OpenCL"):
        props = {"Precision": "mixed"}
    else:
        props = None
    system = pipeline.system_info.psf_object.createSystem(pipeline.system_info.params)
    integrator = openmmtools.integrators.LangevinIntegrator()
    simulation = app.Simulation(
        topology=pipeline.system_info.psf_object.topology,
        system=system,
        integrator=integrator,
        platform=platform,
        platformProperties=props,
    )
    simulation.context.setPositions(pipeline.system_info.crd_object.positions)
    orig_params = []

    f = simulation.context.getSystem().getForces()
    for fs in f:
        if type(fs).__name__ == "NonbondedForce":
            for index in pipeline.system_info.ligand_indices:
                orig_params.append(fs.getParticleParameters(index))

    ghost_ligand(
        simulation=simulation, ligand_indices=pipeline.system_info.ligand_indices
    )
    f = simulation.context.getSystem().getForces()
    for fs in f:
        if type(fs).__name__ == "NonbondedForce":
            for index in pipeline.system_info.ligand_indices:
                assert fs.getParticleParameters(index) == [
                    0 * unit.elementary_charge,
                    0 * unit.nanometer,
                    0 * unit.kilojoule_per_mole,
                ]

    ramp_up_vdw(
        lamda=0.5,
        simulation=simulation,
        ligand_indices=pipeline.system_info.ligand_indices,
        original_parameters=orig_params,
    )
    f = simulation.context.getSystem().getForces()
    for fs in f:
        if type(fs).__name__ == "NonbondedForce":
            for it, index in enumerate(pipeline.system_info.ligand_indices):
                assert fs.getParticleParameters(index) == [
                    0 * unit.elementary_charge,
                    0.5 * orig_params[it][1],
                    0.5 * orig_params[it][2],
                ]

    ramp_up_coulomb(
        lamda=1,
        simulation=simulation,
        ligand_indices=pipeline.system_info.ligand_indices,
        original_parameters=orig_params,
    )
    f = simulation.context.getSystem().getForces()
    for fs in f:
        if type(fs).__name__ == "NonbondedForce":
            for it, index in enumerate(pipeline.system_info.ligand_indices):
                assert fs.getParticleParameters(index) == [
                    1 * orig_params[it][0],
                    0.5 * orig_params[it][1],
                    0.5 * orig_params[it][2],
                ]


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Precision problem when testing on github."
)
def test_sampling():

    # create simulation, system and context
    platform = openmmtools.utils.get_fastest_platform()
    if platform.getName() == ("CUDA" or "OpenCL"):
        props = {"Precision": "mixed"}
    else:
        props = None
    system = create_openmm_system(pipeline.system_info, pipeline.simulation_parameters)
    integrator = openmmtools.integrators.LangevinIntegrator()
    simulation = app.Simulation(
        topology=pipeline.system_info.psf_object.topology,
        system=system,
        integrator=integrator,
        platform=platform,
        platformProperties=props,
    )
    simulation.context.setPositions(pipeline.system_info.crd_object.positions)
    simulation.minimizeEnergy(maxIterations=50)
    simulation.step(5)


def test_system_creation():
    system = create_openmm_system(pipeline.system_info, pipeline.simulation_parameters)
    system = create_openmm_system(
        pipeline.system_info,
        pipeline.simulation_parameters,
        ligand_restraint=True,
        bb_restraints=True,
        path=[Vec3(1, 2, 3)],
    )


def test_barostat_creation():
    system = create_openmm_system(pipeline.system_info, pipeline.simulation_parameters)
    add_barostat(
        system, properties=pipeline.simulation_parameters, membrane_barostat=True
    )
    add_barostat(
        system, properties=pipeline.simulation_parameters, membrane_barostat=False
    )


def test_grid_escape_room_basic():
    grid = Grid(grid=np.zeros(shape=(10, 10, 10), dtype=bool))
    start = GridNode(0, 0, 0)
    end = GridNode(9, 9, 9)
    escape_room = GridEscapeRoom(grid=grid, start=start)


def test_grid_successors():
    grid = Grid(grid=np.zeros(shape=(10, 10, 10), dtype=bool))
    grid.grid[1][1][1] = True
    start = GridNode(0, 0, 0)
    end = GridNode(9, 9, 9)
    escape_room = GridEscapeRoom(grid=grid, start=start)
    children = escape_room.generate_successors(parent=start)
    supposedchildren = [
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
    ]

    for i, c in enumerate(children):
        assert c.get_coordinates() == supposedchildren[i]

    children = escape_room.generate_successors(parent=end)
    supposedchildren = [
        [8, 8, 8],
        [8, 8, 9],
        [8, 9, 8],
        [8, 9, 9],
        [9, 8, 8],
        [9, 8, 9],
        [9, 9, 8],
    ]

    for i, c in enumerate(children):
        assert c.get_coordinates() == supposedchildren[i]


def test_grid_pathfinding():
    grid = Grid.from_files(
        crd=pipeline.system_info.crd_object,
        psf=pipeline.system_info.psf_object,
        gridsize=3 * unit.angstrom,
    )
    node = grid.node_from_files(
        psf=pipeline.system_info.psf_object,
        crd=pipeline.system_info.crd_object,
        name="UNL",
    )
    assert not grid.position_is_blocked(node)
    escape_room = GridEscapeRoom(grid=grid, start=node)
    path = escape_room.escape_room()
    assert path != []


def test_grid_path_partitioning():

    # Generate grid and a star objects

    path1, path2 = [], []
    goal1, goal2 = [], []
    sq3 = 0.5 / math.sqrt(3)
    grid1 = Grid(
        grid=np.zeros(shape=(10, 10, 10), dtype=bool),
        boxlengths=unit.Quantity(value=Vec3(1, 1, 1), unit=unit.angstrom),
        offset=Vec3(0, 0, 0) * unit.angstrom,
    )
    grid2 = Grid(
        grid=np.zeros(shape=(10, 10, 10), dtype=bool),
        boxlengths=unit.Quantity(value=Vec3(2, 2, 2), unit=unit.angstrom),
        offset=Vec3(-9, -5, -6) * unit.angstrom,
    )

    for i in range(5):
        path1.append(GridNode(x=i, y=i, z=i))
        path2.append(GridNode(x=i, y=0, z=1))

    escape_room1 = GridEscapeRoom(grid=grid1, start=GridNode(x=0, y=0, z=0))
    escape_room2 = GridEscapeRoom(grid=grid2, start=GridNode(x=0, y=0, z=0))

    # Generate paths

    escape_room1.shortest_path = path1
    escape_room2.shortest_path = path2

    path1 = escape_room1.get_path_for_sampling(0.05 * unit.nanometer)
    path2 = escape_room2.get_path_for_sampling(0.5 * unit.angstrom)

    # Generate desired outcomes

    for i in range(len(path1)):
        goal1.append(
            unit.Quantity(Vec3(x=i * sq3, y=i * sq3, z=i * sq3), unit=unit.angstrom)
        )

    for i in range(len(path2)):
        goal2.append(
            unit.Quantity(Vec3(x=i / 2 - 9, y=0 - 5, z=2 - 6), unit=unit.angstrom)
        )

    # Check generated paths for tested outcome

    for i in range(len(path1)):
        for j in range(3):
            assert round(path1[i][j].value_in_unit(path1[i].unit), 5) == round(
                goal1[i][j].value_in_unit(goal1[i].unit), 5
            )
    for i in range(len(path2)):
        for j in range(3):
            print(i, j)
            assert round(path2[i][j].value_in_unit(path2[i].unit), 5) == round(
                goal2[i][j].value_in_unit(goal2[i].unit), 5
            )


def test_tree_successor():
    nodes = []
    for i in range(5):
        nodes.append(unit.Quantity(Vec3(i + 1, i + 1, i + 2), unit.nanometer))
    tree = Tree(coordinates=nodes)
    start = unit.Quantity(Vec3(0, 0, 0), unit.nanometer)
    escape_room = TreeEscapeRoom(tree=tree, start=start)
    parent = TreeNode()
    children = escape_room.generate_successors(
        parent=parent, resolution=1, wall_radius=0.12
    )
    supposedchildren = []
    for i in tree.POSSIBLE_NEIGHBOURS:
        supposedchildren.append(
            TreeNode(
                x=i[0] + start.x,
                y=i[1] + start.y,
                z=i[2] + start.z,
            )
        )
    for i, c in enumerate(children):
        assert c.get_grid_coordinates() == supposedchildren[i].get_grid_coordinates()


def test_path_finding():
    escape_room = TreeEscapeRoom.from_files(pipeline.system_info)
    path = escape_room.find_path()
    assert path != []


def test_tree_path_partitioning():
    escape_room = TreeEscapeRoom.from_files(pipeline.system_info)
    path = escape_room.find_path()
    newp = escape_room.get_path_for_sampling()
    for i, p in enumerate(newp):
        try:
            dist = Tree.calculate_euclidean_distance(p, newp[i + 1])
            assert round(dist, 3) == 0.1
        except IndexError:
            pass


def test_load_path():
    pmf = PMFCalculator(
        simulation_parameters=pipeline.simulation_parameters,
        system_info=pipeline.system_info,
        trajectory_directory="UmbrellaPipeline/data",
        original_path_interval=1 * unit.nanometer,
    )
    a = pmf.load_original_path()
    b = pmf.load_sampled_coordinates()
    assert len(a) == 43
    assert len(b) == 43 * 500


def test_pymbar_pmf():
    pmf = PMFCalculator(
        simulation_parameters=pipeline.simulation_parameters,
        system_info=pipeline.system_info,
        trajectory_directory="UmbrellaPipeline/data",
        original_path_interval=1 * unit.nanometer,
    )
    a = pmf.load_original_path()
    b = pmf.load_sampled_coordinates()
    p, e = pmf.calculate_pmf()
    assert len(a) == len(p)
