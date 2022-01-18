import os, pytest, sys, time, math
import openmmtools
from openmm import Vec3, unit, app
import numpy as np

from UmbrellaPipeline import UmbrellaPipeline
from UmbrellaPipeline.path_generation import (
    Tree,
    Grid,
    GridNode,
    TreeNode,
    GridEscapeRoom,
    TreeEscapeRoom,
)
from UmbrellaPipeline.sampling import (
    add_harmonic_restraint,
    ghost_ligand,
    ramp_up_coulomb,
    ramp_up_vdw,
    SamplingHydra,
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

pipeline = UmbrellaPipeline(
    ligand_residue_name="unl",
    toppar_stream_file="UmbrellaPipeline/data/toppar/toppar.str",
    toppar_directory="UmbrellaPipeline/data/toppar",
    psf_file="UmbrellaPipeline/data/step5_input.psf",
    pdb_file="UmbrellaPipeline/data/step5_input.pdb",
)


def test_genbox():
    assert pipeline.system_info.psf_object.boxVectors == None
    minC = gen_pbc_box(
        psf=pipeline.system_info.psf_object, pdb=pipeline.system_info.pdb_object
    )
    assert minC == [
        unit.Quantity(value=-0.5613, unit=unit.nanometer),
        unit.Quantity(value=-0.46090000000000003, unit=unit.nanometer),
        unit.Quantity(value=-0.0634, unit=unit.nanometer),
    ]
    assert pipeline.system_info.psf_object.boxVectors == unit.Quantity(
        value=(
            Vec3(x=11.071, y=0.0, z=0.0),
            Vec3(x=0.0, y=10.882600000000002, z=0.0),
            Vec3(x=0.0, y=0.0, z=10.2086),
        ),
        unit=unit.nanometer,
    )


def test_pipeline():
    path = pipeline.generate_path()


def test_add_harmonic_restraint():

    gen_pbc_box(
        psf=pipeline.system_info.psf_object, pdb=pipeline.system_info.pdb_object
    )
    system = pipeline.system_info.psf_object.createSystem(
        params=pipeline.system_info.params
    )
    ind = get_residue_indices(
        atom_list=pipeline.system_info.psf_object.atom_list, name="unl"
    )
    values = [
        10 * unit.kilocalorie_per_mole / (unit.angstrom ** 2),
        1 * unit.angstrom,
        2 * unit.angstrom,
        3 * unit.angstrom,
    ]
    add_harmonic_restraint(system=system, atom_group=ind, values=values)


def test_script_writing():
    output = [
        "run_umbrella_0.sh",
        "run_umbrella_1.sh",
        "serialized_sys.xml",
    ]
    tree = Tree.from_files(
        pdb=pipeline.system_info.pdb_object, psf=pipeline.system_info.psf_object
    )
    st = tree.node_from_files(
        psf=pipeline.system_info.psf_object,
        pdb=pipeline.system_info.pdb_object,
        name="unl",
    ).get_coordinates()
    path = [st, st]

    sim = SamplingHydra(
        openmm_system=pipeline.system_info.psf_object.createSystem(
            pipeline.system_info.params
        ),
        properties=pipeline.simulation_parameters,
        info=pipeline.system_info,
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
    assert get_centroid_coordinates(
        pipeline.system_info.pdb_object.positions, ind1
    ) == unit.Quantity(
        value=Vec3(x=4.800866666666666, y=5.162369444444445, z=5.116966666666667),
        unit=unit.nanometer,
    )
    assert get_centroid_coordinates(
        pipeline.system_info.pdb_object.positions, ind2
    ) == unit.Quantity(
        value=Vec3(x=4.791909523809522, y=5.152095238095239, z=5.13817619047619),
        unit=unit.nanometer,
    )


def test_com_coords():
    gen_pbc_box(
        pdb=pipeline.system_info.pdb_object, psf=pipeline.system_info.psf_object
    )
    system = pipeline.system_info.psf_object.createSystem(
        params=pipeline.system_info.params,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
    )
    ind = get_residue_indices(
        atom_list=pipeline.system_info.psf_object.atom_list, name="unl"
    )
    assert get_center_of_mass_coordinates(
        positions=pipeline.system_info.pdb_object.positions,
        indices=ind,
        masses=system,
        include_hydrogens=True,
    ) == unit.Quantity(
        value=Vec3(x=4.7843512147078195, y=2.570779540502063, z=1.7248368464666914),
        unit=unit.nanometer,
    )
    assert get_center_of_mass_coordinates(
        positions=pipeline.system_info.pdb_object.positions,
        indices=ind,
        masses=system,
        include_hydrogens=False,
    ) == unit.Quantity(
        value=Vec3(x=4.782878540555002, y=2.569887631028307, z=1.7263107176323071),
        unit=unit.nanometer,
    )


@pytest.mark.skipif("W64" in sys.platform, reason="Bash not supported on Windows.")
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


@pytest.mark.skipif("W64" in sys.platform, reason="Bash not supported on Windows.")
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
        topology=pipeline.system_info.pdb_object.topology,
        system=system,
        integrator=integrator,
        platform=platform,
        platformProperties=props,
    )
    simulation.context.setPositions(pipeline.system_info.pdb_object.getPositions())
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
        pdb=pipeline.system_info.pdb_object,
        psf=pipeline.system_info.psf_object,
        gridsize=3 * unit.angstrom,
    )
    node = grid.node_from_files(
        psf=pipeline.system_info.psf_object,
        pdb=pipeline.system_info.pdb_object,
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
    start = TreeNode(0, 0, 0)
    escape_room = TreeEscapeRoom(tree=tree, start=start)
    children = escape_room.generate_successors(parent=start)
    supposedchildren = []
    for i in tree.POSSIBLE_NEIGHBOURS:
        supposedchildren.append(
            TreeNode(
                x=i[0] * escape_room.stepsize.value_in_unit(tree.unit),
                y=i[1] * escape_room.stepsize.value_in_unit(tree.unit),
                z=i[2] * escape_room.stepsize.value_in_unit(tree.unit),
            )
        )
    for i, c in enumerate(children):
        assert c.get_coordinates() == supposedchildren[i].get_coordinates().in_units_of(
            c.unit
        )


def test_tree_path_finding():
    tree = Tree.from_files(
        pdb=pipeline.system_info.pdb_object, psf=pipeline.system_info.psf_object
    )
    node = tree.node_from_files(
        psf=pipeline.system_info.psf_object,
        pdb=pipeline.system_info.pdb_object,
        name="UNL",
    )
    assert not tree.position_is_blocked(node=node)
    box = []
    for i in range(3):
        box.append(min([row[i] for row in pipeline.system_info.pdb_object.positions]))
        box.append(max([row[i] for row in pipeline.system_info.pdb_object.positions]))
    escape_room = TreeEscapeRoom(tree=tree, start=node, stepsize=0.25 * unit.angstrom)
    path = escape_room.escape_room(box=box)
    assert path != []


def test_tree_path_partitioning():

    # Generate trees and a star objects

    path1, path2 = [], []
    goal1, goal2, goal3 = [], [], []
    sq3 = 1 / math.sqrt(3)
    sq2 = 1 / math.sqrt(2)
    tree = Tree([[0, 0, 0]], unit.angstrom)

    for i in range(5):
        path1.append(TreeNode(x=i, y=i, z=i, unit=unit.angstrom))
        path2.append(TreeNode(x=i, y=-i, z=1, unit=unit.angstrom))

    escape_room1 = TreeEscapeRoom(tree=tree, start=TreeNode(x=0, y=0, z=0))
    escape_room2 = TreeEscapeRoom(tree=tree, start=TreeNode(x=0, y=0, z=0))

    # Generate paths

    escape_room1.shortest_path = path1
    escape_room2.shortest_path = path2

    path1 = escape_room1.get_path_for_sampling(0.05 * unit.nanometer)
    path2 = escape_room2.get_path_for_sampling(0.5 * unit.angstrom)
    path3 = escape_room2.get_path_for_sampling(0.5 * unit.nanometer)

    # Generate desired outcomes

    for i in range(len(path1)):
        goal1.append(
            unit.Quantity(
                Vec3(x=i * sq3 / 2, y=i * sq3 / 2, z=i * sq3 / 2), unit=unit.angstrom
            )
        )

    for i in range(len(path2)):
        goal2.append(
            unit.Quantity(Vec3(x=i * sq2 / 2, y=-i * sq2 / 2, z=1), unit=unit.angstrom)
        )

    for i in range(len(path3)):
        goal3.append(
            unit.Quantity(
                Vec3(x=10 * i * sq2 / 2, y=10 * -i * sq2 / 2, z=1), unit=unit.angstrom
            )
        )

    # Check generated paths for tested outcome

    for i in range(len(path1)):
        for j in range(3):
            assert round(path1[i][j].value_in_unit(path1[i].unit), 5) == round(
                goal1[i][j].value_in_unit(path1[i].unit), 5
            )
    for i in range(len(path2)):
        for j in range(3):
            assert round(path2[i][j].value_in_unit(path2[i].unit), 5) == round(
                goal2[i][j].value_in_unit(path2[i].unit), 5
            )

    for i in range(len(path3)):
        for j in range(3):
            assert round(path3[i][j].value_in_unit(path3[i].unit), 5) == round(
                goal3[i][j].value_in_unit(path3[i].unit), 5
            )
