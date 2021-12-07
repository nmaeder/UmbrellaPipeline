import math
import numpy as np
import openmm.app as app
import openmm.unit as unit
from openmm import Vec3
from UmbrellaPipeline.path_generation import (
    Grid,
    GridEscapeRoom,
    GridNode,
    Tree,
    TreeEscapeRoom,
    TreeNode,
)

pdb = "UmbrellaPipeline/data/step5_input.pdb"
psf = "UmbrellaPipeline/data/step5_input.psf"


def read_pdb(pdb: str = pdb) -> app.PDBFile:
    return app.PDBFile(pdb)


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
    grid = Grid.from_files(pdb=pdb, psf=psf, gridsize=3 * unit.angstrom)
    node = grid.node_from_files(psf=psf, pdb=pdb, name="UNL")
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
    pdb = read_pdb()
    tree = Tree.from_files(pdb=pdb, psf=psf)
    node = tree.node_from_files(psf=psf, pdb=pdb, name="UNL")
    assert not tree.position_is_blocked(node=node)
    box = []
    for i in range(3):
        box.append(min([row[i] for row in pdb.positions]))
        box.append(max([row[i] for row in pdb.positions]))
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
