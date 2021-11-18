from UmbrellaPipeline.pathGeneration import (
    GridAStar,
    TreeAStar,
    Tree,
    Grid,
    TreeNode,
    GridNode,
    gen_box,
    node,
)
import numpy as np
import openmm.unit as unit
import openmm.app as app
from openmm import Vec3
import math

pdb = "UmbrellaPipeline/data/step5_input.pdb"
psf = "UmbrellaPipeline/data/step5_input.psf"


def testGridAStarBasic():
    grid = Grid(grid=np.zeros(shape=(10, 10, 10), dtype=bool))
    start = GridNode(0, 0, 0)
    end = GridNode(9, 9, 9)
    astar = GridAStar(grid=grid, start=start, end=end)

    assert not astar.isEndReached()
    assert not astar.isEndReached(node=start)
    assert astar.isEndReached(node=end)
    astar.shortestPath.append(end)
    assert astar.isEndReached()


def testGridSuccessor():
    grid = Grid(grid=np.zeros(shape=(10, 10, 10), dtype=bool))
    grid.grid[1][1][1] = True
    start = GridNode(0, 0, 0)
    end = GridNode(9, 9, 9)
    astar = GridAStar(grid=grid, start=start, end=end)
    children = astar.generateSuccessors(parent=start)
    supposedchildren = [
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
    ]

    for i, c in enumerate(children):
        assert c.getCoordinates() == supposedchildren[i]

    children = astar.generateSuccessors(parent=end)
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
        assert c.getCoordinates() == supposedchildren[i]


def testGridPathfinding():
    pdbo = app.PDBFile(pdb)
    psfo = app.CharmmPsfFile(psf)
    grid = Grid.gridFromFiles(pdb=pdbo, psf=psfo, gridsize=1 * unit.angstrom)
    node = grid.nodeFromFiles(psf=psfo, pdb=pdbo, name="UNL")
    assert not grid.positionIsBlocked(node)
    astar = GridAStar(grid=grid, start=node)
    path = astar.aStar3D()
    assert path != []


def testTreeSuccessor():
    nodes = []
    for i in range(5):
        nodes.append(unit.Quantity(Vec3(i + 1, i + 1, i + 2), unit.nanometer))
    tree = Tree(coordinates=nodes)
    start = TreeNode(0, 0, 0)
    astar = TreeAStar(tree=tree, start=start)
    children = astar.generateSuccessors(parent=start)
    supposedchildren = []
    for i in tree.possibleNeighbours:
        supposedchildren.append(
            TreeNode(
                x=i[0] * astar.stepsize.value_in_unit(tree.unit),
                y=i[1] * astar.stepsize.value_in_unit(tree.unit),
                z=i[2] * astar.stepsize.value_in_unit(tree.unit),
            )
        )
    for i, c in enumerate(children):
        assert c.getCoordinates() == supposedchildren[i].getCoordinates().in_units_of(
            c.unit
        )


def testTreePathfinding():
    pdbo = app.PDBFile(pdb)
    psfo = app.CharmmPsfFile(psf)
    tree = Tree.treeFromFiles(pdb=pdbo, psf=psfo)
    node = tree.nodeFromFiles(psf=psfo, pdb=pdbo, name="UNL")
    assert not tree.positionIsBlocked(node=node)
    box = []
    for i in range(3):
        box.append(min([row[i] for row in pdbo.positions]))
        box.append(max([row[i] for row in pdbo.positions]))
    astar = TreeAStar(tree=tree, start=node, stepsize=0.5 * unit.angstrom)
    path = astar.aStar3D(box=box)
    assert path != []
