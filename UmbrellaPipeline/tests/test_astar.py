import math
from sys import path
import numpy as np
import openmm.app as app
import openmm.unit as unit
from openmm import Vec3
from UmbrellaPipeline.pathGeneration import (
    Grid,
    GridAStar,
    GridNode,
    Tree,
    TreeAStar,
    TreeNode,
)

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


"""def testGridPathPartitioning():

    #Generate grid and a star objects

    path1, path2 = [], []
    goal1, goal2 = [], []
    sq3 = 1/math.sqrt(3)
    sq2 = 1/math.sqrt(2)
    grid1 = Grid(grid=np.zeros(shape=(10, 10, 10), dtype=bool), boxlengths=unit.Quantity(value=Vec3(1,1,1),unit=unit.angstrom))
    grid2 = Grid(grid=np.zeros(shape=(10, 10, 10), dtype=bool), boxlengths=unit.Quantity(value=Vec3(2,2,2),unit=unit.angstrom), offset=Vec3(-9,-5,-6)*unit.angstrom)

    for i in range(5):
        path1.append(GridNode(x=i, y=i, z=i))
        path2.append(GridNode(x=i, y=i, z=1))

    astar1 = GridAStar(grid=grid1, start=GridNode(x=0, y=0, z=0))
    astar2 = GridAStar(grid=grid2, start=GridNode(x=0, y=0, z=0))

    #Generate paths
    
    astar1.shortestPath = path1
    astar2.shortestPath = path2
    
    path1 = astar1.getPathForSampling(0.05 * unit.nanometer)
    path2 = astar2.getPathForSampling(0.5 * unit.angstrom)

    #Generate desired outcomes
    
    for i in range(len(path1)):
        goal1.append(unit.Quantity(Vec3(x=i*sq3/2, y=i*sq3/2, z=i*sq3/2), unit = unit.angstrom))

    for i in range(len(path2)):
        goal2.append(unit.Quantity(Vec3(x=i*sq2/2, y=i*sq2/2, z=1), unit=unit.angstrom))

    #Check generated paths for tested outcome

    for i in range(len(path1)):
        assert round(path1[i], 5) == round(goal1[i], 5)
    for i in range(1,len(path2)):
        assert round(path2[i], 5) == round(goal2[i], 5)
"""


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


def testTreePathPartitioning():

    # Generate trees and a star objects

    path1, path2 = [], []
    goal1, goal2 = [], []
    sq3 = 1 / math.sqrt(3)
    sq2 = 1 / math.sqrt(2)
    tree = Tree([[0, 0, 0]], unit.angstrom)

    for i in range(5):
        path1.append(TreeNode(x=i, y=i, z=i, _unit=unit.angstrom))
        path2.append(TreeNode(x=i, y=-i, z=1, _unit=unit.angstrom))

    astar1 = TreeAStar(tree=tree, start=TreeNode(x=0, y=0, z=0))
    astar2 = TreeAStar(tree=tree, start=TreeNode(x=0, y=0, z=0))

    # Generate paths

    astar1.shortestPath = path1
    astar2.shortestPath = path2

    path1 = astar1.getPathForSampling(0.05 * unit.nanometer)
    path2 = astar2.getPathForSampling(0.5 * unit.angstrom)

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
