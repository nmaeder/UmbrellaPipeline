import os

import numpy as np
import openmm.app as app
import openmm.unit as unit
from UmbrellaPipeline.pathGeneration import Grid, GridNode

pdb = "UmbrellaPipeline/data/step5_input.pdb"
psf = "UmbrellaPipeline/data/step5_input.psf"


def testGridAndNodeFromFiles():
    grid = Grid.gridFromFiles(
        pdb=pdb, psf=psf, gridsize=1 * unit.angstrom, vdwradius=1.2 * unit.angstrom
    )
    node = grid.nodeFromFiles(pdb=pdb, psf=psf, name="UNL")
    del grid
    del node
    pdbo = app.PDBFile(pdb)
    psfo = app.CharmmPsfFile(psf)
    grid = Grid.gridFromFiles(
        pdb=pdbo, psf=psfo, gridsize=1 * unit.angstrom, vdwradius=1.2 * unit.angstrom
    )
    node = grid.nodeFromFiles(pdb=pdb, psf=psf, name="UNL")


def testInitializeGridFromNumpy():
    grid = Grid(
        grid=np.zeros(shape=(100, 32, 24), dtype=bool),
        boxlengths=[0 * unit.angstrom, 1 * unit.angstrom, 2 * unit.angstrom],
    )
    assert grid.dtype == bool
    assert grid.x == 100
    assert grid.b == 1 * unit.angstrom


def testInitializeFromValues():
    grid = Grid(x=23, y=23, z=23, dtype=bool)
    try:
        failed = Grid(x=23, y=-23, z=23)
    except ValueError:
        pass


def testFunctions():
    grid = Grid(
        grid=np.zeros(shape=(10, 10, 10), dtype=bool),
        boxlengths=[1 * unit.angstrom, 1 * unit.angstrom, 1 * unit.angstrom],
        offset=[0 * unit.angstrom, 1 * unit.angstrom, 2 * unit.angstrom],
    )
    grid.grid[5][5][5] = True

    node1 = GridNode(x=4, y=4, z=4)
    node2 = GridNode(x=5, y=5, z=5)
    node3 = GridNode(x=1, y=1, z=10)

    assert grid.positionIsValid(node=node1) == True
    assert grid.positionIsValid(node=node3) == False

    assert grid.getGridValue(node=node1) == False
    assert grid.getGridValue(node=node2) == True

    assert grid.positionIsBlocked(node=node1) == False
    assert grid.positionIsBlocked(node=node2) == True

    assert grid.toXYZCoordinates() == [
        [
            unit.Quantity(5, unit.angstrom),
            unit.Quantity(6, unit.angstrom),
            unit.Quantity(7, unit.angstrom),
        ]
    ]


def testDistanceCalculations():
    grid = Grid(
        grid=np.zeros(shape=(10, 10, 10), dtype=bool),
        boxlengths=[1 * unit.angstrom, 1 * unit.angstrom, 1 * unit.angstrom],
        offset=[0 * unit.angstrom, 1 * unit.angstrom, 2 * unit.angstrom],
    )
    grid.grid[5][5][5] = True

    node1 = GridNode(x=4, y=5, z=5)
    node2 = GridNode(x=4, y=4, z=5)
    node3 = GridNode(x=4, y=4, z=4)
    node4 = GridNode(x=3, y=3, z=3)
    node5 = GridNode(x=5, y=5, z=5)

    assert grid.getDistanceToTrue(node=node1) == 1
    assert round(grid.getDistanceToTrue(node=node2), 5) == round(
        grid.estimateDiagonalH(node=node2, destination=node5), 5
    )
    assert round(grid.getDistanceToTrue(node=node3), 5) == round(
        grid.estimateDiagonalH(node=node3, destination=node5), 5
    )
    assert round(grid.getDistanceToTrue(node=node4), 5) == round(
        grid.estimateDiagonalH(node=node4, destination=node5), 5
    )

    assert round(grid.estimateDiagonalH(node=node3, destination=node5), 4) == 1.7321
    assert round(grid.estimateDiagonalH(node=node2, destination=node5), 4) == 1.4142
    assert grid.estimateDiagonalH(node=node1, destination=node5) == 1


def testWriteCCP4():
    grid = Grid(grid=np.zeros(shape=(10, 10, 10), dtype=bool))
    grid.toCcp4("test.ccp4")
    assert os.path.exists("test.ccp4")
    os.remove("test.ccp4")
    grid.toCcp4("test")
    assert os.path.exists("test.ccp4")
    os.remove("test.ccp4")
