from UmbrellaPipeline.pathGeneration import Grid
from UmbrellaPipeline.pathGeneration import Node
from simtk import unit

pdb = "UmbrellaPipeline/data/step5_input.pdb"
psf = "UmbrellaPipeline/data/step5_input.psf"


def testInitializeNode():
    node = Node(x=23.4, y=43.6, z=21.6, unit=unit.nanometer)


def testInitializeNodeFromCoordinates():
    node = Node([23.2, 534.2, 1.2])


def testNodeEquality():
    node = Node(x=23.4, y=43.6, z=21.6, unit=unit.nanometer)
    node2 = Node(x=23.4, y=43.6, z=21.6, unit=unit.nanometer)
    node3 = Node(x=22.4, y=43.6, z=21.6, unit=unit.nanometer)
    assert node == node2, "should be equal"
    assert node != node3, "should not be equal"


def test_initialize_grid():
    grid = Grid.gridFromFiles(pdb=pdb, psf=psf, vdwradius=1.2 * unit.angstrom)
