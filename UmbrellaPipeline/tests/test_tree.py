import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3
from UmbrellaPipeline.pathGeneration import Tree, TreeNode

pdb = "data/step5_input.pdb"
psf = "data/step5_input.psf"

def readPDB(pdb:str = pdb) -> app.PDBFile:
    return app.PDBFile(pdb)

def readPSF(psf:str = psf) -> app.CharmmPsfFile:
    return app.CharmmPsfFile(psf)

def testTreeGenration():
    nodes = []
    nodesnu = []
    for i in range(5):
        nodes.append(unit.Quantity(Vec3(i, i + 1, i + 2), unit.nanometer))
        nodesnu.append([i, i + 1, i + 2])
    tree = Tree(coordinates=nodes)
    tree = Tree(coordinates=nodes, _unit=unit.angstrom)
    tree = Tree(coordinates=nodesnu, _unit=unit.angstrom)
    with pytest.raises(ValueError):
        tree = Tree(coordinates=nodesnu)


def testTreeGenerationFromFiles():
    tree = Tree.treeFromFiles(pdb=pdb, psf=psf)
    pdbo = readPDB()
    psfo = readPSF()
    tree = Tree.treeFromFiles(pdb=pdbo, psf=psfo)


def testNodeFromFiles():
    tree = Tree.treeFromFiles(pdb=pdb, psf=psf)
    node = tree.nodeFromFiles(psf=psf, pdb=pdb, name="UNL")
    node = tree.nodeFromFiles(psf=psf, pdb=pdb, name="UNL", includeHydrogens=False)


def testFunctions():
    nodes = []
    for i in range(5):
        nodes.append(unit.Quantity(Vec3(i, i + 1, i + 2), unit.nanometer))
    tree = Tree(coordinates=nodes)

    Node1 = TreeNode(x=1, y=2, z=3)
    Node2 = TreeNode(x=1, y=23, z=3)

    assert tree.positionIsBlocked(node=Node1)
    assert not tree.positionIsBlocked(node=Node2)

    assert tree.distanceToProtein(node=Node1) < 0 * unit.nanometer
    assert tree.distanceToProtein(node=Node2) > 0 * unit.nanometer
