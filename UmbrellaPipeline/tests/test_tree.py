from UmbrellaPipeline.pathGeneration import (
    Tree,
    TreeNode,
)

import openmm.unit as unit
from openmm import Vec3
import openmm.app as app

pdb = "UmbrellaPipeline/data/step5_input.pdb"
psf = "UmbrellaPipeline/data/step5_input.psf"


def testTreeGenration():
    nodes = []
    nodesnu = []
    for i in range(5):
        nodes.append(unit.Quantity(Vec3(i, i + 1, i + 2), unit.nanometer))
        nodesnu.append([i, i + 1, i + 2])
    tree = Tree(coordinates=nodes)
    tree = Tree(coordinates=nodes, _unit=unit.angstrom)
    tree = Tree(coordinates=nodesnu, _unit=unit.angstrom)
    try:
        tree = Tree(coordinates=nodesnu)
    except ValueError:
        pass


def testTreeGenerationFromFiles():
    tree = Tree.treeFromFiles(pdb=pdb, psf=psf)
    pdbo = app.PDBFile(pdb)
    psfo = app.CharmmPsfFile(psf)
    tree = Tree.treeFromFiles(pdb=pdbo, psf=psfo)


def testNodeFromFiles():
    tree = Tree.treeFromFiles(pdb=pdb, psf=psf)
    node = tree.nodeFromFiles(psf=psf, pdb=pdb, name="UNL")


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