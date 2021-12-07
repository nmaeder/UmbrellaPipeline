import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3
from UmbrellaPipeline.path_generation import Tree, TreeNode

pdb = "UmbrellaPipeline/data/step5_input.pdb"
psf = "UmbrellaPipeline/data/step5_input.psf"


def read_pdb(pdb: str = pdb) -> app.PDBFile:
    return app.PDBFile(pdb)


def read_psf(psf: str = psf) -> app.CharmmPsfFile:
    return app.CharmmPsfFile(psf)


def test_tree_generation():
    nodes = []
    nodesnu = []
    for i in range(5):
        nodes.append(unit.Quantity(Vec3(i, i + 1, i + 2), unit.nanometer))
        nodesnu.append([i, i + 1, i + 2])
    tree = Tree(coordinates=nodes)
    tree = Tree(coordinates=nodes, unit=unit.angstrom)
    tree = Tree(coordinates=nodesnu, unit=unit.angstrom)
    with pytest.raises(ValueError):
        tree = Tree(coordinates=nodesnu)


def test_from_files():
    tree = Tree.from_files(pdb=pdb, psf=psf)
    pdbo = read_pdb()
    psfo = read_psf()
    tree = Tree.from_files(pdb=pdbo, psf=psfo)


def test_node_from_files():
    tree = Tree.from_files(pdb=pdb, psf=psf)
    node = tree.node_from_files(psf=psf, pdb=pdb, name="UNL")
    node = tree.node_from_files(psf=psf, pdb=pdb, name="UNL", include_hydrogens=False)


def test_functions():
    nodes = []
    for i in range(5):
        nodes.append(unit.Quantity(Vec3(i, i + 1, i + 2), unit.nanometer))
    tree = Tree(coordinates=nodes)

    Node1 = TreeNode(x=1, y=2, z=3)
    Node2 = TreeNode(x=1, y=23, z=3)

    assert tree.position_is_blocked(node=Node1)
    assert not tree.position_is_blocked(node=Node2)

    assert tree.get_distance_to_protein(node=Node1) < 0 * unit.nanometer
    assert tree.get_distance_to_protein(node=Node2) > 0 * unit.nanometer
