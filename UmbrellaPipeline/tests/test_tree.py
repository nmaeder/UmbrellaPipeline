import openmm.app as app
import openmm.unit as unit
from openmm import Vec3
import pytest

from UmbrellaPipeline.path_generation import Tree, TreeNode


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
