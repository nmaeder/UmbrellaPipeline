from openmm import Vec3, unit
import pytest
from sklearn.utils import column_or_1d

from UmbrellaPipeline.path_finding import Tree, Node


def test_tree_generation():
    nodes = []
    nodes2 = []
    nodesnu = []
    for i in range(5):
        nodes.append(unit.Quantity(Vec3(i, i + 1, i + 2), unit.nanometer))
        nodes2.append(unit.Quantity(Vec3(i, i + 1, i + 2), unit.angstrom))
        nodesnu.append([i, i + 1, i + 2])
    tree = Tree(coordinates=nodes)
    tree = Tree(coordinates=nodes2)
    tree = Tree(coordinates=nodesnu)


def test_functions():
    nodes = []
    for i in range(5):
        nodes.append(unit.Quantity(Vec3(i, i + 1, i + 2), unit.nanometer))
    tree = Tree(coordinates=nodes)

    Node1 = Node(x=1, y=2, z=3)
    Node2 = Node(x=1, y=23, z=3)

    coord1 = Node1.get_coordinates_for_query()
    coord2 = Node2.get_coordinates_for_query()

    assert tree.get_distance_to_wall(coordinates=Node1) < 0
    assert tree.get_distance_to_wall(coordinates=Node2) > 0

    assert tree.get_distance_to_wall(coordinates=coord1) < 0
    assert tree.get_distance_to_wall(coordinates=coord2) > 0