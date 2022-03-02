from re import X
from openmm import Vec3, unit
import pytest

from UmbrellaPipeline.path_finding import Node


def test_node_basics():
    node1 = Node(x=4, y=35, z=-22)
    node2 = Node.from_coords(coords=Vec3(4, 35, -22))
    node3 = Node.from_coords(
        coords=[
            4,
            35,
            -22,
        ]
    )
    assert node1 == node2
    assert node3.z == -22
    assert node3.get_grid_coordinates() == [4, 35, -22]
    assert node3.get_grid_coordinates() != [4, 35, +22]

    with pytest.raises(TypeError):
        node = Node(x="k")

    with pytest.raises(ValueError):
        node = Node(distance_to_wall=0)


def test_functions():
    start = [3.35, 4.234, -0.1234]
    spacing = 0.4
    node1 = Node(x=3, y=5, z=6)

    a = node1.get_coordinates_for_query(start, spacing)
    b = [round(i, 4) for i in a]
    assert b == [4.55, 6.234, 2.2766]

    start = unit.Quantity(value=Vec3(3.35, 4.234, -0.1234), unit=unit.nanometer)
    spacing = 0.4 * unit.nanometer

    a = node1.get_coordinates_for_query(start, spacing)
    b = [round(i, 4) for i in a]
    assert b == [4.55, 6.234, 2.2766]

    start = unit.Quantity(value=Vec3(3.35, 4.234, -0.1234), unit=unit.angstrom)
    spacing = 0.4 * unit.angstrom

    a = node1.get_coordinates_for_query(start, spacing)
    b = [round(i, 5) for i in a]
    assert b == [0.455, 0.6234, 0.22766]


def test_hash_eq_lt():
    node1 = Node(x=4, y=35, z=-22)
    node3 = Node(x=4, y=35, z=-22, distance_to_wall=3)
    node2 = Node.from_coords(coords=Vec3(4, 35, 22))

    assert node1 != node2
    assert hash(node1) != hash(node2)
    assert hash(node1) == hash(node3)
    assert node3 < node2