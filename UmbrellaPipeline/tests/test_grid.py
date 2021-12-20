import os
import numpy as np
import openmm.unit as unit

from UmbrellaPipeline.path_generation import Grid, GridNode


def test_initialize_no_files():
    grid = Grid(
        grid=np.zeros(shape=(100, 32, 24), dtype=bool),
        boxlengths=[0 * unit.angstrom, 1 * unit.angstrom, 2 * unit.angstrom],
    )
    assert grid.dtype == bool
    assert grid.x == 100
    assert grid.b == 1 * unit.angstrom


def test_initialize_from_values():
    grid = Grid(x=23, y=23, z=23, dtype=bool)
    try:
        failed = Grid(x=23, y=-23, z=23)
    except ValueError:
        pass


def test_functions():
    grid = Grid(
        grid=np.zeros(shape=(10, 10, 10), dtype=bool),
        boxlengths=[1 * unit.angstrom, 1 * unit.angstrom, 1 * unit.angstrom],
        offset=[0 * unit.angstrom, 1 * unit.angstrom, 2 * unit.angstrom],
    )
    grid.grid[5][5][5] = True

    node1 = GridNode(x=4, y=4, z=4)
    node2 = GridNode(x=5, y=5, z=5)
    node3 = GridNode(x=1, y=1, z=10)

    assert grid.position_is_valid(node=node1) == True
    assert grid.position_is_valid(node=node3) == False

    assert grid.get_grid_value(node=node1) == False
    assert grid.get_grid_value(node=node2) == True

    assert grid.position_is_blocked(node=node1) == False
    assert grid.position_is_blocked(node=node2) == True

    assert grid.to_cartesian_coordinates() == [
        [
            unit.Quantity(5, unit.angstrom),
            unit.Quantity(6, unit.angstrom),
            unit.Quantity(7, unit.angstrom),
        ]
    ]


def test_distance_calculations():
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

    assert grid.get_distance_to_protein(node=node1) == 1
    assert round(grid.get_distance_to_protein(node=node2), 5) == round(
        grid.calculate_diagonal_distance(node=node2, destination=node5), 5
    )
    assert round(grid.get_distance_to_protein(node=node3), 5) == round(
        grid.calculate_diagonal_distance(node=node3, destination=node5), 5
    )
    assert round(grid.get_distance_to_protein(node=node4), 5) == round(
        grid.calculate_diagonal_distance(node=node4, destination=node5), 5
    )

    assert (
        round(grid.calculate_diagonal_distance(node=node3, destination=node5), 4)
        == 1.7321
    )
    assert (
        round(grid.calculate_diagonal_distance(node=node2, destination=node5), 4)
        == 1.4142
    )
    assert grid.calculate_diagonal_distance(node=node1, destination=node5) == 1


def test_write_ccp4():
    grid = Grid(grid=np.zeros(shape=(10, 10, 10), dtype=bool))
    grid.to_ccp4("UmbrellaPipeline/tests/test.ccp4")
    assert os.path.exists("UmbrellaPipeline/tests/test.ccp4")
    os.remove("UmbrellaPipeline/tests/test.ccp4")
    grid.to_ccp4("UmbrellaPipeline/tests/test")
    assert os.path.exists("UmbrellaPipeline/tests/test.ccp4")
    os.remove("UmbrellaPipeline/tests/test.ccp4")
