import openmm.unit as unit
from openmm import Vec3
from UmbrellaPipeline.path_generation import GridNode, Node, TreeNode


def test_generate_basenode():
    node = Node(x=54, y=2.45, z=45)


def test_basenode():
    node = Node(x=54, y=2.45, z=45)
    assert node.get_coordinates() == [54, 2.45, 45]


def test_treenode_generation():
    treeNode1 = TreeNode(x=4, y=34.1, z=23.2, unit=unit.nanometer)
    treeNode2 = TreeNode.from_coords(
        coords=unit.Quantity(value=Vec3(4, 34.1, 23.2), unit=unit.nanometer)
    )
    treeNode3 = TreeNode.from_coords(
        coords=[
            unit.Quantity(value=4, unit=unit.nanometer),
            unit.Quantity(value=34.1, unit=unit.nanometer),
            unit.Quantity(value=23.2, unit=unit.nanometer),
        ]
    )
    treeNode4 = TreeNode.from_coords(coords=[4, 34.1, 23.2], unit=unit.angstrom)
    assert treeNode4.x == 4
    assert treeNode4.get_coordinates() == unit.Quantity(
        value=Vec3(4, 34.1, 23.2), unit=unit.angstrom
    )
    assert treeNode4.get_coordinates() != unit.Quantity(
        value=Vec3(4, 34.1, 23.2), unit=unit.nanometer
    )


def test_treenode_eq():
    treeNode1 = TreeNode(x=4, y=34.1, z=23.2, unit=unit.nanometer)
    treeNode2 = TreeNode.from_coords(
        coords=unit.Quantity(value=Vec3(4, 34.1, 23.2), unit=unit.nanometer)
    )
    treeNode3 = TreeNode.from_coords(coords=[4, 34.1, 23.2], unit=unit.angstrom)

    assert treeNode1 == treeNode2
    assert treeNode1 != treeNode3


def test_treenode_rnd():
    treeNode1 = TreeNode(x=4.234562345235, y=34.1, z=23.2, unit=unit.nanometer)

    assert round(treeNode1, 2) == unit.Quantity(
        value=Vec3(x=4.23, y=34.10, z=23.20), unit=unit.nanometer
    )


def test_query_coords():
    treeNode1 = TreeNode(x=4.234562345235, y=34.1, z=23.2, unit=unit.nanometer)

    assert treeNode1.get_coordinates_for_query(unit=treeNode1.unit) == [
        unit.Quantity(value=4.234562345235, unit=unit.nanometer).value_in_unit(
            unit.nanometer
        ),
        unit.Quantity(value=34.1, unit=unit.nanometer).value_in_unit(unit.nanometer),
        unit.Quantity(value=23.2, unit=unit.nanometer).value_in_unit(unit.nanometer),
    ]

    assert treeNode1.get_coordinates_for_query(unit=unit.angstrom) == [
        unit.Quantity(value=42.34562345235, unit=unit.angstrom).value_in_unit(
            unit.angstrom
        ),
        unit.Quantity(value=341, unit=unit.angstrom).value_in_unit(unit.angstrom),
        unit.Quantity(value=232, unit=unit.angstrom).value_in_unit(unit.angstrom),
    ]

    assert treeNode1.get_coordinates_for_query(unit=unit.angstrom) != [
        unit.Quantity(value=4.234562345235, unit=unit.nanometer).value_in_unit(
            unit.nanometer
        ),
        unit.Quantity(value=34.1, unit=unit.nanometer).value_in_unit(unit.nanometer),
        unit.Quantity(value=23.2, unit=unit.nanometer).value_in_unit(unit.nanometer),
    ]


def test_gridnode_gen():
    gridNode1 = GridNode(x=42, y=23, z=94)
    gridNode2 = GridNode.from_coords(coords=[42, 23, 94])
    try:
        gridNode3 = GridNode.from_coords(coords=[-1, 0, 0])
    except ValueError:
        pass
    assert gridNode1.get_coordinates() == gridNode2.get_coordinates() == [42, 23, 94]


def test_gridnode_eq():
    gridNode1 = GridNode(x=42, y=23, z=94)
    gridNode2 = GridNode.from_coords(coords=[42, 23, 94])
    gridNode3 = GridNode(x=42, y=233, z=94)

    assert gridNode1 == gridNode2
    assert gridNode1 != gridNode3
