import openmm.unit as unit
from openmm import Vec3
from UmbrellaPipeline.pathGeneration import GridNode, Node, TreeNode


def testGenerate_baseNode():
    node = Node(x=54, y=2.45, z=45)


def testGetCoordinatesBaseNode():
    node = Node(x=54, y=2.45, z=45)
    assert node.getCoordinates() == [54, 2.45, 45]


def testTreeNodeGenerations():
    treeNode1 = TreeNode(x=4, y=34.1, z=23.2, _unit=unit.nanometer)
    treeNode2 = TreeNode.fromCoords(
        coords=unit.Quantity(value=Vec3(4, 34.1, 23.2), unit=unit.nanometer)
    )
    treeNode3 = TreeNode.fromCoords(
        coords=[
            unit.Quantity(value=4, unit=unit.nanometer),
            unit.Quantity(value=34.1, unit=unit.nanometer),
            unit.Quantity(value=23.2, unit=unit.nanometer),
        ]
    )
    treeNode4 = TreeNode.fromCoords(coords=[4, 34.1, 23.2], _unit=unit.angstrom)


def testTeeNodeEquality():
    treeNode1 = TreeNode(x=4, y=34.1, z=23.2, _unit=unit.nanometer)
    treeNode2 = TreeNode.fromCoords(
        coords=unit.Quantity(value=Vec3(4, 34.1, 23.2), unit=unit.nanometer)
    )
    treeNode3 = TreeNode.fromCoords(coords=[4, 34.1, 23.2], _unit=unit.angstrom)

    assert treeNode1 == treeNode2
    assert treeNode1 != treeNode3


def testTreeNodeRound():
    treeNode1 = TreeNode(x=4.234562345235, y=34.1, z=23.2, _unit=unit.nanometer)

    assert round(treeNode1, 3) == unit.Quantity(
        value=Vec3(x=4.235, y=34.100, z=23.200), unit=unit.nanometer
    )


def testTreeNodeQueryCoords():
    treeNode1 = TreeNode(x=4.234562345235, y=34.1, z=23.2, _unit=unit.nanometer)

    assert treeNode1.coordsForQuery(_unit=treeNode1.unit) == [
        unit.Quantity(value=4.234562345235, unit=unit.nanometer).value_in_unit(
            unit.nanometer
        ),
        unit.Quantity(value=34.1, unit=unit.nanometer).value_in_unit(unit.nanometer),
        unit.Quantity(value=23.2, unit=unit.nanometer).value_in_unit(unit.nanometer),
    ]

    assert treeNode1.coordsForQuery(_unit=unit.angstrom) == [
        unit.Quantity(value=42.34562345235, unit=unit.angstrom).value_in_unit(
            unit.angstrom
        ),
        unit.Quantity(value=341, unit=unit.angstrom).value_in_unit(unit.angstrom),
        unit.Quantity(value=232, unit=unit.angstrom).value_in_unit(unit.angstrom),
    ]

    assert treeNode1.coordsForQuery(_unit=unit.angstrom) != [
        unit.Quantity(value=4.234562345235, unit=unit.nanometer).value_in_unit(
            unit.nanometer
        ),
        unit.Quantity(value=34.1, unit=unit.nanometer).value_in_unit(unit.nanometer),
        unit.Quantity(value=23.2, unit=unit.nanometer).value_in_unit(unit.nanometer),
    ]


def testGridNodeGenerations():
    gridNode1 = GridNode(x=42, y=23, z=94)
    gridNode2 = GridNode.fromCoords(coords=[42, 23, 94])
    try:
        gridNode3 = GridNode.fromCoords(coords=[-1, 0, 0])
    except ValueError:
        pass
    assert gridNode1.getCoordinates() == gridNode2.getCoordinates() == [42, 23, 94]


def testGridNodeEquality():
    gridNode1 = GridNode(x=42, y=23, z=94)
    gridNode2 = GridNode.fromCoords(coords=[42, 23, 94])
    gridNode3 = GridNode(x=42, y=233, z=94)

    assert gridNode1 == gridNode2
    assert gridNode1 != gridNode3
