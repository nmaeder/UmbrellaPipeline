from openmm import Vec3
import pytest
from UmbrellaPipeline.path_finding import GridNode, Node, TreeNode


def test_generate_basenode():
    node = Node(x=54, y=2.45, z=45)
    with pytest.raises(TypeError):
        Node(x="lk")


def test_basenode():
    node = Node(x=54, y=2.45, z=45)
    assert node.get_coordinates() == [54, 2, 45]


def test_treenode_generation():
    treeNode1 = TreeNode(x=4, y=34.1, z=23.2)
    treeNode2 = TreeNode.from_coords(coords=Vec3(4, 34.1, 23.2))
    assert treeNode2.y == 34
    assert treeNode2.get_coordinates() == [4, 34, 23]
    assert treeNode2.get_coordinates_for_query(start=[1, -2, 3]) == [5, 32, 26]


def test_treenode_eq():
    treeNode1 = TreeNode(x=4, y=34.1, z=23.2)
    treeNode2 = TreeNode.from_coords(coords=[4, 34.1, 23.2])
    treeNode3 = TreeNode.from_coords(coords=[4, 34.1, 83.2])

    assert treeNode1 == treeNode2
    assert treeNode1 != treeNode3


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
