from openmm import Vec3, unit

from UmbrellaPipeline.path_finding import Tree, TreeNode


def test_tree_generation():
    nodes = []
    nodesa = []
    nodesnu = []
    for i in range(5):
        nodes.append(unit.Quantity(Vec3(i, i + 1, i + 2), unit.nanometer))
        nodesa.append(unit.Quantity(Vec3(i, i + 1, i + 2), unit.angstrom))
        nodesnu.append([i, i + 1, i + 2])
    tree = Tree(coordinates=nodes)
    treea = Tree(coordinates=nodesa)
    treenu = Tree(coordinates=nodesnu)
    for i, d in enumerate(tree.tree.data):
        for j, e in enumerate(d):
            if not j == 0:
                assert e != treea.tree.data[i][j]


def test_functions():
    nodes = []
    for i in range(5):
        nodes.append(unit.Quantity(Vec3(i, i + 1, i + 2), unit.nanometer))
    tree = Tree(coordinates=nodes)


    Node1 = TreeNode(x=1, y=2, z=3)
    Node2 = TreeNode(x=1, y=23, z=3)

    assert tree.get_distance_to_wall(Node1) < 0 
    assert tree.get_distance_to_wall(Node2) > 0 
