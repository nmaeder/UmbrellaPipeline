import math
from typing import List
import openmm.unit as unit

import gemmi
import numpy as np
from grid import Grid
from tree import Tree
from node import (
    TreeNode,
    GridNode,
)


class AStar3D:
    def __init__(
        self,
        start: GridNode or TreeNode,
        end: GridNode or TreeNode,
    ) -> None:

        self.start = start
        self.end = end
        self.shortestPath = []

    def __str__(self) -> str:
        ret = ""
        for i in self.shortestPath:
            ret += f"{str(i)}\n"
        return ret

    def __repr__(self) -> str:
        ret = ""
        for i in self.shortestPath:
            ret += f"{str(i)}\n"
        return ret

    def isGoalreached(self, node: GridNode):
        pass


class GridAStar(AStar3D):
    def __init__(
        self, grid: Grid, start: GridNode, end: GridNode, method: str = "diagonal"
    ) -> None:
        super().__init__(start=start, end=end, method=method)
        self.grid = grid
        self.method = method

    def isGoalReached(self, node: GridNode) -> bool:
        """
        checks if a node has the same coordinates as the end node defined in this class.
        Args:
            node (GridNode): node to be checked
        Returns:
            bool: Returns true if input node is the destination.
        """
        return node == self.end

    def backtracePath(self) -> List[GridNode]:
        """
        removes all the unsuccesfull path legs and returns the direct path from start to end. Does only make sense to run with or after self.aStar3D
        Returns:
            List[GridNode]: direct path from self.start to self.end
        """
        self.shortestPath.reverse()
        new = []
        current = self.shortestPath[0]
        while current:
            new.insert(0, current)
            current = current.parent
        self.shortestPath = new
        return self.shortestPath

    def generateSuccessors(self, parent: GridNode, pathsize: int) -> List[GridNode]:
        """
        generates possible successors for the a star grid
        Args:
            parent (GridNode): parent Node
            pathsize (int): diameter of path that has to be free
        Returns:
            List[GridNode]: list of possible successor nodes
        """
        possibleNeighbours = [
            [-1, -1, -1],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, 0, -1],
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, -1],
            [-1, 1, 0],
            [-1, 1, 1],
            [0, -1, -1],
            [0, -1, 0],
            [0, -1, 1],
            [0, 0, -1],
            [0, 0, 1],
            [0, 1, -1],
            [0, 1, 0],
            [0, 1, 1],
            [1, -1, -1],
            [1, -1, 0],
            [1, -1, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, -1],
            [1, 1, 0],
            [1, 1, 1],
        ]
        ret = []
        for neighbour in possibleNeighbours:
            child = GridNode.fromCoords(
                [a + b for a, b in zip(parent.getCoordinates(), neighbour)]
            )
            if not self.grid.positionIsValid(node=child) or self.grid.positionIsBlocked(
                node=child
            ):
                continue
            child.g = parent.g + self.grid.estimateDiagonalH(
                node=parent, destination=child
            )
            child.h = self.grid.getDistanceToTrue(child)
            child.parent = parent
            ret.append(child)
        return ret

    def aStar3D(self, backtrace: bool = True, pathsize: int = 0) -> List[GridNode]:
        """
        lets a slightly addapted (greedier) version of the A* star algorithm search for the shortest path between start end end point.
        Args:
            backtrace (bool): set to false if you want the function to return all searched nodes instead of the shortet path: Defaults to True
        Returns:
            List[GridNode]: shortest path from a to b if class settint backtrace is true. else returns all searched nodes.
        """
        if self.isGoalReached(node=self.start):
            self.shortestPath.append(self.start)
            return self.shortestPath
        self.start.g = 0
        self.start.h = self.grid.getDistanceToTrue(self.start)
        openList = [self.start]
        while openList:
            q = openList[0]
            for node in openList:
                if node.h > q.h:
                    q = node
            openList.remove(q)
            children = self.generateSuccessors(parent=q, pathsize=pathsize)
            for child in children:
                if not self.grid.getDistanceToTrue(child):
                    self.shortestPath.append(q)
                    self.shortestPath.append(child)
                    return self.shortestPath if not backtrace else self.backtracePath()
                if any(
                    (listEntry == child and listEntry.g <= child.g)
                    for listEntry in openList
                ):
                    continue
                if any(
                    (listEntry == child and listEntry.g <= child.g)
                    for listEntry in self.shortestPath
                ):
                    continue
                else:
                    openList.insert(0, child)
            self.shortestPath.append(q)
        return []

    def pathToCcp4(self, filename: str):
        """
        Write out CCP4 density map of the path in the grid. good for visualization in VMD/pymol.
        Args:
            filename (str): path the file should be written to.
        Returns:
            None: Nothing
        """
        if not filename.endswith(".ccp4"):
            filename += ".ccp4"
        print("Hang in there, this can take a while (~1 Minute)")
        pathgrid = np.zeros(shape=(self.grid.x, self.grid.y, self.grid.z), dtype=bool)
        for i in self.shortestPath:
            pathgrid[i.getCoordinates()[0]][i.getCoordinates()[1]][
                i.getCoordinates()[2]
            ] = True
        ccp4_map = gemmi.Ccp4Map()
        ccp4_map.grid = gemmi.FloatGrid(self.grid.grid.astype(np.float32))
        ccp4_map.update_ccp4_header()
        ccp4_map.write_ccp4_map(filename)
        return None


class TreeAStar(AStar3D):
    def __init__(self, start: TreeNode, end: TreeNode, tree: Tree) -> None:
        super().__init__(start, end)
        self.tree = tree

    def estimateEuclideanH(self, node: GridNode, destination: GridNode) -> float:
        """
        estimates euclidean distance heuristics between node and destination.
        Args:
            node (GridNode): point a
            destination (GridNode): point b
        Returns:
            float: euclidean distance estimate
        """
        return math.sqrt(
            (node.x - destination.x) ** 2
            + (node.y - destination.y) ** 2
            + (node.z - destination.z) ** 2
        )

    def isGoalReached(self, node: TreeNode):
        return self.tree.distanceToProtein(node=node) > 5 * unit.nanometer

    def generateSuccessors(
        self, parent: TreeNode, stepsize: unit.Quantity = 0.01 * unit.nanometer
    ) -> List[TreeNode]:
        """
        generates possible successors for the a star grid
        Args:
            parent (Node): parent Node
            pathsize (int): diameter of path that has to be free
        Returns:
            List[Node]: list of possible successor nodes
        """
        d3 = stepsize / math.sqrt(3)
        d2 = stepsize / math.sqrt(2)
        d1 = stepsize / math.sqrt(1)
        possibleNeighbours = [
            [d1, 0, 0],
            [-d1, 0, 0],
            [0, d1, 0],
            [0, -d1, 0],
            [0, 0, d1],
            [0, 0, -d1],
            [d2, d2, 0],
            [-d2, d2, 0],
            [d2, -d2, 0],
            [-d2, -d2, 0],
            [0, d2, d2],
            [0, -d2, d2],
            [0, d2, -d2],
            [0, -d2, -d2],
            [d2, 0, d2],
            [-d2, 0, d2],
            [d2, 0, -d2],
            [-d2, 0, -d2],
            [d3, d3, d3],
            [-d3, d3, d3],
            [d3, -d3, d3],
            [d3, d3, -d3],
            [-d3, -d3, d3],
            [d3, -d3, -d3],
            [-d3, d3, -d3],
            [-d3, -d3, -d3],
        ]
        ret = []
        for num, neighbour in enumerate(possibleNeighbours):
            child = TreeNode.fromCoords(
                [
                    a + b.value_in_units(parent.unit)
                    for a, b in zip(parent.getCoordinates(), neighbour)
                ],
                unit=parent.unit,
            )
            dist = self.tree.distanceToProtein(node=child)
            child.g = parent.g + self.estimateEuclideanH(parent, child)
            child.h = dist
            child.parent = parent
            ret.append(child)
        return ret

    def aStar3D(self, backtrace: bool = True, pathsize: int = 0) -> List[TreeNode]:
        """
        lets a slightly addapted (greedier) version of the A* star algorithm search for the shortest path between start end end point.
        Args:
            backtrace (bool): set to false if you want the function to return all searched nodes instead of the shortet path: Defaults to True
        Returns:
            List[TreeNode]: shortest path from a to b if class settint backtrace is true. else returns all searched nodes.
        """
        self.start.g = 0
        self.start.h = self.tree.distanceToProtein(node=self.start)
        openList = [self.start]
        while openList:
            q = openList[0]
            for node in openList:
                if node.h > q.h:
                    q = node
            openList.remove(q)
            children = self.generateSuccessors(parent=q, pathsize=pathsize)
            for child in children:
                if self.isGoalReached(child):
                    self.shortestPath.append(q)
                    self.shortestPath.append(child)
                    return self.shortestPath if not backtrace else self.backtracePath()
                if any(
                    (listEntry == child and listEntry.g <= child.g)
                    for listEntry in openList
                ):
                    continue
                if any(
                    (listEntry == child and listEntry.g <= child.g)
                    for listEntry in self.shortestPath
                ):
                    continue
                else:
                    openList.insert(0, child)
            self.shortestPath.append(q)
        return []
