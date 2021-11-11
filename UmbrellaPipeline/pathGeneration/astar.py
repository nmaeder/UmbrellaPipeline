import math
from typing import List

import gemmi
import numpy as np

from grid import Grid, Node


class AStar3D:
    """
    Class that holds and calculates all information as well as the shortest path from start to end by deploying the A* algorithmm
    """

    def __init__(
        self,
        start: Node or List[int],
        end: Node or List[int],
        grid: Grid,
        method: str = "diagonal",
    ) -> None:
        """
        class that holds all information necessary to generate shortest path.

        Args:
            start (NodeorList[int]): start node
            end (NodeorList[int]): destination node
            grid (Grid): grid in which path should be searched
            method (str, optional): give either diagoal or euclidean. Defaults to 'diagonal'.
        """
        self.grid = grid
        self.start = (
            Node.fromCoords(start.getCoordinates())
            if type(start) is not Node
            else start
        )
        self.end = (
            Node.fromCoords(end.getCoordinates()) if type(end) is not Node else end
        )
        self.shortestPath: List[Node] = []
        self.method = method

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

    def estimateDiagonalH(self, node: Node, destination: Node) -> float:
        """
        estimates diagonal distance heuristics between node and destination.

        Args:
            node (Node): point a
            destination (Node): point b

        Returns:
            float: diagonal distance estimate
        """
        dx = abs(node.x - destination.x)
        dy = abs(node.y - destination.y)
        dz = abs(node.z - destination.z)
        dmin = min(dx, dy, dz)
        dmax = max(dx, dy, dz)
        dmid = dx + dy + dz - dmax - dmin
        D3 = math.sqrt(3)
        D2 = math.sqrt(2)
        D1 = math.sqrt(1)
        return (D3 - D2) * dmin + (D2 - D1) * dmid + D1 * dmax

    def estimateEuclideanH(self, node: Node, destination: Node) -> float:
        """
        estimates euclidean distance heuristics between node and destination.

        Args:
            node (Node): point a
            destination (Node): point b

        Returns:
            float: euclidean distance estimate
        """
        return math.sqrt(
            (node.x - destination.x) ** 2
            + (node.y - destination.y) ** 2
            + (node.z - destination.z) ** 2
        )

    def isGoalReached(self, node: Node) -> bool:
        """
        checks if a node has the same coordinates as the end node defined in this class.

        Args:
            node (Node): node to be checked

        Returns:
            bool: Returns true if input node is the destination.
        """
        return node == self.end

    def backtracePath(self) -> List[Node]:
        """
        removes all the unsuccesfull path legs and returns the direct path from start to end. Does only make sense to run with or after self.aStar3D

        Returns:
            List[Node]: direct path from self.start to self.end
        """
        self.shortestPath.reverse()
        new = []
        current = self.shortestPath[0]
        while current:
            new.insert(0, current)
            current = current.parent
        self.shortestPath = new
        return self.shortestPath

    def generateSuccessors(self, parent: Node, pathsize: int) -> List[Node]:
        """
        generates possible successors for the a star grid

        Args:
            parent (Node): parent Node
            pathsize (int): diameter of path that has to be free

        Returns:
            List[Node]: list of possible successor nodes
        """
        possibleNeighbours = [
            [-1, -1, -1],
            [0, -1, -1],
            [1, -1, -1],
            [-1, 0, -1],
            [0, 0, -1],
            [1, 0, -1],
            [-1, 1, -1],
            [0, 1, -1],
            [1, 1, -1],
            [-1, -1, 0],
            [0, -1, 0],
            [1, -1, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [-1, 1, 0],
            [0, 1, 0],
            [1, 1, 0],
            [-1, -1, 1],
            [0, -1, 1],
            [1, -1, 1],
            [-1, 0, 1],
            [0, 0, 1],
            [1, 0, 1],
            [-1, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
        ret = []
        for neighbour in possibleNeighbours:
            child = Node.fromCoords(
                [a + b for a, b in zip(parent.getCoordinates(), neighbour)]
            )
            if not self.grid.positionIsValid(node=child) or self.grid.positionIsBlocked(
                node=child
            ):
                continue
            if pathsize:
                self.grid.areSurroundingsBlocked(node=child, pathsize=pathsize)
            if self.method == "diagonal":
                child.g = parent.g + self.estimateDiagonalH(parent, child)
                child.h = self.estimateDiagonalH(child, self.end)
                child.f = child.g + child.h
            else:
                child.g = parent.g + self.estimateEuclideanH(parent, child)
                child.h = self.estimateEuclideanH(child, self.end)
                child.f = child.g + child.h
            child.parent = parent
            ret.append(child)
        return ret

    def aStar3D(self, backtrace: bool = True, pathsize: int = 0) -> List[Node]:
        """
        lets a slightly addapted (greedier) version of the A* star algorithm search for the shortest path between start end end point.
        TODO: adaptive path size.

        Args:
            backtrace (bool): set to false if you want the function to return all searched nodes instead of the shortet path: Defaults to True

        Returns:
            List[Node]: shortest path from a to b if class settint backtrace is true. else returns all searched nodes.
        """
        if self.isGoalReached(node=self.start):
            self.shortestPath.append(self.start)
            return self.shortestPath
        self.start.g = 0
        self.start.h = (
            self.estimateDiagonalH(node=self.start, destination=self.end)
            if self.method == "diagonal"
            else self.estimateEuclideanH(node=self.start, destination=self.end)
        )
        self.start.f = self.start.h
        openList = [self.start]
        while openList:
            q = openList[0]
            for node in openList:
                if node.f < q.f and node.h < q.h:
                    q = node
            openList.remove(q)
            children = self.generateSuccessors(parent=q, pathsize=pathsize)
            for child in children:
                if self.isGoalReached(child):
                    self.shortestPath.append(q)
                    self.shortestPath.append(child)
                    return self.shortestPath if not backtrace else self.backtracePath()
                if any(
                    (listEntry == child and listEntry.f <= child.f)
                    for listEntry in openList
                ):
                    continue
                if any(
                    (listEntry == child and listEntry.f <= child.f)
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
