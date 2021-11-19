import math
from typing import List

import gemmi
import numpy as np
import openmm.unit as unit
from UmbrellaPipeline.pathGeneration.grid import Grid
from UmbrellaPipeline.pathGeneration.node import GridNode, TreeNode
from UmbrellaPipeline.pathGeneration.tree import Tree


class AStar3D:
    """
    Bass class for the AStar3D class
    """

    def __init__(
        self, start: GridNode or TreeNode, end: GridNode or TreeNode = None
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

    def isEndReached(self, node: TreeNode or GridNode = None):
        """
        Not in use at the moment. Plan is to add support, so one can give a specific endpoint for the path.

        Args:
            node (TreeNodeorGridNode, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if node:
            return node == self.end
        else:
            return self.end in self.shortestPath


class GridAStar(AStar3D):
    def __init__(
        self,
        grid: Grid,
        start: GridNode,
        end: GridNode = None,
    ) -> None:
        super().__init__(start=start, end=end)
        self.grid = grid
        self.shortestPath: List[GridNode] = []

    def isGoalReached(
        self, node: GridNode, distance: unit.Quantity = None
    ) -> bool:
        """
        Checks wether the given node has the distance to the neares True grid point.
        node.h is in "node units" so the distance is divided by the gridcell size to get its value in "grid units" as well.
        If no distance is given, the end point is reached when node is outside the grid
        Args:
            node (GridNode): node to be checked
        Returns:
            bool: Returns true if input node is the destination.
        """
        try: 
            if node.h > distance / self.grid.a:
                return True
        except TypeError:    
            return not self.grid.positionIsValid(node)

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

    def generateSuccessors(self, parent: GridNode) -> List[GridNode]:
        """
        generates possible successors for the a star grid
        Args:
            parent (GridNode): parent Node
            pathsize (int): diameter of path that has to be free
        Returns:
            List[GridNode]: list of possible successor nodes
        """
        ret = []
        for neighbour in self.grid.possibleNeighbours:
            try:
                child = GridNode.fromCoords(
                    [a + b for a, b in zip(parent.getCoordinates(), neighbour)]
                )
            except ValueError:
                continue
            if not self.grid.positionIsValid(node=child):
                continue
            if self.grid.positionIsBlocked(node=child):
                continue
            child.g = parent.g + self.grid.estimateDiagonalH(
                node=parent, destination=child
            )
            child.h = self.grid.getDistanceToTrue(child)
            child.parent = parent
            ret.append(child)
        return ret

    def aStar3D(self, backtrace: bool = True, distance:unit.Quantity = 2*unit.nanometer) -> List[GridNode]:
        """
        lets a slightly addapted (greedier) version of the A* star algorithm search for the shortest path between start end end point.
        Args:
            backtrace (bool): set to false if you want the function to return all searched nodes instead of the shortet path: Defaults to True
            distance (unit.Quantity): If this distance is reached, the algorithm is terminated. if none is given a path is searched until the path leaves the pbc box: Defaults to 2*unit.nanometer
        Returns:
            List[GridNode]: shortest path from a to b if class settint backtrace is true. else returns all searched nodes.
        """
        if self.isGoalReached(node=self.start, distance=distance):
            self.shortestPath.append(self.start)
            return self.shortestPath
        self.start.g = 0
        self.start.h = self.grid.getDistanceToTrue(self.start)
        openList = [self.start]
        while openList:
            q = openList[0]
            for node in openList:
                # prioritize the node that is the fartest from any protein atom. -> will eventually lead out of the case as long as the goal distance is alrge enough.
                if node.h > q.h:
                    q = node
                # if two nodes are equally large apart from the protein, take the one that took less traveling to get there.
                if node.h == q.h and node.g < q.g:
                    q = node
            openList.remove(q)
            children = self.generateSuccessors(parent=q)
            for child in children:
                if self.isGoalReached(node=child, distance=distance):
                    self.shortestPath.append(q)
                    return self.backtracePath() if backtrace else self.shortestPath
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
        ccp4_map.grid = gemmi.FloatGrid(pathgrid.astype(np.float32))
        ccp4_map.update_ccp4_header()
        ccp4_map.write_ccp4_map(filename)
        return None


class TreeAStar(AStar3D):
    def __init__(
        self,
        tree: Tree,
        start: TreeNode,
        end: TreeNode = None,
        pathsize: unit.Quantity = 1.2 * unit.angstrom,
        stepsize: unit.Quantity = 0.01 * unit.nanometer,
    ) -> None:
        super().__init__(start, end)
        self.shortestPath: List[TreeNode] = []
        self.tree = tree
        self.pathsize = pathsize
        self.stepsize = stepsize

    def estimateEuclideanH(self, node: GridNode, destination: GridNode) -> float:
        """
        estimates euclidean distance heuristics between node and destination.
        NOT USED
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

    def isGoalReached(
        self,
        node: TreeNode,
        box: unit.Quantity = None,
        distance:unit.Quantity = None,
    ):
        """
        Checks if the end is reached. either reached when distance bigger than dist_to_protein or, if box is given, when path is outside the box.

        Args:
            node (TreeNode): [description]
            box (unit.Quantity): [description]
            dist_to_protein (unit.Quantity, optional): [description]. Defaults to 2*unit.nanometer.

        Returns:
            [type]: [description]
        """
        try:
            if node.h >= distance:
                return True
        except AttributeError:
            pass
        try:
            return (
                (node.x * node.unit < box[0] or node.x * node.unit > box[1])
                or (node.y * node.unit < box[2] or node.y * node.unit > box[3])
                or (node.z * node.unit < box[4] or node.z * node.unit > box[5]) 
            )
        except (TypeError):
            raise TypeError("Either give distance_to_protein or box mins and maxes!")

    def generateSuccessors(
        self,
        parent: TreeNode,
    ) -> List[TreeNode]:
        """
        generates possible successors for the a star grid
        Args:
            parent (Node): parent Node
            pathsize (int): diameter of path that has to be free
        Returns:
            List[Node]: list of possible successor nodes
        """
        ret = []
        for neighbour in self.tree.possibleNeighbours:
            child = TreeNode.fromCoords(
                [
                    a + b * self.stepsize.value_in_unit(parent.unit)
                    for a, b in zip(parent.coordsForQuery(parent.unit), neighbour)
                ],
                _unit=parent.unit,
                parent=parent,
            )
            dist = self.tree.distanceToProtein(node=child)
            if dist < self.pathsize:
                continue
            child.g = parent.g + self.estimateEuclideanH(node=parent, destination=child)
            child.h = dist
            ret.append(child)
        return ret

    def backtracePath(self) -> List[TreeNode]:
        """
        removes all the unsuccesfull path legs and returns the direct path from start to end. Does only make sense to run with or after self.aStar3D
        Returns:
            List[TreeNode]: direct path from self.start to self.end
        """
        self.shortestPath.reverse()
        new = []
        current = self.shortestPath[0]
        while current:
            new.insert(0, current)
            current = current.parent
        self.shortestPath = new
        return self.shortestPath

    def aStar3D(self, backtrace: bool = True, distance:unit.Quantity = None, box:List[unit.Quantity] = None) -> List[TreeNode]:
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
                if node.h == q.h and node.g < q.g:
                    q = node
            openList.remove(q)
            children = self.generateSuccessors(parent=q)
            for child in children:
                if self.isGoalReached(node=child, box=box, distance=distance):
                    self.shortestPath.append(q)
                    return self.shortestPath if not backtrace else self.backtracePath()
                if any(
                    (round(listEntry, 3) == round(child, 3) and listEntry.g <= child.g)
                    for listEntry in openList
                ):
                    continue
                if any(
                    (round(listEntry, 3) == round(child, 3) and listEntry.g <= child.g)
                    for listEntry in self.shortestPath
                ):
                    continue
                else:
                    openList.insert(0, child)
            self.shortestPath.append(q)
        return []
