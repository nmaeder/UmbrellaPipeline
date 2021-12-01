import copy
import logging
import math
from typing import List

import gemmi
import numpy as np
import openmm.unit as unit
from openmm import Vec3
from UmbrellaPipeline.pathGeneration.grid import Grid
from UmbrellaPipeline.pathGeneration.node import GridNode, TreeNode
from UmbrellaPipeline.pathGeneration.tree import Tree

logger = logging.getLogger(__name__)


class astar_3d:
    """
    Base class for the astar_3d class
    """

    def __init__(
        self, start: GridNode or TreeNode, end: GridNode or TreeNode = None
    ) -> None:

        self.start = start
        self.end = end
        self.shortest_path = []

    def __str__(self) -> str:
        ret = ""
        for i in self.shortest_path:
            ret += f"{str(i)}\n"
        return ret

    def __repr__(self) -> str:
        ret = ""
        for i in self.shortest_path:
            ret += f"{str(i)}\n"
        return ret

    def is_end_reached(self, node: TreeNode or GridNode = None):
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
            return self.end in self.shortest_path


class GridAStar(astar_3d):
    def __init__(
        self,
        grid: Grid,
        start: GridNode,
        end: GridNode = None,
    ) -> None:
        super().__init__(start=start, end=end)
        self.grid = grid
        self.shortest_path: List[GridNode] = []
        logger.warning(
            msg="You are using the grid Version. This is not encouraged. Try to use the tree version whenever possible."
        )

    def is_goal_reached(self, node: GridNode, distance: unit.Quantity = None) -> bool:
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
            return not self.grid.position_is_valid(node)

    def backtrace_path(self) -> List[GridNode]:
        """
        removes all the unsuccesfull path legs and returns the direct path from start to end. Does only make sense to run with or after self.astar_3d
        Returns:
            List[GridNode]: direct path from self.start to self.end
        """
        self.shortest_path.reverse()
        new = []
        current = self.shortest_path[0]
        while current:
            new.insert(0, current)
            current = current.parent
        self.shortest_path = new
        return self.shortest_path

    def generate_successors(self, parent: GridNode) -> List[GridNode]:
        """
        generates possible successors for the a star grid
        Args:
            parent (GridNode): parent Node
            pathsize (int): diameter of path that has to be free
        Returns:
            List[GridNode]: list of possible successor nodes
        """
        ret = []
        for neighbour in self.grid.POSSIBLE_NEIGHBOURS:
            try:
                child = GridNode.from_coords(
                    [a + b for a, b in zip(parent.get_coordinates(), neighbour)]
                )
            except ValueError:
                continue
            if not self.grid.position_is_valid(node=child):
                continue
            if self.grid.position_is_blocked(node=child):
                continue
            child.g = parent.g + self.grid.estimate_diagonal_h(
                node=parent, destination=child
            )
            child.h = self.grid.get_distance_to_protein(child)
            child.parent = parent
            ret.append(child)
        return ret

    def astar_3d(
        self, backtrace: bool = True, distance: unit.Quantity = 2 * unit.nanometer
    ) -> List[GridNode]:
        """
        lets a slightly addapted (greedier) version of the A* star algorithm search for the shortest path between start end end point.
        Args:
            backtrace (bool): set to false if you want the function to return all searched nodes instead of the shortet path: Defaults to True
            distance (unit.Quantity): If this distance is reached, the algorithm is terminated. if none is given a path is searched until the path leaves the pbc box: Defaults to 2*unit.nanometer
        Returns:
            List[GridNode]: shortest path from a to b if class settint backtrace is true. else returns all searched nodes.
        """
        if self.is_goal_reached(node=self.start, distance=distance):
            self.shortest_path.append(self.start)
            return self.shortest_path
        self.start.g = 0
        self.start.h = self.grid.get_distance_to_protein(self.start)
        open_list = [self.start]
        while open_list:
            q = open_list[0]
            for node in open_list:
                # prioritize the node that is the fartest from any protein atom. -> will eventually lead out of the case as long as the goal distance is alrge enough.
                if node.h > q.h:
                    q = node
                # if two nodes are equally large apart from the protein, take the one that took less traveling to get there.
                if node.h == q.h and node.g < q.g:
                    q = node
            open_list.remove(q)
            children = self.generate_successors(parent=q)
            for child in children:
                if self.is_goal_reached(node=child, distance=distance):
                    self.shortest_path.append(q)
                    return self.backtrace_path() if backtrace else self.shortest_path
                if any(
                    (list_entry == child and list_entry.g <= child.g)
                    for list_entry in open_list
                ):
                    continue
                if any(
                    (list_entry == child and list_entry.g <= child.g)
                    for list_entry in self.shortest_path
                ):
                    continue
                else:
                    open_list.insert(0, child)
            self.shortest_path.append(q)
        return []

    def path_to_ccp4(self, filename: str):
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
        for i in self.shortest_path:
            pathgrid[i.get_coordinates()[0]][i.get_coordinates()[1]][
                i.get_coordinates()[2]
            ] = True
        ccp4_map = gemmi.Ccp4Map()
        ccp4_map.grid = gemmi.FloatGrid(pathgrid.astype(np.float32))
        ccp4_map.update_ccp4_header()
        ccp4_map.write_ccp4_map(filename)
        return None

    def get_diff(self, node1: GridNode, node2: GridNode) -> unit.Quantity:
        """
        Helper function for get_path_for_sampling(). Returns the true distance between two grid points.

        Args:
            node1 (GridNode):
            node2 (GridNode):

        Returns:
            [type]: distance between two grid points in units the gridcellsizes have.
        """
        u = self.grid.a.unit
        ret = math.sqrt(
            ((node2.x - node1.x) * self.grid.a.value_in_unit(u)) ** 2
            + ((node2.y - node1.y) * self.grid.b.value_in_unit(u)) ** 2
            + ((node2.z - node1.z) * self.grid.c.value_in_unit(u)) ** 2
        )
        return unit.Quantity(value=ret, unit=u)

    def get_path_for_sampling(
        self, stepsize: unit.Quantity = 1 * unit.angstrom
    ) -> List[unit.Quantity]:
        """
        Generates path of evenly spaced nodes for the sampling. Tree uses grid where diagonal jumps are bigger than nondiagonal jumps, hence this function is needed.
        It can also be used if you want to generated differently spaced paths

        Args:
            stepsize (unit.Quantity, optional): Stepsite of the path you want. Defaults to 1*unit.angstrom.

        Returns:
            List[unit.Quantity]: list of Coordinates.
        """
        ret: list[unit.Quantity] = []
        stride = 1
        path = copy.deepcopy(self.shortest_path)
        iterator = iter(path)
        current = next(iterator)
        new = next(iterator)
        ret.append(
            unit.Quantity(
                value=Vec3(
                    x=current.x * self.grid.a.value_in_unit(self.grid.a.unit)
                    + self.grid.offset[0].value_in_unit(self.grid.a.unit),
                    y=current.y * self.grid.b.value_in_unit(self.grid.a.unit)
                    + self.grid.offset[1].value_in_unit(self.grid.a.unit),
                    z=current.z * self.grid.c.value_in_unit(self.grid.a.unit)
                    + self.grid.offset[2].value_in_unit(self.grid.a.unit),
                ),
                unit=self.grid.a.unit,
            )
        )
        while min(self.grid.a, self.grid.b, self.grid.c) <= stepsize:
            stepsize /= 2
            stride *= 2
        end_reached = False
        diff = self.get_diff(node1=new, node2=current)
        diffo = self.get_diff(node1=new, node2=current)
        factor = stepsize / diff
        newstep = stepsize
        while not end_reached:
            try:
                diff -= newstep
                ret.append(
                    unit.Quantity(
                        value=Vec3(
                            x=((new.x - current.x) * factor + current.x)
                            * self.grid.a.value_in_unit(self.grid.a.unit)
                            + self.grid.offset[0].value_in_unit(self.grid.a.unit),
                            y=((new.y - current.y) * factor + current.y)
                            * self.grid.b.value_in_unit(self.grid.a.unit)
                            + self.grid.offset[1].value_in_unit(self.grid.a.unit),
                            z=((new.z - current.z) * factor + current.z)
                            * self.grid.c.value_in_unit(self.grid.a.unit)
                            + self.grid.offset[2].value_in_unit(self.grid.a.unit),
                        ),
                        unit=self.grid.a.unit,
                    )
                )
                if newstep < stepsize:
                    newstep = stepsize
                factor += stepsize / diffo
                if diff < stepsize:
                    newstep = stepsize - diff
                    current = new
                    new = next(iterator)
                    diff, diffo = self.get_diff(
                        node1=new, node2=current
                    ), self.get_diff(node1=new, node2=current)
                    factor = newstep / diffo
            except StopIteration:
                end_reached = True
        return ret[::stride]


class TreeAStar(astar_3d):
    def __init__(
        self,
        tree: Tree,
        start: TreeNode,
        end: TreeNode = None,
        pathsize: unit.Quantity = 1.2 * unit.angstrom,
        stepsize: unit.Quantity = 0.25 * unit.angstrom,
    ) -> None:
        super().__init__(start, end)
        self.shortest_path: List[TreeNode] = []
        self.tree = tree
        self.pathsize = pathsize
        self.stepsize = stepsize

    def is_goal_reached(
        self,
        node: TreeNode,
        box: unit.Quantity = None,
        distance: unit.Quantity = None,
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
            else:
                return False
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

    def generate_successors(
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
        for neighbour in self.tree.POSSIBLE_NEIGHBOURS:
            child = TreeNode.from_coords(
                [
                    a + b * self.stepsize.value_in_unit(parent.unit)
                    for a, b in zip(
                        parent.get_coordinates_for_query(parent.unit), neighbour
                    )
                ],
                unit_=parent.unit,
                parent=parent,
            )
            dist = self.tree.get_distance_to_protein(node=child)
            if dist < self.pathsize:
                continue
            child.g = parent.g + self.tree.estimate_diagonal_h(
                node=parent, destination=child
            )
            child.h = dist
            ret.append(child)
        return ret

    def backtrace_path(self) -> List[TreeNode]:
        """
        removes all the unsuccesfull path legs and returns the direct path from start to end. Does only make sense to run with or after self.astar_3d
        Returns:
            List[TreeNode]: direct path from self.start to self.end
        """
        self.shortest_path.reverse()
        new = []
        current = self.shortest_path[0]
        while current:
            new.insert(0, current)
            current = current.parent
        self.shortest_path = new
        return self.shortest_path

    def astar_3d(
        self,
        backtrace: bool = True,
        distance: unit.Quantity = None,
        box: List[unit.Quantity] = None,
    ) -> List[TreeNode]:
        """
        lets a slightly addapted (greedier) version of the A* star algorithm search for the shortest path between start end end point.
        Args:
            backtrace (bool): set to false if you want the function to return all searched nodes instead of the shortet path: Defaults to True
        Returns:
            List[TreeNode]: shortest path from a to b if class settint backtrace is true. else returns all searched nodes.
        """
        self.start.g = 0
        self.start.h = self.tree.get_distance_to_protein(node=self.start)
        open_list = [self.start]
        while open_list:
            q = open_list[0]
            for node in open_list:
                if node.h > q.h:
                    q = node
                if node.h == q.h and node.g < q.g:
                    q = node
            open_list.remove(q)
            children = self.generate_successors(parent=q)
            for child in children:
                if self.is_goal_reached(node=child, box=box, distance=distance):
                    self.shortest_path.append(q)
                    return (
                        self.shortest_path if not backtrace else self.backtrace_path()
                    )
                if any(
                    (
                        round(list_entry, 3) == round(child, 3)
                        and list_entry.g <= child.g
                    )
                    for list_entry in open_list
                ):
                    continue
                if any(
                    (
                        round(list_entry, 3) == round(child, 3)
                        and list_entry.g <= child.g
                    )
                    for list_entry in self.shortest_path
                ):
                    continue
                else:
                    open_list.insert(0, child)
            self.shortest_path.append(q)
        return []

    def get_path_for_sampling(
        self, stepsize: unit.Quantity = 1 * unit.angstrom
    ) -> List[unit.Quantity]:
        """
        Generates path of evenly spaced nodes for the sampling. Tree uses grid where diagonal jumps are bigger than nondiagonal jumps, hence this function is needed.
        It can also be used if you want to generated differently spaced paths

        Args:
            stepsize (unit.Quantity, optional): Stepsite of the path you want. Defaults to 1*unit.angstrom.

        Returns:
            List[TreeNode]: list of path nodes.
        """
        stride = 1
        ret = []
        path = copy.deepcopy(self.shortest_path)
        iterator = iter(path)
        current = next(iterator)
        new = next(iterator)
        ret.append(
            unit.Quantity(
                value=Vec3(x=current.x, y=current.y, z=current.z), unit=current.unit
            )
        )
        end_reached = False
        while stepsize > self.stepsize:
            stepsize /= 2
            stride *= 2
        while not end_reached:
            newstep = stepsize
            try:
                if newstep < stepsize:
                    newstep = stepsize
                diff = self.tree.estimate_diagonal_h(current, new) * self.tree.unit
                if diff < stepsize:
                    current = new
                    new = next(iterator)
                    newstep -= diff
                    diff = self.tree.estimate_diagonal_h(current, new) * self.tree.unit
                factor = newstep / diff
                current.x += (new.x - current.x) * factor
                current.y += (new.y - current.y) * factor
                current.z += (new.z - current.z) * factor
                ret.append(
                    unit.Quantity(
                        value=Vec3(
                            x=current.x,
                            y=current.y,
                            z=current.z,
                        ),
                        unit=current.unit,
                    )
                )
            except StopIteration:
                end_reached = True
        del path
        return ret[::stride]
