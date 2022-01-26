import copy, time, logging
from typing import List
import gemmi
import numpy as np
from openmm import Vec3, unit

from UmbrellaPipeline.path_generation import (
    GridNode,
    TreeNode,
    Grid,
    Tree,
)
from UmbrellaPipeline.utils import display_time

logger = logging.getLogger(__name__)


class EscapeRoom3D:
    """
    Base class for the EscapeRoom3d algorithm
    """

    def __init__(
        self,
        start: GridNode or TreeNode,
    ) -> None:

        self.start = start
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


class GridEscapeRoom(EscapeRoom3D):
    def __init__(
        self,
        grid: Grid,
        start: GridNode,
    ) -> None:
        super().__init__(start=start)
        self.grid = grid
        self.shortest_path: List[GridNode] = []
        logger.warning(
            "You are now using the grid version of the escape room module. This is generally not advised and should only be used for visualization purposes. Use the Tree version instead, which is much faster and memory efficient!"
        )

    def is_goal_reached(self, node: GridNode, distance: unit.Quantity = None) -> bool:
        """
        Checks wether the given node has the distance to the neares True grid point.
        node.distance_to_wall is in "node units" so the distance is divided by the gridcell size to get its value in "grid units" as well.
        If no distance is given, the end point is reached when node is outside the grid
        Args:
            node (GridNode): node to be checked
        Returns:
            bool: Returns true if input node is the destination.
        """
        try:
            if node.distance_to_wall > distance / self.grid.a:
                return True
        except TypeError:
            return not self.grid.position_is_valid(node)

    def backtrace_path(self) -> List[GridNode]:
        """
        removes all the unsuccesfull path legs and returns the direct path from start to end. Does only make sense to run with or after self.escape_room
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

    def create_child(self, neighbour: List[int], parent: GridNode) -> GridNode:
        """
        Create child at neighbour position of parent.
        Parameters
        ----------
        neighbour : List[int]
            neighbour position
        parent : GridNode
            parent node

        Returns
        -------
        GridNode
            child node at neighbour position if position is inside grid, else none.
        """
        try:
            child = GridNode.from_coords(
                [a + b for a, b in zip(parent.get_coordinates(), neighbour)]
            )
        except ValueError:
            return None
        child.distance_to_wall = self.grid.get_distance_to_protein(child)
        child.distance_walked = (
            parent.distance_walked
            + self.grid.calculate_diagonal_distance(node=parent, destination=child)
        )
        child.parent = parent
        return child

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
            child = self.create_child(neighbour, parent)
            if child is None:
                continue
            if not self.grid.position_is_valid(node=child):
                continue
            if self.grid.position_is_blocked(node=child):
                continue
            ret.append(child)
        return ret

    def is_child_good(self, child: GridNode, open_list: List[GridNode]) -> None:
        """
        Checks if there is already a child in open list or shortest path that has the same position but had a shorter way to get there.
        Parameters
        ----------
        child : GridNode
            child to be checked.
        open_list : List[GridNode]
            list to search beneath self.shortest_path

        Returns
        -------
        [type]
            [description]
        """
        if any(
            (
                list_entry == child
                and list_entry.distance_walked <= child.distance_walked
            )
            for list_entry in open_list
        ):
            return False
        elif any(
            (
                list_entry == child
                and list_entry.distance_walked <= child.distance_walked
            )
            for list_entry in self.shortest_path
        ):
            return False
        else:
            return True

    def escape_room(
        self, distance: unit.Quantity = 2 * unit.nanometer
    ) -> List[GridNode]:
        """
        lets a slightly addapted (greedier) version of the A* star algorithm search for the shortest path between start end end point.
        Args:
            backtrace (bool): set to false if you want the function to return all searched nodes instead of the shortet path: Defaults to True
            distance (unit.Quantity): If this distance is reached, the algorithm is terminated. if none is given a path is searched until the path leaves the pbc box: Defaults to 2*unit.nanometer
        Returns:
            List[GridNode]: shortest path from a to b if class settint backtrace is true. else returns all searched nodes.
        """
        start = time.time()
        if self.is_goal_reached(node=self.start, distance=distance):
            self.shortest_path.append(self.start)
            logger.warning(
                "Start point does already a goal point. Try using a bigger distance, or even without a distance."
            )
            return self.shortest_path
        self.start.distance_to_wall = self.grid.get_distance_to_protein(self.start)
        open_list = [self.start]
        while open_list:
            q = open_list[0]
            for node in open_list:
                # prioritize the node that is the fartest from any protein atom. -> will eventually lead out of the case as long as the goal distance is alrge enough.
                if node.distance_to_wall > q.distance_to_wall:
                    q = node
                # if two nodes are equally large apart from the protein, take the one that took less traveling to get there.
                if (
                    node.distance_to_wall == q.distance_to_wall
                    and node.distance_walked < q.distance_walked
                ):
                    q = node
            open_list.remove(q)
            children = self.generate_successors(parent=q)
            for child in children:
                if self.is_goal_reached(node=child, distance=distance):
                    self.shortest_path.append(q)
                    logger.info(
                        f"Shortes path was found! Elapsed Time = {display_time(time.time() - start)}"
                    )
                    return self.backtrace_path()
                if self.is_child_good(child=child, open_list=open_list):
                    open_list.insert(0, child)
            self.shortest_path.append(q)
        logger.warning(
            "No way out was found! :( Try again, employing a smaller gridsize when constructing the grid."
        )
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
            List[unit.Quantity]: list of Coordinates.
        """
        ret: list[unit.Quantity] = []
        stride = 1
        path = copy.deepcopy(self.shortest_path)
        iterator = iter(path)
        current = next(iterator)
        new = next(iterator)
        while min(self.grid.a, self.grid.b, self.grid.c) <= stepsize:
            stepsize /= 2
            stride *= 2
        end_reached = False
        diff = self.grid.get_cartesian_distance(node1=new, node2=current)
        diffo = self.grid.get_cartesian_distance(node1=new, node2=current)
        to_go = stepsize / diff
        newstep = stepsize

        ret.append(self.grid.cartesian_coordinates_of_node(current))

        while not end_reached:
            try:
                diff -= newstep
                ret.append(
                    self.grid.cartesian_coordinates_w_increment(current, new, to_go)
                )
                if newstep < stepsize:
                    newstep = stepsize
                to_go += stepsize / diffo
                if diff < stepsize:
                    newstep = stepsize - diff
                    current = new
                    new = next(iterator)
                    diff = self.grid.get_cartesian_distance(node1=new, node2=current)
                    diffo = self.grid.get_cartesian_distance(node1=new, node2=current)
                    to_go = newstep / diffo
            except StopIteration:
                end_reached = True
        return ret[::stride]

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
        logger.info("Hang in there, this can take up to a minute :)")
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


class TreeEscapeRoom(EscapeRoom3D):
    def __init__(
        self,
        tree: Tree,
        start: TreeNode,
        pathsize: unit.Quantity = 1.2 * unit.angstrom,
        stepsize: unit.Quantity = 0.25 * unit.angstrom,
    ) -> None:
        super().__init__(start=start)
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
            if node.distance_to_wall * node.unit >= distance:
                return True
            else:
                return False
        except (TypeError, AttributeError):
            pass
        try:
            return (
                (node.x * node.unit < box[0] or node.x * node.unit > box[1])
                or (node.y * node.unit < box[2] or node.y * node.unit > box[3])
                or (node.z * node.unit < box[4] or node.z * node.unit > box[5])
            )
        except (TypeError):
            raise TypeError("Either give distance_to_protein or box vectors!")

    def create_child(self, neighbour: List[int], parent: TreeNode) -> TreeNode:
        """
        Creates child node at given neighbour position.

        Parameters
        ----------
        neighbour : List[int]
            neighbour position
        parent : TreeNode
            parent node

        Returns
        -------
        TreeNode
            child at given neighbour position
        """
        child = TreeNode.from_coords(
            [
                a + b * self.stepsize.value_in_unit(self.tree.unit)
                for a, b in zip(
                    parent.get_coordinates_for_query(self.tree.unit), neighbour
                )
            ],
            unit=parent.unit,
        )
        child.distance_walked = (
            parent.distance_walked
            + self.tree.calculate_diagonal_distance(
                node=parent, destination=child
            ).value_in_unit(self.tree.unit)
        )
        child.distance_to_wall = self.tree.get_distance_to_protein(
            node=child
        ).value_in_unit(self.tree.unit)
        child.parent = parent
        return child

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
            child = self.create_child(neighbour=neighbour, parent=parent)
            if not child.distance_to_wall == 0 * unit.meter:
                ret.append(child)
        return ret

    def backtrace_path(self) -> List[TreeNode]:
        """
        removes all the unsuccesfull path legs and returns the direct path from start to end. Does only make sense to run with or after self.escape_room
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

    def is_child_good(self, child: TreeNode, open_list: List[TreeNode]) -> bool:
        """
        Checks if a there is already a node with the same coordinates but lower distance_walked in either open_list or shortest_path.

        Args:
            child (TreeNode): node that should be checked
            open_list (List[TreeNode]): list of nodes to go through.

        Returns:
            bool: True if no child with same coordinates and lower distance_walked is in open_list and shortes_path
        """
        if any(
            (
                round(list_entry, 3) == round(child, 3)
                and list_entry.distance_walked <= child.distance_walked
            )
            for list_entry in open_list
        ):
            return False
        elif any(
            (
                round(list_entry, 3) == round(child, 3)
                and list_entry.distance_walked <= child.distance_walked
            )
            for list_entry in self.shortest_path
        ):
            return False
        else:
            return True

    def escape_room(
        self,
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
        start = time.time()
        if self.is_goal_reached(node=self.start, box=box, distance=distance):
            logger.warning(
                "Start point does already a goal point. Try using a bigger distance, or giving the method a box instead of a distance :)."
            )
            self.shortest_path.append(self.start)
            return self.shortest_path
        self.start.distance_to_wall = self.tree.get_distance_to_protein(
            node=self.start
        ).value_in_unit(self.tree.unit)
        self.start.distance_walked = 0
        open_list = [self.start]
        while open_list:
            q = open_list[0]
            for node in open_list:
                if node.distance_to_wall > q.distance_to_wall:
                    q = node
                if (
                    node.distance_to_wall == q.distance_to_wall
                    and node.distance_walked < q.distance_walked
                ):
                    q = node
            open_list.remove(q)
            children = self.generate_successors(parent=q)
            for child in children:
                if self.is_goal_reached(node=child, box=box, distance=distance):
                    self.shortest_path.append(q)
                    logger.info(
                        f"Shortes path was found! Elapsed Time = {display_time(time.time() - start)}"
                    )
                    return self.backtrace_path()
                if not self.is_child_good(child, open_list):
                    continue
                else:
                    open_list.insert(0, child)
            self.shortest_path.append(q)
        logger.warning(
            "No way out was found! :( Try again, using a smaller stepsize when searching a way out."
        )
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
        ret = []
        path = copy.deepcopy(self.shortest_path)
        iterator = iter(path)
        current = next(iterator)
        new = next(iterator)
        ret.append(Vec3(x=current.x, y=current.y, z=current.z))
        end_reached = False
        diff = self.tree.calculate_euclidean_distance(current, new)
        while not end_reached:
            try:
                if stepsize < diff:
                    factor = stepsize / diff
                    current.x += (new.x - current.x) * factor
                    current.y += (new.y - current.y) * factor
                    current.z += (new.z - current.z) * factor
                    ret.append(Vec3(x=current.x, y=current.y, z=current.z))
                    diff = self.tree.calculate_euclidean_distance(current, new)
                else:
                    while stepsize > diff:
                        new = next(iterator)
                        diff = self.tree.calculate_euclidean_distance(current, new)
                    factor = stepsize / diff
                    current.x += (new.x - current.x) * factor
                    current.y += (new.y - current.y) * factor
                    current.z += (new.z - current.z) * factor
                    ret.append(Vec3(x=current.x, y=current.y, z=current.z))
                    diff = self.tree.calculate_euclidean_distance(current, new)
            except StopIteration:
                end_reached = True
        return unit.Quantity(value=ret, unit=self.shortest_path[0].unit)
