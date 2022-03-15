import copy, time, logging, math
from typing import List, Dict, Set, Tuple
import gemmi
import numpy as np
from openmm import Vec3, unit
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from UmbrellaPipeline.path_finding import (
    GridNode,
    TreeNode,
    Grid,
    Tree,
    Queue,
)
from UmbrellaPipeline.utils import (
    SystemInfo,
    display_time,
    get_center_of_mass_coordinates,
    NoWayOutError,
    StartIsFinishError,
)

logger = logging.getLogger(__name__)


class EscapeRoom3D:
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
        """
        This is the Grid implementation of the EscapeRoom Algorithm. Deprecated and not suggested to use.

        Args:
            grid (Grid): _description_
            start (GridNode): _description_
        """
        super().__init__(start=start)
        self.grid = grid
        self.shortest_path: List[GridNode] = []
        logger.warning(
            "You are now using the grid version of the escape room module. This module is deprecated. Use the Tree version instead, which is much faster and memory efficient!"
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

        Args:
            neighbour (List[int]): neighbour position
        parent (GridNode): parent node

        Returns:
            GridNode: child node at neighbour position if position is inside grid, else none.
        """
        try:
            child = GridNode.from_coords(
                [a + b for a, b in zip(parent.get_coordinates(), neighbour)]
            )
        except ValueError:
            return None
        child.distance_to_wall = self.grid.get_distance_to_protein(child)
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

        Args:
            child (GridNode): child to be checked.
            open_list (List[GridNode]): list to search beneath self.shortest_path

        Returns:
            bool: wheter a child is good or not
        """
        if any((list_entry == child) for list_entry in open_list):
            return False
        elif any((list_entry == child) for list_entry in self.shortest_path):
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

    def get_euclidean_distance(self, node1, node2):
        return (
            math.sqrt(
                (node1.x - node2.x) ** 2
                + (node1.y - node2.y) ** 2
                + (node1.z - node2.z) ** 2
            )
            * node1.unit
        )

    def get_path_for_sampling(
        self, stepsize: unit.Quantity = 0.1 * unit.nanometer
    ) -> List[unit.Quantity]:
        """
        Generates path of evenly spaced nodes for the sampling. Tree uses grid where diagonal jumps are bigger than nondiagonal jumps, hence this function is needed.
        It can also be used if you want to generated differently spaced paths

        Args:
            stepsize (unit.Quantity, optional): Stepsite of the path you want. Defaults to 0.1*unit.nanometer.

        Returns:
            List[unit.Quantity]: list of Coordinates.
        """
        ret: list[unit.Quantity] = []
        path = copy.deepcopy(self.shortest_path)
        iterator = iter(path)
        current = self.grid.cartesian_coordinates_of_node(next(iterator))
        new = self.grid.cartesian_coordinates_of_node(next(iterator))
        end_reached = False
        diff = self.get_euclidean_distance(node1=new, node2=current)
        ret.append(
            Vec3(
                x=current.x,
                y=current.y,
                z=current.z,
            )
        )
        while not end_reached:
            try:
                if stepsize < diff:
                    factor = stepsize / diff
                    current.x += (new.x - current.x) * factor
                    current.y += (new.y - current.y) * factor
                    current.z += (new.z - current.z) * factor
                    ret.append(Vec3(x=current.x, y=current.y, z=current.z))
                    diff = self.get_euclidean_distance(current, new)
                else:
                    while stepsize > diff:
                        new = self.grid.cartesian_coordinates_of_node(next(iterator))
                        diff = self.get_euclidean_distance(current, new)
                    factor = stepsize / diff
                    current.x += (new.x - current.x) * factor
                    current.y += (new.y - current.y) * factor
                    current.z += (new.z - current.z) * factor
                    ret.append(Vec3(x=current.x, y=current.y, z=current.z))
                    diff = self.get_euclidean_distance(current, new)
            except StopIteration:
                end_reached = True
        return unit.Quantity(value=ret, unit=self.grid.a.unit)

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
        start: unit.Quantity,
    ) -> None:
        super().__init__(start=start)
        self.tree = tree
        self.shortest_path: List[TreeNode] = []
        self.resolution = 0.01

    @classmethod
    def from_files(cls, system_info: SystemInfo, positions: unit.Quantity = None):
        if not positions:
            positions = system_info.crd_object.positions
        tree = Tree.from_files(positions=positions, psf=system_info.psf_object)
        try:
            start = get_center_of_mass_coordinates(
                positions=positions,
                indices=system_info.ligand_indices,
                masses=system_info.psf_object.system,
            ).value_in_unit(unit.nanometer)
        except AttributeError:
            _ = system_info.psf_object.createSystem(system_info.params)
            start = get_center_of_mass_coordinates(
                positions=positions,
                indices=system_info.ligand_indices,
                masses=system_info.psf_object.system,
            ).value_in_unit(unit.nanometer)
        return cls(tree=tree, start=start)

    def is_goal_reached(
        self,
        node: TreeNode,
        distance: unit.Quantity,
    ):
        """
        Checks wheter a node meets the goal criteria.

        Args:
            node (TreeNode): [description]
            dist_to_protein (unit.Quantity, optional): [description]. Defaults to 2*unit.nanometer.

        Returns:
            [type]: [description]
        """
        try:
            return node.distance_to_wall >= distance.value_in_unit(unit.nanometer)
        except AttributeError:
            return node.distance_to_wall >= distance

    def create_child(
        self,
        neighbour: List[int],
        parent: TreeNode,
        resolution: float,
        wall_radius: float,
    ) -> TreeNode:
        """
        Creates a neighbouring node to parent.

        Args:
            neighbour (List[int]): position offset to parent
            parent (TreeNode): parent TreeNode

        Returns:
            TreeNode: Neighbouring node if its not inside the wall
        """
        child_position = [
            a + b for a, b in zip(parent.get_grid_coordinates(), neighbour)
        ]
        distance_to_wall = self.tree.get_distance_to_wall(
            coordinates=[
                child_position[0] * resolution + self.start.x,
                child_position[1] * resolution + self.start.y,
                child_position[2] * resolution + self.start.z,
            ],
            vdw_radius=wall_radius,
        )
        try:
            return TreeNode.from_coords(
                coords=child_position, distance_to_wall=distance_to_wall, parent=parent
            )
        except ValueError:
            pass

    def generate_successors(
        self,
        parent: TreeNode,
        resolution: float,
        wall_radius: float,
    ) -> List[TreeNode]:
        """
        Creates a max of 6 neibhbouring nodes for the input node.

        Args:
            parent (TreeNode): parent TreeNode

        Returns:
            List[TreeNode]: list of possible successor nodes
        """
        return [
            self.create_child(
                neighbour=neighbour,
                parent=parent,
                resolution=resolution,
                wall_radius=wall_radius,
            )
            for neighbour in self.tree.POSSIBLE_NEIGHBOURS
        ]

    def backtrace_path(
        self,
        closed: Dict,
        key: Tuple,
    ) -> List[TreeNode]:
        """
        removes all the unsuccesfull path legs and returns the direct path from start to end.

        Returns:
            List[TreeNode]: path found by the escape room algorithm
        """
        current = closed[key]
        self.shortest_path = [key]
        while current:
            self.shortest_path.append(current)
            current = closed[current]
        self.shortest_path.reverse()
        return self.shortest_path

    def child_already_exists(
        self, child: TreeNode, open_set: Set, closed_map: Dict
    ) -> bool:
        """
        Checks wether a given node was already looked at.

        Args:
            child (TreeNode): TreeNode to check
            open_set (Set): open set to search
            closed_map (Dict): closed map to search

        Returns:
            bool: True if it already exists
        """
        return child in open_set or child in closed_map

    def find_path(
        self,
        resolution: unit.Quantity = None,
        wall_radius: unit.Quantity = 0.12 * unit.nanometer,
        distance: unit.Quantity = 1.5 * unit.nanometer,
    ) -> List[TreeNode]:
        """
        core function implementing the escape room algorithm

        Args:
            distance (u.Quantity, optional): distance to wall that defines the goal. Defaults to 1.5 u.nanometer.
            resolution (u.Quantity, optional): resolution of the searched grid. Defaults to .25 u.nanometer.

        Returns:
            List[TreeNode]: List of TreeNodes describing the most accessible path oud of 3D object.
        """
        try:
            first_node = TreeNode(
                distance_to_wall=self.tree.get_distance_to_wall(coordinates=self.start)
            )
        except:
            raise StartIsFinishError
        if resolution:
            self.resolution = resolution.value_in_unit(unit.nanometer)

        distance = distance.value_in_unit(unit.nanometer)
        wall_radius = wall_radius.value_in_unit(unit.nanometer)

        open_queue = Queue([first_node])
        open_set = set()
        open_set.add(first_node)
        closed_map = dict()
        st = time.time()
        while open_queue.queue:
            best_node = open_queue.pop()
            open_set.remove(best_node)

            neighbors = self.generate_successors(
                parent=best_node,
                resolution=self.resolution,
                wall_radius=wall_radius,
            )

            for n in neighbors:

                if self.is_goal_reached(n, distance=distance):
                    closed_map[best_node] = best_node.parent
                    closed_map[n] = n.parent
                    self.shortest_path = self.backtrace_path(closed_map, key=n)
                    logger.info(
                        f"Path out of Cavity found! Hurray. Time Elapsed: {display_time(time.time()-st)}"
                    )
                    return self.shortest_path

                if self.child_already_exists(n, open_set, closed_map):
                    continue

                open_queue.push(n)
                open_set.add(n)

            closed_map[best_node] = best_node.parent
        raise NoWayOutError

    def get_path_for_sampling(
        self,
        stepsize: unit.Quantity = 0.1 * unit.nanometer,
    ) -> List[unit.Quantity]:
        """
        Generates path of evenly spaced nodes for the sampling. Tree uses grid where diagonal jumps are bigger than nondiagonal jumps, hence this function is needed.
        It can also be used if you want to generated differently spaced paths

        Args:
            stepsize (unit.Quantity, optional): Stepsite of the path you want. Defaults to .1*unit.nanometer.

        Returns:
            List[TreeNode]: list of path nodes.
        """
        ret = []
        stepsize = stepsize.value_in_unit(unit.nanometer)
        path = [
            [
                i.x * self.resolution + self.start.x,
                i.y * self.resolution + self.start.y,
                i.z * self.resolution + self.start.z,
            ]
            for i in self.shortest_path
        ]
        iterator = iter(path)
        current = next(iterator)
        new = next(iterator)
        ret.append(Vec3(x=current[0], y=current[1], z=current[2]))
        end_reached = False
        diff = self.tree.calculate_euclidean_distance(current, new)
        while not end_reached:
            try:
                if stepsize < diff:
                    factor = stepsize / diff
                    current[0] += (new[0] - current[0]) * factor
                    current[1] += (new[1] - current[1]) * factor
                    current[2] += (new[2] - current[2]) * factor
                    ret.append(Vec3(x=current[0], y=current[1], z=current[2]))
                    diff = self.tree.calculate_euclidean_distance(current, new)
                else:
                    while stepsize > diff:
                        new = next(iterator)
                        diff = self.tree.calculate_euclidean_distance(current, new)
                    factor = stepsize / diff
                    current[0] += (new[0] - current[0]) * factor
                    current[1] += (new[1] - current[1]) * factor
                    current[2] += (new[2] - current[2]) * factor
                    ret.append(Vec3(x=current[0], y=current[1], z=current[2]))
                    diff = self.tree.calculate_euclidean_distance(current, new)
            except StopIteration:
                end_reached = True
        logger.info(
            f"Path for sampling generated. Total number of umbrella windows = {len(ret)}"
        )
        return unit.Quantity(value=ret, unit=unit.nanometer)

    def visualize_path(self, path: unit.Quantity = None):
        """
        Basic visualization of the path generated and the protein.

        Args:
            path (unit.Quantity, optional): Specify if you dont want the classes path to be visualized. Defaults to None.
        """
        pos = path if path else self.shortest_path
        try:
            df = pd.DataFrame(
                pos.value_in_unit(unit.nanometer),
                columns=list("xyz"),
            )
        except AttributeError:
            path = [
                Vec3(
                    i.x * self.resolution + self.start.x,
                    i.y * self.resolution + self.start.y,
                    i.z * self.resolution + self.start.z,
                )
                for i in self.shortest_path
            ]
            df = pd.DataFrame(pos, columns=list("xyz"))
        df2 = pd.DataFrame(self.tree.tree.data, columns=list("abc"))
        a = px.scatter_3d(data_frame=df, x="x", y="y", z="z")
        a.add_trace(
            go.Scatter3d(
                x=df2.a,
                y=df2.b,
                z=df2.c,
                mode="lines",
                line=dict(width=3, color="black"),
            )
        )
        a.show()
