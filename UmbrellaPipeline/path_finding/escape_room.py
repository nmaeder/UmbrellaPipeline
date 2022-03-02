from typing import List, Dict, Set, Tuple
from more_itertools import first
import openmm.unit as u
from openmm import Vec3

from UmbrellaPipeline.path_finding import (
    Queue,
    Node,
    Tree,
)

from UmbrellaPipeline.utils import (
    NoPathFoundError,
    StartError,
    SystemInfo,
    get_center_of_mass_coordinates,
)


class EscapeRoom3D:
    def __init__(
        self,
        tree: Tree,
        start: u.Quantity,
    ) -> None:
        self.tree = tree
        self.start = start
        self.shortest_path: List[Node] = []
        self.resolution = 0.01

    @classmethod
    def from_files(cls, system_info: SystemInfo, positions: u.Quantity = None):
        if not positions:
            positions = system_info.crd_object.positions
        tree = Tree.from_files(positions=positions, psf=system_info.psf_object)
        try:
            start = get_center_of_mass_coordinates(
                positions=positions,
                indices=system_info.ligand_indices,
                masses=system_info.psf_object.system,
            ).value_in_unit(u.nanometer)
        except:
            _ = system_info.psf_object.createSystem(system_info.params)
            start = get_center_of_mass_coordinates(
                positions=positions,
                indices=system_info.ligand_indices,
                masses=system_info.psf_object.system,
            ).value_in_unit(u.nanometer)
        return cls(tree=tree, start=start)

    def is_goal_reached(
        self,
        node: Node,
        distance: u.Quantity,
    ):
        """
        Checks wheter a node meets the goal criteria.

        Args:
            node (Node): [description]
            dist_to_protein (unit.Quantity, optional): [description]. Defaults to 2*unit.nanometer.

        Returns:
            [type]: [description]
        """
        try:
            return node.distance_to_wall >= distance.value_in_unit(u.nanometer)
        except AttributeError:
            return node.distance_to_wall >= distance

    def create_child(
        self, neighbour: List[int], parent: Node, resolution: float, wall_radius: float
    ) -> Node:
        """
        Creates a neighbouring node to parent.

        Args:
            neighbour (List[int]): position offset to parent
            parent (Node): parent Node

        Returns:
            Node: Neighbouring node if its not inside the wall
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
            return Node.from_coords(
                coords=child_position, distance_to_wall=distance_to_wall, parent=parent
            )
        except:
            pass

    def generate_successors(
        self,
        parent: Node,
        resolution: float,
        wall_radius: float,
    ) -> List[Node]:
        """
        Creates a max of 6 neibhbouring nodes for the input node.
        Args:
            parent (Node): parent Node
        Returns:
            List[Node]: list of possible successor nodes
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
    ) -> List[Node]:
        """
        removes all the unsuccesfull path legs and returns the direct path from start to end.
        Returns:
            List[Node]: path found by the escape room algorithm
        """
        current = closed[key]
        self.shortest_path = [key]
        while current:
            self.shortest_path.append(current)
            current = closed[current]
        self.shortest_path.reverse()
        return self.shortest_path

    def child_already_exists(
        self, child: Node, open_set: Set, closed_map: Dict
    ) -> bool:
        """
        Checks wether a given node was already looked at.

        Args:
            child (Node): Node to check
            open_set (Set): open set to search
            closed_map (Dict): closed map to search

        Returns:
            bool: True if it already exists
        """
        return child in open_set or child in closed_map

    def find_path(
        self,
        resolution: u.Quantity = None,
        wall_radius: u.Quantity = 0.12 * u.nanometer,
        distance: u.Quantity = 1.5 * u.nanometer,
    ) -> List[Node]:
        """
        core function implementing the escape room algorithm

        Args:
            distance (u.Quantity, optional): distance to wall that defines the goal. Defaults to 1.5 u.nanometer.
            resolution (u.Quantity, optional): resolution of the searched grid. Defaults to .25 u.nanometer.

        Returns:
            List[Node]: List of Nodes describing the most accessible path oud of 3D object.
        """
        try:
            first_node = Node(
                distance_to_wall=self.tree.get_distance_to_wall(coordinates=self.start)
            )
        except:
            raise StartError
        if resolution:
            self.resolution = resolution.value_in_unit(u.nanometer)
        
        distance = distance.value_in_unit(u.nanometer)
        wall_radius = wall_radius.value_in_unit(u.nanometer)

        open_queue = Queue([first_node])
        open_set = set()
        open_set.add(first_node)
        closed_map = dict()

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
                    return self.shortest_path

                if self.child_already_exists(n, open_set, closed_map):
                    continue

                open_queue.push(n)
                open_set.add(n)

            closed_map[best_node] = best_node.parent

        raise NoPathFoundError

    def get_path_for_sampling(
        self,
        stepsize: u.Quantity = 0.1 * u.nanometer,
    ) -> List[u.Quantity]:
        """
        Generates path of evenly spaced nodes for the sampling. Tree uses grid where diagonal jumps are bigger than nondiagonal jumps, hence this function is needed.
        It can also be used if you want to generated differently spaced paths

        Args:
            stepsize (unit.Quantity, optional): Stepsite of the path you want. Defaults to .1*unit.nanometer.

        Returns:
            List[Node]: list of path nodes.
        """
        ret = []
        stepsize = stepsize.value_in_unit(u.nanometer)
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
        return u.Quantity(value=ret, unit=u.nanometer)
