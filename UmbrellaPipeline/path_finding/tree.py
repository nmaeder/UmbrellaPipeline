import math
from typing import List
import openmm.unit as u
from openmm import Vec3, app
from scipy.spatial import KDTree

from UmbrellaPipeline.utils import get_residue_indices
from UmbrellaPipeline.path_finding import Node


class Tree:
    """
    This class stores the KDTree and all information relevant for writing out coordinates.
    """

    def __init__(
        self,
        coordinates: u.Quantity or List[u.Quantity] or List[float],
    ):
        """

        Args:
            coordinates (u.Quantity or List[u.Quantity] or List[float]): List of coordinates to be added to the tree
        """

        try:
            pos = [i.value_in_unit(u.nanometer) for i in coordinates]
            self.tree = KDTree(pos)
        except AttributeError:
            self.tree = KDTree(coordinates)

        self.POSSIBLE_NEIGHBOURS = [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]

    @property
    def unit(self):
        return u.nanometer

    @classmethod
    def from_files(
        cls,
        positions: u.Quantity or app.CharmmCrdFile,
        psf: str or app.CharmmPsfFile,
        include_membrane: bool = False,
    ):
        """
        creates a k-d tree out of positions and a psf file. only takes the protein (and membrane) atom from the positions.

        Args:
            positions (u.Quantity or app.CharmmCrdFile): positions as unit.Quantity or as path to .crd file
            psf (str or app.CharmmPsfFile): path to psf file or CharmmPsfFile object
            include_membrane (bool): whether to include membrane atoms in the wall.

        Raises:
            ValueError: _description_

        Returns:
            Tree: k-d tree created from input given.
        """
        try:
            psf = app.CharmmPsfFile(psf)
        except TypeError:
            pass

        try:
            positions = app.CharmmCrdFile(positions).positions
        except TypeError:
            pass

        try:
            positions.unit
        except AttributeError:
            raise ValueError(
                "Give atomic positions as a unit.Quantity object or as a list of unit.Quantities"
            )

        indices = get_residue_indices(psf.atom_list)
        coords = [
            Vec3(
                x=positions[i].x,
                y=positions[i].y,
                z=positions[i].z,
            )
            for i in indices
        ]
        return cls(coordinates=u.Quantity(value=coords, unit=positions.unit))

    @staticmethod
    def calculate_euclidean_distance(node: Node, destination: Node) -> u.Quantity:
        """
        calculates euclidean distance between node and destination.
        Args:
            node (Node): point a
            destination (Node): point b
        Returns:
            float: euclidean distance
        """
        try:
            return math.sqrt(
                (node.x - destination.x) ** 2
                + (node.y - destination.y) ** 2
                + (node.z - destination.z) ** 2
            )
        except AttributeError:
            return math.sqrt(
                (node[0] - destination[0]) ** 2
                + (node[1] - destination[1]) ** 2
                + (node[2] - destination[2]) ** 2
            )

    @staticmethod
    def calculate_diagonal_distance(node: Node, destination: Node) -> u.Quantity:
        """
        calculates diagonal distance between node and destination.
        Args:
            node (Node): point a
            destination (Node): point b
        Returns:
            float: diagonal distance
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
        return u.Quantity(
            value=(D3 - D2) * dmin + (D2 - D1) * dmid + D1 * dmax, unit=node.unit
        )

    def get_distance_to_wall(
        self,
        coordinates: List[float] = None,
        vdw_radius: float = 0.12,
    ) -> u.Quantity:
        """
        Returns the distance to the nearest wall.

        Args:
            node (Node, optional): Node for which to get the nearest distance. Defaults to None.
            coordinates (List[float], optional): position in 3d space to get the nearest distance. Defaults to None.
            vdw_radius (u.Quantity, optional): Radius to set for  walls.. Defaults to 0.12*u.nanometer.

        Returns:
            u.Quantity: Distance to the nearest wall.
        """
        try:
            dist, _ = self.tree.query(x=coordinates.get_coordinates_for_query(), k=1)
        except AttributeError:
            dist, _ = self.tree.query(x=coordinates, k=1)
        return dist - vdw_radius

    def get_nearest_neighbour_index(self, coords: u.Quantity) -> int:
        """
        Returns the index of the nearest neighbour in the tree. used for the PMFCalculator binning.

        Args:
            coords (u.Quantity): coordinate for which to get the nearest neighbour

        Returns:
            (int): index of the nearest neighbour in the tree.
        """
        _, i = self.tree.query(x=[coords.x, coords.y, coords.z], k=1)
        return i
