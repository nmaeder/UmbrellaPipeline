import math
from typing import List
import openmm.app as app
import openmm.unit as u
from openmm import Vec3
from scipy.spatial import KDTree

from UmbrellaPipeline.utils import (
    get_residue_indices,
    get_centroid_coordinates,
)
from UmbrellaPipeline.path_generation import TreeNode


class Tree:
    """
    This class stores the KDTree and all information relevant for writing out coordinates.
    """

    def __init__(
        self,
        coordinates: u.Quantity or List[u.Quantity] or List[float],
        unit: u.Unit = None,
    ):
        """

        Args:
            coordinates (u.Quantity or List[u.Quantity] or List[float]): List of coordinates to be added to the tree
            unit (u.Unit): unit of coordinates if they are given without any.
        """
        if unit:
            try:
                self.unit = unit
                pos = []
                for i in coordinates:
                    pos.append(list(i.value_in_unit(i.unit)))
                self.tree = KDTree(pos)
            except AttributeError:
                self.unit = unit
                self.tree = KDTree(coordinates)
        else:
            try:
                self.unit = coordinates[0].unit
                pos = []
                for i in coordinates:
                    pos.append(list(i.value_in_unit(self.unit)))
                self.tree = KDTree(pos)
            except AttributeError:
                raise ValueError("no unit provided.")
        self.POSSIBLE_NEIGHBOURS = [
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

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value: u.Unit):
        if not value.is_compatible(u.nanometer):
            raise TypeError("The unit of Tree has to be a length.")
        else:
            self._unit = value

    @classmethod
    def from_files(
        cls,
        pdb: str or app.PDBFile,
        psf: str or app.CharmmPsfFile,
    ):
        """
        Constructor for grid. takes in psf and pdb files generated from charmmgui and generates a grid where all points with a protein atom are true. every other gridpoint is False.

        Args:
            pdbfile (str): give either path to pdb file as string or an openmm.app.PDBFile object.
            psffile (str): give either path to psf file as string or an openmm.app.CharmmPsfFile object.
        Returns:
            Grid: KDTree with the protein positions used for the adapted A* algorithm
        """
        try:
            pdb = app.PDBFile(pdb)
        except TypeError:
            pdb = pdb
        try:
            psf = app.CharmmPsfFile(psf)
        except TypeError:
            psf = psf

        indices = get_residue_indices(psf.atom_list)
        coords = []
        unit = pdb.positions.unit
        for i in indices:
            coords.append(list(pdb.positions[i].value_in_unit(unit)))

        return cls(unit=unit, coordinates=coords)

    def node_from_files(
        self, psf: str, pdb: str, name: str, include_hydrogens: bool = True
    ) -> TreeNode:
        """
        calculates the centroid coordinates of the ligand and returns the grid node closest to the centriod Cordinates.

        Args:
            psf (str): give either path to pdb file as string or an openmm.app.PDBFile object.
            pdb (str): give either path to psf file as string or an openmm.app.CharmmPsfFile object.
            name (str): name of the residue that is the starting point.

        Returns:
            Node: grid node closest to ligand centroid.

        TODO: add support for center of mass -> more complicated since it needs an initialized system.
        """
        try:
            pdb = app.PDBFile(pdb)
        except TypeError:
            pdb = pdb
        try:
            psf = app.CharmmPsfFile(psf)
        except TypeError:
            psf = psf

        indices = get_residue_indices(
            atom_list=psf.atom_list, name=name, include_hydrogens=include_hydrogens
        )
        coordinates = get_centroid_coordinates(positions=pdb.positions, indices=indices)
        return TreeNode.from_coords(
            [
                coordinates[0],
                coordinates[1],
                coordinates[2],
            ]
        )

    def position_is_blocked(
        self,
        node: TreeNode = None,
        coordinates: List[float] = None,
        unit: u.Unit = u.nanometer,
        vdw_radius: u.Quantity = 1.2 * u.angstrom,
    ) -> bool:
        """
        Checks if a Node is the vdw_radius of a protein Atom in the tree

        Args:
            node (TreeNode, optional): TreeNode type object. Defaults to None.
            coordinates (u.Wuantity, optional): grid cell coordinates. Defaults to None.
            unit (u.Unit, optional): unit of coordinates. Defaults to u.nanometer.
            vdw_radius (u.Quantity): vdw_radius given to each protein atom. Defaults to 1.2 * u.angstrom

        Returns:
            bool: True if Node is within grid
        """
        try:
            dist, i = self.tree.query(x=node.get_coordinates_for_query(self.unit), k=1)
        except AttributeError:
            coords = u.Quantity(
                value=Vec3(coordinates[0], coordinates[1], coordinates[2]), unit=unit
            )
            dist, i = self.tree.query(x=coords.value_in_unit(self.unit))
        return dist * self.unit < vdw_radius

    def get_distance_to_protein(
        self,
        node: TreeNode = None,
        coordinates: List[float] = None,
        unit: u.Unit = u.nanometer,
        vdw_radius: u.Quantity = 1.2 * u.angstrom,
    ) -> u.Quantity:
        """get_distance_to_protein
        returns distance to the nearest protein atom

        Parameters
        ----------
        node : Node
            node for which the distance should be calculated
        vdw_radius : u.Quantity, optional
            vdw_radius to be used for protein atoms, defaults to 1.2*u.angstrom

        Returns
        -------
        u.Quantity: distance to nearest protein atom
        """
        try:
            dist, i = self.tree.query(x=node.get_coordinates_for_query(self.unit), k=1)
        except AttributeError:
            coords = u.Quantity(
                value=Vec3(coordinates[0], coordinates[1], coordinates[2]), unit=unit
            )
            dist, i = self.tree.query(x=coords.value_in_unit(self.unit), k=1)
        dist = dist * self.unit - vdw_radius.in_units_of(self.unit)
        return dist

    def get_nearest_neighbour_index(self, coords: u.Quantity):
        dist, i = self.tree.query(x=[coords.x, coords.y, coords.z], k=1)
        return i

    def calculate_euclidean_distance(
        self, node: TreeNode, destination: TreeNode
    ) -> u.Quantity:
        """
        calculates euclidean distance heuristics between node and destination.
        NOT USED
        Args:
            node (TreeNode): point a
            destination (TreeNode): point b
        Returns:
            float: euclidean distance
        """
        return u.Quantity(
            value=math.sqrt(
                (node.x - destination.x) ** 2
                + (node.y - destination.y) ** 2
                + (node.z - destination.z) ** 2
            ),
            unit=node.unit,
        )

    def calculate_diagonal_distance(
        self, node: TreeNode, destination: TreeNode
    ) -> u.Quantity:
        """
        calculates diagonal distance heuristics between node and destination.
        Args:
            node (TreeNode): point a
            destination (TreeNode): point b
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
