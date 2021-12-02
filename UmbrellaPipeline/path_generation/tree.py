import math
from typing import List

import openmm.app as app
import openmm.unit as unit
from openmm import Vec3
from scipy.spatial import KDTree
from UmbrellaPipeline.path_generation.path_helper import (
    get_residue_indices,
    get_centroid_coordinates,
)
from UmbrellaPipeline.path_generation.node import TreeNode


class Tree:
    """
    This class stores the KDTree and all information relevant for writing out coordinates.
    """

    def __init__(
        self,
        coordinates: unit.Quantity or List[unit.Quantity] or List[float],
        unit_: unit.Unit = None,
    ):
        """

        Args:
            coordinates (unit.Quantity or List[unit.Quantity] or List[float]): List of coordinates to be added to the tree
            unit (unit.Unit): unit of coordinates if they are given without any.
        """
        if unit_:
            try:
                self.unit = unit_
                pos = []
                for i in coordinates:
                    pos.append(list(i.value_in_unit(i.unit)))
                self.tree = KDTree(pos)
            except AttributeError:
                self.unit = unit_
                self.tree = KDTree(coordinates)
        else:
            try:
                self.unit = coordinates[0].unit
                pos = []
                for i in coordinates:
                    pos.append(list(i.value_in_unit(i.unit)))
                self.tree = KDTree(pos)
            except AttributeError:
                raise ValueError("no unit_ provided.")
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

        return cls(unit_=unit, coordinates=coords)

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
        unit_: unit.Unit = unit.nanometer,
        vdw_radius: unit.Quantity = 1.2 * unit.angstrom,
    ) -> bool:
        """
        Checks if a Node is the vdw_radius of a protein Atom in the tree

        Args:
            node (TreeNode, optional): TreeNode type object. Defaults to None.
            coordinates (unit.Wuantity, optional): grid cell coordinates. Defaults to None.
            unit (unit.Unit, optional): unit of coordinates. Defaults to unit.nanometer.
            vdw_radius (unit.Quantity): vdw_radius given to each protein atom. Defaults to 1.2 * unit.angstrom

        Returns:
            bool: True if Node is within grid
        """
        try:
            dist, i = self.tree.query(x=node.get_coordinates_for_query(self.unit), k=1)
        except AttributeError:
            coords = unit.Quantity(
                value=Vec3(coordinates[0], coordinates[1], coordinates[2]), unit=unit_
            )
            dist, i = self.tree.query(x=coords.value_in_unit(self.unit))
        return dist * self.unit < vdw_radius

    def get_distance_to_protein(
        self,
        node: TreeNode = None,
        coordinates: List[float] = None,
        unit_: unit.Unit = unit.nanometer,
        vdw_radius: unit.Quantity = 1.2 * unit.angstrom,
    ) -> unit.Quantity:
        """get_distance_to_protein
        returns distance to the nearest protein atom

        Parameters
        ----------
        node : Node
            node for which the distance should be calculated
        vdw_radius : unit.Quantity, optional
            vdw_radius to be used for protein atoms, defaults to 1.2*unit.angstrom

        Returns
        -------
        unit.Quantity: distance to nearest protein atom
        """
        try:
            dist, i = self.tree.query(x=node.get_coordinates_for_query(self.unit), k=1)
        except AttributeError:
            coords = unit.Quantity(
                value=Vec3(coordinates[0], coordinates[1], coordinates[2]), unit=unit_
            )
            dist, i = self.tree.query(x=coords.value_in_unit(self.unit), k=1)
        dist = dist * self.unit - vdw_radius.in_units_of(self.unit)
        return dist

    def estimate_euclidean_h(self, node: TreeNode, destination: TreeNode) -> float:
        """
        estimates euclidean distance heuristics between node and destination.
        NOT USED
        Args:
            node (TreeNode): point a
            destination (TreeNode): point b
        Returns:
            float: euclidean distance estimate
        """
        return math.sqrt(
            (node.x - destination.x) ** 2
            + (node.y - destination.y) ** 2
            + (node.z - destination.z) ** 2
        )

    def estimate_diagonal_h(self, node: TreeNode, destination: TreeNode) -> float:
        """
        estimates diagonal distance heuristics between node and destination.
        Args:
            node (TreeNode): point a
            destination (TreeNode): point b
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
