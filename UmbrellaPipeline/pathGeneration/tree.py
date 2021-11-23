from typing import List

import openmm.app as app
import openmm.unit as unit
from openmm import Vec3
from scipy.spatial import KDTree
from UmbrellaPipeline.pathGeneration.pathHelper import get_indices, getCentroidCoordinates
from UmbrellaPipeline.pathGeneration.node import TreeNode


class Tree:
    """
    This class stores the KDTree and all information relevant for writing out coordinates.
    """

    def __init__(
        self,
        coordinates: unit.Quantity or List[unit.Quantity] or List[float],
        _unit: unit.Unit = None,
    ):
        """

        Args:
            coordinates (unit.Quantity or List[unit.Quantity] or List[float]): List of coordinates to be added to the tree
            unit (unit.Unit): unit of coordinates if they are given without any.
        """
        if _unit:
            try:
                self.unit = _unit
                pos = []
                for i in coordinates:
                    pos.append(list(i.value_in_unit(i.unit)))
                self.tree = KDTree(pos)
            except AttributeError:
                self.unit = _unit
                self.tree = KDTree(coordinates)
        else:
            try:
                self.unit = coordinates[0].unit
                pos = []
                for i in coordinates:
                    pos.append(list(i.value_in_unit(i.unit)))
                self.tree = KDTree(pos)
            except AttributeError:
                raise ValueError("no _unit provided.")
        self.possibleNeighbours = [
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
    def treeFromFiles(
        cls, pdb: str or app.PDBFile, psf: str or app.CharmmPsfFile,
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

        indices = get_indices(psf.atom_list)
        coords = []
        unit = pdb.positions.unit
        for i in indices:
            coords.append(list(pdb.positions[i].value_in_unit(unit)))

        return cls(_unit=unit, coordinates=coords)

    def nodeFromFiles(self, psf: str, pdb: str, name: str) -> TreeNode:
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

        indices = get_indices(atom_list=psf.atom_list, name=name)
        coordinates = getCentroidCoordinates(positions=pdb.positions, indices=indices)
        return TreeNode.fromCoords([coordinates[0], coordinates[1], coordinates[2],])

    def positionIsBlocked(
        self,
        node: TreeNode = None,
        coordinates: List[float] = None,
        _unit: unit.Unit = unit.nanometer,
        vdwRadius: unit.Quantity = 1.2 * unit.angstrom,
    ) -> bool:
        """
        Checks if a Node is the vdwRadius of a protein Atom in the tree

        Args:
            node (TreeNode, optional): TreeNode type object. Defaults to None.
            coordinates (unit.Wuantity, optional): grid cell coordinates. Defaults to None.
            unit (unit.Unit, optional): unit of coordinates. Defaults to unit.nanometer.
            vdwRadius (unit.Quantity): vdwRadius given to each protein atom. Defaults to 1.2 * unit.angstrom

        Returns:
            bool: True if Node is within grid
        """
        try:
            dist, i = self.tree.query(x=node.coordsForQuery(self.unit), k=1)
        except AttributeError:
            coords = unit.Quantity(
                value=Vec3(coordinates[0], coordinates[1], coordinates[2]), unit=_unit
            )
            dist, i = self.tree.query(x=coords.value_in_unit(self.unit))
        return dist * self.unit < vdwRadius

    def distanceToProtein(
        self,
        node: TreeNode = None,
        coordinates: List[float] = None,
        _unit: unit.Unit = unit.nanometer,
        vdwRadius: unit.Quantity = 1.2 * unit.angstrom,
    ) -> unit.Quantity:
        """distanceToProtein
        returns distance to the nearest protein atom

        Parameters
        ----------
        node : Node
            node for which the distance should be calculated
        vdwRadius : unit.Quantity, optional
            vdwRadius to be used for protein atoms, defaults to 1.2*unit.angstrom

        Returns
        -------
        unit.Quantity: distance to nearest protein atom
        """
        try:
            dist, i = self.tree.query(x=node.coordsForQuery(self.unit), k=1)
        except AttributeError:
            coords = unit.Quantity(
                value=Vec3(coordinates[0], coordinates[1], coordinates[2]), unit=_unit
            )
            dist, i = self.tree.query(x=coords.value_in_unit(self.unit), k=1)
        dist = dist * self.unit - vdwRadius.in_units_of(self.unit)
        return dist
