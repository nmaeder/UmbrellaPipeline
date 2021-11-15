import math
from itertools import product
from typing import List
from scipy.spatial import KDTree

import gemmi
import numpy as np
import openmm.app as app
import openmm.unit as unit

from UmbrellaPipeline.pathGeneration.helper import (
    gen_box,
    get_indices,
    getCentroidCoordinates,
)
from UmbrellaPipeline.pathGeneration.node import Node


class Grid:
    """
    This class stores the grid and all information relevant for writing out coordinates.
    """

    def __init__(
        self,
        coordinates: unit.Quantity or List[unit.Quantity] or List[float],
        unit: unit.Unit = None,
    ):
        """

        Args:
            coordinates (unit.Quantity or List[unit.Quantity] or List[float]): List of coordinates to be added to the tree
            unit (unit.Unit): unit of coordinates if they are given without any.
        """
        try:
            self.unit = coordinates.unit
            pos = []
            for i in coordinates:
                pos.append(list(i.value_in_unit(i.unit)))
            self.tree = KDTree(pos)

        except TypeError:
            self.unit = unit
            self.tree = KDTree(coordinates)

    @classmethod
    def gridFromFiles(
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

        indices = get_indices(psf.atom_list)
        coords = []
        unit = pdb.position.unit
        for i in indices:
            coords.append(list(pdb.positions[i].value_in_unit(unit)))

        return cls(unit=unit, coordinates=coords)

    def nodeFromFiles(self, psf: str, pdb: str, name: str) -> Node:
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
        return Node.fromCoords(
            [
                math.floor((coordinates[0] - self.offset[0]) / self.a),
                math.floor((coordinates[1] - self.offset[1]) / self.a),
                math.floor((coordinates[2] - self.offset[2]) / self.a),
            ]
        )

    def getGridValue(self, node: Node = None, coordinates: List[int] = None) -> bool:
        """
        returns Value of gridcell.
        Args:
            node (Node, optional): Node type object. Defaults to None.
            coordinates (List[int], optional): Node coordinates. Defaults to None.

        Returns:
            bool: value of gridcell
        """
        return (
            self.grid[node.x][node.y][node.z]
            if node
            else self.grid[coordinates[0]][coordinates[1]][coordinates[2]]
        )

    def positionIsValid(
        self,
        node: Node = None,
        coordinates: List[int] = None,
        vdwRadius: unit.Quantity = 1.2 * unit.angstrom,
    ) -> bool:
        """
        Checks if a Node is within the grid
        depreceated

        Args:
            node (Node, optional): Node type object. Defaults to None.
            coordinates (List[int], optional): grid cell coordinates. Defaults to None.

        Returns:
            bool: True if Node is within grid
        """
        if node:
            return self.tree.query()
        if coordinates:
            return (
                0 <= coordinates[0] < self.x
                and 0 <= coordinates[1] < self.y
                and 0 <= coordinates[2] < self.z
            )
        return False

    def positionIsBlocked(
        self, node: Node, vdwRadius: unit.Quantity = 1.2 * unit.angstrom
    ) -> bool:
        """Returns true if a gridcell is occupied with a protien atom.

        Args:
            node (Node, optional): Node type object. Defaults to None.
            coordinates (List[int], optional): grid cell coordinates. Defaults to None.

        Returns:
            bool: True if position is occupied by protein
        """
        dist, key = self.tree.query(x=node.getCoordinateValuesInUnit(self.unit), k=1)
        return not dist * self.unit > vdwRadius

    def distanceToProtein(
        self,
        node: Node,
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
        dist, key = self.tree.query(x=node.getCoordinateValuesInUnit(self.unit))
        dist = dist * vdwRadius.unit - vdwRadius
        return dist.in_units_of(self.unit)

    def areSurroundingsBlocked(self, node: Node, pathsize: int) -> bool:
        """
        checks if surroundigns in a given radius are occupied. only checks the 16 outmost points.
        depreceated

        Args:
            node (Node): Node type object
            pathsize (int): Radius in which surroundings are checked.

        Returns:
            bool: True if a surrounding blocv is true
        """
        for dx, dy, dz in product(
            [pathsize, int(pathsize / 2)],
            [pathsize, int(pathsize / 2)],
            [pathsize, int(pathsize / 2)],
        ):
            if not self.positionIsValid(
                coordinates=[node.x + dx, node.y + dy, node.z + dz]
            ):
                continue
            if any(
                term
                for term in [
                    self.grid[node.x - dx][node.y - dy][node.z - dz],
                    self.grid[node.x - dx][node.y - dy][node.z + dz],
                    self.grid[node.x - dx][node.y + dy][node.z - dz],
                    self.grid[node.x - dx][node.y + dy][node.z + dz],
                    self.grid[node.x + dx][node.y - dy][node.z - dz],
                    self.grid[node.x + dx][node.y - dy][node.z + dz],
                    self.grid[node.x + dx][node.y + dy][node.z - dz],
                    self.grid[node.x + dx][node.y + dy][node.z + dz],
                ]
            ):
                return True
        return False

    def toCcp4(self, filename: str):
        """
        Write out CCP4 density map of the grid. good for visualization in VMD/pymol.
        depreceatred

        Args:
            filename (str): path the file should be written to.

        Returns:
            None: Nothing
        """
        if not filename.endswith(".ccp4"):
            filename += ".ccp4"
        print("Hang in there, this can take a while (~1 Minute)")
        ccp4_map = gemmi.Ccp4Map()
        ccp4_map.grid = gemmi.FloatGrid(self.grid.astype(np.float32))
        ccp4_map.update_ccp4_header()
        ccp4_map.write_ccp4_map(filename)
        return None

    def toXYZCoordinates(self) -> List[float]:
        """
        returns list of cartesian coordinates that are true in the grid.
        depreceated

        Returns:
            List[float]: List of cartesian coordinates with value true.
        """
        ret = []
        for x, y, z in product(self.x, self.y, self.z):
            if self.grid[x][y][z]:
                ret.append(
                    [
                        x * self.a + self.offset[0],
                        y * self.b + self.offset[0],
                        z * self.c + self.offset[0],
                    ]
                )
        return ret
