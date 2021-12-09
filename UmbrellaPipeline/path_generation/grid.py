import math
from itertools import product
from typing import List, ValuesView
from openmm import Vec3

import gemmi
import numpy as np
import openmm.app as app
import openmm.unit as unit
from UmbrellaPipeline.path_generation.path_helper import (
    gen_pbc_box,
    get_residue_indices,
    get_centroid_coordinates,
)
from UmbrellaPipeline.path_generation.node import GridNode


class Grid:
    """
    This class stores the grid and all information relevant for writing out coordinates.
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        z: int = 0,
        dtype: type = bool,
        grid: np.array = None,
        boxlengths: unit.Quantity or List[unit.Quantity] = None,
        offset: unit.Quantity = unit.Quantity(Vec3(0, 0, 0), unit=unit.nanometer),
    ):
        """
        Args:
            x (int, optional): number of grid cells in x direction. Defaults to 0.
            y (int, optional): number of grid cells in x direction. Defaults to 0.
            z (int, optional): number of grid cells in x direction. Defaults to 0.
            dtype (type, optional): datatype for the grid to be. Defaults to bool.
            grid (np.array, optional): already existing numpy array. Defaults to None.
            boxlengths (unit.Quantity or List[unit.Quantity], optional): gridcell size. Defaults to None.
            offset (List[unit.Quantity], optional): if gridpoint (0,0,0) does not correspond to the cartesian (0,0,0). Defaults to None.
        """
        try:
            self.grid = grid
            self.x = grid.shape[0]
            self.y = grid.shape[1]
            self.z = grid.shape[2]
            self.dtype = grid.dtype
        except AttributeError:
            try:
                self.grid = np.zeros(shape=(x, y, z), dtype=dtype)
                self.x = x
                self.y = y
                self.z = z
                self.dtype = dtype
            except IndexError as err:
                raise err
        try:
            self.a, self.b, self.c = boxlengths[0], boxlengths[1], boxlengths[2]
        except TypeError:
            self.a, self.b, self.c = boxlengths, boxlengths, boxlengths
        self.offset = offset
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
        gridsize: unit.Quantity or List[unit.Quantity] = 0.1 * unit.angstrom,
        vdw_radius: unit.Quantity = 1.2 * unit.angstrom,
        add_vdw: bool = True,
    ):
        """
        Constructor for grid. takes in psf and pdb files generated from charmmgui and generates a grid where all points with a protein atom are true. every other gridpoint is False.
        Args:
            pdbfile (str): give either path to pdb file as string or an openmm.app.PDBFile object.
            psffile (str): give either path to psf file as string or an openmm.app.CharmmPsfFile object.
            gridsize (unit.QuantityorList[unit.Quantity], optional): [description]. Defaults to .1*unit.angstrom.
            vdw_radius (unit.Quantity, optional): VDW radius of the protein atoms in the grid. Defaults to 1.2*unit.angstrom.
            add_vdw (bool, optional): Whether or not the protein atoms should have a VDW radius in the grid. Defaults to True.
        Returns:
            Grid: Boolean grid where protein positions are True.
        """
        try:
            pdb = app.PDBFile(pdb)
        except TypeError:
            pdb = pdb
        try:
            psf = app.CharmmPsfFile(psf)
        except TypeError:
            psf = psf

        inx = get_residue_indices(psf.atom_list)
        if psf.boxVectors == None:
            min_c = gen_pbc_box(psf, pdb)

        n = [
            round(psf.boxLengths[0] / gridsize),
            round(psf.boxLengths[1] / gridsize),
            round(psf.boxLengths[2] / gridsize),
        ]
        l = [
            psf.boxLengths[0] / n[0],
            psf.boxLengths[1] / n[1],
            psf.boxLengths[2] / n[2],
        ]
        numadd = (
            [0, 0, 0]
            if not vdw_radius
            else [
                round(vdw_radius / l[0]),
                round(vdw_radius / l[1]),
                round(vdw_radius / l[2]),
            ]
        )
        grid = np.zeros(shape=(n[0], n[1], n[2]), dtype=bool)
        for index in inx:
            x, y, z = (
                math.floor((pdb.positions[index][0] - min_c[0]) / l[0]),
                math.floor((pdb.positions[index][1] - min_c[1]) / l[1]),
                math.floor((pdb.positions[index][2] - min_c[2]) / l[2]),
            )
            grid[x][y][z] = True
            if add_vdw:
                for dx, dy, dz in product(
                    range(numadd[0] + 1), range(numadd[1] + 1), range(numadd[2] + 1)
                ):
                    if dx == dy == dz == 0:
                        continue
                    if math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) > numadd[0]:
                        continue
                    grid[x - dx][y - dy][z - dz] = True
                    grid[x - dx][y - dy][z + dz] = True
                    grid[x - dx][y + dy][z - dz] = True
                    grid[x - dx][y + dy][z + dz] = True
                    grid[x + dx][y - dy][z - dz] = True
                    grid[x + dx][y - dy][z + dz] = True
                    grid[x + dx][y + dy][z - dz] = True
                    grid[x + dx][y + dy][z + dz] = True

        return cls(grid=grid, boxlengths=[l[0], l[1], l[2]], offset=min_c)

    def node_from_files(self, psf: str, pdb: str, name: str) -> GridNode:
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

        indices = get_residue_indices(atom_list=psf.atom_list, name=name)
        coordinates = get_centroid_coordinates(positions=pdb.positions, indices=indices)
        ret = GridNode(
            x=math.floor((coordinates[0] - self.offset[0]) / self.a),
            y=math.floor((coordinates[1] - self.offset[1]) / self.a),
            z=math.floor((coordinates[2] - self.offset[2]) / self.a),
        )
        if not self.position_is_valid(ret):
            raise ValueError("Node not within grid")
        return ret

    def get_grid_value(
        self, node: GridNode = None, coordinates: List[int] = None
    ) -> bool:
        """
        returns Value of gridcell.
        Args:
            node (Node, optional): Node type object. Defaults to None.
            coordinates (List[int], optional): Node coordinates. Defaults to None.
        Returns:
            bool: value of gridcell
        """
        try:
            return (
                self.grid[node.x][node.y][node.z]
                if node
                else self.grid[coordinates[0]][coordinates[1]][coordinates[2]]
            )
        except IndexError as err:
            raise err

    def position_is_valid(
        self, node: GridNode = None, coordinates: List[int] = None
    ) -> bool:
        """
        Checks if a Node is within the grid.
        Args:
            node (Node, optional): Node type object. Defaults to None.
            coordinates (List[int], optional): grid cell coordinates. Defaults to None.
        Returns:
            bool: True if Node is within grid
        """
        if node:
            return node.x < self.x and node.y < self.y and node.z < self.z
        if coordinates:
            return (
                0 <= coordinates[0] < self.x
                and 0 <= coordinates[1] < self.y
                and 0 <= coordinates[2] < self.z
            )
        return False

    def position_is_blocked(
        self, node: GridNode = None, coordinates: List[int] = None
    ) -> bool:
        """Returns true if a gridcell is occupied with a protien atom.
        Args:
            node (Node, optional): Node type object. Defaults to None.
            coordinates (List[int], optional): grid cell coordinates. Defaults to None.
        Returns:
            bool: True if position is occupied by protein
        """
        return self.get_grid_value(node=node, coordinates=coordinates)

    def calculate_diagonal_distance(
        self, node: GridNode, destination: GridNode
    ) -> float:
        """
        calculates diagonal distance between node and destination.
        Args:
            node (GridNode): point a
            destination (GridNode): point b
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

    def get_distance_to_protein(self, node: GridNode) -> float:
        """
        Args:
            node (GridNode): [description]

        Returns:
            [type]: [description]

        """
        for i in range(1, min(self.x, self.y, self.z), 1):
            for n in self.POSSIBLE_NEIGHBOURS:
                x, y, z = node.x + n[0] * i, node.y + n[1] * i, node.z + n[2] * i
                try:
                    node2 = GridNode(x, y, z)
                except ValueError:
                    continue
                try:
                    if self.get_grid_value(node=node2):
                        return self.calculate_diagonal_distance(
                            node=node, destination=node2
                        )
                except IndexError:
                    continue
        return 0

    def to_ccp4(self, filename: str):
        """
        Write out CCP4 density map of the grid. good for visualization in VMD/pymol.
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

    def get_cartesian_distance(self, node1: GridNode, node2: GridNode) -> unit.Quantity:
        """
        Helper function for get_path_for_sampling(). Returns the true distance between two grid points.

        Args:
            node1 (GridNode):
            node2 (GridNode):

        Returns:
            [type]: distance between two grid points in units the gridcellsizes have.
        """
        u = self.a.unit
        ret = math.sqrt(
            ((node2.x - node1.x) * self.a.value_in_unit(u)) ** 2
            + ((node2.y - node1.y) * self.b.value_in_unit(u)) ** 2
            + ((node2.z - node1.z) * self.c.value_in_unit(u)) ** 2
        )
        return unit.Quantity(value=ret, unit=u)

    def cartesian_coordinates_of_node(self, node: GridNode) -> unit.Quantity:
        """
        returns the cartesian coordinates of a node in a grid.

        Parameters
        ----------
        node : GridNode
            node you want the cartesian coordinates from

        Returns
        -------
        unit.Quantity
            cartesian coordinates of the node.

        Raises
        ------
        ValueError
            if given node is outside the grid
        """
        if not self.position_is_valid(node):
            raise ValueError("Node outside of grid!")
        u = self.a.unit
        return unit.Quantity(
            value=Vec3(
                x=node.x * self.a.value_in_unit(u) + self.offset[0].value_in_unit(u),
                y=node.y * self.b.value_in_unit(u) + self.offset[1].value_in_unit(u),
                z=node.z * self.c.value_in_unit(u) + self.offset[2].value_in_unit(u),
            ),
            unit=u,
        )

    def cartesian_coordinates_w_increment(
        self, node1: GridNode, node2: GridNode, to_go: float
    ):
        """
        returns the coordinates between node 1 and 2 where factor gives the total increment in all 3 dimenstions.
        factor has to be smaller than the distance between the two nodes!

        Parameters
        ----------
        node1 : GridNode
            node1
        node2 : GridNode
            node1
        factor : float
            distance to be traveled

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        ValueError
            [description]
        """
        u = self.a.unit
        if to_go * u > self.get_cartesian_distance(node1=node1, node2=node2):
            raise ValueError("to_go is bigger than distance between two nodes")
        if not self.position_is_valid(node1) or not self.position_is_valid(node2):
            raise ValueError("Either Node is outside of the grid")
        return unit.Quantity(
            value=Vec3(
                x=((node2.x - node1.x) * to_go + node1.x) * self.a.value_in_unit(u)
                + self.offset[0].value_in_unit(u),
                y=((node2.y - node1.y) * to_go + node1.y) * self.b.value_in_unit(u)
                + self.offset[1].value_in_unit(u),
                z=((node2.z - node1.z) * to_go + node1.z) * self.c.value_in_unit(u)
                + self.offset[2].value_in_unit(u),
            ),
            unit=u,
        )

    def to_cartesian_coordinates(self) -> List[float]:
        """
        returns list of cartesian coordinates that are true in the grid.
        Returns:
            List[float]: List of cartesian coordinates with value true.
        """
        ret = []
        for x, y, z in product(range(self.x), range(self.y), range(self.z)):
            if self.grid[x][y][z]:
                ret.append(
                    [
                        x * self.a + self.offset[0],
                        y * self.b + self.offset[1],
                        z * self.c + self.offset[2],
                    ]
                )
        return ret
