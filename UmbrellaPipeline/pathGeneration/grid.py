import numpy as np
import gemmi
from helper import *
from node import Node
import math
from itertools import product
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from typing import List

class Grid:
    """
    This class stores the grid and all information relevant for writing out coordinates.
    """
    def __init__(self, x:int = 0, y:int = 0, z:int = 0, dtype:type = bool, grid:np.array = None, boxlengths:unit.Quantity or List[unit.Quantity] = None, offset:List[unit.Quantity] = None):
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
        
        if grid:
            self.grid = grid 
            self.x = grid.shape[0]
            self.y = grid.shape[1] 
            self.z = grid.shape[2]
        else:
            self.grid = np.zeros(shape=(x,y,z), dtype=bool) 
            self.x = x
            self.y = y
            self.z = z
        self.a, self.b, self.c = boxlengths[0], boxlengths[1], boxlengths[2]
        self.offset = offset
        
        
    @classmethod    
    def gridFromFiles(cls, pdb:str or app.PDBFile, psf:str or app.CharmmPsfFile, gridsize:unit.Quantity or List[unit.Quantity] = .1*unit.angstrom, vdwradius:unit.Quantity = 1.2*unit.angstrom, addVDW:bool = True):
        """
        Constructor for grid. takes in psf and pdb files generated from charmmgui and generates a grid where all points with a protein atom are true. every other gridpoint is False.

        Args:
            pdbfile (str): give either path to pdb file as string or an openmm.app.PDBFile object.
            psffile (str): give either path to psf file as string or an openmm.app.CharmmPsfFile object.
            gridsize (unit.QuantityorList[unit.Quantity], optional): [description]. Defaults to .1*unit.angstrom.
            vdwradius (unit.Quantity, optional): VDW radius of the protein atoms in the grid. Defaults to 1.2*unit.angstrom.
            addVDW (bool, optional): Whether or not the protein atoms should have a VDW radius in the grid. Defaults to True.

        Returns:
            Grid: Boolean grid where protein positions are True.
        """
        if type(pdb) is str: pdb = app.PDBFile(pdb)
        if type(psf) is str: psf = app.CharmmPsfFile(psf)    
        inx = get_indices(psf.atom_list)
        min_c = gen_box(psf, pdb)
        n = [round(psf.boxLengths[0]/gridsize), round(psf.boxLengths[1]/gridsize), round(psf.boxLengths[2]/gridsize)]
        l = [psf.boxLengths[0]/n[0], psf.boxLengths[1]/n[1], psf.boxLengths[2]/n[2]]
        numadd = [0,0,0] if not vdwradius else [round(vdwradius/l[0]),round(vdwradius/l[1]),round(vdwradius/l[2])]
        grid = np.zeros(shape=(n[0],n[1],n[2]), dtype=bool)
        for index in inx:
            x, y, z = math.floor((pdb.positions[index][0] - min_c[0]) / l[0]), math.floor((pdb.positions[index][1] - min_c[1]) / l[1]), math.floor((pdb.positions[index][2] - min_c[2]) / l[2])
            grid[x][y][z] = True
            if addVDW:
                for dx, dy, dz in product(range(numadd[0]+1), range(numadd[1]+1), range(numadd[2]+1)):
                    if dx == dy == dz == 0: continue
                    if math.sqrt(dx**2 + dy**2 + dz**2) > numadd[0]: continue
                    grid[x-dx][y-dy][z-dz] = True
                    grid[x-dx][y-dy][z+dz] = True
                    grid[x-dx][y+dy][z-dz] = True
                    grid[x-dx][y+dy][z+dz] = True
                    grid[x+dx][y-dy][z-dz] = True
                    grid[x+dx][y-dy][z+dz] = True
                    grid[x+dx][y+dy][z-dz] = True
                    grid[x+dx][y+dy][z+dz] = True
        return cls(grid=grid, boxlengths=[l[0],l[1],l[2]], offset=min_c)

    def nodeFromFiles(self, psf:str, pdb:str, name:str) -> Node:
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
        if type(pdb) is str: pdb = app.PDBFile(pdb)
        if type(psf) is str: psf = app.CharmmPsfFile(psf)    
        indices = get_indices(atom_list=psf.atom_list, name=name)
        coordinates = getCentroidCoordinates(positions=pdb.positions, indices=indices)
        return Node.fromCoords([math.floor((coordinates[0] - self.offset[0]) / self.a), math.floor((coordinates[1] - self.offset[1]) / self.a), math.floor((coordinates[2] - self.offset[2]) / self.a)])

    def getGridValue(self, node:Node = None, coordinates:List[int] = None) -> bool:
        """
        returns Value of gridcell.
        Args:
            node (Node, optional): Node type object. Defaults to None.
            coordinates (List[int], optional): Node coordinates. Defaults to None.

        Returns:
            bool: value of gridcell
        """
        return self.grid[node.x][node.y][node.z] if node else self.grid[coordinates[0]][coordinates[1]][coordinates[2]]

    def positionIsValid(self, node:Node = None, coordinates:List[int] = None) -> bool:
        """
        Checks if a Node is within the grid

        Args:
            node (Node, optional): Node type object. Defaults to None.
            coordinates (List[int], optional): grid cell coordinates. Defaults to None.

        Returns:
            bool: True if Node is within grid
        """
        if node:
            return (0 <= node.x < self.x and 0 <= node.y < self.y and 0 <= node.z < self.z)
        if coordinates:
            return (0 <= coordinates[0] < self.x and 0 <= coordinates[1] < self.y and 0 <= coordinates[2] < self.z)
        return False

    def positionIsBlocked(self, node:Node = None, coordinates:List[int] = None) -> bool:
        """Returns true if a gridcell is occupied with a protien atom.

        Args:
            node (Node, optional): Node type object. Defaults to None.
            coordinates (List[int], optional): grid cell coordinates. Defaults to None.

        Returns:
            bool: True if position is occupied by protein
        """
        return self.getGridValue(node=node, coordinates=coordinates)

    def areSurroundingsBlocked(self, node:Node, pathsize:int) -> bool:
        """
        checks if surroundigns in a given radius are occupied. only checks the 16 outmost points.

        Args:
            node (Node): Node type object
            pathsize (int): Radius in which surroundings are checked.

        Returns:
            bool: True if a surrounding blocv is true
        """
        for dx, dy, dz in product([pathsize, int(pathsize/2)], [pathsize, int(pathsize/2)], [pathsize, int(pathsize/2)]):
            if not self.positionIsValid(coordinates=[node.x+dx, node.y+dy, node.z+dz]): continue
            if any (term for term in [self.grid[node.x-dx][node.y-dy][node.z-dz], self.grid[node.x-dx][node.y-dy][node.z+dz], self.grid[node.x-dx][node.y+dy][node.z-dz], self.grid[node.x-dx][node.y+dy][node.z+dz], self.grid[node.x+dx][node.y-dy][node.z-dz], self.grid[node.x+dx][node.y-dy][node.z+dz], self.grid[node.x+dx][node.y+dy][node.z-dz], self.grid[node.x+dx][node.y+dy][node.z+dz]]):
                return True
        return False

    def toCcp4(self, filename:str):
        """
        Write out CCP4 density map of the grid. good for visualization in VMD/pymol.

        Args:
            filename (str): path the file should be written to.

        Returns:
            None: Nothing
        """
        if not filename.endswith('.ccp4'): filename += '.ccp4'
        print('Hang in there, this can take a while (~1 Minute)')
        ccp4_map = gemmi.Ccp4Map()
        ccp4_map.grid = gemmi.FloatGrid(self.grid.astype(np.float32))
        ccp4_map.update_ccp4_header()
        ccp4_map.write_ccp4_map(filename)
        return None

    
    def toXYZCoordinates(self) -> List[float]:
        """
        returns list of cartesian coordinates that are true in the grid.

        Returns:
            List[float]: List of cartesian coordinates with value true.
        """
        ret = []
        for x,y,z in product(self.x, self.y, self.z):
            if self.grid[x][y][z]:
                ret.append([x*self.a+self.offset[0], y*self.b+self.offset[0], z*self.c+self.offset[0]])
        return ret