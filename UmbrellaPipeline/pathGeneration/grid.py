import numpy as np
from helper import *
from node import Node
import math
from itertools import product
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from typing import Typing

class Grid:

    def __init__(self, x:int = 0, y:int = 0, z:int = 0, dtype:type = bool, grid:np.array = None, boxlengths:unit.Quantity or List[unit.Quantity] = None, offset:List[unit.Quantity] = None):
        if type(grid) != None:
            self.grid = grid 
            self.x = grid.shape[0]
            self.y = grid.shape[1] 
            self.z = grid.shape[2]
        else:
            self.grid = grid 
            self.x = x
            self.y = y
            self.z = z
        self.a, self.b, self.c = boxlengths[0], boxlengths[1], boxlengths[2]
        self.offset = offset
        
        
    @classmethod    
    def gridFromFiles(cls, pdbfile:str, psffile:str, gridsize:unit.Quantity or List[unit.Quantity] = .1*unit.angstrom, vdwradius:float = 1.2*unit.angstrom, addVDW:bool = True):
        pdb, psf = app.PDBFile(pdbfile), app.CharmmPsfFile(psffile)
        inx, min_c = get_indices(psf.atom_list), gen_box(psf, pdb)
        n = [round(psf.boxLengths[0]/gridsize), round(psf.boxLengths[1]/gridsize), round(psf.boxLengths[2]/gridsize)]
        l = [psf.boxLengths[0]/n[0], psf.boxLengths[1]/n[1], psf.boxLengths[2]/n[2]]
        numadd = 0 if not vdwradius else [round(vdwradius/l[0]),round(vdwradius/l[1]),round(vdwradius/l[2])]
        grid = np.zeros(shape=(n[0],n[1],n[2]), dtype=bool)
        for index in inx:
            x, y, z = math.floor((pdb.positions[index][0] - min_c[0]) / l[0]), math.floor((pdb.positions[index][1] - min_c[1]) / l[1]), math.floor((pdb.positions[index][2] - min_c[2]) / l[2])
            grid[z][y][x] = True
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
        pdb, psf = app.PDBFile(pdb), app.CharmmPsfFile(psf)    
        indices = get_indices(atom_list=psf.atom_list, name=name)
        coordinates = getCentroidCoordinates(positions=pdb.positions, indices=indices)
        return Node.fromCoords([math.floor((coordinates[0] - self.offset[0]) / self.a), math.floor((coordinates[1] - self.offset[1]) / self.a), math.floor((coordinates[2] - self.offset[2]) / self.a)])

    def getGridValue(self, node:Node = None, coordinates:List[int] = None) -> bool:
        return self.grid[node.x][node.y][node.z] if node else self.grid[coordinates[0]][coordinates[1]][coordinates[2]]

    def positionIsValid(self, node:Node = None, coordinates:List[int] = None) -> bool:
        if node:
            return (0 <= node.x < self.x and 0 <= node.y < self.y and 0 <= node.z < self.z)
        if coordinates:
            return (0 <= coordinates[0] < self.x and 0 <= coordinates[1] < self.y and 0 <= coordinates[2] < self.z)
        return False

    def positionIsBlocked(self, node:Node = None, coordinates:List[int] = None) -> bool:
        return self.getGridValue(node=node, coordinates=coordinates)

    def toXYZCoordinates(self):
        ret = []
        for x,y,z in product(self.x, self.y, self.z):
            if self.grid[x][y][z]:
                ret.append([x*self.a+self.offset[0], y*self.b+self.offset[0], z*self.c+self.offset[0]])
        return ret