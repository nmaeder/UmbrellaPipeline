from typing import List
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import os
from itertools import product

aa_list = ['ala', 'arg', 'asn', 'asp', 'cys', 'gln', 'glu', 'gly', 'his', 'ile', 'leu', 'lys', 'met', 'phe', 'pro', 'pyl', 'ser', 'sec', 'thr', 'trp', 'tyr', 'val', 'asx', 'glx', 'xaa', 'xle']

def get_indices(atom_list: app.internal.charmm.topologyobjects.AtomList, name: List[str] or str = aa_list) -> List[int]:
    ret = []
    for i, atom in enumerate(atom_list): 
        if name is str:
            if name.lower() in str(atom).lower() : ret.append(i)
        else:
            if any (aa.lower() in str(atom).lower() for aa in name): ret.append(i)

def getCentroidCoordinates(positions:unit.Quantity, indices:List[int]) -> unit.Quantity:
    ret = [0*unit.nanometer, 0*unit.nanometer, 0*unit.nanometer]
    for coordinate in range(3):
        for i in indices:
            ret[coordinate] += positions[i][coordinate]
        ret[coordinate] / len(indices)
    return ret

def getCenterOfMassCoordinates(positions:unit.Quantity, indices:unit.Quantity, masses:mm.openmm.System) -> unit.Quantity:
    ret = [0*unit.nanometer,0*unit.nanometer,0*unit.nanometer]
    mass = 0*unit.dalton
    for coordinate in range(3):
        for atomnr in indices:
            ret[coordinate] += positions[atomnr][coordinate] * masses.getParticleMass(atomnr)
            mass += masses.getParticleMass(atomnr)
        ret[coordinate] /= mass
    
def get_params(param_directory:str, param_str_file:str) -> app.charmmparameterset.CharmmParameterSet:
    current_dir = os.getcwd()
    os.chdir(param_directory)
    parFiles = ()
    for line in open(param_str_file, 'r'):
        if '!' in line: line = line.split('!')[0]
        parfile = line.strip()
        if len(parfile) != 0: parFiles += (parfile, )
    ret = app.CharmmParameterSet( *parFiles )
    os.chdir(current_dir)
    return ret

def gen_box(psf, crd):
    coords = crd.positions

    min_crds = [coords[0][0], coords[0][1], coords[0][2]]
    max_crds = [coords[0][0], coords[0][1], coords[0][2]]

    for coord in coords:
        min_crds[0] = min(min_crds[0], coord[0])
        min_crds[1] = min(min_crds[1], coord[1])
        min_crds[2] = min(min_crds[2], coord[2])
        max_crds[0] = max(max_crds[0], coord[0])
        max_crds[1] = max(max_crds[1], coord[1])
        max_crds[2] = max(max_crds[2], coord[2])

    boxlx = max_crds[0]-min_crds[0]
    boxly = max_crds[1]-min_crds[1]
    boxlz = max_crds[2]-min_crds[2]

    psf.setBox(boxlx, boxly, boxlz)
    return psf
