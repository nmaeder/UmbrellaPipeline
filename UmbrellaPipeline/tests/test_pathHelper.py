import openmm.unit as unit
import os
from openmm import Vec3
from UmbrellaPipeline.pathGeneration import (
    genBox,
    getIndices,
    getCenterOfMassCoordinates,
    getCentroidCoordinates,
    getParams,
)
import openmm.app as app

pdb = "UmbrellaPipeline/data/step5_input.pdb"
psf = "UmbrellaPipeline/data/step5_input.psf"

def readPDB(pdb:str = pdb) -> app.PDBFile:
    return app.PDBFile(pdb)

def readPSF(psf:str = psf) -> app.CharmmPsfFile:
    return app.CharmmPsfFile(psf)




def testLigandIndices():
    psf = readPSF()
    indices = getIndices(psf.atom_list, name="unl")
    goal = list(range(8478, 8514, 1))
    assert indices == goal

    indices = getIndices(psf.atom_list, name="unl", includeHydrogens=False)
    goal = list(range(8478, 8499, 1))
    assert indices == goal


def testProteinIndices():
    psf = readPSF()
    indices = getIndices(psf.atom_list)
    goal = list(range(0, 8478, 1))
    assert indices == goal


def testgenBox():
    psf = readPSF()
    pdb = readPDB()
    assert psf.boxVectors == None
    minC = genBox(psf=psf, pdb=pdb)
    assert minC == [
        unit.Quantity(value=-0.5613, unit=unit.nanometer),
        unit.Quantity(value=-0.46090000000000003, unit=unit.nanometer),
        unit.Quantity(value=-0.0634, unit=unit.nanometer),
    ]
    assert psf.boxVectors == unit.Quantity(
        value=(
            Vec3(x=11.071, y=0.0, z=0.0),
            Vec3(x=0.0, y=10.882600000000002, z=0.0),
            Vec3(x=0.0, y=0.0, z=10.2086),
        ),
        unit=unit.nanometer,
    )

def testGetParams():
    params = getParams(
        topparDirectory="UmbrellaPipeline/data/toppar", topparStrFile="toppar.str"
    )


def testCentroidCoords():
    psf = readPSF()
    pdb = readPDB()
    ind1 = getIndices(atom_list=psf.atom_list, name="unl")
    ind2 = getIndices(atom_list=psf.atom_list, name="unl", includeHydrogens=False)
    assert getCentroidCoordinates(pdb.positions, ind1) == unit.Quantity(
        value=Vec3(x=4.800866666666666, y=5.162369444444445, z=5.116966666666667),
        unit=unit.nanometer,
    )
    assert getCentroidCoordinates(pdb.positions, ind2) == unit.Quantity(
        value=Vec3(x=4.791909523809522, y=5.152095238095239, z=5.13817619047619),
        unit=unit.nanometer,
    )


def testCOMCoords():
    pdb = readPDB()
    psf = readPSF()
    genBox(pdb=pdb, psf=psf)
    params = getParams(
        topparDirectory="UmbrellaPipeline/data/toppar", topparStrFile="toppar.str"
    )
    system = psf.createSystem(
        params=params,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
    )
    ind = getIndices(atom_list=psf.atom_list, name="unl")
    assert getCenterOfMassCoordinates(
        positions=pdb.positions, indices=ind, masses=system, includeHydrogens=True
    ) == unit.Quantity(
        value=Vec3(x=4.7843512147078195, y=2.570779540502063, z=1.7248368464666914),
        unit=unit.nanometer,
    )
    assert getCenterOfMassCoordinates(
        positions=pdb.positions, indices=ind, masses=system, includeHydrogens=False
    ) == unit.Quantity(
        value=Vec3(x=4.782878540555002, y=2.569887631028307, z=1.7263107176323071),
        unit=unit.nanometer,
    )
