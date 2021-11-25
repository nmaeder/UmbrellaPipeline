from UmbrellaPipeline.pathGeneration import(
    gen_box,
    get_indices,
    getCenterOfMassCoordinates,
    getCentroidCoordinates,
)
import openmm.app as app

pdb = "UmbrellaPipeline/data/step5_input.pdb"
psf = "UmbrellaPipeline/data/step5_input.psf"

pdb = app.PDBFile(pdb)
psf = app.CharmmPsfFile(psf)

def testLigandIndices():
    indices = get_indices(psf.atom_list, name="unl")
    goal = list(range(8478,8514,1))
    assert indices == goal

def testProteinIndices():
    indices = get_indices(psf.atom_list)
    goal = list(range(0,8478,1))
    assert indices == goal

def testgenBox():
    "TODO"
    pass

def testCentroidCoords():
    "TODO"
    pass

def testCOMCoords():
    "TODO"
    pass
