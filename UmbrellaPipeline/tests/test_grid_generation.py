from UmbrellaPipeline.pathGeneration import Grid
from simtk import unit

pdb = "UmbrellaPipeline/data/step5_input.pdb"
psf = "UmbrellaPipeline/data/step5_input.psf"


def test_initialize_grid():
    grid = Grid.gridFromFiles(pdb=pdb, psf=psf, vdwradius=1.2 * unit.angstrom)

