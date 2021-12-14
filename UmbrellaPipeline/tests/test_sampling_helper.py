import openmm.app as app
import openmm.unit as unit

from UmbrellaPipeline.sampling import add_harmonic_restraint
from UmbrellaPipeline.path_generation import (
    gen_pbc_box,
    parse_params,
    get_residue_indices,
)


def test_add_harmonic_restraint():
    toppar_stream_file = "UmbrellaPipeline/data/toppar/toppar.str"
    toppar_directory = "UmbrellaPipeline/data/toppar"
    psf = "UmbrellaPipeline/data/step5_input.psf"
    pdb = "UmbrellaPipeline/data/step5_input.pdb"

    pdb = app.PDBFile(pdb)
    psf = app.CharmmPsfFile(psf)

    gen_pbc_box(psf=psf, pdb=pdb)
    params = parse_params(
        toppar_directory=toppar_directory, toppar_str_file=toppar_stream_file
    )
    system = psf.createSystem(params=params)
    ind = get_residue_indices(atom_list=psf.atom_list, name="unl")
    values = [
        10 * unit.kilocalorie_per_mole / (unit.angstrom ** 2),
        1 * unit.angstrom,
        2 * unit.angstrom,
        3 * unit.angstrom,
    ]
    add_harmonic_restraint(system=system, atom_group=ind, values=values)
