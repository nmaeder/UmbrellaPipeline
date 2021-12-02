import openmm.unit as unit
import os
from openmm import Vec3
from UmbrellaPipeline.path_generation import (
    gen_pbc_box,
    get_residue_indices,
    get_center_of_mass_coordinates,
    get_centroid_coordinates,
    parse_params,
)
import openmm.app as app

pdb = "UmbrellaPipeline/data/step5_input.pdb"
psf = "UmbrellaPipeline/data/step5_input.psf"


def read_pdb(pdb: str = pdb) -> app.PDBFile:
    return app.PDBFile(pdb)


def read_psf(psf: str = psf) -> app.CharmmPsfFile:
    return app.CharmmPsfFile(psf)


def test_ligand_indices():
    psf = read_psf()
    indices = get_residue_indices(psf.atom_list, name="unl")
    goal = list(range(8478, 8514, 1))
    assert indices == goal

    indices = get_residue_indices(psf.atom_list, name="unl", include_hydrogens=False)
    goal = list(range(8478, 8499, 1))
    assert indices == goal


def test_protein_indices():
    psf = read_psf()
    indices = get_residue_indices(psf.atom_list)
    goal = list(range(0, 8478, 1))
    assert indices == goal


def test_genbox():
    psf = read_psf()
    pdb = read_pdb()
    assert psf.boxVectors == None
    minC = gen_pbc_box(psf=psf, pdb=pdb)
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


def test_param_parser():
    params = parse_params(
        toppar_directory="UmbrellaPipeline/data/toppar",
        toppar_str_file="UmbrellaPipeline/data/toppar/toppar.str",
    )


def test_centroid_coords():
    psf = read_psf()
    pdb = read_pdb()
    ind1 = get_residue_indices(atom_list=psf.atom_list, name="unl")
    ind2 = get_residue_indices(
        atom_list=psf.atom_list, name="unl", include_hydrogens=False
    )
    assert get_centroid_coordinates(pdb.positions, ind1) == unit.Quantity(
        value=Vec3(x=4.800866666666666, y=5.162369444444445, z=5.116966666666667),
        unit=unit.nanometer,
    )
    assert get_centroid_coordinates(pdb.positions, ind2) == unit.Quantity(
        value=Vec3(x=4.791909523809522, y=5.152095238095239, z=5.13817619047619),
        unit=unit.nanometer,
    )


def test_com_coords():
    pdb = read_pdb()
    psf = read_psf()
    gen_pbc_box(pdb=pdb, psf=psf)
    params = parse_params(
        toppar_directory="UmbrellaPipeline/data/toppar",
        toppar_str_file="UmbrellaPipeline/data/toppar/toppar.str",
    )
    system = psf.createSystem(
        params=params,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
    )
    ind = get_residue_indices(atom_list=psf.atom_list, name="unl")
    assert get_center_of_mass_coordinates(
        positions=pdb.positions, indices=ind, masses=system, include_hydrogens=True
    ) == unit.Quantity(
        value=Vec3(x=4.7843512147078195, y=2.570779540502063, z=1.7248368464666914),
        unit=unit.nanometer,
    )
    assert get_center_of_mass_coordinates(
        positions=pdb.positions, indices=ind, masses=system, include_hydrogens=False
    ) == unit.Quantity(
        value=Vec3(x=4.782878540555002, y=2.569887631028307, z=1.7263107176323071),
        unit=unit.nanometer,
    )
