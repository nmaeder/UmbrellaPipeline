from UmbrellaPipeline import UmbrellaPipeline


def test_interface_gen():
    interface = UmbrellaPipeline(
        ligand_residue_name="unl",
        toppar_stream_file="UmbrellaPipeline/data/toppar/toppar.str",
        toppar_directory="UmbrellaPipeline/data/toppar",
        psf_file="UmbrellaPipeline/data/step5_input.psf",
        pdb_file="UmbrellaPipeline/data/step5_input.pdb",
    )


def test_interface_path_gen():
    interface = UmbrellaPipeline(
        ligand_residue_name="unl",
        toppar_stream_file="UmbrellaPipeline/data/toppar/toppar.str",
        toppar_directory="UmbrellaPipeline/data/toppar",
        psf_file="UmbrellaPipeline/data/step5_input.psf",
        pdb_file="UmbrellaPipeline/data/step5_input.pdb",
    )
    path = interface.generate_path()
