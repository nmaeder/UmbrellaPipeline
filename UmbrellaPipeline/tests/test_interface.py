from UmbrellaPipeline.interface import SamplingInterface


def test_interface_gen():
    interface = SamplingInterface(
        ligand_residue_name="unl",
        toppar_stream_file="UmbrellaPipeline/data/toppar/toppar.str",
        toppar_directory="UmbrellaPipeline/data/toppar",
        psf="UmbrellaPipeline/data/step5_input.psf",
        pdb="UmbrellaPipeline/data/step5_input.pdb",
        number_eq_steps=100,
        number_prod_steps=1000,
        io_frequency=10,
        trajectory_output_path="",
        conda_environment="openmm",
    )


def test_interface_path_gen():
    interface = SamplingInterface(
        ligand_residue_name="unl",
        toppar_stream_file="UmbrellaPipeline/data/toppar/toppar.str",
        toppar_directory="UmbrellaPipeline/data/toppar",
        psf="UmbrellaPipeline/data/step5_input.psf",
        pdb="UmbrellaPipeline/data/step5_input.pdb",
        number_eq_steps=100,
        number_prod_steps=1000,
        io_frequency=10,
        trajectory_output_path=None,
        conda_environment="openmm",
    )
    path = interface.generate_path()
