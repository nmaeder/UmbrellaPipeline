Examples
========

The code presented in this tool is easy to use, yet customizable if needed. The following two examples show a no brainer, nun customized working example 
whereas the second example shows more in depth, how this tool can be used.

Here we go with the no-brainer:

.. code-block:: python

    # Before starting, prepare your system with CHARMM-GUI and make sure to also create the openmm folder by CHARMM-GUI

    # import the important classes and module for the nobrainer

    from UmbrellaPipeline import UmbrellaPipeline
    from UmbrellaPipeline.utils import SystemInfo, SimulationParameters

    # this object stores the info to your system.
    a = SystemInfo(
        psf_file = "charmm-gui/openmm/step5_input.psf",
        crd_file = "charmm-gui/openmm/step5_input.crd",
        ligand_name = "_ligand_name",
        toppar_stream_file = "charmm-gui/openmm/toppar/toppar.str",
        toppar_directory = "charmm-gui/openmm/toppar",
    )

    # this object stores the info to your system.
    system_info = SystemInfo(
        psf_file = "charmm-gui/openmm/step5_input.psf",
        crd_file = "charmm-gui/openmm/step5_input.crd",
        ligand_name = "_ligand_name",
        toppar_stream_file = "charmm-gui/openmm/toppar/toppar.str",
        toppar_directory = "charmm-gui/openmm/toppar",
    )

    #create the UmbrellaPipeline object.
    pipeline = UmbrellaPipeline(system_info=system_info)
    
    #run equilibration, pathfinding, production and analysis all in one go :)
    pipeline.run_simulations_local()

And here is a more in depth how_to.