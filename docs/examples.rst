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

.. code-block:: python

    # Before starting, prepare your system with CHARMM-GUI and make sure to also create the openmm folder by CHARMM-GUI

    # import the important classes and module for the nobrainer

    from UmbrellaPipeline.utils import SystemInfo, SimulationParameters
    from UmbrellaPipeline.path_finding import TreeEscapeRoom
    from UmbrellaPipeline.sampling import UmbrellaSampling
    from UmbrellaPipeline.analysis import PMFCalculator

    from openmm import unit

    # this object stores the info to your system.
    system_info = SystemInfo(
        psf_file = "charmm-gui/openmm/step5_input.psf",
        crd_file = "charmm-gui/openmm/step5_input.crd",
        ligand_name = "_ligand_name",
        toppar_stream_file = "charmm-gui/openmm/toppar/toppar.str",
        toppar_directory = "charmm-gui/openmm/toppar",
    )

    #set parameters for the sampling
    simulation_parameters = SimulationParameters(
        temperature = 310 * u.kelvin,
        pressure = 1 * u.bar, #only used for equilibration
        time_step = 2 * u.femtoseconds,
        force_constant = 1 * u.kilocalorie_per_mole / (u.angstrom**2),
        friction_coefficient = 1 / u.picosecond,
        n_equilibration_steps = 500000,
        n_production_steps = 2500000,
        write_out_frequency = 5000,
    )

    #create the sampling object for equilibration

    sampling = UmbrellaSampling(
        simulation_parameters = simulation_parameters,
        system_info = system_info,
        traj_write_path = <path for trajectories>,
        restrain_protein_backbone = True, # lets say we want to restrain the backbone in the equilibration
    )

    #run equilibration
    state = sampling.run_equilibration(
        use_membrane_barostat = True # now it uses a monte carlo membrane barostat instead of a monte carlo barostat.
    )

    #generate dissociation pathway

    escape_room = TreeEscapeRoom.from_files(system_info=system_info, positions = state.GetPositions())
    escape_room.find_path(resolution = .1*unit.angstrom, distance = 2*unit.nanometer)

    path = escape_room.get_path_for_sampling(stepsize=0.5*unit.angstrom)

    # run sampling along pathway

    sampling.run_production(
        path = path,
        state = state,
    )

    # run analysis:
    analysis = PMFCalculator(
        simulation_parameters = simulation_parameters,
        system_info = system_info,
        trajectory_directory = sampling.trajectory_directory,
        original_path_interval = 0.5*unit.angstrom,
    )

    #creating a file containing all 

    analysis.calculate_pmf()
    analysis.plot()
