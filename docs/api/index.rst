Python API
==========

This package contains 3 submodules and a utils module, that implements functions needed for the 3 submodules. 
It also relies on two datastructures to sort the input and keep everything neat and clear.

UmbrellaPipeline Module
~~~~~~~~~~~~~~~~~~~~~~~

This is the main module provided. It combines the submodules listed below, and makes quick and easy use of this tool possible. Of course, you don't need to 
use this tool like this, but you can use the sub-modules how you want. See :docs:`examples` for more details.

.. toctree::
   :maxdepth: 1

   umbrella_pipeline

Sub-Modules
~~~~~~~~~~~~

The ``path_finding`` sub-module contains the EscapeRoom Classes, which implement the Escape Room Algorithm, that is used for pathway generation.
Two versions are currently supported, the Grid and the Tree version, of which the use of the TreeEscapeRoom is highly encouraged, since it is much faster 
and more memory efficient.

The ``sampling`` submodule contains everything to run the umbrella sampling. It uses the OpenMM python api to run any simulations. The ligand restraints are 
realized using the OpenMM CustomCentroidBondForce. Different input flags can be given to the Sampling classes, to bring the simulation parameters in accordance
with your needs. 

In the ``analysis`` part, the Potential of Mean force is calculated from the trajectories generatd. The MBAR method is employed and the equations are solved using the pymbar solver.

.. toctree::
   :maxdepth: 1

   path_finding
   sampling
   analysis

Utilitary functions
~~~~~~~~~~~~~~~~~~~

These are functions and exceptions that are accessed by multiple of the above sub-modules.

.. toctree::
   :maxdepth: 1

   utils

Input Data Structures
~~~~~~~~~~~~~~~~~~~~~~

To structure the various input data needed by this tool, two handy data structures are supplied, that simplify the handling of said input.

The ``SimulationParameters`` data structure holds all information like temperature, pressure, etc. that is deployed in the umbrella sampling.

The ``SystemInfo`` class contains all the information about the system that is simulated, as the force field data, ligand name, starting conformation etc.

.. toctree::
   :maxdepth: 1

   simulation_parameters
   system_info
