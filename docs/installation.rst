Installation
============

At the moment, only installation is only possible from source code. For this, either manually downoload the files or clone the github repo.
Use

.. code-block:: bash

    $ python setup.py install

Before installation make sure you have install all the requirements:

* python
* pip
* numpy
* gemmi
* openmm
* openmmtools
* scipy
* pymbar
* matplotlib
* plotly
* pytest

To run all the tests (can take up to 5 minutes), go to the root directory of the package and run:

.. code-block:: bash

    $ pytest -v UmbrellaPipeline/tests