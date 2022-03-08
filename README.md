UmbrellaPipeline
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/nmaeder/UmbrellaPipeline/workflows/CI/badge.svg)](https://github.com/nmaeder/UmbrellaPipeline/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/nmaeder/UmbrellaPipeline/branch/master/graph/badge.svg)](https://codecov.io/gh/nmaeder/UmbrellaPipeline/branch/master)


Umbrella Sampling Pipeline for PMF of Protein Ligand Unbinding

## Theory

This package estimates the sterically least hindered protein ligand dissociation pathway, conducts umbrella sampling along this pathway and subsequently calculates the Potential of Mean Force (PMF) for the protein ligand unbinding. No alignement of the protein cavity or manually placing ligands necessary.

# Path Finding
Using a modified version of the A* path finding algorithm, this tool takes a docked ligand as a starting point, and then finds (if existing) the sterically least hindreds way out of the protein cavity by following a simple best-first search algorihtm. 

![Alt text](docs/_static/escape_room_no_bottleneck.eps?raw=true "Title")

# Sampling

All sampling is done using openmm. After a equilibration period, the ligand is then placed at evenly spaced positions along the dissociation pathway and umbrella sampling is conducted.

# PMF Calculations

The trajectories are analysed and the PMF is calculated using either pymbar or FastMBAR.

## How-To

Visit the documentation.

## Requirements

- Python 3.7, 3.8, 3.9
- numpy
- openmm
- openmmtools
- scipy
- pymbar
- matplotlib
- plotly
- FastMBAR
- gemmi

## Installation
Either download the source code or clone the repository to the desired destination and run 
```
pip install .
```
inside the downloaded folder. To install the dependencies use
```
pip install -r docs/requirements.yaml
```

### Maintainer

- Niels Maeder <maedern@student.ethz.ch> (ETH Zürich)

### Copyright

Copyright (c) 2022, Niels Maeder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.