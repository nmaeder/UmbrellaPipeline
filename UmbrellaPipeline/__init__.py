"""Umbrella Sampling Pipeline for PMF of Protein Ligand Unbinding"""

# Add imports here
from .umbrella_pipeline import UmbrellaPipeline

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

import logging

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)1s()] %(message)s"
logging.basicConfig(format=FORMAT, datefmt="%d-%m-%-Y:%H:%M", level=logging.INFO)
