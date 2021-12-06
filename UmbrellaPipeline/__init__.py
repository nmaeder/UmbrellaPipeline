"""Umbrella Sampling Pipeline for PMF of Protein Ligand Unbinding"""

# Add imports here
from .test import *
from .umbrella_pipeline import UmbrellaPipeline

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
