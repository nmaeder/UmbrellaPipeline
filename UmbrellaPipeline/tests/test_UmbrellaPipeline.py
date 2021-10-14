"""
Unit and regression test for the UmbrellaPipeline package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import UmbrellaPipeline


def test_UmbrellaPipeline_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "UmbrellaPipeline" in sys.modules
