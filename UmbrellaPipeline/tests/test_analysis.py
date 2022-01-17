import pytest, os
import openmm.unit as unit

from UmbrellaPipeline.analysis import PMFCalculator
from UmbrellaPipeline.utils import SimulationProperties, SimulationSystem


"""
All normal cases below. go further down for the altered simulation setups.

insert filenames in plot_pytest() to save the figures you create.

fürs nachbauen, die calculate_pmf() funktion ist was man mit pymbar nachbauen könnte :) jeder step ist mit kommentaren versehen :) 
"""


@pytest.mark.skipif(os.getenv("CI") == "true", reason="not real tests")
def test_ghost_105():

    trajectory_directory = (
        "/data/shared/projects/enhanced_sampling/trajectories/ghost_ce_105/traj"
    )
    number_of_bins = None  # TODO: if NONE, number of bins = number of umbrella windows

    sim_sys = SimulationSystem(
        psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/openmm/step5_input.psf",
        pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/openmm/step5_input.pdb",
        toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/toppar",
        toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/openmm/toppar.str",
        ligand_name="UNL",
    )

    pmf = PMFCalculator(
        simulation_properties=SimulationProperties(),
        simulation_system=sim_sys,
        original_path_interval=2 * unit.angstrom,
        trajectory_directory=trajectory_directory,
        n_bins=number_of_bins,
    )

    pmf.load_original_path()
    pmf.load_sampled_coordinates()
    pmf.calculate_pmf()
    pmf.plot_pytest()


@pytest.mark.skipif(os.getenv("CI") == "true", reason="not real tests")
def test_ghost_110():
    trajectory_directory = (
        "/data/shared/projects/enhanced_sampling/trajectories/ghost_ce_110/traj"
    )
    number_of_bins = None  # TODO: if NONE, number of bins = number of umbrella windows

    sim_sys = SimulationSystem(
        psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.psf",
        pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.pdb",
        toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/toppar",
        toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/toppar.str",
        ligand_name="UNL",
    )

    pmf = PMFCalculator(
        simulation_properties=SimulationProperties(),
        simulation_system=sim_sys,
        original_path_interval=2 * unit.angstrom,
        trajectory_directory=trajectory_directory,
        n_bins=number_of_bins,
    )

    pmf.load_original_path()
    pmf.load_sampled_coordinates()
    pmf.calculate_pmf()
    pmf.plot_pytest()


@pytest.mark.skipif(os.getenv("CI") == "true", reason="not real tests")
def test_ghost_115():
    trajectory_directory = (
        "/data/shared/projects/enhanced_sampling/trajectories/ghost_ce_115/traj"
    )
    number_of_bins = None  # TODO: if NONE, number of bins = number of umbrella windows

    sim_sys = SimulationSystem(
        psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/03-CE-115/charmm-gui/openmm/step5_input.psf",
        pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/03-CE-115/charmm-gui/openmm/step5_input.pdb",
        toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/03-CE-115/charmm-gui/toppar",
        toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/03-CE-115/charmm-gui/openmm/toppar.str",
        ligand_name="UNL",
    )

    pmf = PMFCalculator(
        simulation_properties=SimulationProperties(),
        simulation_system=sim_sys,
        original_path_interval=2 * unit.angstrom,
        trajectory_directory=trajectory_directory,
        n_bins=number_of_bins,
    )

    pmf.load_original_path()
    pmf.load_sampled_coordinates()
    pmf.calculate_pmf()
    pmf.plot_pytest()


@pytest.mark.skipif(os.getenv("CI") == "true", reason="not real tests")
def test_ghost_116():
    trajectory_directory = (
        "/data/shared/projects/enhanced_sampling/trajectories/ghost_ce_116/traj"
    )
    number_of_bins = None  # TODO: if NONE, number of bins = number of umbrella windows

    sim_sys = SimulationSystem(
        psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/04-CE-116/charmm-gui/openmm/step5_input.psf",
        pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/04-CE-116/charmm-gui/openmm/step5_input.pdb",
        toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/04-CE-116/charmm-gui/toppar",
        toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/04-CE-116/charmm-gui/openmm/toppar.str",
        ligand_name="UNL",
    )

    pmf = PMFCalculator(
        simulation_properties=SimulationProperties(),
        simulation_system=sim_sys,
        original_path_interval=2 * unit.angstrom,
        trajectory_directory=trajectory_directory,
        n_bins=number_of_bins,
    )

    pmf.load_original_path()
    pmf.load_sampled_coordinates()
    pmf.calculate_pmf()
    pmf.plot_pytest()


@pytest.mark.skipif(os.getenv("CI") == "true", reason="not real tests")
def test_ghost_117():
    trajectory_directory = (
        "/data/shared/projects/enhanced_sampling/trajectories/ghost_ce_117/traj"
    )
    number_of_bins = None  # TODO: if NONE, number of bins = number of umbrella windows

    sim_sys = SimulationSystem(
        psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/05-CE-117/charmm-gui/openmm/step5_input.psf",
        pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/05-CE-117/charmm-gui/openmm/step5_input.pdb",
        toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/05-CE-117/charmm-gui/toppar",
        toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/05-CE-117/charmm-gui/openmm/toppar.str",
        ligand_name="UNL",
    )

    pmf = PMFCalculator(
        simulation_properties=SimulationProperties(),
        simulation_system=sim_sys,
        original_path_interval=2 * unit.angstrom,
        trajectory_directory=trajectory_directory,
        n_bins=number_of_bins,
    )

    pmf.load_original_path()
    pmf.load_sampled_coordinates()
    pmf.calculate_pmf()
    pmf.plot_pytest()


"""
Special cases. all for the 02-CE-110 system.
"""


@pytest.mark.skipif(os.getenv("CI") == "true", reason="not real tests")
def test_lower_k():
    trajectory_directory = (
        "/data/shared/projects/enhanced_sampling/trajectories/110_lower_k/traj"
    )
    number_of_bins = None  # TODO: if NONE, number of bins = number of umbrella windows

    sim_sys = SimulationSystem(
        psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.psf",
        pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.pdb",
        toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/toppar",
        toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/toppar.str",
        ligand_name="UNL",
    )

    pmf = PMFCalculator(
        simulation_properties=SimulationProperties(
            force_constant=10 * unit.kilocalorie_per_mole / (unit.angstrom ** 2)
        ),
        simulation_system=sim_sys,
        original_path_interval=2 * unit.angstrom,
        trajectory_directory=trajectory_directory,
        n_bins=number_of_bins,
    )

    pmf.load_original_path()
    pmf.load_sampled_coordinates()
    pmf.calculate_pmf()
    pmf.plot_pytest()


@pytest.mark.skipif(os.getenv("CI") == "true", reason="not real tests")
def test_ghost_110():
    trajectory_directory = (
        "/data/shared/projects/enhanced_sampling/trajectories/110_lower_smaller/traj"
    )
    number_of_bins = None  # TODO: if NONE, number of bins = number of umbrella windows

    sim_sys = SimulationSystem(
        psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.psf",
        pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.pdb",
        toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/toppar",
        toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/toppar.str",
        ligand_name="UNL",
    )

    pmf = PMFCalculator(
        simulation_properties=SimulationProperties(
            force_constant=10 * unit.kilocalorie_per_mole / (unit.angstrom ** 2)
        ),
        simulation_system=sim_sys,
        original_path_interval=1 * unit.angstrom,
        trajectory_directory=trajectory_directory,
        n_bins=number_of_bins,
    )

    pmf.load_original_path()
    pmf.load_sampled_coordinates()
    pmf.calculate_pmf()
    pmf.plot_pytest()


@pytest.mark.skipif(os.getenv("CI") == "true", reason="not real tests")
def test_ghost_110():
    trajectory_directory = (
        "/data/shared/projects/enhanced_sampling/trajectories/110_smaller_d/traj"
    )
    number_of_bins = None  # TODO: if NONE, number of bins = number of umbrella windows

    sim_sys = SimulationSystem(
        psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.psf",
        pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.pdb",
        toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/toppar",
        toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/toppar.str",
        ligand_name="UNL",
    )

    pmf = PMFCalculator(
        simulation_properties=SimulationProperties(),
        simulation_system=sim_sys,
        original_path_interval=1 * unit.angstrom,
        trajectory_directory=trajectory_directory,
        n_bins=number_of_bins,
    )

    pmf.load_original_path()
    pmf.load_sampled_coordinates()
    pmf.calculate_pmf()
    pmf.plot_pytest()
