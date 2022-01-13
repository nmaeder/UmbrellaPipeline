import openmm.unit as unit
from openmm import Vec3
import numpy as np

from UmbrellaPipeline.analysis import PMFCalculator
from UmbrellaPipeline.path_generation import Tree, TreeNode, TreeEscapeRoom
from UmbrellaPipeline.utils import SimulationProperties, SimulationSystem

"""
    folgende testsysteme sind gelaufen, alle zu finden auf /data/cluster/projects/enhanced_sampling/niels/master_thesis/
    habe dir gleich simulationSystem und simulationproperty objekte vorbereitet, die du dann unten einfach einsetzen musst. hoffe das macht sinn.

    alle sachen die du brauchen must sind in test marcus, alles da man anfassen muss hat kommentar mit TODO dahinter, der rest läuft von selbst. 

    ghost_ce_105/traj: umbrella simulations mit 2 angstrom abstand der umbrella windows und force constants von je 100 kilocal/mol/a².
    ghost_ce_110 für die ersten 5 systeme aus dem dat folder. trajectorien jeweils im traj unter ordner. restraint punkte im coordinates.dat file
    ghost_ce_115
    ghost_ce_116
    ghost_ce_117 
    110_lower_k: ce_110 system, auch 2 angstrom window abstände, aber nur 10 kcal/mol/a² force constant
    
    110_lower_smaller: ce_110 system mit 10 force constant aber nur 1 angstrom abstand. heisst doppelt soviele windows

    110_smaller_d: wieder 100kcal/mol/a² forceconstant, aber nur 1 angstrom abstand zwischen den windows
"""

sim_properties_ghost = SimulationProperties()

# use this second variant for the pmfcalculator when analysing the 110_lower versions, the above for everything else
sim_properties_lower = SimulationProperties(
    force_constant=10 * unit.kilocalorie_per_mole / (unit.angstrom ** 2)
)

# here below the simulation systems to use, use according.
# sim_sys_105 = SimulationSystem(
#    psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/openmm/step5_input.psf",
#    pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/openmm/step5_input.pdb",
#    toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/toppar",
#    toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/openmm/toppar.str",
#    ligand_name="UNL",
# )

sim_sys_110 = SimulationSystem(
    psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.psf",
    pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.pdb",
    toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/toppar",
    toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/toppar.str",
    ligand_name="UNL",
)

"""
sim_sys_115 = SimulationSystem(
    psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/03-CE-115/charmm-gui/openmm/step5_input.psf",
    pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/03-CE-115/charmm-gui/openmm/step5_input.pdb",
    toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/03-CE-115/charmm-gui/toppar",
    toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/03-CE-115/charmm-gui/openmm/toppar.str",
    ligand_name="UNL",
)

sim_sys_116 = SimulationSystem(
    psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/04-CE-116/charmm-gui/openmm/step5_input.psf",
    pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/04-CE-116/charmm-gui/openmm/step5_input.pdb",
    toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/04-CE-116/charmm-gui/toppar",
    toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/04-CE-116/charmm-gui/openmm/toppar.str",
    ligand_name="UNL",
)

sim_sys_117 = SimulationSystem(
    psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/05-CE-117/charmm-gui/openmm/step5_input.psf",
    pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/05-CE-117/charmm-gui/openmm/step5_input.pdb",
    toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/05-CE-117/charmm-gui/toppar",
    toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/05-CE-117/charmm-gui/openmm/toppar.str",
    ligand_name="UNL",
)
"""

normal_interval = 2 * unit.angstrom
short_interval = 1 * unit.angstrom


def test_marcus():

    trajectory_directory = "/data/shared/projects/enhanced_sampling/110_lower_k/traj"  # TODO: path to traj folder.
    number_of_bins = 25  # TODO:number of bins

    # calculates positions of binpoints that do not equal restraint points.

    pmf = PMFCalculator(
        simulation_properties=sim_properties_lower,  # TODO: give correct simulation property object. only use the sim_properties_lower for the trajectories with lower force constant!!
        simulation_system=sim_sys_110,  # TODO: use the sim_sys_# depending on which system you are looking at. match the numbers
        original_path_stepsize=normal_interval,  # TODO: change to short_interval if you analyze trajectories with more windows, otherwise use normal_interval
        original_path=None,
        trajectory_directory=trajectory_directory,
        n_bins=number_of_bins,
    )
    pmf.system_info.psf_object.createSystem(pmf.system_info.params)

    pmf.parse_trajectories()

    # i've implemented both versions now, one that bins according to nearest neighbour, and one that makes the spherical bins around the binpoints.
    pmf.calculate_pmf(nearest_neighbour_method=True)
    # pmf.create_pdf("pdffilepath")
    pmf.create_pdf("~/test.pdf")
    # for testing purposes ive made the matrices A and B class attributes of PMFCalculator so you can easily access them via pmf.A in this case. only usefull after deploying calculate_pmf of course :)
