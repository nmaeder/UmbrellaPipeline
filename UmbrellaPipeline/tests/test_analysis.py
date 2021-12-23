import openmm.unit as unit
from openmm import Vec3
import numpy as np

from UmbrellaPipeline.analysis import PMFCalculator
from UmbrellaPipeline.path_generation import Tree, TreeNode, TreeEscapeRoom
from UmbrellaPipeline.utils import SimulationProperties, SimulationSystem

"""
    folgende testsysteme sind gelaufen, alle zu finden auf /data/cluster/projects/enhanced_sampling/niels/master_thesis/

    ghost_ce_105: umbrella simulations mit 2 angstrom abstand der umbrella windows und force constants von je 100 kilocal/mol/a².
    ghost_ce_110 für die ersten 5 systeme aus dem dat folder. trajectorien jeweils im traj unter ordner. restraint punkte im coordinates.dat file
    ghost_ce_115
    ghost_ce_116
    ghost_ce_117

    110_lower_k: ce_110 system, auch 2 angstrom window abstände, aber nur 10 kcal/mol/a² force constant
    
    110_lower_smaller: ce_110 system mit 10 force constant aber nur 1 angstrom abstand. heisst doppelt soviele windows

    110_smaller_d: wieder 100kcal/mol/a² forceconstant, aber nur 1 angstrom abstand zwischen den windows
"""


def create_extra_bin_points(path, stepsize):
    """
    helper function, see usage below.
    """
    tree = Tree(path)
    er = TreeEscapeRoom(tree=Tree, start=TreeNode)
    newp = []
    for i in path:
        newp.append(
            TreeNode(
                x=i.x,
                y=i.y,
                z=i.z,
                unit=i.unit,
            )
        )
    er.shortest_path = newp
    return er.get_path_for_sampling(stepsize=stepsize)


def test_marcus():

    sim_properties_ghost = SimulationProperties()

    # use this second variant for the pmfcalculator when analysing the 110_lower versions, the above for everything else
    sim_properties_lower = SimulationProperties(
        force_constant=10 * unit.kilocalorie_per_mole / (unit.angstrom ** 2)
    )

    sim_shorty = SimulationProperties(
        n_equilibration_steps=2000, n_production_steps=2000, write_out_frequency=100
    )

    # here below the simulation systems to use, use according.
    sim_sys_105 = SimulationSystem(
        psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/openmm/step5_input.psf",
        pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/openmm/step5_input.pdb",
        toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/toppar",
        toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/01-CE-105/charmm-gui/openmm/toppar.str",
        ligand_name="UNL",
    )

    sim_sys_110 = SimulationSystem(
        psf_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.psf",
        pdb_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/step5_input.pdb",
        toppar_directory="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/toppar",
        toppar_stream_file="/data/shared/projects/DAT_enhanced_sampling/00DATA/02-CE-110/charmm-gui/openmm/toppar.str",
        ligand_name="UNL",
    )

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

    normal_interval = 2 * unit.angstrom
    short_interval = 1 * unit.angstrom

    coordinate_file = "/data/shared/projects/enhanced_sampling/test/coordinates.dat"

    dat = np.loadtxt(fname=coordinate_file, delimiter=",", skiprows=1)
    path = []
    for i in range(len(dat)):
        path.append(
            unit.Quantity(
                Vec3(
                    x=dat[i][1],
                    y=dat[i][2],
                    z=dat[i][3],
                ),
                unit=unit.nanometer,
            )
        )

    # use the accordint property and system object generated above for the right trajectory

    trajectory_directory = "/data/shared/projects/enhanced_sampling/test"

    bin_path = path
    number_of_bins = 24

    # calculates positions of binpoints that do not equal restraint points.

    pmf = PMFCalculator(
        simulation_properties=sim_shorty,
        simulation_system=sim_sys_105,
        path=path,
        path_interval=normal_interval,
        trajectory_directory=trajectory_directory,
        n_bins=number_of_bins,
    )
    pmf.system_info.psf_object.createSystem(pmf.system_info.params)
    if not number_of_bins == pmf.n_windows:
        new_size = pmf.n_windows * normal_interval / number_of_bins
        bin_path = create_extra_bin_points(path=pmf.path, stepsize=new_size)

    pmf.parse_trajectories()

    # i've implemented both versions now, one that bins according to nearest neighbour, and one that makes the spherical bins around the binpoints. use true if you want the nearest neighbour binning.
    # this is not tested yet, but i'm pretty sure it does what it is supposed to. falls es nicht geht auf false setzen, das geht auf jeden fall!!
    pmf.calculate_pmf(bin_path=bin_path, nearest_neighbour_method=False)
    # pmf.create_pdf("pdffilepath")

    # for testing purposes ive made the matrices A and B class attributes of PMFCalculator so you can easily access them via pmf.A in this case. only usefull after deploying calculate_pmf of course :)
