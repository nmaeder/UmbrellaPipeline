import openmm.unit as unit
import re

from UmbrellaPipeline.utils import (
    SimulationProperties,
    SimulationSystem,
)

units = {
    "kelvin": unit.kelvin,
    "bar": unit.bar,
    "femtosecond": unit.femtosecond,
    "/picosecond": 1 / unit.picosecond,
    "kilocalorie/(angstrom**2*mole)": unit.kilocalorie_per_mole / (unit.angstrom ** 2),
    "angstrom": unit.angstrom,
    "nanometer": unit.nanometer,
}

def parse_input_file(file):
    pass

def serialize_for_analysis(
    sim_props: SimulationProperties,
    sim_sys: SimulationSystem,
    traj_dir: str,
    path_interval: unit.Quantity,
    grid_spacing: unit.Quantity,
    filename: str,
):

    order = [
        [
            "temperature",
            f"{sim_props.temperature._value} {sim_props.temperature.unit.get_name()}",
        ],
        [
            "pressure",
            f"{sim_props.pressure._value} {sim_props.pressure.unit.get_name()}",
        ],
        [
            "time_step",
            f"{sim_props.time_step._value} {sim_props.time_step.unit.get_name()}",
        ],
        [
            "force_constant",
            f"{sim_props.force_constant._value} {sim_props.force_constant.unit.get_name()}",
        ],
        [
            "fric_coefficient",
            f"{sim_props.friction_coefficient._value} {sim_props.friction_coefficient.unit.get_name()}",
        ],
        ["n_eq_steps", sim_props.n_equilibration_steps],
        ["n_p_steps", sim_props.n_production_steps],
        ["io_frequency", sim_props.write_out_frequency],
        ["psf_file", sim_sys.psf_file],
        ["pdb_file", sim_sys.pdb_file],
        ["toppar_dir", sim_sys.toppar_dir],
        ["toppar_str", sim_sys.toppar_str],
        ["lig_name", sim_sys.ligand_name],
        ["traj_dir", traj_dir],
        ["path_interval", f"{path_interval._value} {path_interval.unit.get_name()}"],
        ["grid_spacing", f"{grid_spacing._value} {grid_spacing.unig.get_name()}"],
    ]

    with open(file=filename, mode="w") as f:
        for name in order:
            f.write("{0:<20} = {1:<50}\n".format(name[0], name[1]))


def deserialize_for_analysis(filename):
    with open(file=filename, mode="r") as f:
        data = f.read()
    t = re.search(r"temperature\s*=\s(\d+)\s(\w+)", data).group(1, 2)
    p = re.search(r"pressure\s*=\s(\d+)\s(\w+)", data).group(1, 2)
    dt = re.search(r"time_step\s*=\s(\d+)\s(\w+)", data).group(1, 2)
    k = re.search(r"force_constant\s*=\s(\d+)\s(\w+..\w+..\d.\w+.)", data).group(1, 2)
    f = re.search(r"fric_coefficient\s*=\s(\d+)\s(/\w+)", data).group(1, 2)
    ne = re.search(r"n_eq_steps\s*=\s(\d+)", data).group(1)
    np = re.search(r"n_p_steps\s*=\s(\d+)", data).group(1)
    io = re.search(r"io_frequency\s*=\s(\d+)", data).group(1)
    sf = re.search(r"psf_file\s*=\s([\w/.-]+)", data).group(1)
    db = re.search(r"pdb_file\s*=\s([\w/.-]+)", data).group(1)
    td = re.search(r"toppar_dir\s*=\s([\w/.-]+)", data).group(1)
    ts = re.search(r"toppar_str\s*=\s([\w/.-]+)", data).group(1)
    l = re.search(r"lig_name\s*=\s(\w+)", data).group(1)
    tr = re.search(r"traj_dir\s*=\s([\w/.-]+)", data).group(1)
    pi = re.search(r"path_interval\s*=\s(\d+)\s(\w+)", data).group(1, 2)
    gs = re.search(r"grid_spacing\s*=\s(\d+)\s(/\w+)", data).group(1, 2)
    print(sf)
    simprop = SimulationProperties(
        temperature=unit.Quantity(value=float(t[0]), unit=units[t[1]]),
        pressure=unit.Quantity(value=float(p[0]), unit=units[p[1]]),
        time_step=unit.Quantity(value=float(dt[0]), unit=units[dt[1]]),
        force_constant=unit.Quantity(value=float(k[0]), unit=units[k[1]]),
        friction_coefficient=unit.Quantity(value=float(f[0]), unit=units[f[1]]),
        n_equilibration_steps=int(ne),
        n_production_steps=int(np),
        write_out_frequency=int(io),
    )

    simsys = SimulationSystem(
        psf_file=sf,
        pdb_file=db,
        toppar_directory=td,
        toppar_stream_file=ts,
        ligand_name=l,
    )

    return (
        simprop,
        simsys,
        tr,
        unit.Quantity(value=pi[0], unit=units[pi[1]]),
        unit.Quantity(value=gs[0], unit=units[gs[1]]),
    )
