import os
from typing import List
from openmm import unit, app
import logging

logger = logging.getLogger(__name__)


def parse_params(
    toppar_directory: str, toppar_str_file: str
) -> app.charmmparameterset.CharmmParameterSet:
    """
    Use to read in top and par files generated by the CHARMM-GUI. IMPORTANT: allways give the path to the openmm (created by charmm gui) folder within the charmm-gui folder.

    Args:
        param_directory (str): folder created by charmm_gui

    Returns:
        app.charmmparameterset.CharmmParameterSet: [description]
    """
    toppar_str_file = os.path.abspath(toppar_str_file)
    current_dir = os.getcwd()
    os.chdir(toppar_directory)
    parFiles = ()
    for line in open(toppar_str_file, "r"):
        if "!" in line:
            line = line.split("!")[0]
        parfile = line.strip()
        if len(parfile) != 0:
            parFiles += (parfile,)
    ret = app.CharmmParameterSet(*parFiles)
    os.chdir(current_dir)
    logger.info("Charmm Parameters read in.")
    return ret


def gen_pbc_box(
    psf: str or app.CharmmPsfFile,
    pos: unit.Quantity,
) -> List[unit.Quantity]:
    """
    Generates pbc box and adds it to the psf. returns offset of the box from 0,0,0.

    Args:
        psf (strorapp.CharmmPsfFile): psf file or path to psf file
        crd (strorapp.CharmmCrdFile): crd file or path to crd file

    Returns:
        List[unit.Quantity]: offset of box from 0,0,0
    """

    if isinstance(psf, str):
        psf = app.CharmmPsfFile(psf)
    coords = pos

    min_crds = [coords[0][0], coords[0][1], coords[0][2]]
    max_crds = [coords[0][0], coords[0][1], coords[0][2]]

    for coord in coords:
        min_crds[0] = min(min_crds[0], coord[0])
        min_crds[1] = min(min_crds[1], coord[1])
        min_crds[2] = min(min_crds[2], coord[2])
        max_crds[0] = max(max_crds[0], coord[0])
        max_crds[1] = max(max_crds[1], coord[1])
        max_crds[2] = max(max_crds[2], coord[2])

    boxlx = max_crds[0] - min_crds[0]
    boxly = max_crds[1] - min_crds[1]
    boxlz = max_crds[2] - min_crds[2]

    psf.setBox(boxlx, boxly, boxlz)
    logger.info(
        f"Periodic boundary conditions added to psf object with size {boxlx}, {boxly}, {boxlz}."
    )
    return min_crds
