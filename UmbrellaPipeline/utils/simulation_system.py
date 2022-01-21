from openmm import app
import os
from typing import Tuple, List

from UmbrellaPipeline.utils import (
    parse_params,
    get_residue_indices,
)


class SimulationSystem:
    """
    This class stores all paths, pdb, psf and parameter objects.
    """

    def __init__(
        self,
        psf_file: str,
        pdb_file: str,
        toppar_stream_file: str,
        toppar_directory: str,
        ligand_name: str,
    ) -> None:
        """

        Args:
            psf_file (str): path to psf_file
            pdb_file (str): path to pdb_file
            toppar_stream_file (str): path to toppar.str generated by charmm_gui
            toppar_directory (str): path to toppar dir generated by charmm_gui
            trajectory_path (str): outputpath for the trajectories to be written to.
        """
        self.psf_file = psf_file
        self.pdb_file = pdb_file
        self.psf_object = psf_file
        self.pdb_object = pdb_file
        self.params = (toppar_directory, toppar_stream_file)
        self.ligand_name = ligand_name
        self.ligand_indices_from_name = ligand_name

    # Getters
    @property
    def psf_file(self) -> str:
        return self._psf_file

    @property
    def pdb_file(self) -> str:
        return self._pdb_file

    @property
    def psf_object(self) -> app.CharmmPsfFile:
        return self._psf_object

    @property
    def pdb_object(self) -> app.PDBFile:
        return self._pdb_object

    @property
    def params(self) -> app.CharmmParameterSet:
        return self._params

    @property
    def ligand_name(self) -> str:
        return self._ligand_name

    @property
    def ligand_indices(self) -> List[int]:
        return self._ligand_indices

    # Setters
    @psf_file.setter
    def psf_file(self, value: str) -> None:
        try:
            self._psf_file = os.path.abspath(value)
        except:
            raise FileNotFoundError

    @pdb_file.setter
    def pdb_file(self, value: str) -> None:
        try:
            self._pdb_file = os.path.abspath(value)
        except:
            raise FileNotFoundError

    @psf_object.setter
    def psf_object(self, value: str) -> None:
        try:
            self._psf_object = app.CharmmPsfFile(os.path.abspath(value))
        except:
            raise FileNotFoundError

    @pdb_object.setter
    def pdb_object(self, value: str) -> None:
        try:
            self._pdb_object = app.PDBFile(os.path.abspath(value))
        except:
            raise FileNotFoundError

    @params.setter
    def params(self, value: Tuple[str, str]) -> None:
        try:
            dir, file = value
            self._params = parse_params(toppar_directory=dir, toppar_str_file=file)
        except:
            raise FileNotFoundError

    @ligand_name.setter
    def ligand_name(self, value: str) -> None:
        if value == None:
            raise ValueError("LigandName cannot be None")
        else:
            self._ligand_name = value

    @ligand_indices.setter
    def ligand_indices(self, value: List[int] or int):
        if isinstance(value, int):
            self._ligand_indices = [value]
        else:
            self._ligand_indices = value

    @ligand_indices.setter
    def ligand_indices_from_name(self, value: str) -> None:
        ind = get_residue_indices(atom_list=self.psf_object.atom_list, name=value)
        if ind == []:
            raise LookupError("No residue with given name Found")
        else:
            self._ligand_indices = ind
