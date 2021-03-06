from openmm import app, unit
import os
from typing import Tuple, List

from UmbrellaPipeline.utils import (
    parse_params,
    get_residue_indices,
)


class SystemInfo:
    def __init__(
        self,
        psf_file: str,
        crd_file: str,
        toppar_stream_file: str,
        toppar_directory: str,
        ligand_name: str,
    ) -> None:
        """
        This class stores all paths, crd, psf and parameter objects. It also automatically generates the ligand residue indices, that are used
        throughout the whole package.

        Args:
            psf_file (str): path to psf_file
            crd_file (str): path to crd_file
            toppar_stream_file (str): path to toppar.str generated by charmm_gui
            toppar_directory (str): path to toppar dir generated by charmm_gui
            ligand_name (str): Name of the ligand residue.
        """

        self.params = (toppar_directory, toppar_stream_file)
        self.psf_file = psf_file
        self.crd_file = crd_file
        self.psf_object = psf_file
        self.crd_object = crd_file
        self.params = (toppar_directory, toppar_stream_file)
        self.ligand_name = ligand_name
        self.ligand_indices_from_name = ligand_name

    # Getters
    @property
    def psf_file(self) -> str:
        return self._psf_file

    @property
    def crd_file(self) -> str:
        return self._crd_file

    @property
    def psf_object(self) -> app.CharmmPsfFile:
        return self._psf_object

    @property
    def crd_object(self) -> app.CharmmCrdFile:
        return self._crd_object

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
    @params.setter
    def params(self, value: Tuple[str]) -> None:
        try:
            dir, file = value
            self._params = parse_params(toppar_directory=dir, toppar_str_file=file)
        except:
            raise FileNotFoundError

    @psf_file.setter
    def psf_file(self, value: str) -> None:
        try:
            self._psf_file = os.path.abspath(value)
        except:
            raise FileNotFoundError

    @crd_file.setter
    def crd_file(self, value: str) -> None:
        try:
            self._crd_file = os.path.abspath(value)
        except:
            raise FileNotFoundError

    @psf_object.setter
    def psf_object(self, value: str) -> None:
        try:
            self._psf_object = app.CharmmPsfFile(os.path.abspath(value))
            self._psf_object.createSystem(self.params)
        except:
            raise FileNotFoundError

    @crd_object.setter
    def crd_object(self, value: str) -> None:
        try:
            self._crd_object = app.CharmmCrdFile(os.path.abspath(value))
            self._crd_object.positions = self._crd_object.positions.in_units_of(
                unit.nanometer
            )
            self._crd_object.positions.unit = unit.nanometer
        except:
            raise FileNotFoundError

    @ligand_name.setter
    def ligand_name(self, value: str) -> None:
        if value is None:
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
