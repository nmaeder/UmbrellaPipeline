from typing import List
from openmm import vec3
import openmm.unit as unit


class Node:
    def __init__(
        self,
        x: float = -1,
        y: float = -1,
        z: float = -1,
        unit: unit.Unit = unit.nanometer,
        f: float = 0.0,
        g: float = 0.0,
        h: float = 0.0,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.unit = unit
        self.f = f
        self.g = g
        self.h = h
        self.parent: Node = None

    @classmethod
    def fromCoords(cls, coords: List[float], unit: unit.Unit):
        """
        constructor creates node form list of node coordinates

        Args:
            coords (List[int]): list of node coordinates
        """

        x, y, z = coords[0], coords[1], coords[2]
        return cls(x, y, z)

    def __str__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def __repr__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def __eq__(self, o: object) -> bool:
        return self.getCoordinates() == o.getCoordinates().in_units_of(self.unit)

    def getCoordinates(self) -> unit.Quantity:
        """
        returns Quantity of the Cordinates in the unit the node is

        Returns:
            unit.Quantity: Quantity of the Cordinates in the unit of the node
        """
        return unit.Quantity(vec3(self.x, self.y, self.z), unit=self.unit)

    def getCoordinatesInUnitsOf(self, unit: unit.Unit) -> unit.Quantity:
        """
        returns Quantity of the Cordinates in the unit provided

        Returns:
            unit.Quantity: Quantity of the Cordinates in the unit provided
        """
        return unit.Quantity(vec3(self.x, self.y, self.z), unit=self.unit).in_units_of(
            unit
        )

    def getCoordinateValuesInUnit(self, unit: unit.Unit) -> List[float]:
        """
        returns list of the coordinate values in the provided unit

        Returns:
            unit.Quantity: coordinate values in the provided unit
        """
        return [
            self.x.value_in_unit(unit),
            self.y.value_in_unit(unit),
            self.z.value_in_unit(unit),
        ]
