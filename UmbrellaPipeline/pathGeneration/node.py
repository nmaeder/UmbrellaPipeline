from typing import List
from openmm import vec3
import openmm.unit as unit


class Node:
    def __init__(self, x=-1, y=-1, z=-1, g: float = 0, h: float = 0, f: float = 0):
        self.x = x
        self.y = y
        self.z = z
        self.g = g
        self.h = h
        self.f = f
        self.parent: Node = None

    def __str__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def __repr__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"


class TreeNode(Node):
    def __init__(
        self,
        x: float = -1,
        y: float = -1,
        z: float = -1,
        g: float = 0,
        h: float = 0,
        f: float = 0,
        unit: unit.Unit = unit.nanometer,
    ):
        super().__init__(x=x, y=y, z=z, g=g, h=h, f=f)
        self.unit = unit

    @classmethod
    def fromCoords(
        cls,
        coords: List[unit.Quantity] or unit.Quantity or List[float],
        unit: unit.Unit = None,
    ):
        try:
            x, y, z = coords[0], coords[1], coords[2]
            unit = coords.unit
            return cls(x=x, y=y, z=z, unit=unit)
        except TypeError:
            x, y, z = coords[0], coords[1], coords[2]
            unit = unit
            return cls(x=x, y=y, z=z, unit=unit)

    def __eq__(self, o: object) -> bool:
        return self.getCoordinates() == o.getCoordinates()

    def getCoordinates(self) -> unit.Quantity:
        return unit.Quantity(value=vec3(self.x, self.y, self.z), unit=self.unit)


class GridNode(Node):
    def __init__(
        self,
        x: int = -1,
        y: int = -1,
        z: int = -1,
        g: float = 0,
        h: float = 0,
        f: float = 0,
    ):
        super().__init__(x=x, y=y, z=z, g=g, h=h, f=f)

    def __eq__(self, o: object) -> bool:
        return self.getCoordinates() == o.getCoordinates()

    @classmethod
    def fromCoords(cls, coords: List[int]):
        """
        constructor creates node form list of node coordinates
        Args:
            coords (List[int]): list of node coordinates
        """

        x, y, z = coords[0], coords[1], coords[2]
        return cls(x=x, y=y, z=z)

    def getCoordinates(self) -> List[int]:
        """
        returns list of node coordinates
        Returns:
            List[int]: List of node coordinates
        """
        return [self.x, self.y, self.z]
