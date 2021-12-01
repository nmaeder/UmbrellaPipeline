from typing import List

import openmm.unit as unit
from openmm import Vec3


class Node:
    def __init__(
        self, x=-1, y=-1, z=-1, g: float = 0, h: float = 0, f: float = 0, parent=None
    ):
        self.x = x
        self.y = y
        self.z = z
        self.g = g
        self.h = h
        self.f = f
        self.parent: Node = parent

    def __str__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def __repr__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def get_coordinates(self):
        return [self.x, self.y, self.z]


class TreeNode(Node):
    """
    Used in context with Tree and TreeAStar
    """

    def __init__(
        self,
        x: float = -1,
        y: float = -1,
        z: float = -1,
        g: float = 0,
        h: float = 0,
        f: float = 0,
        unit_: unit.Unit = unit.nanometer,
        parent=None,
    ):
        super().__init__(x=x, y=y, z=z, g=g, h=h, f=f)
        self.unit = unit_
        self.parent: TreeNode = parent

    @classmethod
    def from_coords(
        cls,
        coords: List[unit.Quantity] or unit.Quantity or List[float],
        unit_: unit.Unit = None,
        parent=None,
    ):
        try:
            u = coords[0].unit
            x, y, z = (
                coords[0].value_in_unit(u),
                coords[1].value_in_unit(u),
                coords[2].value_in_unit(u),
            )
            p = parent
        except AttributeError:
            x, y, z = coords[0], coords[1], coords[2]
            u = unit_
            p = parent
        return cls(x=x, y=y, z=z, unit_=u, parent=p)

    def __str__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def __repr__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def __eq__(self, o: object) -> bool:
        return self.get_coordinates() == o.get_coordinates()

    def __round__(self, decimals: int = 3) -> unit.Quantity:
        return unit.Quantity(
            value=Vec3(
                round(self.x, decimals),
                round(self.y, decimals),
                round(self.z, decimals),
            ),
            unit=self.unit,
        )

    def get_coordinates(self) -> unit.Quantity:
        return unit.Quantity(value=Vec3(self.x, self.y, self.z), unit=self.unit)

    def get_coordinates_for_query(self, unit_) -> List[float]:
        x = unit.Quantity(value=self.x, unit=self.unit)
        y = unit.Quantity(value=self.y, unit=self.unit)
        z = unit.Quantity(value=self.z, unit=self.unit)
        return [x.value_in_unit(unit_), y.value_in_unit(unit_), z.value_in_unit(unit_)]


class GridNode(Node):
    """
    Used in combination with Grid and GridAStar.
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        z: int = 0,
        g: float = 0,
        h: float = 0,
        f: float = 0,
        parent=None,
    ):
        if any(i < 0 for i in [x, y, z]):
            raise ValueError(
                "Negative values for x, y or z are not allowed for type GridNode"
            )
        super().__init__(x=x, y=y, z=z, g=g, h=h, f=f)
        self.parent: GridNode = parent

    def __eq__(self, o: object) -> bool:
        return self.get_coordinates() == o.get_coordinates()

    def __str__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def __repr__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    @classmethod
    def from_coords(cls, coords: List[int]):
        """
        constructor creates node form list of node coordinates
        Args:
            coords (List[int]): list of node coordinates
        """

        x, y, z = coords[0], coords[1], coords[2]
        return cls(x=x, y=y, z=z)

    def get_coordinates(self) -> List[int]:
        """
        returns list of node coordinates
        Returns:
            List[int]: List of node coordinates
        """
        return [self.x, self.y, self.z]
