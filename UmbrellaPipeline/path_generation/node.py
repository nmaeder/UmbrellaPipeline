from typing import List
import openmm.unit as u
from openmm import Vec3


class Node:
    def __init__(
        self,
        x=0,
        y=0,
        z=0,
        distance_to_wall: float = 0,
        distance_walked: float = float("inf"),
        parent=None,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.distance_to_wall = distance_to_wall
        self.distance_walked = distance_walked
        self.parent: Node = parent

    def __str__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def __repr__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def __eq__(self, o: object) -> bool:
        return self.get_coordinates() == o.get_coordinates()

    def get_coordinates(self):
        return [self.x, self.y, self.z]

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def distance_to_wall(self):
        return self._distance_to_wall

    @property
    def distance_walked(self):
        return self._distance_walked

    @x.setter
    def x(self, value: float):
        try:
            self._x = float(value)
        except TypeError as err:
            raise err

    @y.setter
    def y(self, value: float):
        try:
            self._y = float(value)
        except TypeError as err:
            raise err

    @z.setter
    def z(self, value: float):
        try:
            self._z = float(value)
        except TypeError as err:
            raise err

    @distance_to_wall.setter
    def distance_to_wall(self, value: float):
        self._distance_to_wall = 0 if value < 0 else value

    @distance_walked.setter
    def distance_walked(self, value: float):
        if value < 0:
            raise ValueError("Walked distance can not be negative!")
        else:
            self._distance_walked = value


class TreeNode(Node):
    """
    Used in context with Tree and TreeEscapeRoom
    """

    def __init__(
        self,
        x=0,
        y=0,
        z=0,
        distance_to_wall: float = 0,
        distance_walked: float = float("inf"),
        unit=u.nanometer,
        parent=None,
    ) -> None:
        super().__init__(
            x=x,
            y=y,
            z=z,
            distance_walked=distance_walked,
            distance_to_wall=distance_to_wall,
        )
        self.parent: TreeNode = parent
        self.unit = unit

    @property
    def unit(self) -> u.Unit:
        return self._unit

    @unit.setter
    def unit(self, value: u.Unit):
        if value.is_compatible(u.nanometer):
            self._unit = value
        else:
            raise TypeError("The unit of tree node has to be a Length.")

    @classmethod
    def from_coords(
        cls,
        coords: List[u.Quantity] or u.Quantity or List[float],
        unit: u.Unit = None,
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
            u = unit
            p = parent
        return cls(x=x, y=y, z=z, unit=u, parent=p)

    def __round__(self, decimals: int = 3) -> u.Quantity:
        return u.Quantity(
            value=Vec3(
                round(self.x, decimals),
                round(self.y, decimals),
                round(self.z, decimals),
            ),
            unit=self.unit,
        )

    def get_coordinates(self) -> u.Quantity:
        return u.Quantity(value=Vec3(self.x, self.y, self.z), unit=self.unit)

    def get_coordinates_for_query(self, unit) -> List[float]:
        x = u.Quantity(value=self.x, unit=self.unit)
        y = u.Quantity(value=self.y, unit=self.unit)
        z = u.Quantity(value=self.z, unit=self.unit)
        try:
            return [x.value_in_unit(unit), y.value_in_unit(unit), z.value_in_unit(unit)]
        except:
            raise TypeError("Input argument unit has to have the dimension Length.")


class GridNode(Node):
    """
    Used in combination with Grid and GridEscapeRoom.
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        z: int = 0,
        distance_to_wall: float = 0,
        distance_walked: float = 0,
        parent=None,
    ):
        super().__init__(
            x=x,
            y=y,
            z=z,
            distance_to_wall=distance_to_wall,
            distance_walked=distance_walked,
        )
        self.parent: GridNode = parent

    @classmethod
    def from_coords(cls, coords: List[int]):
        """
        constructor creates node form list of node coordinates
        Args:
            coords (List[int]): list of node coordinates
        """
        x, y, z = coords[0], coords[1], coords[2]
        return cls(x=x, y=y, z=z)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @x.setter
    def x(self, value: int):
        if value < 0:
            raise ValueError("GridNode coordinate has to be non-negative Integer!")
        try:
            self._x = int(value)
        except TypeError:
            raise TypeError("GridNode coordinate has to be non-negative Integer!")

    @z.setter
    def z(self, value: int):
        if value < 0:
            raise ValueError("GridNode coordinate has to be non-negative Integer!")
        try:
            self._z = int(value)
        except TypeError:
            raise TypeError("GridNode coordinate has to be non-negative Integer!")

    @y.setter
    def y(self, value: int):
        if value < 0:
            raise ValueError("GridNode coordinate has to be non-negative Integer!")
        try:
            self._y = int(value)
        except TypeError:
            raise TypeError("GridNode coordinate has to be non-negative Integer!")
