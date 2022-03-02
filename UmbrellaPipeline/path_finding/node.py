from typing import List
import openmm.unit as u


class Node:
    """
    Used in context with Tree and TreeEscapeRoom
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        z: int = 0,
        distance_to_wall=0.1,
        parent=None,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.distance_to_wall = distance_to_wall
        self.parent: Node = parent

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __eq__(self, other: object) -> bool:
        return self.get_grid_coordinates() == other.get_grid_coordinates()

    def __lt__(self, other: object) -> bool:
        return self.distance_to_wall > other.distance_to_wall

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    @property
    def z(self) -> int:
        return self._z

    @property
    def distance_to_wall(self) -> float:
        return self._distance_to_wall

    @x.setter
    def x(self, value: int) -> None:
        try:
            self._x = int(value)
        except (TypeError, ValueError):
            raise TypeError("Coordinates for Node need to be of type int")

    @y.setter
    def y(self, value: int) -> None:
        try:
            self._y = int(value)
        except (TypeError, ValueError):
            raise TypeError("Coordinates for Node need to be of type int")

    @z.setter
    def z(self, value: int) -> None:
        try:
            self._z = int(value)
        except (TypeError, ValueError):
            raise TypeError("Coordinates for Node need to be of type int")

    @distance_to_wall.setter
    def distance_to_wall(self, value) -> None:
        if value <= 0:
            raise ValueError("Position of node would be inside Wall.")
        else:
            self._distance_to_wall = value

    @classmethod
    def from_coords(cls, coords, distance_to_wall: float = 0.1, parent: object = None):
        x, y, z = coords
        return cls(x=x, y=y, z=z, distance_to_wall=distance_to_wall, parent=parent)

    def get_grid_coordinates(self) -> List[float]:
        return [self.x, self.y, self.z]

    def get_coordinates_for_query(
        self, start: u.Quantity = [0, 0, 0], spacing: u.Quantity = 1 * u.nanometer
    ) -> List[float]:
        if isinstance(spacing, u.Quantity):
            spacing = spacing.value_in_unit(u.nanometer)
        try:
            x = self.x * spacing + start[0].value_in_unit(u.nanometer)
            y = self.y * spacing + start[1].value_in_unit(u.nanometer)
            z = self.z * spacing + start[2].value_in_unit(u.nanometer)
        except AttributeError:
            x = self.x * spacing + start[0]
            y = self.y * spacing + start[1]
            z = self.z * spacing + start[2]
        return [x, y, z]
