from typing import List
import openmm.unit as u


class Node:
    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        z: int = 0,
        distance_to_wall: float = 0.1,
        parent=None,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.distance_to_wall = distance_to_wall
        self.parent: Node = parent

    def __str__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def __repr__(self) -> str:
        return f"[{self.x},{self.y},{self.z}]"

    def __eq__(self, o: object) -> bool:
        return self.get_coordinates() == o.get_coordinates()

    # Getter
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

    # Setter
    @x.setter
    def x(self, value: int):
        try:
            self._x = int(value)
        except (TypeError, ValueError):
            raise TypeError

    @y.setter
    def y(self, value: float):
        try:
            self._y = int(value)
        except (TypeError, ValueError):
            raise TypeError

    @z.setter
    def z(self, value: float):
        try:
            self._z = int(value)
        except (TypeError, ValueError):
            raise TypeError

    @distance_to_wall.setter
    def distance_to_wall(self, value) -> None:
        if value <= 0:
            raise ValueError("Position of node would be inside Wall.")
        else:
            self._distance_to_wall = value

    def get_coordinates(self):
        return [self.x, self.y, self.z]


class TreeNode(Node):
    """
    Used in context with Tree and TreeEscapeRoom
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        z: int = 0,
        distance_to_wall: float = 0.1,
        parent=None,
    ) -> None:
        super().__init__(
            x=x,
            y=y,
            z=z,
            distance_to_wall=distance_to_wall,
            parent=parent
        )

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __lt__(self, other: object) -> bool:
        return self.distance_to_wall > other.distance_to_wall

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
        parent=None,
    ):
        super().__init__(
            x=x,
            y=y,
            z=z,
            distance_to_wall=distance_to_wall,
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

    # Getter
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

    # Setter
    @x.setter
    def x(self, value: int):
        if value < 0:
            raise ValueError("GridNode coordinate has to be non-negative!")
        try:
            self._x = int(value)
        except TypeError:
            raise TypeError("GridNode coordinate has to be non-negative Integer!")

    @z.setter
    def z(self, value: int):
        if value < 0:
            raise ValueError("GridNode coordinate has to be non-negative!")
        try:
            self._z = int(value)
        except TypeError:
            raise TypeError("GridNode coordinate has to be non-negative Integer!")

    @y.setter
    def y(self, value: int):
        if value < 0:
            raise ValueError("GridNode coordinate has to be non-negative!")
        try:
            self._y = int(value)
        except TypeError:
            raise TypeError("GridNode coordinate has to be non-negative Integer!")

    @distance_to_wall.setter
    def distance_to_wall(self, value: float):
        self._distance_to_wall = 0 if value < 0 else value
