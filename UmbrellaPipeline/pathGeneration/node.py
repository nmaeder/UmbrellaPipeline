from typing import List

class Node:

    def __init__(self, x: int = -1, y: int = -1, z: int = -1, f: float = float('inf'), g: float = float('inf')):
        self.x = x
        self.y = y
        self.z = z
        self.f = f
        self.g = g 
        self.parent:Node = None 

    @classmethod
    def fromCoords(cls, coords:List[int]):
        x, y, z = coords[0], coords[1], coords[2]
        return cls(x,y,z)        

    def __str__(self) -> str:
        return f'[{self.x},{self.y},{self.z}]'

    def __repr__(self) -> str:
        return f'[{self.x},{self.y},{self.z}]'

    def __eq__(self, o:object) -> bool:
        return self.getCoordinates() == o.getCoordinates()

    def getCoordinates(self) -> List[int]:
        return [self.x, self.y, self.z]