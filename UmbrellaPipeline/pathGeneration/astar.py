from typing import List
from node import *
from grid import *
from helper import *
import math


class AStar3D:

    def __init__(self, start: Node or List[int], end: Node or List[int], grid:Grid, method:str = 'diagonal', backtrace:bool = True) -> None:
        self.grid = grid
        self.start = Node.fromCoords(start.getCoordinates()) if type(start) is not Node else start
        self.end = Node.fromCoords(end.getCoordinates()) if type(end) is not Node else end
        self.shortestPath: List[Node] = []
        self.method = method
        self.backtrace = True


    def __str__(self):
        ret = ''
        for i in self.shortestPath:
            ret += f'{str(i)}\n'
        return ret


    def diagonal_h(self, node:Node, destination:Node) -> float:
        dx = abs(node.x - destination.x)
        dy = abs(node.y - destination.y)
        dz = abs(node.z - destination.z)
        dmin = min(dx,dy,dz)
        dmax = max(dx,dy,dz)
        dmid = dx + dy + dz - dmax - dmin
        D3 = math.sqrt(3)
        D2 = math.sqrt(2)
        D1 = math.sqrt(1)
        return (D3 - D2) * dmin + (D2 - D1) * dmid + D1 * dmax
    
    def euclidean_h(self, node:Node, destination:Node) -> float:
        return math.sqrt((node.x - destination.x)**2 + (node.y - destination.y)**2 + (node.z - destination.z)**2)
    
    def is_goal_reached(self, node:Node) -> bool:
        return node == self.end 
    
    def backtracePath(self) -> List[Node]:
        self.shortestPath.reverse()
        new = []
        current = self.shortestPath[0]
        while current:
            new.insert(0,current)
            current = current.parent
        self.shortestPath = new
        return self.shortestPath
    
    def generateSuccessors(self, parent:Node, end:Node, grid:Grid) -> List[Node]:
        possibleNeighbours = [[-1,-1,-1],[0,-1,-1],[1,-1,-1],[-1,0,-1],[0,0,-1],[1,0,-1],[-1,1,-1],[0,1,-1],[1,1,-1],[-1,-1,0],[0,-1,0],[1,-1,0],[-1,0,0],[1,0,0],[-1,1,0],[0,1,0],[1,1,0],[-1,-1,1],[0,-1,1],[1,-1,1],[-1,0,1],[0,0,1],[1,0,1],[-1,1,1],[0,1,1],[1,1,1]]
        ret = []
        for neighbour in possibleNeighbours:
            child = Node.fromCoords([a+b for a,b in zip(parent.getCoordinates(), neighbour)])
            if not self.grid.positionIsValid(node=child) or self.grid.positionIsBlocked(node=child): continue
            if self.method == 'diagonal':
                child.g = parent.g + self.diagonal_h(parent, child)
                child.f = child.g + self.diagonal_h(child, end)
            else:
                child.g = parent.g + self.euclidean_h(parent, child)
                child.f = child.g + self.euclidean_h(child, end)
            child.parent = parent
            ret.append(child)
        return ret

    def aStar3D(self) -> List[Node]:
        if self.is_goal_reached(node=self.start): 
            self.shortestPath.append(self.start)
            return self.shortestPath
        self.start.g = 0
        self.start.f = self.diagonal_h(node=self.start, destination=self.end) if self.method == 'diagonal' else self.euclidean_h(node=self.start, destination=self.end)
        openList = [self.start]
        while openList:
            q = openList[0]
            for node in openList:
                if node.f < q.f : q = node
            openList.remove(q)
            children = self.generateSuccessors(parent=q, end=self.end, grid=self.grid)
            for child in children:
                if self.is_goal_reached(child):
                    self.shortestPath.append(q)
                    self.shortestPath.append(child)
                    print(self.shortestPath)
                    return self.shortestPath if not self.backtrace else self.backtracePath()
                if any ((listEntry == child and listEntry.f < child.f) for listEntry in (openList or self.shortestPath)): continue
                openList.insert(0,child)
            self.shortestPath.append(q)
        return []