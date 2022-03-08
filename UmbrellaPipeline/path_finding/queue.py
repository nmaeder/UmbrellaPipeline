from heapq import heapify, heappop, heappush
from typing import List
from UmbrellaPipeline.path_finding import Node


class Queue:
    """
    Basic class to implement the priority queue for the A* and the Escape Room Algorithm.
    """

    def __init__(self, existing_list: List[Node] = None) -> None:
        """
        Creates either an empty or turns an existing List into a priority Queue
        Args:
            existing_list (List[Node], optional): Give a list you want to turn into a priority queue. Defaults to None.
        """
        try:
            self.queue = existing_list
            heapify(existing_list)
        except:
            self.queue = []

    def push(self, value: Node) -> None:
        """
        Adds an element to the priority queue
        Args:
            value (Node): element to add.
        """
        try:
            heappush(self.queue, value)
        except (ValueError, TypeError):
            raise TypeError("Queue members need to be of type float")

    def pop(self) -> Node:
        """
        Gets and removes the 0th element from the Queue
        Raises:
            IndexError: Raised when the queue is empty.
        Returns:
            Node: Queue entry at position 0
        """
        try:
            return heappop(self.queue)
        except IndexError:
            raise IndexError("Queue is empty")
