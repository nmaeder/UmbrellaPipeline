from typing import Type
import pytest
from UmbrellaPipeline.path_finding import Queue

def test_queue():
    q1 = Queue([-2,4,6,2,2,1,-43,87])
    q2 = Queue()

    with pytest.raises(IndexError):
        q2.pop()

    assert q1.pop() == -43
    assert q1.pop() == -2

    q1.push(-200)
    q1.push(200)

    assert q1.pop() == -200

    with pytest.raises(TypeError):
        q1.push("K")