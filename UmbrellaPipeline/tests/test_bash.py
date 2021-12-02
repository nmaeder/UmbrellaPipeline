from UmbrellaPipeline.utils import (
    execute_bash,
    execute_bash_parallel,
)
import time, os, sys
import pytest

@pytest.mark.skipif("win" in sys.platform, reason="Bash not supported on Windows.")
def test_execute_bash():
    command1 = "echo Hello World"
    command2 = ["echo", "Hello World"]
    command3 = "sleep 12"
    stderr = "testerr.log"
    stdout = "testout.log"
    ret1 = execute_bash(command=command1)
    ret2 = execute_bash(command=command2, stdout_file=stdout)
    with pytest.raises(TimeoutError):
        execute_bash(command=command3, kill_after_wait=True, stderr_file=stderr)
    assert ret1 == "Hello World\n"
    assert ret2 == "Hello World\n"
    os.remove(stderr)
    os.remove(stdout)

@pytest.mark.skipif("win" in sys.platform, reason="Bash not supported on Windows.")
def test_parallel_bash():
    commands = ["sleep 3", "sleep 3", "sleep 3", "echo World"]
    start = time.time()
    o = execute_bash_parallel(command=commands)
    end = time.time() - start
    assert end < 5
    assert o[3] == "World\n"
