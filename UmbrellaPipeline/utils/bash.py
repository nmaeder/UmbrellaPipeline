import subprocess

import time
from typing import List


def execute_bash(
    command: str or List[str],
    kill_after_wait: bool = False,
    stderr_file: str = None,
    stdout_file: str = None,
) -> str:
    """
    Bash Wrapper. Gives option to timeout and to write files out of stdout and/or stderr

    Args:
        command (str or List[str]): command to be executed.
        kill_after_wait (bool, optional): if true, command is aborted after 10s. Defaults to False.
        stderr_file (str, optional): file for stderr. Defaults to None.
        stdout_file (str, optional): file for stdout. Defaults to None.

    Raises:
        et: if a timeout of the command occurs that is not raised by the bash console.
        ee: if any other exception occurs that is thrown by the bash console
        OSError: error if returncode from the bash console is non-zero.

    Returns:
        str: stdout of the bash console
    """
    stderr, stdout = None, None

    if isinstance(command, str):
        command = command.split()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if kill_after_wait:
                process.kill()
                raise TimeoutError

        try:
            while process.poll() is None:
                time.sleep(0.1)
        except KeyboardInterrupt:
            process.kill()

    except TimeoutError as et:
        stdout, stderr = process.communicate()
        if stderr_file:
            with open(file=stderr_file, mode="w") as err:
                err.writelines(stderr)
        raise et

    except Exception as ee:
        stdout, stderr = process.communicate()
        if stderr_file:
            with open(file=stderr_file, mode="w") as err:
                err.writelines(stderr)
        raise ee

    stdout, stderr = process.communicate()

    if stdout_file:
        with open(file=stdout_file, mode="w") as out:
            out.writelines(stdout)
    return stdout


def execute_bash_parallel(
    command: List[List[str]],
    stdout_file: List[str] = None,
) -> List[str]:
    """
    Executes a number of bash scripts in parallel. returs output of all of them. if a list of outputfiles is given, the output is written to these files.

    Args:
        command (List[List[str]] or List[str]): list of commands to execute.
        stdout_file (List[str], optional): List of files where the stdout should be store. Defaults to None.

    Raises:
        e: any exception occuring during the bashcommand.
        StopIteration: if length of stderr is not eqaul to the number of commands

    Returns:
        List[str]: List of the stdoutputs given by each process.
    """
    procs: List[str] = []
    newc = []
    for coms in command:
        newc.append(coms.split())
    command = newc

    try:
        procs = [
            subprocess.Popen(
                com,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            for com in command
        ]

    except Exception as e:
        raise e

    stdout: List[str] = []
    stderr: List[str] = []
    for process in procs:
        o, e = process.communicate()
        stdout.append(o)
        stderr.append(e)

    if stdout_file:
        try:
            for i, out in enumerate(stdout):
                with open(file=stdout_file[i], mode="w") as out:
                    out.writelines(stdout[i])
        except StopIteration:
            raise StopIteration(
                "You have given a list in stdout_file that doesnt match the number of Commands!"
            )
    return stdout
