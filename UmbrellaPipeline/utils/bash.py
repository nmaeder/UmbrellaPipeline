from os import kill
import subprocess
import logging
import time
from typing import List

logger = logging.getLogger(__name__)


def executeBashCommand(
    command: str or List[str],
    ignoreReturn: bool = False,
    killAfterWait: bool = False,
    stderrFile: str = None,
    stdoutFile: str = None,
) -> str:
    """
    Bash Wrapper. Gives option to timeout and to write files out of stdout and/or stderr

    Args:
        command (str or List[str]): command to be executed.
        ignoreReturn (bool, optional): if return value of bash should be ignored. Defaults to False.
        killAfterWait (bool, optional): if true, command is aborted after 10s. Defaults to False.
        stderrFile (str, optional): file for stderr. Defaults to None.
        stdoutFile (str, optional): file for stdout. Defaults to None.

    Raises:
        et: if a timeout of the command occurs that is not raised by the bash console.
        ee: if any other exception occurs that is thrown by the bash console
        OSError: error if returncode from the bash console is non-zero.

    Returns:
        str: stdout of the bash console
    """
    if isinstance(command, str):
        command = command.split()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = process.communicate()
        try:
            process.wait(timeout=10)
        except:
            if killAfterWait:
                process.kill()
                logger.warning(
                    "Process got killed, since it took to long. If its a long job, set killAfterWait to False"
                )
        try:
            while process.poll() is None:
                time.sleep(0.1)
        except KeyboardInterrupt:
            process.kill()

    except TimeoutError as et:
        raise et
    except Exception as ee:
        raise ee

    if stdoutFile:
        with open(file=stdoutFile, mode="a") as out:
            out.writelines(stdout.decode("utf-8"))
    else:
        logger.info(stdout.dcode("utf-8"))
    if stderrFile:
        with open(file=stderrFile, mode="a") as err:
            err.writelines(stderr.decode("utf-8"))
    else:
        if stderr:
            logger.error(stderr.decode("utf-8"))

    if process.returncode > 0 and not ignoreReturn:
        raise OSError(
            f"Return of Bash command is non-Zero: {process.returncode}\n"
            f"OUT: {stdout}\n"
            f"ERR: {stderr}\n"
        )

    return stdout.decode("utf-8")
