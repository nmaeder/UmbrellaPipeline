class NoWayOutError(Exception):
    """
    This Exception is thrown if the escape room algorithm cannot find a way out of the cavity.

    """

    def __init__(self) -> None:

        message = "There was no Escape Path found. Try narrowing the walls or use a smaller grid spacing."
        super().__init__(message)


class StartIsFinishError(Exception):
    """
    This exception is thrown if the starting point for the escape room algorithm already satisfies the endpoint condition.

    """

    def __init__(self) -> None:
        message = "The starting point already meets the goal criteria. Increase the distance to protein."
        super().__init__(message)
