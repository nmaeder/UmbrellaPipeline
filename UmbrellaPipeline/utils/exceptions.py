class NoWayOutError(Exception):
    def __init__(self) -> None:
        message = "There was no Escape Path found. Try narrowing the walls or use a smaller grid spacing."
        super().__init__(message)


class StartIsFinishError(Exception):
    def __init__(self) -> None:
        message = "The starting point already meets the goal criteria. Increase the distance to protein."
        super().__init__(message)
