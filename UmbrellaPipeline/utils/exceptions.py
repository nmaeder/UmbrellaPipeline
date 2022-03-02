class NoPathFoundError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "There was no Escape Path found. Try narrowing the walls or using a smaller grid spacing"
        )


class StartError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "The Starting point is inside the wall, try narrowing the walls."
        )
