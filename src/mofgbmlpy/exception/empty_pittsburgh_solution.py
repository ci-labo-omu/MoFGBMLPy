class EmptyPittsburghSolution(Exception):
    def __init__(self, message: str = "A Pittsburgh solution has no michigan solution left"):
        super().__init__(message)
