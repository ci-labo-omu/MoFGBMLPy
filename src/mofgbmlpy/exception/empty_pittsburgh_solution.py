class EmptyPittsburghSolution(Exception):
    """Exception raised when a Pittsburgh solution has no Michigan solution left."""

    def __init__(self, message: str = "A Pittsburgh solution has no michigan solution left"):
        """Constructor

        Args:
            message (str): The error message to be displayed when the exception is raised.
        """
        super().__init__(message)
