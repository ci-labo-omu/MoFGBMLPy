class ExceededMaxTrialsNumber(Exception):
    """Exception raised when the number of trials exceeds the maximum allowed."""

    def __init__(self, message: str = "Exceeded the maximum number of trials"):
        """Constructor

        Args:
            message (str): The error message to be displayed when the exception is raised.
        """
        super().__init__(message)
