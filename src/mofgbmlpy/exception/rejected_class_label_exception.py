class RejectedClassLabelException(Exception):
    """Exception raised when a class label is rejected during the learning process."""

    def __init__(self, message: str = "The class label is rejected"):
        """Constructor

        Args:
            message (str): The error message to be displayed when the exception is raised.
        """
        super().__init__(message)
