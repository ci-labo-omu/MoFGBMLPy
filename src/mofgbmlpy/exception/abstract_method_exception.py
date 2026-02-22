class AbstractMethodException(Exception):
    """Exception raised when an abstract method is called without being implemented in a concrete class."""

    def __init__(self, message: str = "This method is abstract. Please use a concrete class."):
        """Constructor

        Args:
            message (str): The error message to be displayed when the exception is raised.
        """
        super().__init__(message)
