class UninitializedKnowledgeException(Exception):
    """Exception raised when knowledge is not yet initialized (no fuzzy set)."""

    def __init__(self, message: str = "Knowledge is not yet initialized (no fuzzy set)"):
        """Constructor

        Args:
            message (str): The error message to be displayed when the exception is raised.
        """
        super().__init__(message)
