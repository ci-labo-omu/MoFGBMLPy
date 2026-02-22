class InvalidSolutionTypeException(Exception):
    """Exception raised when a solution is of an invalid type."""

    def __init__(self, expected_type: str):
        """Constructor

        Args:
            expected_type (str): Expected type of the solution
        """
        super().__init__(f"Solution must be of type {expected_type}")
