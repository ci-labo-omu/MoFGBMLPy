class InvalidSolutionTypeException(Exception):
    def __init__(self, expected_type: str):
        super().__init__(f"Solution must be of type {expected_type}")
