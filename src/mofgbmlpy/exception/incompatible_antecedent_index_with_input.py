class IncompatibleAntecedentIndexWithInput(Exception):
    """Exception raised when the antecedent index is inconsistent with the input vector value at a specific dimension."""

    def __init__(self, dim: int, input_value: float, antecedent_index: int):
        """Constructor

        Args:
            dim (int): The dimension at which the inconsistency occurs
            input_value (float): The value of the input vector at the inconsistent dimension
            antecedent_index (int): The antecedent index that is inconsistent with the input value
        """
        super().__init__(
            f"The antecedent index ({antecedent_index}) at the dimension {dim} is inconsistent with the "
            f"input vector value at this dimension ({input_value}). One is categorical and the other is "
            f"not."
        )
