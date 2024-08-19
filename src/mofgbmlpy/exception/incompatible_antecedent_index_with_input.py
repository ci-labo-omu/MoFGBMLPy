class IncompatibleAntecedentIndexWithInput(Exception):
    def __init__(self, dim: int, input_value: float, antecedent_index: int):
        super().__init__(f"The antecedent index ({antecedent_index}) at the dimension {dim} is inconsistent with the "
                         f"input vector value at this dimension ({input_value}). One is categorical and the other is "
                         f"not.")
