from pymoo.core.repair import Repair


class PittsburghRepair(Repair):
    """Repair Pittsburgh solutions by removing those with no winners"""
    def _do(self, problem, Z, **kwargs):
        """Run the repair operator

        Args:
            problem (PittsburghProblem):
            Z (Population): Population of Pittsburgh solutions
            **kwargs (dict): Other Pymoo arguments

        Returns:
            Population: Repaired population
        """
        return problem.remove_no_winner_michigan_solution(Z)