from pymoo.core.repair import Repair


class PittsburghRepair(Repair):
    def _do(self, problem, Z, **kwargs):
        return problem.remove_no_winner_michigan_solution(Z)