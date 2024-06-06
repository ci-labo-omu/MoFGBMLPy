```mermaid
classDiagram
    note for Dataset "Store a list of patterns"
    note for DatasetManager "Singleton storing datasets (note: maybe put it \nin a big context/environment class with the other singletons)"
    note for Input "Methods to load data"
    note for Pattern "One pattern (one data row)"

    namespace data {
        class Dataset {
        }

        class DatasetManager {
        }

        class Input {
        }

        class Output {
        }

        class Pattern {
        }
    }

    note for FuzzySetFactory "Create membership function objects for new fuzzy set"
    Knowledge --o "*" LinguisticVariable
    FuzzySet --o "*" LinguisticVariable

    namespace simpful {
        class LinguisticVariable {
            +get_values(float v) List~float~
        }

        class FuzzySet {
        }
    }

    SingleWinnerRuleSelection --|> AbstractClassification
    AllCombinationAntecedentFactory --|> AbstractAntecedentFactory
    HeuristicAntecedentFactory --|> AbstractAntecedentFactory

    namespace fuzzy {
    %%    knowledge
        class Knowledge {
            -fuzzy_sets: List~LinguisticVariable~
            -fuzzy_sets_lengths: List~int~
        }

    %%    fuzzy_term
        class FuzzySetFactory {
        }

        class DontCare {
        }

    %%    rule
        class RuleFactory {
            -antecedent_factory
            -consequent_factory
        }

        class Rule {
        }

    %%          rule.antecedent
        class Antecedent {
        }

        class AbstractAntecedentFactory {
        }

        class AllCombinationAntecedentFactory {
        }

        class HeuristicAntecedentFactory {
        }

    %%          rule.consequent

        class Consequent {
        }

        class AbstractConsequentFactory {
        }

    %%    classifier

        class Classifier {
        }

        class AbstractClassification {
        }

        class SingleWinnerRuleSelection {
        }
    }

    BasicDuplicateElimination --|> ElementwiseDuplicateElimination
    MichiganSolution --|> AbstractSolution~T~
    PittsburghSolution --|> AbstractSolution
    PittsburghProblem --|> Problem
    MichiganProblem --|> Problem
    BasicMutation --|> Mutation
    MichiganMutation --|> Mutation
    PittsburghMutation --|> Mutation
    UniformCrossover --|> Crossover
    MichiganCrossover --|> Crossover
    PittsburghCrossover --|> Crossover
    HybridGBMLCrossover --|> Crossover
    HybridGBMLSampling --|> Sampling
    note for MichiganSolution "bounds is a pair of int"

    namespace gbml {
        class BasicDuplicateElimination {
        }

    %%    solution
        class AbstractSolution~T~ {
            <<abstract>>
            -objectives: List~float~
            -variables: List~T~
            -constraints: List~float~
            +Solution(int num_variables, int num_objectives, int num_constraints=0)
            +set_objective(int index, float value)
            +get_objective(int index) float
            +get_objectives() float
            +set_variable(int index)
            +get_variable(int index) T
            +get_variables() List~T~
            +set_constraint(int index, float value)
            +get_constraint(int index) float
            +get_constraints() float
            +get_num_variables() int
            +get_num_objectives() int
            +get_num_constraints() int
            +__eq__(other) bool
            +__hash__() int
            +__copy__() Solution~T~
        }

        class MichiganSolution {
            -rule: Rule
            -consequent_factory: ConsequentFactory
            -bounds: tuple
            +MichiganSolution(tuple bounds, int num_objectives, int num_constraints, ConsequentFactory consequent_factory)
        }

        class MichiganSolutionFactory {
            -bounds: tuple
            -num_objectives: int
            -num_constraints: int
            -rule_factory: RuleFactory
            +MichiganSolutionFactory(tuple bounds, int num_objectives, int num_constraints, RuleFactory rule_factory)
        }

        class PittsburghSolution {
            -classifier: Classifier
            +PittsburghSolution(int num_variables, int num_objectives, int num_constraints, ConsequentFactory consequent_factory, Classifier classifier)
        %%        learning() update the consequent part of each rule
            +learning()
            +classifiy(Pattern pattern) MichiganSolution
            +
        }

        class PittsburghSolutionFactory {
            -classifier: Classifier
            -num_objectives: int
            -num_constraints: int
            -michigan_solution_factory: MichiganSolutionFactory
            +MichiganSolutionFactory(int num_variables, int variables, int num_objectives, int num_constraints, MichiganSolutionFactory michigan_solution_factory)
        }

    %%    operator

    %%          crossover
        class UniformCrossover {
        }

        class MichiganCrossover {
        }

        class PittsburghCrossover {
        }

        class HybridGBMLCrossover {
        }

    %%    mutation
    %%TEMPORARY CLASS
        class BasicMutation {
        }

        class MichiganMutation {
        }

        class PittsburghMutation {
        }

    %%    problem
        class PittsburghProblem {
        }

        class MichiganProblem {
        }

    %%    sampling
        class HybridGBMLSampling {
        }
    }

    namespace main {
        class Consts {
        }

        class ExperienceParameters {
        }

        class MofgbmlBasicMain {
        }
    }

    namespace pymoo {
        class Problem {
        }
        class Crossover {
        }
        class Mutation {
        }
        class Sampling {
        }
        class ElementwiseDuplicateElimination {
        }
        class NSGA2 {
        }
    }

```