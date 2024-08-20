import cython
from mofgbmlpy.data.pattern import Pattern
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory cimport HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.factory.abstract_antecedent_factory cimport AbstractAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
cimport numpy as cnp

from mofgbmlpy.fuzzy.rule.consequent.abstract_consequent cimport AbstractConsequent

cdef class RuleBuilderCore:
    """Rule builder

    Attributes:
        _antecedent_factory (AbstractAntecedentFactory): Antecedent factory
        _consequent_factory (AbstractLearning):  Consequent factory
        _knowledge (Knowledge): Knowledge base
    """
    def __init__(self, antecedent_factory, consequent_factory, knowledge):
        """Constructor

        Args:
            antecedent_factory (AbstractAntecedentFactory): Antecedent factory
            consequent_factory (AbstractLearning):  Consequent factory
            knowledge (Knowledge): Knowledge base
        """
        self._antecedent_factory = antecedent_factory
        self._consequent_factory = consequent_factory
        self._knowledge = knowledge


    cdef int[:,:] create_antecedent_indices(self, int num_rules=1, Pattern pattern=None):
        """Create antecedent indices from a pattern or not. The two arguments can't be given at the same time
        
        Args:
            num_rules (int): Number of antecedents to be generated 
            pattern (Pattern): Pattern used to generate the antecedent

        Returns:
            int[,]: Array of array of antecedent indices
        """
        cdef AbstractAntecedentFactory factory = self._antecedent_factory
        cdef HeuristicAntecedentFactory heuristic_factory

        if pattern is None:
            return factory.create_antecedent_indices(num_rules)
        else:
            if not isinstance(factory, HeuristicAntecedentFactory):
                raise TypeError("The antecedent factory must be HeuristicAntecedentFactory if a pattern is provided")
            heuristic_factory = factory
            return heuristic_factory.create_antecedent_indices_from_pattern(pattern)

    cdef Antecedent create_antecedent_from_indices(self, int[:] antecedent_indices):
        """Create an antecedent from indices
        
        Args:
            antecedent_indices (int[]): Antecedent indices 

        Returns:
            Antecedent: Antecedent
        """
        return Antecedent(antecedent_indices, self._knowledge)

    cdef AbstractConsequent create_consequent(self, Antecedent antecedent, Dataset dataset=None):
        """Create a consequent from the antecedent and eventually a dataset different from the one given
        
        Args:
            antecedent (Antecedent): Antecedent
            dataset (Dataset): Optional dataset (if not given, use the one defined in this class). Used by the Scikit-learn wrapper for Pittsburgh solutions 

        Returns:
            AbstractConsequent: New consequent
        """
        return self._consequent_factory.learning(antecedent, dataset=dataset)

    cpdef Knowledge get_knowledge(self):
        """Get the knowledge
        
        Returns:
            Knowledge: Knowledge base
        """
        return self._knowledge

    cpdef Dataset get_training_dataset(self):
        """Get the training set
        
        Returns:
            Dataset: Training set
        """
        return self._consequent_factory.get_training_set()