
from mofgbmlpy.data.dataset_density cimport DatasetWithDensity
from mofgbmlpy.exception.abstract_method_exception import AbstractMethodException
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.abstract_consequent cimport AbstractConsequent



cdef class AbstractLearningWithDensity:
    """Abstract class for the consequent factory (learning)

    Attributes:
        _train_ds (DatasetWithDensity): Training dataset used to generate the consequent
    """

    def __init__(self, DatasetManager dataset_manager):
        """Constructor

        Args:
            training_dataset (DatasetWithDensity): Training dataset used to generate the consequent
        """
        training_dataset = dataset_manager.current_dataset
        if training_dataset is None:
            raise TypeError("The training dataset cannot be None")
        self.__train_ds = training_dataset

    cpdef AbstractConsequent learning(self, Antecedent antecedent, DatasetWithDensity dataset=None, double reject_threshold=0):
        """Learn a consequent from the antecedent and dataset
        
        Args:
            antecedent (Antecedent): Antecedent whose consequent part is learnt
            dataset (DatasetWithDensity): Training dataset
            reject_threshold (double): Threshold for the rule weight under which the rule is considered rejected

        Returns:
            AbstractConsequent: Created consequent
        """
        raise AbstractMethodException()

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        raise AbstractMethodException()
