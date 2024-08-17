import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.class_label.class_label_multi import ClassLabelMulti
from mofgbmlpy.data.dataset import Dataset
from mofgbmlpy.data.pattern import Pattern


class PittsburghScikitClassifier(BaseEstimator, ClassifierMixin):
    """Pittsburgh solution wrapper for Scikit-learn

    Attributes:
        pittsburgh_solution (PittsburghSolution): Pittsburgh solution wrapped
    """

    def __init__(self, pittsburgh_solution):
        """Constructor

        Args:
            pittsburgh_solution (PittsburghSolution): Pittsburgh solution wrapped
        """
        self.pittsburgh_solution = pittsburgh_solution
        self._training_dataset = None

    def fit(self, X, y):
        """Train the classifier on the given dataset

        Args:
            X (list): Array of arrays of attributes values
            y (list): Array of labels

        Returns:
            PittsburghScikitClassifier: Fitted PittsburghScikitClassifier
        """
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        training_dataset = PittsburghScikitClassifier.dataset_from_x_y(X, y)
        self.pittsburgh_solution.learning(training_dataset)
        return self

    def predict(self, X):
        """Predict the labels of a list of attributes vectors

        Args:
            X (list): Array of arrays of attributes values

        Returns:
            list: Array of predicted labels
        """
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        y = []

        for i in range(len(X)):
            y.append(self._predict_one(X[i]))
        return np.asarray(y)

    def _predict_one(self, x):
        """Predict the labels of an attribute vectors

        Args:
            x (list): Array of attributes values

        Returns:
            object: Predicted label
        """
        class_label = self.pittsburgh_solution.predict(Pattern(0, x, ClassLabelBasic(0)))
        if class_label is None:
            return -1
        return class_label.get_class_label_value()

    @staticmethod
    def dataset_from_x_y(X, y):
        """Load a Dataset object from attributes vectors and class labels arrays

        Args:
            X (list): Array of arrays of attributes values
            y (list): Array of labels

        Returns:
            Dataset: Created dataset
        """
        size = len(X)
        n_dim = X.shape[1]
        c_num = len(unique_labels(y))
        patterns = np.empty(size, object)

        is_multi_label = y.ndim == 2

        for i in range(size):
            if is_multi_label:
                class_label = ClassLabelMulti(y[i])
            else:
                class_label = ClassLabelBasic(y[i])
            patterns[i] = Pattern(i, X[i], class_label)

        return Dataset(size, n_dim, c_num, patterns)
