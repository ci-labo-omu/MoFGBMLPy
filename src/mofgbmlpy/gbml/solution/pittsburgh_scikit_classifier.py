import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import minmax_scale
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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
            patterns[i] = Pattern(i, X[i].astype(np.float32), class_label)

        return Dataset(size, n_dim, c_num, patterns)

    def plot_decision_boundaries(self, X, y, title="Decision Boundaries", fixed_vals=None, num_points_per_dim=100):
        X = X.astype(np.float32)
        if X.shape[1] > 2 and (fixed_vals is None or len(fixed_vals) != X.shape[1]):
            raise NotImplementedError("Decision boundary plot is only implemented for 2D datasets.")

        class_colors = ["red", "green", "blue", "gray"]
        class_labels = ["Class 0", "Class 1", "Class 2", "Unclassified"]

        feature_1, feature_2 = np.meshgrid(
            np.linspace(0, 1, num=num_points_per_dim), np.linspace(0, 1, num=num_points_per_dim)
        )

        if fixed_vals is not None and fixed_vals.count(None) != 2:
            raise ValueError("fixed_vals must contain exactly two None values.")

        grid = np.vstack([feature_1.ravel(), feature_2.ravel()], dtype=np.float32).T

        grid_full = np.zeros((grid.shape[0], X.shape[1]), dtype=np.float32)
        grid_i = 0

        if fixed_vals is None:
            grid_full = grid
        else:
            var_indices = []
            title += " (fixed:"
            for i in range(len(fixed_vals)):
                if fixed_vals[i] is None:
                    var_indices.append(i)
                else:
                    title += f" x_{i},"
            title = title[:-1] + ")"

            for i in range(X.shape[1]):
                if fixed_vals[i] is not None:
                    grid_full[:, i] = fixed_vals[i]
                else:
                    grid_full[:, i] = grid[:, grid_i]
                    grid_i += 1

        predictions = self.predict(grid_full)
        predictions = predictions.reshape(feature_1.shape)

        predictions, cmap = self._get_db_plot_values(predictions, class_colors)

        display = DecisionBoundaryDisplay(xx0=feature_1, xx1=feature_2, response=predictions)
        fig, ax = plt.subplots(figsize=(8, 5))

        display.plot(ax=ax, cmap=cmap, alpha=0.1)

        if fixed_vals is None:
            plt.xlabel("x_0")
            plt.ylabel("x_1")
        else:
            plt.xlabel(f"x_{var_indices[0]}")
            plt.ylabel(f"x_{var_indices[1]}")

        # y_pred = self.predict(X)
        y, cmap = self._get_db_plot_values(y, class_colors)

        display.ax_.scatter(X[:, var_indices[0]], X[:, var_indices[1]], c=y, edgecolor="black", cmap=cmap)

        # Legend
        background_handles = [
            Patch(facecolor=class_colors[i], edgecolor="none", alpha=0.3, label=f"{class_labels[i]} (predicted)")
            for i in range(len(class_colors))
        ]

        point_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=class_colors[i],
                markeredgecolor="black",
                label=f"{class_labels[i]} (true)",
            )
            for i in range(len(class_colors) - 1)
        ]

        handles = background_handles + point_handles
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_conf_matrix(self, X, y, title="Confusion Matrix", ignore_unclassified=True):
        X = X.astype(np.float32)
        y_pred = self.predict(X)
        num_unclassified = np.count_nonzero(y_pred == -1)
        title += f" (Unclassified: {num_unclassified})"

        if ignore_unclassified:
            mask = y_pred != -1
            y = y[mask]
            y_pred = y_pred[mask]
            labels = np.unique(y)
        else:
            labels = np.unique(np.concatenate([y, y_pred]))

        cm = confusion_matrix(y, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.viridis)
        plt.title(title)
        plt.show()

    @staticmethod
    def _get_db_plot_values(y_values, class_colors):
        init_shape = y_values.shape
        if y_values.ndim == 2:
            y_values = y_values.ravel()
        unique_classes = np.unique(y_values)
        new_y_values = np.empty_like(y_values)
        new_class_colors = []
        class_mapping = {}

        for new_idx, old_class in enumerate(unique_classes):
            class_mapping[old_class] = new_idx
            new_class_colors.append(class_colors[old_class])

        for i in range(new_y_values.shape[0]):
            new_y_values[i] = class_mapping[y_values[i]]

        if init_shape != new_y_values.shape:
            new_y_values = new_y_values.reshape(init_shape)

        return new_y_values, ListedColormap(new_class_colors)
