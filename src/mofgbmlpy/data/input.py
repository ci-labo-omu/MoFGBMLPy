from mofgbmlpy.data.dataset import Dataset
from mofgbmlpy.data.pattern import Pattern
from mofgbmlpy.data.class_label.class_label_multi import ClassLabelMulti
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
import numpy as np
import csv


class Input:
    """Class of static methods used to read files and load datasets """
    @staticmethod
    def input_data_set(file_name, is_multi_label):
        """Load a dataset from a file name

        Args:
            file_name (str): Name of the file containing the dataset
            is_multi_label (bool):  True if the dataset is multi label and False otherwise

        Returns:
            Dataset: Dataset read from the file
        """
        if is_multi_label:
            return Input.input_data_set_multi(file_name)
        else:
            return Input.input_data_set_basic(file_name)

    @staticmethod
    def input_data_set_multi(file_name):
        """Load a multi label dataset from a file name

        Args:
            file_name (str): Name of the file containing the dataset

        Returns:
            Dataset: Dataset read from the file
        """
        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            size, num_dim, num_classes = int(header[0]), int(header[1]), int(header[2])

            pattern_id = 0
            patterns = []
            for row in reader:
                in_vector = np.zeros(num_dim, dtype=np.float64)
                c_vector = np.zeros(num_classes, dtype=np.int_)

                for i in range(len(in_vector)):
                    in_vector[i] = float(row[i])

                j = num_dim
                for i in range(len(c_vector)):
                    c_vector[i] = int(float(row[j]))
                    j += 1

                class_labels = ClassLabelMulti(c_vector)

                patterns.append(Pattern(pattern_id, in_vector, class_labels))
                pattern_id += 1

            patterns = np.array(patterns, dtype=object)
            dataset = Dataset(size, num_dim, num_classes, patterns)

        return dataset

    @staticmethod
    def input_data_set_basic(file_name):
        """ Load a (mono label) dataset from a file name

        Args:
            file_name (str): Name of the file containing the dataset

        Returns:
            Dataset: Dataset read from the file
        """
        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            size, num_dim, num_classes = int(header[0]), int(header[1]), int(header[2])

            pattern_id = 0

            patterns = []

            for row in reader:
                in_vector = np.zeros(num_dim, dtype=np.float64)

                for i in range(len(in_vector)):
                    in_vector[i] = float(row[i])

                class_label = ClassLabelBasic(int(float(row[num_dim])))

                patterns.append(Pattern(pattern_id, in_vector, class_label))
                pattern_id += 1

        patterns = np.array(patterns, dtype=object)
        dataset = Dataset(size, num_dim, num_classes, patterns)

        return dataset

    @staticmethod
    def get_train_test_files(arguments):
        """Load the train and test datasets from filenames specified in an Arguments object

        Args:
            arguments (Arguments): Object containing the TRAIN_FILE, TEST_FILE and IS_MULTI_LABEL keys

        Returns:
            Dataset: Training dataset read from the file
            Dataset: Test dataset read from the file
        """
        if (arguments is None or
                not arguments.has_key("TRAIN_FILE") or
                not arguments.has_key("TEST_FILE") or
                not arguments.has_key("IS_MULTI_LABEL")):
            raise ValueError("Invalid arguments")

        train_file_name = arguments.get("TRAIN_FILE")
        test_file_name = arguments.get("TEST_FILE")
        is_multi_label = arguments.get("IS_MULTI_LABEL")

        if train_file_name is None or test_file_name is None or is_multi_label is None:
            raise ValueError("Invalid arguments")

        training_data = Input.input_data_set(train_file_name, is_multi_label)

        arguments.set("DATA_SIZE", training_data.get_size())
        arguments.set("ATTRIBUTE_NUMBER", training_data.get_num_dim())
        arguments.set("CLASS_LABEL_NUMBER", training_data.get_num_classes())

        test_data = Input.input_data_set(test_file_name, is_multi_label)
        return training_data, test_data
