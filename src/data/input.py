from src.data.dataset import Dataset
from src.data.pattern import Pattern
from src.data.class_label.class_label_multi import ClassLabelMulti
from src.data.class_label.class_label_basic import ClassLabelBasic
from src.main.consts import Consts
import numpy as np
import csv


class Input:
    @staticmethod
    def input_data_set(file_name, is_multi_label):
        if is_multi_label:
            return Input.input_data_set_multi(file_name)
        else:
            return Input.input_data_set_basic(file_name)

    @staticmethod
    def input_data_set_multi(file_name):
        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            size, num_dim, num_classes = int(header[0]), int(header[1]), int(header[2])

            pattern_id = 0
            patterns = []
            for row in reader:
                in_vector = np.zeros(num_dim, dtype=np.float_)
                c_vector = np.zeros(num_classes, dtype=np.int_)

                for i in range(len(in_vector)):
                    in_vector[i] = float(row[i])

                j = num_dim
                for i in range(len(c_vector)):
                    c_vector[i] = int(row[j])
                    j += 1

                class_labels = ClassLabelMulti(c_vector)

                patterns.append(Pattern(pattern_id, in_vector, class_labels))
                pattern_id += 1
            dataset = Dataset(size, num_dim, num_classes, patterns)

        return dataset

    @staticmethod
    def input_data_set_basic(file_name):
        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            size, num_dim, num_classes = int(header[0]), int(header[1]), int(header[2])

            pattern_id = 0

            patterns = []

            for row in reader:
                in_vector = np.zeros(num_dim, dtype=np.float_)

                for i in range(len(in_vector)):
                    in_vector[i] = float(row[i])

                class_label = ClassLabelBasic(int(row[num_dim]))

                patterns.append(Pattern(pattern_id, in_vector, class_label))
                pattern_id += 1

        dataset = Dataset(size, num_dim, num_classes, patterns)

        return dataset

    @staticmethod
    def get_train_test_files(train_file_name, test_file_name, is_multi_label):
        # manager = DataSetManager.get_instance()

        training_data = Input.input_data_set(train_file_name, is_multi_label)
        # manager.add_train(training_data)

        Consts.DATA_SIZE = training_data.get_size()  # TODO: check if it's really changed
        Consts.ATTRIBUTE_NUMBER = training_data.get_num_dim()
        Consts.CLASS_LABEL_NUMBER = training_data.get_num_classes()

        test_data = Input.input_data_set(test_file_name, is_multi_label)
        # manager.add_test(test_data)
        return training_data, test_data
