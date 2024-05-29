from src.main.experience_parameter import ExperienceParameter
from src.data.dataset import Dataset
from src.data.pattern import Pattern
from src.data.attribute_vector import AttributeVector
from src.fuzzy.rule.consequent.classLabel.class_label_multi import ClassLabelMulti
from src.fuzzy.rule.consequent.classLabel.class_label_basic import ClassLabelBasic
from src.data.dataset_manager import DataSetManager
from src.main.consts import Consts
import numpy as np
import csv


class Input:
    @staticmethod
    def input_data_set(file_name):
        params = ExperienceParameter.get_instance()
        if params.get_class_label_type() == ExperienceParameter.ClassLabelType.SINGLE:
            return Input.input_data_set_multi(file_name)
        else:
            return Input.input_data_set_basic(file_name)

    @staticmethod
    def input_data_set_multi(file_name):
        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            dataset = Dataset(int(header[0]), int(header[1]), int(header[2]))

            id = 0
            for row in reader:
                vector = np.zeros(dataset.get_n_dim(), dtype=np.float_)
                c_vector = np.zeros(dataset.get_c_num(), dtype=np.int_)

                for i in range(len(vector)):
                    vector[i] = float(row[i])

                j = dataset.get_n_dim()
                for i in range(len(c_vector)):
                    c_vector[i] = int(row[j])
                    j += 1

                in_vector = AttributeVector(vector)
                class_labels = ClassLabelMulti(c_vector)

                dataset.add_pattern(Pattern(id, in_vector, class_labels))
                id += 1

        return dataset

    @staticmethod
    def input_data_set_basic(file_name):
        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            dataset = Dataset(int(header[0]), int(header[1]), int(header[2]))

            id = 0
            for row in reader:
                vector = np.zeros(dataset.get_n_dim(), dtype=np.float_)

                for i in range(len(vector)):
                    vector[i] = float(row[i])

                in_vector = AttributeVector(vector)
                class_label = ClassLabelBasic(int(row[dataset.get_n_dim()]))

                dataset.add_pattern(Pattern(id, in_vector, class_label))
                id += 1

        return dataset

    @staticmethod
    def load_train_test_files(train_file_name, test_file_name):
        manager = DataSetManager.get_instance()

        training_data = Input.input_data_set(train_file_name)
        manager.add_train(training_data)

        Consts.DATA_SIZE = training_data.get_size()
        Consts.ATTRIBUTE_NUMBER = training_data.get_n_dim()
        Consts.CLASS_LABEL_NUMBER = training_data.get_c_num()

        test_data = Input.input_data_set(test_file_name)
        manager.add_test(test_data)
