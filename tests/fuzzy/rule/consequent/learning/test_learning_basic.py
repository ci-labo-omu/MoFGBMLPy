import copy

from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.main.arguments import Arguments


def get_training_set():
    args = Arguments()
    args.set("TRAIN_FILE", "../dataset/iris/a0_0_iris-10tra.dat")
    args.set("TEST_FILE", "../dataset/iris/a0_0_iris-10tra.dat")
    args.set("IS_MULTI_LABEL", False)
    train, _ = Input.get_train_test_files(args)
    return train


def test_deep_copy():
    # Just check if it raises an exception
    obj = LearningBasic(get_training_set())
    _ = copy.deepcopy(obj)

    assert True