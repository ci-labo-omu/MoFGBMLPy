import copy

from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.main.arguments import Arguments
from util import get_a0_0_iris_train_test

train, _ = get_a0_0_iris_train_test()


def test_deep_copy():

    # Just check if it raises an exception
    obj = LearningBasic(train)
    _ = copy.deepcopy(obj)

    assert True