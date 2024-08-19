import copy

import numpy as np
import pytest

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic


def test_label_value_none():
    with pytest.raises(TypeError):
        _ = ClassLabelBasic(None)


def test_label_value_valid():
    label = ClassLabelBasic(0)
    assert label.get_class_label_value() == 0


def test_label_value_float():
    label = ClassLabelBasic(1.0)
    assert label.get_class_label_value() == 1


def test_set_label_value_none():
    label = ClassLabelBasic(0)
    with pytest.raises(TypeError):
        label.set_class_label_value(None)


def test_set_label_value_valid():
    label = ClassLabelBasic(0)
    label.set_class_label_value(1)
    assert label.get_class_label_value() == 1


def test_set_label_value_float():
    label = ClassLabelBasic(0)
    label.set_class_label_value(1.0)
    assert label.get_class_label_value() == 1


def test_rejected():
    label = ClassLabelBasic(0)
    before = label.is_rejected()
    label.set_rejected()
    assert before is False and label.is_rejected() is True


def test_deep_copy():
    label = ClassLabelBasic(0)
    label_copy = copy.deepcopy(label)

    v1 = label.get_class_label_value()
    v2 = label_copy.get_class_label_value()
    assert v1 == v2


def test_eq_true():
    label1 = ClassLabelBasic(0)
    label2 = ClassLabelBasic(0)
    assert label1 == label2


def test_eq_false():
    label1 = ClassLabelBasic(0)
    label2 = ClassLabelBasic(1)
    assert label1 != label2

