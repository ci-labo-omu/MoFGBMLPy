import copy

import numpy as np
import pytest
from mofgbmlpy.data.class_label.class_label_multi import ClassLabelMulti


def test_label_value_none():
    label = ClassLabelMulti(None)
    assert label.get_class_label_value().shape[0] == 0


def test_label_value_valid():
    label_value = np.array([0, 1, 2], dtype=int)
    label = ClassLabelMulti(label_value)
    assert np.array_equal(label.get_class_label_value(), label_value)


def test_label_value_empty():
    label_value = np.array([], dtype=int)
    label = ClassLabelMulti(label_value)
    assert np.array_equal(label.get_class_label_value(), label_value) and label.get_length() == 0


def test_invalid_label_value_list():
    label_value = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _ = ClassLabelMulti(label_value)


def test_invalid_label_value_float():
    label_value = [0, 1, 2]
    with pytest.raises(TypeError):
        _ = ClassLabelMulti(label_value)


def test_set_label_value_none():
    label = ClassLabelMulti(np.array([0, 1, 2], dtype=int))
    with pytest.raises(TypeError):
        label.set_class_label_value(None)


def test_set_label_value_valid():
    label = ClassLabelMulti(np.array([0, 1, 2], dtype=int))
    label_value = np.array([0, 1, 2], dtype=int)
    label.set_class_label_value(label_value)
    assert np.array_equal(label.get_class_label_value(), label_value)


def test_set_label_value_empty():
    label = ClassLabelMulti(np.array([0, 1, 2], dtype=int))
    label_value = np.array([], dtype=int)
    label.set_class_label_value(label_value)
    assert np.array_equal(label.get_class_label_value(), label_value) and label.get_length() == 0


def test_set_invalid_label_value_list():
    label = ClassLabelMulti(np.array([0, 1, 2], dtype=int))
    label_value = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        label.set_class_label_value(label_value)


def test_rejected():
    label = ClassLabelMulti(np.array([0, 1, 2], dtype=int))
    before = label.is_rejected()
    label.set_rejected()
    assert before is False and label.is_rejected() is True


def test_get_length_full():
    label = ClassLabelMulti(np.array([0, 1, 2], dtype=int))
    assert label.get_length() == 3


def test_get_length_empty():
    label = ClassLabelMulti(np.array([], dtype=int))
    assert label.get_length() == 0


def test_get_length_none():
    label = ClassLabelMulti(None)
    assert label.get_length() == 0


def test_deep_copy():
    label = ClassLabelMulti(np.array([0, 1, 2], dtype=int))
    label_copy = copy.deepcopy(label)

    v1 = label.get_class_label_value()
    v2 = label_copy.get_class_label_value()
    
    assert np.array_equal(v1, v2)
    assert id(v1) != id(v2)

    label.set_class_label_value_at(0, -1)
    v1_modified = label.get_class_label_value_at(0)
    v2_after_modification = label_copy.get_class_label_value_at(0)

    assert v1_modified == -1
    assert v2_after_modification == 0
    



def test_eq_none():
    label1 = ClassLabelMulti(None)
    label2 = ClassLabelMulti(None)
    assert label1 == label2


def test_eq_true():
    label1 = ClassLabelMulti(np.array([0, 1, 2], dtype=int))
    label2 = ClassLabelMulti(np.array([0, 1, 2], dtype=int))
    assert label1 == label2


def test_eq_false_different_length():
    label1 = ClassLabelMulti(np.array([0, 1, 2], dtype=int))
    label2 = ClassLabelMulti(np.array([0, 2], dtype=int))
    assert label1 != label2


def test_eq_false_same_length():
    label1 = ClassLabelMulti(np.array([0, 1, 2], dtype=int))
    label2 = ClassLabelMulti(np.array([0, 2, 1], dtype=int))
    assert label1 != label2
