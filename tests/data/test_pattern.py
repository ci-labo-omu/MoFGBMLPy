import copy

import numpy as np
import pytest

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.class_label.class_label_multi import ClassLabelMulti
from mofgbmlpy.data.input import Input
from mofgbmlpy.data.pattern import Pattern


def test_negative_id():
    with pytest.raises(ValueError):
        _ = Pattern(-1, np.array([1.0, 2.0, 3.0], dtype=np.float64), ClassLabelBasic(0))


def test_none_id():
    with pytest.raises(Exception):
        _ = Pattern(None, np.array([1.0, 2.0, 3.0], dtype=np.float64), ClassLabelBasic(0))


def test_none_attribute_vector():
    _ = Pattern(0, None, ClassLabelBasic(0))


def test_empty_attribute_vector():
    _ = Pattern(0, np.array([], dtype=np.float64), ClassLabelBasic(0))
    assert True


def test_none_target_class():
    with pytest.raises(Exception):
        _ = Pattern(0, np.array([1.0, 2.0, 3.0], dtype=np.float64), None)


def test_get_attribute_value_negative_index():
    p = Pattern(0, np.array([1.0, 2.0, 3.0], dtype=np.float64), ClassLabelBasic(0))
    with pytest.raises(IndexError):
        p.get_attribute_value(-4)  # Here e.g. -1 is accepted (it's 3.0)


def test_get_attribute_value_too_big_index():
    p = Pattern(0, np.array([1.0, 2.0, 3.0], dtype=np.float64), ClassLabelBasic(0))
    with pytest.raises(IndexError):
        p.get_attribute_value(3)


def test_get_num_dim():
    p = Pattern(0, np.array([1.0, 2.0, 3.0], dtype=np.float64), ClassLabelBasic(0))
    assert p.get_num_dim() == 3


def test_deep_copy():
    p = Pattern(0, np.array([0.0, 1.0, 2.0], dtype=np.float64), ClassLabelBasic(0))
    p_copy = copy.deepcopy(p)

    assert p == p_copy
    assert id(p) != id(p_copy)

    old_val = p.get_attribute_value(0)
    new_val = -1.0
    p.set_attribute_value(0, new_val)

    assert p.get_attribute_value(0) == np.float64(new_val)
    assert p_copy.get_attribute_value(0) == np.float64(old_val)


def test_eq_true():
    p1 = Pattern(0, np.array([0.0, 1.0, 2.0], dtype=np.float64), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([0.0, 1.0, 2.0], dtype=np.float64), ClassLabelBasic(0))

    # assert p1.get_id() == p2.get_id()
    # assert p1.get_target_class() == p2.get_target_class()
    # assert np.array_equal(p1.get_attributes_vector(), p2.get_attributes_vector())

    assert p1 == p2


def test_eq_different_patterns_order():
    p1 = Pattern(0, np.array([0.0, 1.0, 1.0], dtype=np.float64), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([0.0, 2.0, 2.0], dtype=np.float64), ClassLabelBasic(0))

    assert p1 != p2


def test_eq_different_id():
    vector = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    label = ClassLabelBasic(0)
    p1 = Pattern(0, vector, label)
    p2 = Pattern(1, vector, label)

    assert p1 != p2


def test_eq_different_target_class_same_type():
    vector = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    p1 = Pattern(1, vector, ClassLabelBasic(0))
    p2 = Pattern(2, vector, ClassLabelBasic(1))

    assert p1 != p2
    # sometimes: E       assert [id:0, input:{0.000000, 1.000000, 2.000000}, Class:1] != [id:0, input:{0.000000, 1.000000, 2.000000}, Class:1]


def test_eq_different_target_class_different_type():
    vector = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    p1 = Pattern(0, vector, ClassLabelBasic(0))
    p2 = Pattern(0, vector, ClassLabelMulti(np.array([0, 1], dtype=int)))

    assert p1 != p2


def test_eq_different_vector_same_size():
    p1 = Pattern(0, np.array([0.0, 1.0, 2.0], dtype=np.float64), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([0.0, 1.0, 1.0], dtype=np.float64), ClassLabelBasic(0))

    assert p1 != p2


def test_eq_different_vector_different_size():
    p1 = Pattern(0, np.array([0.0, 1.0, 2.0], dtype=np.float64), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float64), ClassLabelBasic(0))

    assert p1 != p2
