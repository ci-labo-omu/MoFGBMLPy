import copy

import numpy as np
import pytest

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.dataset import Dataset
from mofgbmlpy.data.pattern import Pattern


def test_negative_size():
    p = np.array([Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))], dtype=object)
    with pytest.raises(ValueError):
        Dataset(-1, 3, 1, p)


def test_null_size():
    p = np.array([Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))], dtype=object)
    with pytest.raises(ValueError):
        Dataset(0, 3, 1, p)


def test_size_different_from_pattern_length():
    p = np.array([Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))], dtype=object)
    with pytest.raises(ValueError):
        Dataset(2, 3, 1, p)


def test_negative_num_dim():
    p = np.array([Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))], dtype=object)
    with pytest.raises(ValueError):
        Dataset(1, -1, 1, p)


def test_null_num_dim():
    p = np.array([Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))], dtype=object)
    with pytest.raises(ValueError):
        Dataset(1, 0, 1, p)


def test_num_dim_different_from_pattern_dim():
    p = np.array([Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))], dtype=object)
    with pytest.raises(ValueError):
        Dataset(1, 1, 1, p)


def test_negative_num_classes():
    p = np.array([Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))], dtype=object)
    with pytest.raises(ValueError):
        Dataset(1, 1, -1, p)


def test_null_num_classes():
    p = np.array([Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))], dtype=object)
    with pytest.raises(ValueError):
        Dataset(1, 1, 0, p)


def test_none_patterns():
    with pytest.raises(TypeError):
        Dataset(1, 1, 1, None)


def test_get_pattern():
    p1 = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([3.0, 4.0, 5.0]), ClassLabelBasic(1))
    ds = Dataset(2, 3, 1, np.array([p1, p2], dtype=object))
    assert ds.get_pattern(0) == p1 and ds.get_pattern(1) == p2


def test_get_patterns():
    p1 = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([3.0, 4.0, 5.0]), ClassLabelBasic(1))
    ds = Dataset(2, 3, 1, np.array([p1, p2], dtype=object))
    patterns = ds.get_patterns()
    assert patterns[0] == p1 and patterns[1] == p2


def test_get_int_params():
    p = np.array([Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))], dtype=object)
    ds = Dataset(1, 3, 1, p)
    assert ds.get_size() == 1 and ds.get_num_dim() == 3 and ds.get_num_classes() == 1


def test_deep_copy():
    p1 = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([3.0, 4.0, 5.0]), ClassLabelBasic(1))
    ds = Dataset(2, 3, 1, np.array([p1, p2], dtype=object))
    ds_copy = copy.deepcopy(ds)

    assert ds == ds_copy


def test_eq_true():
    p1 = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))
    ds1 = Dataset(1, 3, 1, np.array([p1], dtype=object))
    ds2 = Dataset(1, 3, 1, np.array([p2], dtype=object))

    assert ds1 == ds2


def test_eq_false_different_patterns_order():
    p1 = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([0.0, 2.0, 1.0]), ClassLabelBasic(0))

    ds1 = Dataset(2, 3, 1, np.array([p1, p2], dtype=object))
    ds2 = Dataset(2, 3, 1, np.array([p2, p1], dtype=object))

    assert ds1 != ds2


def test_eq_different_patterns():
    p1 = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([0.0, 0.0, 2.0]), ClassLabelBasic(0))
    ds1 = Dataset(1, 3, 1, np.array([p1], dtype=object))
    ds2 = Dataset(1, 3, 1, np.array([p2], dtype=object))

    assert ds1 != ds2


def test_eq_different_size():
    p1 = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([0.0, 0.0, 2.0]), ClassLabelBasic(0))
    ds1 = Dataset(1, 3, 1, np.array([p1], dtype=object))
    ds2 = Dataset(2, 3, 1, np.array([p1, p2], dtype=object))

    assert ds1 != ds2


def test_eq_different_num_dim():
    p1 = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))
    p2 = Pattern(0, np.array([0.0, 0.0]), ClassLabelBasic(0))
    ds1 = Dataset(1, 3, 1, np.array([p1], dtype=object))
    ds2 = Dataset(1, 2, 1, np.array([p2], dtype=object))

    assert ds1 != ds2
