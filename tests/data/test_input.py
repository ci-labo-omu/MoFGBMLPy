import os

import numpy as np
import pytest

from mofgbmlpy.data.input import Input
from mofgbmlpy.main.arguments.arguments import Arguments
from util import get_datasets


datasets = get_datasets()

# datasets_multi = datasets[""]
# del datasets[""]
datasets = np.concatenate(list(datasets.values()))
#
# #
# @pytest.mark.parametrize("file_name", datasets_multi)
# def test_input_data_set_multi(file_name):
#     _ = Input.input_data_set(file_name, True)
#     assert True
#
#
# @pytest.mark.parametrize("file_name", datasets)
# def test_input_data_set_basic(file_name):
#     _ = Input.input_data_set(file_name, False)
#     assert True


def test_get_train_test_files_none_args():
    with pytest.raises(ValueError):
        _ = Input.get_train_test_files(None)


def test_get_train_test_files_none_train_file():
    args = Arguments()
    args.set("TRAIN_FILE", None)
    args.set("TEST_FILE", datasets[0])
    args.set("IS_MULTI_LABEL", False)
    with pytest.raises(ValueError):
        _ = Input.get_train_test_files(args)


def test_get_train_test_files_unknown_train_file():
    args = Arguments()
    args.set("TRAIN_FILE", "InexistingFile@=+")
    args.set("TEST_FILE", datasets[0])
    args.set("IS_MULTI_LABEL", False)
    with pytest.raises(Exception):
        _ = Input.get_train_test_files(args)


def test_get_train_test_files_none_test_file():
    args = Arguments()
    args.set("TRAIN_FILE", datasets[0])
    args.set("TEST_FILE", datasets[0])
    args.set("IS_MULTI_LABEL", None)
    with pytest.raises(ValueError):
        _ = Input.get_train_test_files(args)


def test_get_train_test_files_unknown_test_file():
    args = Arguments()
    args.set("TRAIN_FILE", datasets[0])
    args.set("TEST_FILE", "InexistingFile@=+")
    args.set("IS_MULTI_LABEL", False)
    with pytest.raises(Exception):
        _ = Input.get_train_test_files(args)


def test_get_train_test_files_none_is_multi_label():
    args = Arguments()
    args.set("TRAIN_FILE", datasets[0])
    args.set("TEST_FILE", datasets[0])
    args.set("IS_MULTI_LABEL", None)
    with pytest.raises(ValueError):
        _ = Input.get_train_test_files(args)
