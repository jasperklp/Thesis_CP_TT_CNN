from ...Experiment_1 import calc_expectation
import pytest


def test_check_list_or_int_float_int():
    assert calc_expectation.check_list_or_int_float(1) == [1]

def test_check_list_or_int_float_float():
    assert calc_expectation.check_list_or_int_float(1.2) == [1.2]

def test_check_list_or_int_float_list():
    assert calc_expectation.check_list_or_int_float([1,1]) == [1,1]

def test_check_list_or_int_float_tuple():
    with pytest.raises(ValueError):
        calc_expectation.check_list_or_int_float((1,2))

def test_check_list_or_int_float_str():
    with pytest.raises(ValueError):
        calc_expectation.check_list_or_int_float('a')