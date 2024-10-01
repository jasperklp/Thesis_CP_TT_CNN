from Experiment_1 import exp1
import pytest


def test_check_list_or_int_float_int():
    assert exp1.check_list_or_int_float(1) == [1]

def test_check_list_or_int_float_float():
    assert exp1.check_list_or_int_float(1.2) == [1.2]

def test_check_list_or_int_float_list():
    assert exp1.check_list_or_int_float([1,1]) == [1,1]

def test_check_list_or_int_float_tuple():
    with pytest.raises(ValueError):
        exp1.check_list_or_int_float((1,2))

def test_check_list_or_int_float_str():
    with pytest.raises(ValueError):
        exp1.check_list_or_int_float('a')