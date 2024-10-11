from ...Experiment_1 import experiment_helper_functions as helper
import pytest

def test_name_number_type():
    with pytest.raises(TypeError,match=r"Number should have type int, but it has type .*"):
        helper.name_number(1.1)

def test_name_number_val_size():
    assert helper.name_number(1395864371) == "1.30Gb"

def test_name_number_size():
    assert helper.name_number(1395864371,False, True) == "Gb"

def test_name_number_val():
    assert helper.name_number(1395864371,True, False) == "1.30"

def test_name_number_no_output():
    with pytest.raises(ValueError):
        helper.name_number(1395864371, False, False)