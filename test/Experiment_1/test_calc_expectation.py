import pytest
from Experiment_1 import calc_expectation

# Test outcomes
def test_uncomp():
    assert calc_expectation.ram_estimation_2d(10,20,3,[30,40], 'uncomp' , 1,1,1) == 1209600
    

#Test whether al input ar inserted correctly

def test_kernel_size_type_int():
    assert calc_expectation.ram_estimation_2d(10,20,3,[30,40], 'uncomp' , 1,1,1) == 1209600

def test_kernel_size_type_list_of_2_int():
    assert calc_expectation.ram_estimation_2d(10,20,[3,3],[30,40], 'uncomp' , 1,1,1) == 1209600

def test_kernel_size_type_list_of_3_int():
    with pytest.raises(ValueError, match=r".* should either be a single int or a list of two ints"):
        calc_expectation.ram_estimation_2d(10,20,[3,3,3],[30,40], 'uncomp' , 1,1,1)

def test_kernel_size_type_float():
    with pytest.raises(ValueError, match=r".* should either be a single int or a list of two ints"):
        calc_expectation.ram_estimation_2d(10,20,3.0,[30,40], 'uncomp' , 1,1,1)

#Check whether rank is required
def test_rank_requirement():
    with pytest.raises(ValueError, match=r".* rank cannot be None\nPlease insert a rank"):
        calc_expectation.ram_estimation_2d(10,20,[3,3],[30,40], 'cp' , 1,1,1)