import pytest
from Experiment_1 import calc_expectation


#Test whether al input ar inserted correctly
def test_check_input_int():
    assert calc_expectation.check_int_or_list_of_int(3,'kernel_size') == [3,3]

def test_check_input_list_of_2_int():
    assert calc_expectation.check_int_or_list_of_int([3,3],'kernel_size') == [3,3]

def test_check_input_list_of_3_int():
    with pytest.raises(ValueError, match=r".* should either be a single int or a list of two ints"):
        calc_expectation.check_int_or_list_of_int([3,3,3],'kernel_size')

def test_check_input_type_float():
    with pytest.raises(ValueError, match=r".* should either be a single int or a list of two ints"):
        calc_expectation.check_int_or_list_of_int(3.0,'kernel_size')

#Check whether rank is required
def test_rank_requirement():
    with pytest.raises(ValueError, match=r".* rank cannot be None\nPlease insert a rank"):
        calc_expectation.ram_estimation_2d(10,20,[3,3],[30,40], 'cp' , 1,1,1)


# Test outcomes memory uncomp
def test_uncomp_same_input_output_image_size():
    assert calc_expectation.ram_estimation_2d(10,20,3,[30,40], 'uncomp' , 1,1,1) == 1209600

def test_uncomp_dif_input_output_image_size():
    assert calc_expectation.ram_estimation_2d(10,20,5,[30,40], 'uncomp' , 1,1,1) == 38280 * 32

def test_uncomp_no_square_kernel():
    assert calc_expectation.ram_estimation_2d(10,20,[3,5],[30,40], 'uncomp' , 1,1,1) == 37800 * 32


# Test outcomes memory CP
def test_cp_same_input_output_image_size():
    assert calc_expectation.ram_estimation_2d(10,20,3,[30,40], 'cp', 1,1,1,rank = 5) == 54180 * 32 # nr of elements times number of bits.

def test_cp_dif_input_output_image_size():
    assert calc_expectation.ram_estimation_2d(10,20,5,[30,40], 'cp', 1,1,1,rank = 5) == 50500 * 32

def test_cp_no_square_kernel():
    assert calc_expectation.ram_estimation_2d(10,20,[3,5],[30,40], 'cp' , 1,1,1,rank = 5) == 52390 * 32
    

# Test outcomes MAC 
def test_MAC_uncomp_same_input_output_image_size():
    assert calc_expectation.MAC_estimation_2d(10,20,3,[30,40], 'uncomp' , 1,1,1) == 3 * 3 * 10 * 20 * 30 * 40

def test_MAC_uncomp_dif_input_output_image_size():
    assert calc_expectation.MAC_estimation_2d(10,20,5,[30,40], 'uncomp' , 1,1,1) == 5 * 5 * 10 * 20 * 28 * 38

def test_MAC_uncomp_no_square_kernel():
    assert calc_expectation.MAC_estimation_2d(10,20,[3,5],[30,40], 'uncomp' , 1,1,1) == 3 * 5 * 10 * 20 * 30 * 38 

# Test outcomes MAC CP
def test_MAC_cp_same_input_output_image_size():
    assert calc_expectation.MAC_estimation_2d(10,20,3,[30,40], 'cp', 1,1,1,rank = 5) ==  216000
def test_MAC_cp_diff_input_output_image_size():
    assert calc_expectation.MAC_estimation_2d(10,20,5,[30,40], 'cp', 1,1,1,rank = 5) == 221500
def test_MAC_cp_non_square_kernel():
    assert calc_expectation.MAC_estimation_2d(10,20,[3,5],[30,40], 'cp' , 1,1,1,rank = 5) == 219600



