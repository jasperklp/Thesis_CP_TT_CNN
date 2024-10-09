import pytest
from ...Experiment_1 import calc_expectation

#Test whether al input ar inserted correctly
def test_check_input_int():
    assert calc_expectation.check_int_or_tuple_of_int(3,'kernel_size') == (3,3)

def test_check_input_list_of_2_int():
    assert calc_expectation.check_int_or_tuple_of_int((3,3),'kernel_size') == (3,3)

def test_check_input_list_of_3_int():
    with pytest.raises(ValueError, match=r".* should either be a single int or a tuple of two ints"):
        calc_expectation.check_int_or_tuple_of_int((3,3,3),'kernel_size')

def test_check_input_type_float():
    with pytest.raises(ValueError, match=r".* should either be a single int or a tuple of two ints"):
        calc_expectation.check_int_or_tuple_of_int(3.2,'kernel_size')

#Check whether rank is required
def test_rank_requirement():
    with pytest.raises(ValueError, match=r".* rank cannot be None\nPlease insert a rank"):
        calc_expectation.ram_estimation_2d(10,20,(3,3),(30,40), 'cp' , 1,1,1)

def test_rank_calc_cp_int():
    [_,_,_,_,_,rank] = calc_expectation.validate_MAC_or_RAM_calc_input((3,3), (3,3),(3,3),(3,3), (30,30),'cp', rank = 5, in_channel=20,out_channel=20)
    assert  (rank == 5) & (isinstance(rank, int))

def test_rank_calc_cp_float():
    [_,_,_,_,_,rank] = calc_expectation.validate_MAC_or_RAM_calc_input((3,3), (3,3),(3,3),(3,3), (30,30),'cp', rank = 0.1, in_channel=20,out_channel=20) 
    assert  (rank == 7) & (isinstance(rank, int))

def test_rank_calc_cp_float_atleast1():
    [_,_,_,_,_,rank] = calc_expectation.validate_MAC_or_RAM_calc_input((3,3), (3,3),(3,3),(3,3), (30,30),'cp', rank = 1e-99, in_channel=20,out_channel=20)
    assert  (rank == 1) & (isinstance(rank, int))

# Test outcomes memory uncomp
def test_uncomp_same_input_output_image_size():
    assert calc_expectation.ram_estimation_2d(10,20,3,(30,40), 'uncomp' , 1,1,1) == 1209600

def test_uncomp_dif_input_output_image_size():
    assert calc_expectation.ram_estimation_2d(10,20,5,(30,40), 'uncomp' , 1,1,1) == 38280 * 32

def test_uncomp_no_square_kernel():
    assert calc_expectation.ram_estimation_2d(10,20,(3,5),(30,40), 'uncomp' , 1,1,1) == 37800 * 32


# Test outcomes memory CP
def test_cp_same_input_output_image_size():
    assert calc_expectation.ram_estimation_2d(10,20,3,(30,40), 'cp', 1,1,1,rank = 5) == 54180 * 32 # nr of elements times number of bits.

def test_cp_dif_input_output_image_size():
    assert calc_expectation.ram_estimation_2d(10,20,5,(30,40), 'cp', 1,1,1,rank = 5) == 50500 * 32

def test_cp_no_square_kernel():
    assert calc_expectation.ram_estimation_2d(10,20,(3,5),(30,40), 'cp' , 1,1,1,rank = 5) == 52390 * 32
    

# Test outcomes MAC 
def test_MAC_uncomp_same_input_output_image_size():
    assert calc_expectation.MAC_estimation_2d(10,20,3,(30,40), 'uncomp' , 1,1,1) == 3 * 3 * 10 * 20 * 30 * 40

def test_MAC_uncomp_dif_input_output_image_size():
    assert calc_expectation.MAC_estimation_2d(10,20,5,(30,40), 'uncomp' , 1,1,1) == 5 * 5 * 10 * 20 * 28 * 38

def test_MAC_uncomp_no_square_kernel():
    assert calc_expectation.MAC_estimation_2d(10,20,(3,5),(30,40), 'uncomp' , 1,1,1) == 3 * 5 * 10 * 20 * 30 * 38 

# Test outcomes MAC CP
def test_MAC_cp_same_input_output_image_size():
    assert calc_expectation.MAC_estimation_2d(10,20,3,(30,40), 'cp', 1,1,1,rank = 5) ==  216000
def test_MAC_cp_diff_input_output_image_size():
    assert calc_expectation.MAC_estimation_2d(10,20,5,(30,40), 'cp', 1,1,1,rank = 5) == 221500
def test_MAC_cp_non_square_kernel():
    assert calc_expectation.MAC_estimation_2d(10,20,(3,5),(30,40), 'cp' , 1,1,1,rank = 5) == 219600

#Test outcomes RAM split uncomp
def test_RAM_split_uncomp():
    assert calc_expectation.ram_estimation_2d(10,20,3,(30,40),'uncomp',1,1,1,output_total=False) == [384000,[57600],[0], 768000]

def test_RAM_split_cp():
    assert calc_expectation.ram_estimation_2d(10,20,3,(30,40),'cp',1,1,1,rank=5,output_total=False) == [12000*32,[50*32,15*32,15*32,100*32],[6000*32,6000*32,6000*32], 24000*32]


#Test outcomes RAM bytes
def test_RAM_in_bytes_together_uncomp():
    uncomp_bytes = calc_expectation.ram_estimation_2d(10,20,3,(30,40),'uncomp',1,1,1,output_total=True, output_in_bytes=True)
    uncomp_bits = calc_expectation.ram_estimation_2d(10,20,3,(30,40),'uncomp',1,1,1,output_total = True, output_in_bytes=False)

    assert uncomp_bytes == uncomp_bits // 8

def test_RAM_in_bytes_split_uncomp():
    uncomp_bytes = calc_expectation.ram_estimation_2d(10,20,3,(30,40),'uncomp',1,1,1,output_total=False, output_in_bytes=True)
    uncomp_bits = calc_expectation.ram_estimation_2d(10,20,3,(30,40),'uncomp',1,1,1,output_total = False, output_in_bytes=False)

    assert uncomp_bytes == [i / 8 if type(i) == int else [j/8 for j in i] for i in uncomp_bits ]

def test_RAM_in_bytes_split_cp():
    cp_bytes = calc_expectation.ram_estimation_2d(10,20,3,(30,40),'cp',1,1,1,rank=5,output_in_bytes=True,output_total=False)
    cp_bits = calc_expectation.ram_estimation_2d(10,20,3,(30,40),'cp',1,1,1,rank=5,output_total=False)

    assert cp_bytes == [i / 8 if type(i) == int else [j/8 for j in i] for i in cp_bits ]

#Test other outcomes MAC
def test_MAC_splittotal_cp():
    cp_MAC_split = calc_expectation.MAC_estimation_2d(10,20,3,(30,40), 'cp', 1,1,1,rank = 5, output_total=False)
    cp_MAC_total = calc_expectation.MAC_estimation_2d(10,20,3,(30,40), 'cp', 1,1,1,rank = 5)
    assert cp_MAC_total == sum(cp_MAC_split)

def test_MAC_split_cp():
    cp_MAC_split = calc_expectation.MAC_estimation_2d(10,20,3,(30,40), 'cp', 1,1,1,rank = 5, output_total=False)
    assert cp_MAC_split == [10*5*30*40, 3*5*30*40,3*5*30*40,20*5*30*40]

#Test combination
def test_MACRAM_uncomp_dif_no_square_kernel_input_output_image():
    assert calc_expectation.MAC_and_ram_estimation_2d(10,20,(3,7), (30,40), 'uncomp', 1, 2, 3) == [calc_expectation.MAC_estimation_2d(10,20,(3,7), (30,40), 'uncomp', 1, 2, 3), calc_expectation.ram_estimation_2d(10,20,(3,7), (30,40), 'uncomp', 1, 2, 3)]

def test_MACRAM_uncomp_dif_no_square_kernel_input_output_image():
    assert calc_expectation.MAC_and_ram_estimation_2d(10,20,(3,7), (30,40), 'cp', 1, 2, 3, rank=5,output_in_bytes=True, output_total=False) == [calc_expectation.MAC_estimation_2d(10,20,(3,7), (30,40),  'cp', 1, 2, 3, rank=5, output_total=False), calc_expectation.ram_estimation_2d(10,20,(3,7), (30,40), 'cp', 1, 2, 3, rank=5,output_in_bytes=True, output_total=False)]


