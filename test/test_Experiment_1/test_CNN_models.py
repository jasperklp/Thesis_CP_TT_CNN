from ...Experiment_runner import CNN_models
from ...Experiment_runner import calc_expectation
import torch


def test_MAC_and_RAM_uncomp_model():
    image = (30,40)
    model = CNN_models.uncomp_model(10,20,3,1,2,3,1)
    MACRAM_model = model.MAC_and_RAM(image)
    MACRAM_direct = calc_expectation.MAC_and_ram_estimation_2d(10,20,3,image,'uncomp',1,2,3)
    assert MACRAM_model == MACRAM_direct

def test_MAC_and_RAM_cp_model():
    image = (30,40)
    rank = 5
    model = CNN_models.cp_tensorly_model(10,20,3,rank,1,2,3,1)
    MACRAM_model = model.MAC_and_RAM(image)
    MACRAM_direct = calc_expectation.MAC_and_ram_estimation_2d(10,20,3,image,'cp',1,2,3, rank=rank)
    assert MACRAM_model == MACRAM_direct

def test_MAC_and_RAM_cp_model_reconstructed():
    image = (30,40)
    rank = 5 
    model = CNN_models.cp_tensorly_model(10,20,3,rank,1,2,3,1,implementation='reconstructed')
    MACRAM_model = model.MAC_and_RAM(image)
    MACRAM_direct = calc_expectation.MAC_and_ram_estimation_2d(10,20,3,image,'uncomp',1,2,3)
    assert MACRAM_model == MACRAM_direct

def test_MAC_and_RAM_cp_model_split():
    image = (30,40)
    rank = 5
    model = CNN_models.cp_tensorly_model(10,20,3,rank,1,2,3,1)
    MACRAM_model = model.MAC_and_RAM(image,output_total=False, output_in_bytes=True)
    MACRAM_direct = calc_expectation.MAC_and_ram_estimation_2d(10,20,3,image,'cp',1,2,3, rank=rank,output_total=False, output_in_bytes=True)
    assert MACRAM_model == MACRAM_direct