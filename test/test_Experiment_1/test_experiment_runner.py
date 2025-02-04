from ...Experiment_runner import experiment_runner as runner
from ...Experiment_runner import CNN_models

#Do forall tests a simple short run s.t. it can be verified that all functionallity works in the end :)
def test_for_uncomp_model():
    model_1 = CNN_models.uncomp_model(16,16,3)
    runner.model_runner(model_1, 10, 4, verbose=True)

def test_for_cp_model():
    model_1 = CNN_models.cp_tensorly_model(16,16,3, rank = 0.5)
    MAC,RAM = model_1.MAC_and_RAM(4)
    runner.model_runner(model_1, 10, 4, verbose=True)

def test_for_GIL_model():
    model_1 = CNN_models.cp_GIL_model(16,16,3, rank = 0.5)
    MAC,RAM = model_1.MAC_and_RAM(4)
    runner.model_runner(model_1, 10, 4, verbose=True)

def test_for_tt_model():
    model_1 = CNN_models.tt_tensorly_model(16,16,3, rank = (1,2,3,4,1))
    MAC,RAM = model_1.MAC_and_RAM(4)
    runner.model_runner(model_1,10,4,verbose=True)