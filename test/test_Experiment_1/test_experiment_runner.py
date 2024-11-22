from ...Experiment_1 import experiment_runner as runner
from ...Experiment_1 import CNN_models

#Do forall tests a simple short run s.t. it can be verified that all functionallity works in the end :)
def test_for_uncomp_model():
    model_1 = CNN_models.uncomp_model(256,256,3)
    runner.model_runner(model_1, 100, 4, verbose=True)

def test_for_cp_model():
    model_1 = CNN_models.cp_tensorly_model(256,256,3, rank = 0.5)
    MAC,RAM = model_1.MAC_and_RAM(4)
    runner.model_runner(model_1, 100, 4, verbose=True)

def test_for_GIL_model():
    model_1 = CNN_models.cp_GIL_model(256,256,3, rank = 0.5)
    MAC,RAM = model_1.MAC_and_RAM(4)
    runner.model_runner(model_1, 100, 4, verbose=True)
