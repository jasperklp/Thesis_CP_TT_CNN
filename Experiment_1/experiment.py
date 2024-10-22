import experiment_runner as runner
import experiment_helper_functions as helper
import CNN_models
import torch
import json
import datetime
import os

model_1 = CNN_models.uncomp_model(256,512,3)
model_2 = CNN_models.cp_tensorly_model(256,512,3,rank=0.1)
model_3 = CNN_models.cp_GIL_model(256,512,3,rank=0.1)

image_size_uncomp   = 40
image_size_cp       = 40

experiment_outputs = []

if __name__ == "__main__":
    [start_date,start_time] = f"{datetime.datetime.now()}".split()
    start_time = start_time.replace(":",".")

    experiment_outputs.append(runner.model_runner(model_2, 1, 40, verbose=True))
    

    outfile =  open(f"{os.getcwd()}\\data\\data\\test_experiment\\{start_date}_{start_time}.json", "w")
    json.dump(experiment_outputs,outfile, indent=6)
    outfile.close()
        
    

  
    


