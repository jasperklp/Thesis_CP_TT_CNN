import experiment_runner as runner
import experiment_helper_functions as helper
import CNN_models
import torch
import json

model_1 = CNN_models.uncomp_model(256,512,3,dtype=torch.float32)
model_2 = CNN_models.cp_tensorly_model(256,512,3,rank=0.1,dtype=torch.float32)

image_size_uncomp   = 40
image_size_cp       = 40





#def compare_found_and_theory(RAM_theory, RAM_found):




if __name__ == "__main__":
    time,profiler_values_1 = runner.model_runner(model_1, 1 ,40, verbose=True)
    time,profiler_values_2 = runner.model_runner(model_2, 1, 40, verbose=True)

    [MAC, RAM] = model_1.MAC_and_RAM(40,output_in_bytes=True, output_total=False)
    #RAM[1] = sum(RAM[1])
    RAM = helper.print_RAM(RAM)
    print(f"RAM uncomp_model = {RAM}")

    [MAC, RAM] = model_2.MAC_and_RAM(40,output_in_bytes=True, output_total=False)
    #RAM[1] = sum(RAM[1])
    RAM = helper.print_RAM(RAM)
    print(f"RAM CP_model = {RAM}")

    
    
    with open("trace_CP_tensorly.json") as json_file:
        data =  json.load(json_file)
        helper.json_get_memory_changes_per_model_ref(data)


