import torch
import numpy as np
import calc_expectation
import CNN_models
import time
import datetime
import tqdm
import random
import copy
from torch.profiler import profile, record_function, ProfilerActivity
import os
import json
from pathlib import Path
import experiment_helper_functions as helper
chosen_seed = 100



def model_runner(model, epochs : int, image_size : int|tuple, device : str = 'cpu', verbose : bool = False):
    [start_date,start_time] = f"{datetime.datetime.now()}".split()
    start_time = start_time.replace(":",".")
    
    if verbose == True:
        print(f"Running a model")

    #All paramteres will be validated.
    image_size = calc_expectation.check_int_or_tuple_of_int(image_size, "image_size")
    if not(issubclass(type(model),torch.nn.Module)):
        raise TypeError(r'Model should be an instance of a subclass of torch.nn.Module')
    
    try:
        model_information = model.get_output_data()
        in_channels = model_information["in_channels"]
    except: 
        raise AttributeError("Could not get model information")
    
    if (verbose == True):
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print("\t",param_tensor, "\t", model.state_dict()[param_tensor].size())

    if not(isinstance(device, str)):
        raise TypeError("Device type is not a string")
    else:  
        torch.device(device)
    total_time = 0

    #Ensure the code is reproducible
    torch.manual_seed(chosen_seed)
    random.seed(chosen_seed)
    np.random.seed(chosen_seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic == False
    torch.use_deterministic_algorithms(True)

    if verbose == True:
        print("Start experiment")

    with profile(activities=[ProfilerActivity.CPU], 
                 profile_memory=True,
                 record_shapes=False, 
                 with_stack=False,
                 #on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model.name}'),
                 #with_flops=True
                 with_modules=False
                 ) as prof:

            for i in tqdm.tqdm(range(epochs)):
                with record_function("Input_image"):
                    input = torch.randn(1,in_channels, image_size[0], image_size[1],dtype=torch.float32)
                    print(input.nbytes)
                with record_function("model_size"):
                    model_test = copy.deepcopy(model)
                with torch.no_grad():
                    start = time.time()
                    output = model_test(input)
                    end =   time.time()

                # with record_function("Output_image"):
                #     output_test = copy.deepcopy(output)

                wall_time = end - start

                total_time += wall_time
                prof.step()

    if verbose == True:
        print(prof.key_averages().table(sort_by="cpu_memory_usage"))
    
    print(start_time)
    tracefile = f"{os.getcwd()}\\data\\data_raw\\{start_date}_{start_time}_{model.name}.json"
    prof.export_chrome_trace(tracefile)

    [end_date,end_time] = f"{datetime.datetime.now()}".split()
    end_time = end_time.replace(":",".")


    

    #Create model output
    output = {
        "model_name"    : model.name,
        "model_type"    : model.model_type,
        "in_channels"   : model_information.get("in_channels"),
        "out_channels"  : model_information.get("out_channels"),
        "kernel_size"   : model_information.get("kernel_size"),
        "stride"        : model_information.get("stride"),
        "padding"       : model_information.get("padding"),
        "image_size"    : image_size,

        "measurement_start_time"    : [start_date, start_time],
        "measurement_end_time"   	: [end_date, end_time],

        "Inference duration" : total_time        
    }

    [expected_MAC, expected_RAM] = model.MAC_and_RAM(image_size, True, False)
    [expected_MAC_total, expected_RAM_total] = model.MAC_and_RAM(image_size, True, True)

    output["Expected MAC"] = expected_MAC
    output["Expected MAC total"] = expected_MAC_total
    output["Expected RAM"] = expected_RAM
    output["Expected RAM total"] = expected_RAM_total

    if output.get("modeltype") == "cp":
        output["rank"]      = model_information.get("rank")
        output["rank_int"]  = model_information.get("rank_int")
    
    json_file = open(tracefile)
    data = json.load(json_file)
    events = data["traceEvents"]
    (output["Peak allocated RAM"],output["Total allocated RAM"]) = helper.get_peak_and_total_alloc_memory(events)
    output["Filter per model"] = helper.json_get_memory_changes_per_model_ref(data,True)
    json_file.close()


    return (output)







