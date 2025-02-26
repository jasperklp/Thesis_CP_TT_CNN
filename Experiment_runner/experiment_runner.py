import sys
import os
#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import calc_expectation as calc_expectation
import CNN_models  as CNN_models
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
import logging

logger = logging.getLogger(__name__)



def model_runner(model, epochs : int, image_size : int|tuple, device : str = 'cpu', verbose : bool = False):
    [start_date,start_time] = f"{datetime.datetime.now()}".split()
    start_time = start_time.replace(":",".")
    
    if verbose == True:
        print(f"Running a model")

    logger.info(f"Started a new model at {start_date} {start_time}")
    try:
        model_information = model.get_output_data()
        in_channels = model_information["in_channels"]
    except: 
        raise AttributeError("Could not get model information")

    logger.info(f"Model is {model.name} with settings:  \
                \n \t in_channels ={model_information["in_channels"]} , \
                \n \t out_channels = {model_information["in_channels"]}, \
                \n \t kernel_size = {model_information["kernel_size"]},\
                \n \t padding = {model_information["padding"]},\
                \n \t image_size = {image_size}\n"
               )

    if model_information.get("rank") != None:
        logger.info(f"Rank = {model_information["rank"]}")
    #All paramteres will be validated.
    image_size = calc_expectation.check_int_or_tuple_of_int(image_size, "image_size")
    if not(issubclass(type(model),torch.nn.Module)):
        raise TypeError(r'Model should be an instance of a subclass of torch.nn.Module')
    
    
    
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
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic == False
    torch.use_deterministic_algorithms(True)


    if verbose == True:
        print("Start experiment")

    measurements = []
    with torch.no_grad():
        with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
            for i in range(epochs):
                with profile(activities=[ProfilerActivity.CPU], 
                    profile_memory=True,
                    record_shapes=False, 
                    with_stack=False,
                    with_modules=False
                    ) as prof:
                    chosen_seed = 100 + i 
                    torch.manual_seed(chosen_seed)
                    random.seed(chosen_seed)
                    np.random.seed(chosen_seed)
                    
                    with record_function("Input_image"):
                        input = torch.randn(1,in_channels, image_size[0], image_size[1],dtype=torch.float32)
                    with record_function("model_size"):
                        model_test = copy.deepcopy(model)
                    
                    start = time.time()
                    output = model_test(input)
                    end =   time.time()

                    wall_time = end - start

                    total_time += wall_time
                    del input
                    del model_test
                    del output

                if verbose == True:
                    print(prof.key_averages().table(sort_by="cpu_memory_usage"))
                
    
    
        tracefile = f"{os.getcwd()}\\data\\data_raw\\{start_date}_{start_time}_{model.name}.json"
        prof.export_chrome_trace(tracefile)
        measurements.append(tracefile)

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

        "Inference duration"    : total_time,
        "nr of measurements"    : epochs,
        "measurements"          : []
    }

    [expected_MAC, expected_RAM] = model.MAC_and_RAM(image_size, True, False)
    [expected_MAC_total, expected_RAM_total] = model.MAC_and_RAM(image_size, True, True)

    output["Expected MAC"] = expected_MAC
    output["Expected MAC total"] = expected_MAC_total
    output["Expected RAM"] = expected_RAM
    output["Expected RAM total"] = expected_RAM_total

    if output.get("model_type") == "cp":
        output["rank"]      = model_information.get("rank")
        output["rank_int"]  = model_information.get("rank_int")
    
    for tracefile in measurements:
        measurement = {}
        json_file = open(tracefile)
        data = json.load(json_file)
        events = data["traceEvents"]
        (measurement["Peak allocated RAM"],measurement["Total allocated RAM"]) = helper.get_peak_and_total_alloc_memory(events, verbose=verbose)
        measurement["Filter per model"] = helper.json_get_memory_changes_per_model_ref(data, verbose=verbose)
        json_file.close()
        output["measurements"].append(measurement)


    clock_date, clock_time = helper.get_date_time()
    logger.info(f"Ended model at {clock_date} {clock_time}")
    return (output)







