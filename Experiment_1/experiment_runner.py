import torch
import numpy as np
import calc_expectation
import CNN_models
import time
import tqdm
import random
import copy
from torch.profiler import profile, record_function, ProfilerActivity
import os
from pathlib import Path
chosen_seed = 100



def model_runner(model, epochs : int, image_size : int|tuple, device : str = 'cpu', verbose : bool = False):
    if verbose == True:
        print(f"Running a model")

    #All paramteres will be validated.
    image_size = calc_expectation.check_int_or_tuple_of_int(image_size, "image_size")
    if not(issubclass(type(model),torch.nn.Module)):
        raise TypeError(r'Model should be an instance of a subclass of torch.nn.Module')
    
    try:
        in_channels, out_channels = model.get_in_and_out_channels()
    except: 
        raise AttributeError("Could not get in and out channel from model using get_in_and_out_channels method")
    
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
                    #with record_function("model_inference"):
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
    
    prof.export_chrome_trace(f"trace_{model.name}.json")
    #prof.export_memory_timeline(f"trace2_{model.name}.html")

    return (total_time, prof.key_averages())







