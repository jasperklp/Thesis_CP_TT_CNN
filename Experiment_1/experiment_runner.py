import torch
import numpy as np
import calc_expectation
import CNN_models
import time
import tqdm
import random
import copy
from torch.profiler import profile, record_function, ProfilerActivity
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
        raise NotImplementedError("get_in_and_out_channels method is not implemented or functioning")
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
    
    
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        
            for i in tqdm.tqdm(range(epochs)):
                input = torch.randn(in_channels, out_channels, image_size[0], image_size[1])
                with torch.no_grad():
                    with record_function("model_inference"):
                        #model_test = copy.deepcopy(model)
                        start = time.time()
                        model(input)
                        end =   time.time()

                wall_time = end - start

                total_time += wall_time

    print(prof.key_averages().table(sort_by="cpu_memory_usage"))
    return total_time






