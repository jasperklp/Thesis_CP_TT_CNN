import statistics
import json
import os
import numpy as np
import sys

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Experiment_1.experiment_helper_functions import measurement


def get_mathplotlib_colours():
    mathplotlib_colours = ['tab:blue', 'tab:orange','tab:green', 'tab:red','tab:purple', 'tab:brown', 'tab:pink','tab:grey', 'tab:olive','tab:cyan']
    return mathplotlib_colours

def second_util():
    print(1)



def verify_if_measurements_are_detministic(outcomes_data, verbose: bool = False):
    """
        Function test if all memory transactions and their function names are the same for each measurement

    Args:
        outcomes_data: outcomes of a measurment (see JSON file experiment JSON file structure)
        verbose: (default = False) if true prints for each measurement the avarage and std of the peak memory and total allocated mememory.
    Returns:
        bool : Returns true if for each measurement the byte transactions and names were in the same order and equally sized. Additionally, the peak memory and total allocated memory should be the same.
                If verbose = True it finishes all measurments
                If verbose = Fasle it immediatly returns upon detection.
    """
    #Assume true, if verbose is true all data will be handeld. Otherwise a quick return will be initiated.
    return_value = True

    for measurement_number, j in enumerate(outcomes_data):
        peak_RAM_data = []
        total_alloc_mem_data = []
        byte_data = []
        operation_name  = [] 
        for i in j["measurements"]:
            #Obtains a list of all peak memory values
            peak_RAM_data.append(i["Peak allocated RAM"])
            total_alloc_mem_data.append(i["Total allocated RAM"])

            #Obtain all bytes and names ordered for each list.
            byte_data.append([k.get("Bytes") for l in i["Filter per model"] for k in l["Events"] if k.get("Bytes") != None])
            operation_name.append([k.get("Operation name")  for l in i["Filter per model"] for k in l["Events"] if k.get("Bytes") != None])

        #Checks if all ordered bytes and names are the same 
        # if so it prints nothing else it prints That not all names or bytes are equal.
        for i in range(len(byte_data[0])):
            if not (all(point[i] == byte_data[0][i] for point in byte_data)):
                if verbose == True:
                    print("Not all byte values are equal.")
                    return_value = False
                else:
                    return False
            if not (all(point[i] == byte_data[0][i] for point in byte_data)):
                if verbose == True:
                    print("Not all names are equal")
                    return_value = False
                else:
                    return False
            
        #Calculates the mean of peak memory and peak RAM
        mean_peak_RAM = sum(peak_RAM_data)/len(peak_RAM_data)
        mean_total_alloc = sum(total_alloc_mem_data)/len(total_alloc_mem_data)

        #Calculates the standard deviation of peak RAM and total alloc memory if there is more then one measurement
        if len(byte_data) !=1:
            std_peak_RAM = statistics.stdev(peak_RAM_data)
            std_total_alloc = statistics.stdev(total_alloc_mem_data)

        
        if verbose == True:
            print(f"This is measurment number {measurement_number}")
            #Print out the means and standard deviation for all peak alloc data and memory
            print(f"For peak data the avarage is {mean_peak_RAM} with standard deviation {std_peak_RAM}")
            print(f"For total allocated RAM the avarage is {mean_total_alloc} with standard deviation {std_total_alloc}")
            print("") #Get empty line

        if (len(byte_data) > 1):
            if (std_peak_RAM != 0.0):
                if verbose == True:
                    print("Peak ram standard deviation is not zero")
                    return_value = False
                else:
                    return False
            if (std_total_alloc != 0.0):
                if verbose == True:
                    print("Total alloc standard deviation is not zero")
                    return_value = False
                else:
                    return False
    
    #Return if no measurment encouters problems or verbose = True
    return return_value

def preprocess_measurement_data_single(read_file, folder, measurement_variable):
    with open(f"{os.getcwd()}\\data\\data\\{folder}\\{read_file}.json") as json_file:
        data = json.load(json_file)

    print(f"Data is deterministic is {verify_if_measurements_are_detministic(data["outcomes"], False)}")

    #Re obtain the measurement class to know which elements are present
    measurement_parameters = measurement.from_dict(data["Setup_data"])

    model_types = ["uncomp"] + measurement_parameters.rank

    results = np.zeros((2,len(model_types), len(measurement_parameters.in_channel)),int)

    #Sort all images for plotting
    for dif_parameter_result in data["outcomes"]: 
        first_index = None
        second_index = None
        #First get the first index which is the model type
        if dif_parameter_result["model_type"] == "uncomp":
            first_index = 0
        else:
            first_index = model_types.index(dif_parameter_result["rank"])

        second_index = measurement_parameters.in_channel.index(dif_parameter_result[f"{measurement_variable}"])

        if first_index | second_index == None:
            raise ValueError(f"Not all indices are valid. First index = {first_index} and second index = {second_index}")
        
        results[0,first_index,second_index] = dif_parameter_result["measurements"][0]["Total allocated RAM"] #As all data points are deterministic and do take the same values. We can just take the first measurement
        results[1,first_index,second_index] = dif_parameter_result["Expected RAM total"]

    return (results, measurement_parameters, model_types)
