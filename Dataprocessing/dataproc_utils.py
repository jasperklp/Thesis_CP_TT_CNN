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

def preprocess_measurement_data(read_file, folder, measurement_variable, measurement_variable2 = None):
    """
        This puts all measurement data in a fixed sturcture.

    Args:
        read_file:  This is the filename of the JSON measurement file
        folder:     This is the folder in which the data file is located. Note thate this always gets appended with ~/*Project_folder*\\data\\data
        measurement_variable:   The name of the variable which should be iterated over
        measurement_variable2   The name of the second variable which could be iterated over
    Returns:
        results:                Numpy array of the data with structure [i][j][k]([l])
                                    i   = 0 For Measured data, 1 for expected data, which is also extracted form the measurement file
                                    j   = The measurement rank parameter, [0] is reserved for the model which is not compressed. The the rest is e.g. [0.01, 0.05, 0.1 ect.
                                    k   = The first iterable measurment parameter i.e. if the measurement is iterated over [1,2,4.8] Then k represent this index
                                    l   = Same as with k. Note that this is only present in the case a measurement paramteter is inserted.
        measurment_parameters:  This will return a measurement struct which is retrieved from the JSON file
        model_types:            This will returned the parameters which are iterated over i.e. uncomp and different ranks for CP
    Raises
    """
    with open(f"{os.getcwd()}\\data\\data\\{folder}\\{read_file}.json") as json_file:
        data = json.load(json_file)

    print(f"Data is deterministic is {verify_if_measurements_are_detministic(data["outcomes"], False)}")

    #Re obtain the measurement class to know which elements are present
    measurement_parameters = measurement.from_dict(data["Setup_data"])
    model_types = ["uncomp"] + measurement_parameters.rank

    if measurement_variable2 == None:
        results = np.zeros((2,len(model_types), len(getattr(measurement_parameters, measurement_variable))),int)
    else:
        results = np.zeros((2,len(model_types), len(getattr(measurement_parameters, measurement_variable)), len(getattr(measurement_parameters, measurement_variable2))),int)
   
    #Sort all images for plotting
    for dif_parameter_result in data["outcomes"]: 
        first_index = None
        second_index = None
        #First get the first index which is the model type
        if dif_parameter_result["model_type"] == "uncomp":
            first_index = 0
        else:
            first_index = model_types.index(dif_parameter_result["rank"])

        #Get second index, which is the index that is looped over during measurment.
        datapoint = dif_parameter_result[f"{measurement_variable}"]
        datapoint = datapoint[0] if isinstance(datapoint,list) else datapoint
        second_index = getattr(measurement_parameters, measurement_variable).index(datapoint)

        if measurement_variable2 == None:
            if first_index | second_index == None:
                raise ValueError(f"Not all indices are valid. First index = {first_index} and second index = {second_index}")
            
            results[0,first_index,second_index] = dif_parameter_result["measurements"][0]["Total allocated RAM"] #As all data points are deterministic and do take the same values. We can just take the first measurement
            results[1,first_index,second_index] = dif_parameter_result["Expected RAM total"]
        else:
            third_index = None
            datapoint2 = dif_parameter_result[f"{measurement_variable2}"]
            datapoint2 = datapoint2[0] if isinstance(datapoint2,list) else datapoint2
            third_index = getattr(measurement_parameters, measurement_variable2).index(datapoint2)

            if first_index | second_index | third_index == None:
                raise ValueError(f"Not all indices are valid. First index = {first_index} and second index = {second_index}")
            
            results[0,first_index,second_index,third_index] = dif_parameter_result["measurements"][0]["Total allocated RAM"] #As all data points are deterministic and do take the same values. We can just take the first measurement
            results[1,first_index,second_index,third_index] = dif_parameter_result["Expected RAM total"]

    return (results, measurement_parameters, model_types)


def check_to_list(measurement_variable, measurement_range):
    if measurement_variable in {"image_size", "kernel_size", "padding", "stride", "dilation"}:
        return [[i, i] if not isinstance(i,list) else i for i in measurement_range ]
    else:
        return measurement_range


def get_model_types(data, filter_ranks = None, filter_models = None) : 
    """
        Gets the different model types form a JSON file

        Args:
            data    :   data from a JSON file
        Returns:
            model_types : List of "uncomp" or "decomposition + rank"

    """

    # Get measurement dictionary
    measurement_parameters = measurement.from_dict(data["Setup_data"])

    #If no ranks are given, all ranks that are in the measurement data are valid.
    if filter_ranks is None:
        used_ranks = measurement_parameters.rank
    #Else only the ranks that are in the input and in the measurement data
    else:
        used_ranks = list(set(measurement_parameters.rank) & set(filter_ranks))

    #Same for models
    if filter_models is None:
        used_models = measurement_parameters.models
    else:
        used_models = list(set(measurement_parameters.models) & set(filter_models))

    
    print(measurement_parameters.models)
    #Determine number of different measurements
    nr_of_models = len(used_ranks) * len(used_models) 
    model_types = [f"{model} {rank}" for rank in used_ranks for model in used_models if model not in "uncomp"]
    if "uncomp" in used_models:
        nr_of_models = nr_of_models - len(used_ranks) + 1
        model_types = ["uncomp"] + model_types
        
    return model_types



def preprocess_time_data(read_file, folder, measurement_variable = None, measurement_variable2 = None, iterator_routine = None, used_ranks = None, used_models = None):
    """"
        Processes measurment data

        Processes measurment data in a numpy format. It accepts multiple decomposed types.

        Args:


        Returns:
            results     :   3(4)-D Numpy array conatining all results (datanr , modelnr, first indexnr (,2nd index nr))
                                datanr (1 : Inference duration, 2: Total allocated RAM, 3: Expected MAC total)

            model_types : An array with model types
    """

    with open(f"{os.getcwd()}\\data\\data\\{folder}\\{read_file}.json") as json_file:
        data = json.load(json_file)


    model_types = get_model_types(data, used_ranks, used_models) 
    nr_of_models = len(model_types)
    
    measurement_parameters = measurement.from_dict(data["Setup_data"])

    #Determine number of total tests
    nr_of_tests = measurement_parameters.amount_of_measurements(iterator_routine)

    #Determine the measurment variables
    measurement_range = getattr(measurement_parameters, measurement_variable)
    measurement_range = check_to_list(measurement_variable, measurement_range)
    if measurement_variable2 is not None:
        measurement_range2 = getattr(measurement_parameters,measurement_variable2)
        measurement_range2 = check_to_list(measurement_variable2, measurement_range2)
    print(measurement_range)
    
    if measurement_variable2 is None:
        results = np.zeros((3, nr_of_models, len(measurement_range)))
    else:
        results = np.zeros((3, nr_of_models, len(measurement_range), len(measurement_range2)))

    for test in data["outcomes"]:
        # Exception is used when a setup is in data, but will not be in plot.
        try:
            try:
                modelnr = model_types.index(f"{test["model_type"]} {test["rank"]}")
            except KeyError:
                modelnr = model_types.index(f"{test["model_type"]}")
        except ValueError:
            continue

        if measurement_variable2 is None:
            results[0, modelnr, measurement_range.index(test[f"{measurement_variable}"])] = statistics.mean(test["Inference duration"])
            results[1, modelnr, measurement_range.index(test[f"{measurement_variable}"])] = test["measurements"][0]["Total allocated RAM"]
            results[2, modelnr, measurement_range.index(test[f"{measurement_variable}"])] = test["Expected MAC Total"]
        elif measurement_variable2 is not None:
            results[0, modelnr, measurement_range.index(test[f"{measurement_variable}"]), measurement_range2.index(test[f"{measurement_variable2}"])] = statistics.mean(test["Inference duration"])
            results[1, modelnr, measurement_range.index(test[f"{measurement_variable}"]), measurement_range2.index(test[f"{measurement_variable2}"])] = test["measurements"][0]["Total allocated RAM"]
            results[2, modelnr, measurement_range.index(test[f"{measurement_variable}"]), measurement_range2.index(test[f"{measurement_variable2}"])] = test["Expected MAC total"]

    return results, model_types

        






    
