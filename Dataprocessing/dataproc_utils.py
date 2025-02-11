import statistics
import json
import os
import numpy as np
import pandas as pd
import sys

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))



from Experiment_runner.experiment_helper_functions import measurement
from Experiment_runner.calc_expectation import ram_estimation_2d

def uncomp_alternative_name():
    return "regular"


def get_mathplotlib_colours(i = None):
    """
        Returns a color or an array of colors

        Args:
            i   : if None returns whole array of colors (Default)
                : if int returns the ith color. It does wrap around the end
    """
    mathplotlib_colours = ['tab:blue', 'tab:orange','tab:green', 'tab:red','tab:purple', 'tab:brown', 'tab:pink','tab:grey', 'tab:olive','tab:cyan', 'lightpink', 'bisque', 'yellow','lime', 'midnightblue', ]
    
    if i is not None:
        return mathplotlib_colours[i%len(mathplotlib_colours)]
        
    return mathplotlib_colours

def list_intersection(list1 : list, list2 : list):
    """
        Returns the intersection of two lists, with the ordering of the first list
    """

    return [i for i in list1 if i in list2]


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

def preprocess_measurement_data(read_file, folder, measurement_variable, measurement_variable2 = None, used_ranks = None, used_models = None):
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
    model_types = get_model_types(data, filter_models=used_models, filter_ranks=used_ranks)
    if measurement_variable2 == None:
        results = np.zeros((2,len(model_types), len(getattr(measurement_parameters, measurement_variable))),int)
    else:
        results = np.zeros((2,len(model_types), len(getattr(measurement_parameters, measurement_variable)), len(getattr(measurement_parameters, measurement_variable2))),int)
   
    #Sort all images for plotting
    for dif_parameter_result in data["outcomes"]: 
        first_index = None
        second_index = None
        #First get the first index which is the model type
        # Exception is used when a setup is in data, but will not be in plot.
        try:
            try:
                first_index = model_types.index(f"{dif_parameter_result["model_type"]} {dif_parameter_result["rank"]}")
            except KeyError:
                first_index = model_types.index(f"{dif_parameter_result["model_type"]}")
        except ValueError:
            continue

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
                        : "Default order/default accepted = ["uncomp", "cp", "tt"]

    """

    # Get measurement dictionary
    measurement_parameters = measurement.from_dict(data["Setup_data"])

    #If no ranks are given, all ranks that are in the measurement data are valid.
    if filter_ranks is None:
        used_ranks = measurement_parameters.rank
    #Else only the ranks that are in the input and in the measurement data
    else:
        used_ranks = list_intersection(filter_ranks, measurement_parameters.rank)

    #Same for models
    if filter_models is None:
        used_models = list_intersection(["uncomp", "cp", "tt"],measurement_parameters.models)
    else:
        used_models = list_intersection(filter_models, measurement_parameters.models)

    
    #Determine number of different measurements
    nr_of_models = len(used_ranks) * len(used_models) 
    model_types = [f"{model} {rank}"  for model in used_models for rank in used_ranks if model not in "uncomp"]
    
    if "uncomp" in used_models:
        nr_of_models = nr_of_models - len(used_ranks) + 1
        model_types = ["uncomp"] + model_types
    return model_types



def preprocess_time_data(read_file, folder, measurement_variable = None, measurement_variable2 = None, iterator_routine = None, used_ranks = None, used_models = None):
    """"
        Processes measurment data

        Processes measurment data in a numpy format. It accepts multiple decomposed types.
        Sorts measurements by the measurement variables. If there are two viable options it overwrites.

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
            results[0, modelnr, measurement_range.index(test[f"{measurement_variable}"])] = statistics.median(test["Inference duration"])
            results[1, modelnr, measurement_range.index(test[f"{measurement_variable}"])] = test["measurements"][0]["Total allocated RAM"]
            results[2, modelnr, measurement_range.index(test[f"{measurement_variable}"])] = test["Expected MAC total"]
        elif measurement_variable2 is not None:
            results[0, modelnr, measurement_range.index(test[f"{measurement_variable}"]), measurement_range2.index(test[f"{measurement_variable2}"])] = statistics.median(test["Inference duration"])
            results[1, modelnr, measurement_range.index(test[f"{measurement_variable}"]), measurement_range2.index(test[f"{measurement_variable2}"])] = test["measurements"][0]["Total allocated RAM"]
            results[2, modelnr, measurement_range.index(test[f"{measurement_variable}"]), measurement_range2.index(test[f"{measurement_variable2}"])] = test["Expected MAC total"]

    return results, model_types

def preprocess_time_all_combinations(read_file, folder, measurement_variable = None, iterator_routine = None, used_ranks = None, used_models = None):
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
    results = np.zeros((3, nr_of_models, nr_of_tests))
    i = -1
    for test in data["outcomes"]:
        # Exception is used when a setup is in data, but will not be in plot.
        try:
            try:
                modelnr = model_types.index(f"{test["model_type"]} {test["rank"]}")
            except KeyError:
                modelnr = model_types.index(f"{test["model_type"]}")
        except ValueError:
            continue

        i += 1
        if measurement_variable is None:
            results[0, modelnr, i//nr_of_models] = statistics.median(test["Inference duration"])
            results[1, modelnr, i//nr_of_models] = test["measurements"][0]["Total allocated RAM"]
            results[2, modelnr, i//nr_of_models] = test["Expected MAC total"]
    return results, model_types, nr_of_tests


def get_theoretical_results_in_preprocess_measurement_format(read_file, folder, iteration_method :str = None, measurement_variable = None, measurement_variable2 = None, iterator_routine = None, used_ranks = None, used_models = None):
    """
        Gets the theoretical amount of memory in the sameformat as the process and time. 
    """

    with open(f"{os.getcwd()}\\data\\data\\{folder}\\{read_file}.json") as json_file:
        data = json.load(json_file)

    measurement_parameters = measurement.from_dict(data["Setup_data"])
    model_types = get_model_types(data, filter_models=used_models, filter_ranks=used_ranks)
    if iteration_method == None:
        iterator = measurement_parameters.__iter__()
    elif iteration_method == "same_in_out":
        iterator = measurement_parameters.iter_same_in_out()
    elif iteration_method == "same_in_out_same_kernel_pad":
        iterator = measurement_parameters.iter_same_in_out_same_kernel_pad()
    else:
        raise ValueError("Routine is not valid. See the main function for valid options or give no option for default routine")

    with open(f"{os.getcwd()}\\data\\data\\{folder}\\{read_file}.json") as json_file:
        data = json.load(json_file)

   

    if measurement_variable2 == None:
        results = np.zeros((1,len(model_types), len(getattr(measurement_parameters, measurement_variable))),int)
    else:
        results = np.zeros((1,len(model_types), len(getattr(measurement_parameters, measurement_variable)), len(getattr(measurement_parameters, measurement_variable2))),int)
    
    for (in_channel, out_channel, kernel_size, stride, padding, dilation, image_size, ranks, _ , models) in iterator:
        # print(models)
        for rank in ranks:
            for model in models:

                data_dict = {
                    "in_channel"        : in_channel,
                    "out_channel"       : out_channel,
                    "kernel_size"       : kernel_size,
                    "stride"            : stride,
                    "padding"           : padding,
                    "dilation"          : dilation,
                    "image_size"        : image_size,
                    "rank"              : rank,
                    "model"             : model   
                }
                # print(model)
                try:
                    if model == "uncomp":
                        first_index = model_types.index(f"{model}")
                    else:
                        first_index = model_types.index(f"{model} {rank}")                       
                except ValueError:
                    continue
                
                # print(model)

                datapoint = data_dict[f"{measurement_variable}"]
                datapoint = datapoint[0] if isinstance(datapoint,list) else datapoint
                second_index = getattr(measurement_parameters, measurement_variable).index(datapoint)

                if measurement_variable2 is None:
                    results[0,first_index,second_index] = ram_estimation_2d(in_channel, out_channel, kernel_size, image_size, model, stride, padding, dilation, rank, 32, True, True)
                else :
                    datapoint = data_dict[f"{measurement_variable2}"]
                    datapoint = datapoint[0] if isinstance(datapoint,list) else datapoint
                    third_index = getattr(measurement_parameters, measurement_variable2).index(datapoint)
                    results[0,first_index,second_index,third_index] = ram_estimation_2d(in_channel, out_channel, kernel_size, image_size, model, stride, padding, dilation, rank, 32, True, True)

    return results, measurement_parameters, model_types




def get_pandas_infernce_memory_pairs(data, used_ranks = None, used_models = None):
    """
        Makes a pandas frame for all memory time pairs
    """

    headers = { 
         "model"    : [],
         "memory"   : [],
         "MAC"      : [],
         "duration" : [] 
    }

    df = pd.DataFrame(headers)

    model_types = get_model_types(data, used_ranks, used_models)

    measurement_parameters = measurement.from_dict(data["Setup_data"])
    iterator_routine = None
    nr_of_tests = measurement_parameters.amount_of_measurements(iterator_routine)
    print(nr_of_tests)
    print(len(data["outcomes"]))
    i = 0
    for test in data["outcomes"]:
        i += 1
        #If the model is not in the filtered list continue
        try:
            try:
                modelnr = model_types.index(f"{test["model_type"]} {test["rank"]}")
            except KeyError:
                modelnr = model_types.index(f"{test["model_type"]}")
        except ValueError:
            continue
    
        duration = test["Inference duration"]
        MAC_exp = test["Expected MAC total"]
        memory = test["Expected RAM total"]

        # print(len(duration))

        data = {
            "model"    : model_types[modelnr],
            "memory"   : memory,
            "MAC"      : MAC_exp,
            "duration" : statistics.median(duration)
        }
        
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        #print(df)

    print(i)

    return df

            
        





        






    
