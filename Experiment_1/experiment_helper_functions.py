import json
import fnmatch


def name_number(number : int, add_number = True, add_name_size_string = True):
    """
    This function converts an integer value to string containg the value in b, Kb, Mb, Gb, Tb
    
    Args:     
        number: This is the number to be converted
        add_number:     This indicates whether the number should be added in the result
        add_name_size_string    This indicates whether the name should be added in the result.

    Returns:
        A string containg the size and the unit. If one of those is turned of it is just not prented
        So 1.30 Gb, Gb and 1.30 could all be correct answers.

    Raises:
        TypeError: If the type is not an int
        ValueError: If both add_number and add_name_size_string are false as then no output should be printed.
    """
    if not isinstance(number, int):
        raise TypeError(f'Number should have type int, but it has type {type(number)}')

    name = ["b", "Kb", "Mb", "Gb", "Tb"]
    i = 0
    while(abs(number) > 1024):
        number = number / 1024
        i = i + 1
    
    if  (add_number == True)&(add_name_size_string == True):
        if i <= len(name):
            return f'{number:.2f}{name[i]}'
        else:
            raise NotImplemented('No name is present for such large number')
    elif (add_number == True) & (add_name_size_string == False):
        return f'{number:.2f}'
    elif (add_number == False) & (add_name_size_string == True):
        if  i<= len(name):
            return  name[i]
        else:
            raise ValueError('No name is present for such large number')
    else:
        raise ValueError("Nothing will be returned both the value and name of size are turned off")
    
    

def print_RAM(RAM_list : list):
    """" 
    Function which converts all values of the input list to human readable string. 
    
    The inputs (which are in bytes) to sizes in Kb, Mb, ect. to make the list more human readable.
    
    Args:
        RAM_list: A list containing integers or lists with integers
    Returns:   
        A list with sizes in xb for example Kb or Mb if an entry contains a list then a list containing these is returned.

    """
    return [name_number(i) if type(i) == int else [name_number(j) for j in i] for i in RAM_list]




def extract_profiler_memory(profiler_output):
    """
    Function which extracts the allocated memory for the total model and images from a memory profiler output.
    
    Args: 
        profiler_output: torch.profiler.key_averages() object
    
    Output:  
        A list with:
            RAM[0] is size of the input image in bytes
            RAM[1] is size of the model parameters for all kernels combined in bytes
            RAM[2] a list with the images sizes of all in between images in bytes
            RAM[3] the size of the last filters image in bytes

    """

    RAM = [0,0,[],0]
    for item in profiler_output:
        if item.key == "Input_image":
            RAM[0] = item.cpu_memory_usage
        if item.key == "model_size":
            RAM[1] = item.cpu_memory_usage
        if item.key.startswith('Filter_image'):
            pos = int(item.key[-1]) - 1
            while 1:
                try:
                    RAM[2][pos] = item.cpu_memory_usage
                    break
                except:
                    RAM[2].append(0)
    RAM[3] = RAM[2].pop()
    return RAM

def get_function_call_for_mem_ref(events):
    """
    This function adds the function which requestes/allocates the memory

    This function adds to the dictinary of the memory allocation event, the CPU operation that is summoned last to get more insights in memory allocation

    Args:
        Events: These are all the events in the trace events element of the Chrome JSON file exports.
    Returns:
        Void. Python is by refernce thus the references of the operations are added to the events. 
    """
    memory_events = []

    #Obtain a list of memory events
    for i in events:
        #All memory events have a name called memory
        #So if no name is present, go to the next
        if i.get("name") != "[memory]":
            continue

        #If all conditions are not met append i
        memory_events.append(i)

    #Now go though all cpu_op events and link the one that has started but is not yet closed the last before the memory event.
    for j in memory_events:
        print(j)
        j["Operation name"] = {"name" : "No operation could be assigned."}
        dif_opt = -1
        for i in events:
            #If no cpu_op event is not valid
            if i.get("cat") != "cpu_op":
                continue
            
            #If the memory event happend before or after the cpu_op took place, continue
            if (i["ts"] > j["ts"]) | (j["ts"] >(i["ts"] + i["dur"])):
                continue

            #Empty events are not that interesting as they dont't tell that much therefore they're excluded.
            if fnmatch.fnmatch(i["name"],"*empty*"):
                continue

            dif = j["ts"] - i["ts"]
            
            if (dif < dif_opt) | (dif_opt == -1):
                dif_opt = dif
                j["Operation name"] = i
        
        if (dif_opt == -1):
            print("There are memory operations without an assigned name.")
        
        

def json_get_memory_changes_per_model_ref(data, verbose: bool = True):
    """
    This function prints the amount of memory per memory record to the terminal.

    This function adds for each memory record a user places. The allocated or deallocated memory to the terminal and adds the caling function with it.
    If the memory is allocated by a function containing empty then callee of that function is inserted to increase meaning.

    Args:
        data: This is the data that is obtained from the JSON file which has to be imported with the JSON.load function.
        verbose: (default: True) send output to terminal if false data manumpulation could still be done.
    Returns:
        Output to terminal.
    Raises:
        ValueError: If profile_memory is not set to one.
        RunTimeWarning: If not all memory events are added, because there was no record function to attach it to.
    """
    #Check if the memory has been profiled.
    if(data.get("profile_memory") != 1):
        raise ValueError("Memory is not profiled.")
    
    #Get events data
    events = data["traceEvents"]

    get_function_call_for_mem_ref(events)
    
    #Get the record functions in a list and add extra dictonairy entry
    user_events = []
    for i in events:
        if i.get("cat") is not None :
            if (i['cat'] == "user_annotation"):
                i["Memory_event"] = []
                user_events.append(i)           


    #For all events. Get the memory events and add them to an enrty
    for i in events:
        if(i.get("name") != "[memory]"):
            continue
        
        added_to_entry = 0
        
        for j in user_events:
            if (i['ts'] > j['ts']) & (i['ts'] < (j['ts'] + j['dur'])):
                assert added_to_entry == 0
                j["Memory_event"].append(i)
                added_to_entry == 1

        if added_to_entry == 0:
            RuntimeWarning("Not all memory events are added to a memory record")

    if(verbose == True):
        #Print the outcomes of the memory event.
        print("Printing events")
        for j in user_events:
            print(f"\t{j["name"]}")
            for i in j["Memory_event"]:
                print(f"\t\t {name_number(i["args"]["Bytes"])} \t for operation {i["Operation name"]["name"]}")

                        
