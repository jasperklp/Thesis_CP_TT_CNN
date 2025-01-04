import json
import fnmatch
import warnings
import copy
import datetime
import math
import logging
from dataclasses import dataclass, asdict
logger = logging.getLogger(__name__)

@dataclass
class measurement:
    #Data for CNN
    in_channel  : int | list
    out_channel : int |list
    kernel_size : int|tuple | list
    image_size  : int|tuple | list
    rank        : int|tuple | list
    padding     : int|tuple | list      = 1
    stride      : int|tuple | list      = 1
    dilation    : int|tuple | list      = 1 
    models      : list                  = None
    
    #Nr of epochs
    epochs      : int                   = 1

    def __post_init__(self):
        to_list = lambda x : [x] if not isinstance(x,list) else x
        self.in_channel     = to_list(self.in_channel)
        self.out_channel    = to_list(self.out_channel)
        self.kernel_size    = to_list(self.kernel_size)
        self.image_size     = to_list(self.image_size)
        self.rank           = to_list(self.rank)
        self.padding        = to_list(self.padding)
        self.stride         = to_list(self.stride)
        self.dilation       = to_list(self.dilation)
        if self.models is None:
            self.models = ["uncomp", "tt", "cp"] 


        #Check if padding and the kernel size have approximately the same shape as they go together
        len_kernel_size = 1 if (isinstance(self.kernel_size, tuple) | isinstance(self.padding, int)) else len(self.kernel_size)
        len_padding = 1 if (isinstance(self.padding, tuple) | isinstance(self.padding, int)) else len(self.padding)

        if len_kernel_size != len_padding:
            raise ValueError(f"The length of the kernel_sizes and padding should be equal, but the padding list has length {len_padding} and kernel_size_list has length {len_kernel_size}")

    def amount_of_measurements(self, func):
        """
            Deterimens the amount of measurements that are done based on the iterator given in func

        Args:
            func:   An iterator of the class
        Returns:
            The amount of measurements that will be done. Based on the iterator (routine)
        Raises:
            ValueError  : If the iterator does not exist.
        """
        if func is None:
            func_name = "__iter__"

        elif func in {"iter_same_in_out", "iter_same_in_out_same_kernel_pad"}:
            func_name = func
        else:
            func_name = func.__name__

        if func_name == "__iter__":
            to_be_measured = [self.in_channel, self.out_channel, self.kernel_size, self.padding, self.stride, self.dilation, self.image_size]
        elif func_name == "iter_same_in_out":
            to_be_measured =  [self.in_channel, self.kernel_size, self.padding, self.stride, self.dilation, self.image_size]
        elif func_name == "iter_same_in_out_same_kernel_pad":
            to_be_measured =  [self.in_channel, self.kernel_size, self.stride, self.dilation, self.image_size]
        else: 
            raise ValueError(f"Expected an iterator of the measurement dataclass, instead got {func}")
        
        return math.prod([len(i) for i in to_be_measured])

    def __iter__(self):
        for in_channel in self.in_channel:
            for out_channel in self.out_channel:
                for kernel_size in self.kernel_size:
                    for stride in self.stride:
                        for padding in self.padding:
                            for dilation in self.dilation:
                                for image_size in self.image_size:
                                    #print(in_channel, out_channel, kernel_size, stride, padding, dilation, image_size,self.rank,self.epochs)
                                    yield (in_channel, out_channel, kernel_size, stride, padding, dilation, image_size,self.rank,self.epochs, self.models)

    def iter_same_in_out(self):
        if (len(self.in_channel) != len(self.out_channel)):
            raise ValueError(f"List of in_channels and out_channels should be of the same length.\n Instead {self.in_channel} in_channels are present and {self.out_channel} out channels")
        
        for in_channel,out_channel in zip(self.in_channel,self.out_channel):
            for kernel_size in self.kernel_size:
                    for stride in self.stride:
                        for padding in self.padding:
                            for dilation in self.dilation:
                                for image_size in self.image_size:
                                    #print(in_channel, out_channel, kernel_size, stride, padding, dilation, image_size,self.rank,self.epochs)
                                    yield (in_channel, out_channel, kernel_size, stride, padding, dilation, image_size,self.rank,self.epochs, self.models)

    def iter_same_in_out_same_kernel_pad(self):
        if (len(self.in_channel) != len(self.out_channel)):
            raise ValueError(f"List of in_channels and out_channels should be of the same length.\n Instead {self.in_channel} in_channels are present and {self.out_channel} out channels")
        
        if (len(self.in_channel) != len(self.out_channel)):
            raise ValueError(f"List of kernelsizes and paddings should be of the same length.\n Instead {self.kernel_size} kernel_sizes are present and {self.padding} different paddings")
        
        
        for in_channel,out_channel in zip(self.in_channel,self.out_channel):
            for kernel_size,padding in  zip(self.kernel_size,self.padding):
                for stride in self.stride:
                    for dilation in self.dilation:
                        for image_size in self.image_size:
                            # print(in_channel, out_channel, kernel_size, stride, padding, dilation, image_size,self.rank,self.epochs)
                            yield (in_channel, out_channel, kernel_size, stride, padding, dilation, image_size,self.rank,self.epochs, self.models)


    def as_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls,dictionary):
        in_channel     = dictionary.get("in_channel")
        out_channel    = dictionary.get("out_channel")
        kernel_size    = dictionary.get("kernel_size")
        image_size     = dictionary.get("image_size")
        rank           = dictionary.get("rank")
        padding        = dictionary.get("padding")
        stride         = dictionary.get("stride")
        dilation       = dictionary.get("dilation")
        epochs         = dictionary.get("epochs")
        models         = dictionary.get("models")
        return cls(in_channel,out_channel,kernel_size,image_size,rank,padding,stride,dilation,models,epochs)
    




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
    all_memory_events_are_good = True
    for j in memory_events:
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
            #warnings.warn("There are memory events without an assigned cpu operation")
            all_memory_events_are_good = False
        
    # if all_memory_events_are_good == False:
    #     logger.warning("There are memory events without an assigned cpu operation")
        
def add_start_end_and_in_between_events(user_events, add_start_event :bool = True, add_end_event: bool = True, add_in_between_events: bool = True):
    """
    Creates a before event, after event and in between event for all user defined events.

    Args:
        user_events: All the record function events from the JSON file
        start_event: (default == True) If true add start event
        end_event: (default == True) If true addes end event
        in_between_events: (default == True) If true adds in between events
    Returns:
        user_events: It returns the inserted user events.
    Raises:
        waring --> If two userfunctions overlap.
        A way to handle this is not implemented.
    """

    events_to_add = []

    #Make sure that the user events are sorted in a time order.
    user_events = sorted(user_events, key=lambda x:x["ts"])

    if add_in_between_events == True:
        in_between_events = [{"cat" : "user_annotation",
                            "name" : f"Between {user_events[i]["name"]} and {user_events[i+1]["name"]}",
                            "ts" : user_events[i]["ts"] + user_events[i]["dur"] + 0.001,
                            "dur" : user_events[i+1]["ts"] - (user_events[i]["ts"] + user_events[i]["dur"])-0.002,
                            "Memory_event": []

            } for i in range(len(user_events)- 1)

        ]

        events_to_add.extend(in_between_events)
    
    #Append a start event before the first event
    if add_start_event == True:
        events_to_add.append({"cat" : "user_annotation",
                "name" : "Start",
                "ts"   : 0,
                "dur" : user_events[0]["ts"],
                "Memory_event" : [],
                })
    
    #Append a end event after the last event.
    if add_end_event == True:
        events_to_add.append({"cat" : "user_annotation",
                "name" : "End",
                "ts"   : user_events[-1]["ts"] + user_events[-1]["dur"]+0.001,
                "dur" : user_events[-1]["ts"],
                "Memory_event" : [],
                })

    #Make sure that the user events are sorted in a time order.
    return sorted(user_events + events_to_add, key=lambda x:x["ts"])
                        

def json_get_memory_changes_per_model_ref(data, verbose: bool = False):
    """
    This function prints the amount of memory per memory record to the terminal.

    This function adds for each memory record a user places. The allocated or deallocated memory to the terminal and adds the caling function with it.
    If the memory is allocated by a function containing empty then callee of that function is inserted to increase meaning.

    Args:
        data: This is the data that is obtained from the JSON file which has to be imported with the JSON.load function.
        verbose: (default: False) send output to terminal if false data manipulation could still be done.
    Returns:
        A list containing dictionaries of all memory record events and user created events in between. Aside from this name, a list of memory events is included this list contains summary of the memory event.
        The structure overview is shown below
        [Record function and inbetween dicts] 
        dicts --> {"name of record function" ,[memory event list]}
        [memory event list] --> {Bytes:int, "name of function if determined", Address : int}
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

    user_events = add_start_end_and_in_between_events(user_events)

    #For all events. Get the memory events and add them to an enrty
    all_memory_events_are_added_to_a_record = True
    for i in events:
        if(i.get("name") != "[memory]"):
            continue
        
        added_to_entry = 0
        
        for j in user_events:
            if (i['ts'] >= j['ts']) & (i['ts'] <= (j['ts'] + j['dur'])):
                j["Memory_event"].append(i)
                added_to_entry = 1

        if added_to_entry == 0:
            warnings.warn("Not all memory events are added to a memory record")
            all_memory_events_are_added_to_a_record = False
            print(i)
    if all_memory_events_are_added_to_a_record == False:
        logger.warning("Not all memory events are added to a memory record")

    if(verbose == True):
        #Print the outcomes of the memory event.
        print("Printing events")
        for j in user_events:
            print(f"\t{j["name"]}")
            for i in j["Memory_event"]:
                print(f"\t\t{name_number(i["args"]["Bytes"])}\tfor operation {i["Operation name"]["name"]}")

    output = []
    for j in user_events:
        event_item = {}
        event_item["name"] = copy.deepcopy(j["name"])
        event_item["Events"] = []
        for i in j["Memory_event"]:
            memory_event = {}
            memory_event["Bytes"] = copy.deepcopy(i["args"]["Bytes"])
            memory_event["Operation name"] = copy.deepcopy(i["Operation name"]["name"])
            memory_event["Adress"] = copy.deepcopy(i["args"]["Addr"])
            event_item["Events"].append(memory_event)
        output.append(event_item)

    return output



def get_peak_and_total_alloc_memory(events, verbose = False):
    """
    This function obtains the total allocated memory and peak allocated memory.

    This function obtains the total sum of all allocations, which is called the total allocated memory. 
    Additionally, this function also finds the maximum amount of memory that was assigned in a single time instance.
    During the operations, thus all memory which is allocated before the first recorded memory event is omitted.

    Args:
        events: These are the memory events considered (all non-memory events are filtered)
        verbose: (default = False) If True the function prints the peak allocated memory and the total allocated memory.
    Returns:
        A tuple containing the peak allocated memory and total allocated memory.
    """
    peak_memory = 0
    total_alloc_memory = 0
    ts = -1
    offset = 0
    
    for i in events:
        #Filters out the memory events
        if i.get("name") != "[memory]":
            continue

        #Sometimes memory is already allocated before the run. This does not count towards the result of the model. 
        #And will thus be deducted.
        if (ts == -1) | (i["ts"] < ts) : 
            ts = i["ts"]
            offset =  i["args"]["Total Allocated"] - i["args"]["Bytes"]

        #Obtains the peak value
        if peak_memory < i["args"]["Total Allocated"]:
            peak_memory = i["args"]["Total Allocated"]

        #If it is an allocation the amount of bytes is positive. Hence only those values are added.
        if i["args"]["Bytes"] > 0:
            total_alloc_memory += i["args"]["Bytes"]
    
    peak_memory = peak_memory - offset

    if verbose == True:
        print(f"Total allocated memory = {name_number(total_alloc_memory)}")
        print(f"Peak memory = {name_number(peak_memory)}")
    
    return peak_memory, total_alloc_memory


def get_date_time(delete_microseconds : bool = False):
    [date, time] = f"{datetime.datetime.now()}".split()
    if delete_microseconds == True:
        [time, _] = time.split(".")
    time = time.replace(":",".")
    return (date , time)

def get_total_mem_per_filter(Filter_per_model,verbose : bool = False):
    output = []
    for record_function in Filter_per_model:
        memory_total_alloc = 0
        for memory_event in record_function["Events"]:
            memory = memory_event.get("Bytes")
            if memory > 0:
                memory_total_alloc += memory
        output.append({record_function.get("name") : memory_total_alloc})
    return output
        
