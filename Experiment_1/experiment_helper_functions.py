
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
    while(number > 1024):
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