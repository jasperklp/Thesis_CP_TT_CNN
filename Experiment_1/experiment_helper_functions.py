def name_number(number, add_number = True, add_name_size_string = True):
    name = ["b", "Kb", "Mb", "Gb", "Tb"]
    i = 0
    while(number > 1024):
        number = number / 1024
        i = i + 1
    
    if (add_name_size_string == True) & (add_number == True):
        if i <= len(name):
            return f'{number:.2f}{name[i]}'
        else:
            raise NotImplemented(r'No name is present for such large number')
    elif (add_name_size_string == False) & (add_number == True):
        return i
    elif (add_name_size_string == True) & (add_number == False):
        if  i<= len(name):
            return  name[i]
        else:
            raise ValueError('No name is present for such large number')
    else:
        raise ValueError("Nothing will be returned both the value and name of size are turned off")
    
    
def print_RAM(RAM_list : list):
    return [name_number(i) if type(i) == int else [name_number(j) for j in i] for i in RAM_list]