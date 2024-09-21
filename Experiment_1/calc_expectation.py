import math

#Checks whether the input is an integer or a list of 2 ints. If it is an integer it returns a list of two. If it's neither of both then raises exception.
#Input:     The value of the variable to be checked
#           The name of that variable
#Output:    A list containing of two ints with the size of the variable.

def check_int_or_list_of_int(value_to_check, name_of_value_to_check):
    if (type(value_to_check) is int):
        return [value_to_check, value_to_check]
    elif (type(value_to_check) is list):
        if not(any(not isinstance(x,int) for x in value_to_check) | (len(value_to_check) != 2)) :
            return value_to_check
    raise ValueError(f'{name_of_value_to_check} should either be a single int or a list of two ints')

#Inputs certain variables of a convolutional neural layer and outputs the expected required amount of RAM necassary for the kernel and the feature images.
def ram_estimation_2d(in_channel : int, out_channel : int, kernel_size, image, method, stride, padding, dilitation, rank='None', bits_per_value : int = 32):
    #Check input parameters which could ether be an int or a list of ints.
    kernel_size = check_int_or_list_of_int(kernel_size,   "kernel_size")
    stride      = check_int_or_list_of_int(stride,        "stride")
    padding     = check_int_or_list_of_int(padding,       "padding")
    dilitation  = check_int_or_list_of_int(dilitation,    "dilitation")
    image       = check_int_or_list_of_int(image,         "image")

    if (method in {'cp','tucker','tt'}) & (rank == 'None'):
        raise ValueError(f'For {method} rank cannot be None\nPlease insert a rank')
        
    
    
    #Calculate the output image as it will always have the same shape
    image_out = []
    for i in range(len(image)):
            image_out.append(math.floor((image[i]  + 2 * padding[i] - dilitation[i] * (kernel_size[i] - 1) - 1)/stride[i]+1))
    
    
    #Based on the mehtod, calculate the RAM required.
    if method == 'uncomp':
        input_image_size    = in_channel * math.prod(image)
        kernal_storage_size = in_channel * out_channel * math.prod(kernel_size)
        output_image_size   = out_channel * math.prod(image_out)

        return (input_image_size + kernal_storage_size + output_image_size) * bits_per_value
    
    elif method == 'cp' :
        kernal_storage_size = []
        filter_storage_size = []
        input_image_size    = in_channel * math.prod(image)
        output_image_size   = out_channel * math.prod(image_out)
        
        #Add the storage size for each of the four kernels.
        kernal_storage_size.append(in_channel * rank)
        kernal_storage_size.append(kernel_size[0] * rank)
        kernal_storage_size.append(kernel_size[1] * rank)
        kernal_storage_size.append(out_channel * rank)
        
        #Insert the storage sizes of the consecutive filter images in a list
        filter_storage_size.append(image[0] * image[1] * rank)
        filter_storage_size.append(image[0] * image[1] * rank)
        filter_storage_size.append(image[0] * image[1] * rank)
        
        return input_image_size + math.prod(kernal_storage_size) + math.prod(filter_storage_size) + output_image_size
       
    elif method == 'tucker':
        raise(NotImplementedError)
    elif method == 'tt':
        raise(NotImplementedError)
    else : 
        print(f'Give a valid method')
        print(f'Valid methods are :')
        print(f'uncomp, cp, tt, tucker')
        raise(ValueError)