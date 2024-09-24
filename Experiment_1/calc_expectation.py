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

#Calculates the dimensions of the output image from the input image.
#The inputs should be given as list of two ints or as a single int this case it will be used for both input dimensions.
#Input:         image       : This are the dimensions of the input image
#               stride      : This is the stride of the CNN
#               padding     : This is the padding of the CNN
#               dilitation  : This is the dilitation of the CNN
#               k
def calc_output_image_dim(kernel_size : list, stride : list, padding : list, dilitation: list, in_image: list):
    kernel_size = check_int_or_list_of_int(kernel_size,   "kernel_size")
    stride      = check_int_or_list_of_int(stride,        "stride")
    padding     = check_int_or_list_of_int(padding,       "padding")
    dilitation  = check_int_or_list_of_int(dilitation,    "dilitation")
    in_image    = check_int_or_list_of_int(in_image,      "image")
    
    image_out = []
    for i in range(len(in_image)):
            image_out.append(math.floor((in_image[i]  + 2 * padding[i] - dilitation[i] * (kernel_size[i] - 1) - 1)/stride[i]+1))
    return image_out

def validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilitation, image, method, rank):
    kernel_size = check_int_or_list_of_int(kernel_size,   "kernel_size")
    stride      = check_int_or_list_of_int(stride,        "stride")
    padding     = check_int_or_list_of_int(padding,       "padding")
    dilitation  = check_int_or_list_of_int(dilitation,    "dilitation")
    image       = check_int_or_list_of_int(image,         "image") 

    if (method in {'cp','tucker','tt'}) & (rank == 'None'):
        raise ValueError(f'For {method} rank cannot be None\nPlease insert a rank')
    
    return [kernel_size, stride, padding, dilitation, image]


#Inputs certain variables of a convolutional neural layer and outputs the expected required amount of RAM necassary for the kernel and the feature images.
#Input:         in_channel      : Number of in_channels
#               out_channel     : Number of out_channels
#               kernel_size     : Kernel size of the CNN layer, could be square or single integer
#               image           : Size of the image. This is assumed to be 2D. Input can be a single integer or a list of two ints.
#               method          : Method which is used inside the CNN layer. This could be uncomp (for uncompressed), cp (for canoncial polyadic), tucker (for tucker) or tt for tensor train
#               stride          : This is the stride used in the CNN. This could be a list of two ints or a single integer which is applied to both sides.
#               padding         : Give the padding of the CNN. This could be a List of two ints or one int which is applied to both sides of the input image.
#               dilitation      : Gives the dilitation of the CNN. This could be a list of two ints or one int which is applied to both sizes of the input image.
#kwargs
#               rank            : Defaults ('None') user is required to give a (list of) ranks for the cp, tucker and tt decomposition. For uncompressed this is ignored.
#               bits_per_element: States the number of bits in memory for each element. Defaults to 32 (for 32 bit floating point), but can be adjusted.
def ram_estimation_2d(in_channel : int, out_channel : int, kernel_size, image, method, stride, padding, dilitation, rank='None', bits_per_element : int = 32):
    #Check input parameters which could ether be an int or a list of ints.
    [kernel_size, stride, padding, dilitation, image] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilitation, image, method, rank)
    
    #Calculate the output image as it will always have the same shape
    image_out = calc_output_image_dim(kernel_size,stride, padding, dilitation, image)
    
    
    #Based on the mehtod, calculate the RAM required.
    if method == 'uncomp':
        input_image_size    = in_channel * math.prod(image)
        kernal_storage_size = in_channel * out_channel * math.prod(kernel_size)
        output_image_size   = out_channel * math.prod(image_out)

        return (input_image_size + kernal_storage_size + output_image_size) * bits_per_element
    
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
        filter_storage_size.append(image[0] * image_out[1] * rank)
        filter_storage_size.append(image_out[0] * image_out[1] * rank)
        
        total_elements = input_image_size + sum(kernal_storage_size) + sum(filter_storage_size) + output_image_size

        return total_elements * bits_per_element
       
    elif method == 'tucker':
        raise NotImplementedError
    elif method == 'tt':
        raise NotImplementedError
    else : 
        raise ValueError(f'Give a valid method\nValid methods are:\nuncomp, cp, tt, tucker')
    


def MAC_estimation_2d(in_channel : int, out_channel : int, kernel_size, image, method, stride, padding, dilitation, rank='None', bits_per_element : int = 32):
    #Check input parameters which could ether be an int or a list of ints.
    [kernel_size, stride, padding, dilitation, image] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilitation, image, method, rank)
    
    #Calculate the output image as it will always have the same shape
    image_out = calc_output_image_dim(kernel_size,stride, padding, dilitation, image)

    #Based on the input method calculate the number of MACs required.
    if method == 'uncomp':
        raise NotImplementedError
    elif method == 'cp':
        raise NotImplementedError
    elif method == 'tucker':
        raise NotImplementedError
    elif method == 'tt':
        raise NotImplementedError
    else:
        raise ValueError(f'Give a valid method\nValid methods are:\nuncomp, cp, tt, tucker')
    