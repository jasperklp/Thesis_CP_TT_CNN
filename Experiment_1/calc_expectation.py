import math
import functools

#Defaults

bits_per_element_default = 32 #Gives the default number of bits for a memeory operation.





#Checks whether the input is an integer or a tuple of 2 ints. If it is an integer it returns a tuple of two ints. If it's neither of both then raises exception.
#Input:     The value of the variable to be checked
#           The name of that variable
#Output:    A tuple containing of two ints with the size of the variable.
@functools.lru_cache
def check_int_or_tuple_of_int(value_to_check : int | tuple, name_of_value_to_check : str) -> tuple:  
    if (isinstance(value_to_check, int)):
        return (value_to_check, value_to_check)
    elif (isinstance(value_to_check, tuple)) :
          if ((len(value_to_check) == 2) & (all(isinstance(i,int) for i in value_to_check))):
            return value_to_check
    raise ValueError(f'{name_of_value_to_check} should either be a single int or a tuple of two ints')

#Calculates the dimensions of the output image from the input image.
#The inputs should be given a tuple of two ints or as a single int this case it will be used for both input dimensions.
#Input:         image       : This are the dimensions of the input image
#               stride      : This is the stride of the CNN
#               padding     : This is the padding of the CNN
#               dilitation  : This is the dilitation of the CNN
#               k
@functools.lru_cache
def calc_output_image_dim(kernel_size : tuple, stride : tuple, padding : tuple, dilitation: tuple, in_image: tuple):
    image_out = []
    for i in range(2):
            image_out.append(math.floor((in_image[i]  + 2 * padding[i] - dilitation[i] * (kernel_size[i] - 1) - 1)/stride[i]+1))
    return image_out

def validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilitation, image, method, rank, in_channel, out_channel):
    kernel_size = check_int_or_tuple_of_int(kernel_size,   "kernel_size")
    stride      = check_int_or_tuple_of_int(stride,        "stride")
    padding     = check_int_or_tuple_of_int(padding,       "padding")
    dilitation  = check_int_or_tuple_of_int(dilitation,    "dilitation")
    image       = check_int_or_tuple_of_int(image,         "image") 

    if (method in {'cp','tucker','tt'}) & (rank == None):
        raise ValueError(f'For {method} rank cannot be None\nPlease insert a rank')
    elif(method == 'cp'):
        if isinstance(rank,int):
            rank = rank
        elif isinstance(rank,float):
            rank = math.floor(rank * in_channel * out_channel / sum(in_channel + out_channel + sum(kernel_size)))
        else :
            raise ValueError(f'CP rank must be an integer or a float')
    return [kernel_size, stride, padding, dilitation, image, rank]


#Inputs certain variables of a convolutional neural layer and outputs the expected required amount of RAM necassary for the kernel and the feature images.
#Input:         in_channel      : Number of in_channels
#               out_channel     : Number of out_channels
#               kernel_size     : Kernel size of the CNN layer, could be square or single integer
#               image           : Size of the image. This is assumed to be 2D. Input can be a single integer or a tuple of two ints.
#               method          : Method which is used inside the CNN layer. This could be uncomp (for uncompressed), cp (for canoncial polyadic), tucker (for tucker) or tt for tensor train
#               stride          : This is the stride used in the CNN. This could be a tuple of two ints or a single integer which is applied to both sides.
#               padding         : Give the padding of the CNN. This could be a tuple of two ints or one int which is applied to both sides of the input image.
#               dilitation      : Gives the dilitation of the CNN. This could be a tuple of two ints or one int which is applied to both sizes of the input image.
#kwargs
#               rank            : Defaults ('None') user is required to give a (tuple of) ranks for the cp, tucker and tt decomposition. For uncompressed this is ignored.
#                               : If float is given the amount of parameters will be scaled.
#               bits_per_element: States the number of bits in memory for each element. Defaults to 32 (for 32 bit floating point), but can be adjusted.
def ram_estimation_2d(in_channel : int, out_channel : int, kernel_size : int | tuple[int,int], image: int | tuple[int,int], method: int | tuple[int,int], stride: int | tuple[int,int], padding: int | tuple[int,int], dilitation: int | tuple[int,int], rank=None, bits_per_element : int = bits_per_element_default):
    #Check input parameters which could ether be an int or a tuple of ints.
    [kernel_size, stride, padding, dilitation, image, rank] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilitation, image, method, rank, in_channel, out_channel)
    print(1)
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
        
        #Insert the storage sizes of the consecutive filter images in a tuple
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
    

#Inputs certain variables of a convolutional neural layer and outputs the expected required amount of RAM necassary for the kernel and the feature images.
#Input:         in_channel      : Number of in_channels
#               out_channel     : Number of out_channels
#               kernel_size     : Kernel size of the CNN layer, could be square or single integer
#               image           : Size of the image. This is assumed to be 2D. Input can be a single integer or a tuple of two ints.
#               method          : Method which is used inside the CNN layer. This could be uncomp (for uncompressed), cp (for canoncial polyadic), tucker (for tucker) or tt for tensor train
#               stride          : This is the stride used in the CNN. This could be a tuple of two ints or a single integer which is applied to both sides.
#               padding         : Give the padding of the CNN. This could be a tuple of two ints or one int which is applied to both sides of the input image.
#               dilitation      : Gives the dilitation of the CNN. This could be a tuple of two ints or one int which is applied to both sizes of the input image.

def MAC_estimation_2d(in_channel : int, out_channel : int, kernel_size: int | tuple[int,int], image: int | tuple[int,int], method: int | tuple[int,int], stride: int | tuple[int,int], padding: int | tuple[int,int], dilitation: int | tuple[int,int], rank=None):
    #Check input parameters which could ether be an int or a tuple of ints.
    [kernel_size, stride, padding, dilitation, image, rank] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilitation, image, method, rank, in_channel, out_channel)
    
    #Calculate the output image as it will always have the same shape
    image_out = calc_output_image_dim(kernel_size,stride, padding, dilitation, image)

    #Based on the input method calculate the number of MACs required.
    if method == 'uncomp':
        return kernel_size[0] * kernel_size[1] * in_channel * out_channel * image_out[0] * image_out[1]
    elif method == 'cp':
        filter_operations = []
        #Append the operations per kernel.
        filter_operations.append(in_channel * rank * image[0] * image[1])
        filter_operations.append(kernel_size[1] * rank * image[0] * image_out[1])
        filter_operations.append(kernel_size[0] * rank * image_out[0] * image_out[1])
        filter_operations.append(out_channel * rank * image_out[0] * image_out[1])
        
        return sum(filter_operations)
    elif method == 'tucker':
        raise NotImplementedError
    elif method == 'tt':
        raise NotImplementedError
    else:
        raise ValueError(f'Give a valid method\nValid methods are:\nuncomp, cp, tt, tucker')



#This function passes trough a lot of variables MAC and RAM estimation for CNN's have in common. For an explanation of the inputs see reqested functions.

def MAC_and_ram_estimation_2d(in_channel : int, out_channel : int, kernel_size, image: int | tuple[int,int], method: int | tuple[int,int], stride: int | tuple[int,int], padding: int | tuple[int,int], dilitation: int | tuple[int,int], rank=None, bits_per_element : int = bits_per_element_default):
    [kernel_size, stride, padding, dilitation, image] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilitation, image, method, rank, in_channel, out_channel)
    
    MAC = MAC_estimation_2d(in_channel, out_channel, kernel_size, image, method, stride, padding, dilitation, rank=rank)
    RAM = ram_estimation_2d(in_channel, out_channel, kernel_size, image, method, stride, padding, dilitation, rank=rank, bits_per_element=bits_per_element)

    return [MAC, RAM]
    