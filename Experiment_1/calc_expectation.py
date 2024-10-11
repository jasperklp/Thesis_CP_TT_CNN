import math
import functools
import numpy as np

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
#               dilation  : This is the dilation of the CNN
#               k
@functools.lru_cache
def calc_output_image_dim(kernel_size : tuple, stride : tuple, padding : tuple, dilation: tuple, in_image: tuple):
    image_out = []
    for i in range(2):
            image_out.append(math.floor((in_image[i]  + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1)/stride[i]+1))
    return image_out

def CP_rank_from_compratio(compressionratio : float, in_channel: int, out_channel : int, kernel_size : tuple, rounding = "round"):
    if (rounding == "round"):
        rounding_fun = round
    elif (rounding == "ceil"):
        rounding_fun = math.ceil
    elif (rounding == "floor"):
        rounding_fun = math.floor
    else:
        raise ValueError(f"Rounding should be round, ceil or floor, but got {rounding} instead")

    rank =  rounding_fun(compressionratio * in_channel * out_channel * math.prod(kernel_size) / (in_channel + out_channel + sum(kernel_size)))
    if (rank == 0):
        return 1
    else:
        return rank


def validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilation, image, method, rank, in_channel, out_channel):
    kernel_size = check_int_or_tuple_of_int(kernel_size,   "kernel_size")
    stride      = check_int_or_tuple_of_int(stride,        "stride")
    padding     = check_int_or_tuple_of_int(padding,       "padding")
    dilation    = check_int_or_tuple_of_int(dilation,      "dilation")
    image       = check_int_or_tuple_of_int(image,         "image") 

    if (method in {'cp','tucker','tt'}) & (rank == None):
        raise ValueError(f'For {method} rank cannot be None\nPlease insert a rank')
    elif(method == 'cp'):
        if isinstance(rank,int):
            rank = rank
        elif isinstance(rank,float):
            rank = CP_rank_from_compratio(rank, in_channel, out_channel, kernel_size)
            if rank <= 0:
                rank = 1
        else :
            raise ValueError(f'CP rank must be an integer or a float')
    return [kernel_size, stride, padding, dilation, image, rank]


#Inputs certain variables of a convolutional neural layer and outputs the expected required amount of RAM necassary for the kernel and the feature images.
#Input:         in_channel      : Number of in_channels
#               out_channel     : Number of out_channels
#               kernel_size     : Kernel size of the CNN layer, could be square or single integer
#               image           : Size of the image. This is assumed to be 2D. Input can be a single integer or a tuple of two ints.
#               method          : Method which is used inside the CNN layer. This could be uncomp (for uncompressed), cp (for canoncial polyadic), tucker (for tucker) or tt for tensor train
#               stride          : This is the stride used in the CNN. This could be a tuple of two ints or a single integer which is applied to both sides.
#               padding         : Give the padding of the CNN. This could be a tuple of two ints or one int which is applied to both sides of the input image.
#               dilation      : Gives the dilation of the CNN. This could be a tuple of two ints or one int which is applied to both sizes of the input image.
#kwargs
#               rank            : Defaults ('None') user is required to gCP_rank_from_compratio(rank, in_channel, out_channel, kernel_size)ive a (tuple of) ranks for the cp, tucker and tt decomposition. For uncompressed this is ignored.
#                               : If float is given the amount of parameters will be scaled.
#               bits_per_element: States the number of bits in memory for each element. Defaults to 32 (for 32 bit floating point), but can be adjusted.
def ram_estimation_2d(in_channel : int, out_channel : int, kernel_size : int | tuple, image: int | tuple, method: int | tuple, stride = 1, padding = 1 , dilation = 1, rank=None, bits_per_element : int = bits_per_element_default , output_total : bool = True, output_in_bytes : bool = False):
    #Check input parameters which could ether be an int or a tuple of ints.
    [kernel_size, stride, padding, dilation, image, rank] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilation, image, method, rank, in_channel, out_channel)
    #Calculate the output image as it will always have the same shape
    image_out = calc_output_image_dim(kernel_size,stride, padding, dilation, image)
    
    kernal_storage_size = []
    filter_storage_size = []
    input_image_size    = in_channel * math.prod(image)
    output_image_size   = out_channel * math.prod(image_out)
    
    #Based on the mehtod, calculate the RAM required.
    if method == 'uncomp':
        kernal_storage_size.append(in_channel * out_channel * math.prod(kernel_size))
        filter_storage_size.append(0)
   
    elif method == 'cp' :
      
        #Add the storage size for each of the four kernels.
        kernal_storage_size.append(in_channel * rank)
        kernal_storage_size.append(kernel_size[0] * rank)
        kernal_storage_size.append(kernel_size[1] * rank)
        kernal_storage_size.append(out_channel * rank)
        
        #Insert the storage sizes of the consecutive filter images in a tuple
        filter_storage_size.append(image[0] * image[1] * rank)
        filter_storage_size.append(image[0] * image_out[1] * rank)
        filter_storage_size.append(image_out[0] * image_out[1] * rank)
        
   
    elif method == 'tucker':
        raise NotImplementedError
    elif method == 'tt':
        raise NotImplementedError
    else : 
        raise ValueError(f'Give a valid method\nValid methods are:\nuncomp, cp, tt, tucker')
    
    per_element_multiplier = bits_per_element // 8 if output_in_bytes else bits_per_element 

    if output_total :
        output_elements = input_image_size + sum(kernal_storage_size) + sum(filter_storage_size) + output_image_size
        return output_elements * per_element_multiplier
    else:
        return [input_image_size*per_element_multiplier, [i*per_element_multiplier for i in kernal_storage_size], [i*per_element_multiplier for i in filter_storage_size],output_image_size*per_element_multiplier]
    

#Inputs certain variables of a convolutional neural layer and outputs the expected required amount of RAM necassary for the kernel and the feature images.
#Input:         in_channel      : Number of in_channels
#               out_channel     : Number of out_channels
#               kernel_size     : Kernel size of the CNN layer, could be square or single integer
#               image           : Size of the image. This is assumed to be 2D. Input can be a single integer or a tuple of two ints.
#               method          : Method which is used inside the CNN layer. This could be uncomp (for uncompressed), cp (for canoncial polyadic), tucker (for tucker) or tt for tensor train
#               stride          : This is the stride used in the CNN. This could be a tuple of two ints or a single integer which is applied to both sides.
#               padding         : Give the padding of the CNN. This could be a tuple of two ints or one int which is applied to both sides of the input image.
#               dilation      : Gives the dilation of the CNN. This could be a tuple of two ints or one int which is applied to both sizes of the input image.

def MAC_estimation_2d(in_channel : int, out_channel : int, kernel_size: int | tuple[int,int], image: int | tuple[int,int], method: int | tuple[int,int], stride: int | tuple[int,int], padding: int | tuple[int,int], dilation: int | tuple[int,int], rank=None, output_total : bool = True):
    #Check input parameters which could ether be an int or a tuple of ints.
    [kernel_size, stride, padding, dilation, image, rank] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilation, image, method, rank, in_channel, out_channel)
    
    #Calculate the output image as it will always have the same shape
    image_out = calc_output_image_dim(kernel_size,stride, padding, dilation, image)
    filter_operations = []

    #Based on the input method calculate the number of MACs required.
    if method == 'uncomp':
        filter_operations.append(kernel_size[0] * kernel_size[1] * in_channel * out_channel * image_out[0] * image_out[1])
    elif method == 'cp':
        #Append the operations per kernel.
        filter_operations.append(in_channel * rank * image[0] * image[1])
        filter_operations.append(kernel_size[1] * rank * image[0] * image_out[1])
        filter_operations.append(kernel_size[0] * rank * image_out[0] * image_out[1])
        filter_operations.append(out_channel * rank * image_out[0] * image_out[1])
        
    elif method == 'tucker':
        raise NotImplementedError
    elif method == 'tt':
        raise NotImplementedError
    else:
        raise ValueError(f'Give a valid method\nValid methods are:\nuncomp, cp, tt, tucker')
    
    if output_total == True:
        return sum(filter_operations)
    else:
        return filter_operations
    
#This function passes trough a lot of variables MAC and RAM estimation for CNN's have in common. For an explanation of the inputs see reqested functions.

def MAC_and_ram_estimation_2d(in_channel : int, out_channel : int, kernel_size, image: int | tuple[int,int], method: int | tuple[int,int], stride: int | tuple[int,int], padding: int | tuple[int,int], dilation: int | tuple[int,int], rank=None, bits_per_element : int = bits_per_element_default, output_total : bool = True, output_in_bytes : bool = False):
    [kernel_size, stride, padding, dilation, image,rank] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilation, image, method, rank, in_channel, out_channel)
    
    MAC = MAC_estimation_2d(in_channel, out_channel, kernel_size, image, method, stride, padding, dilation, rank=rank, output_total=output_total)
    RAM = ram_estimation_2d(in_channel, out_channel, kernel_size, image, method, stride, padding, dilation, rank=rank, bits_per_element=bits_per_element, output_total=output_total,output_in_bytes=output_in_bytes)

    return [MAC, RAM]


def check_list_or_int_float(value_to_check): 
    if (isinstance(value_to_check, list)) :
        if(all((isinstance(i, int) | isinstance(i, float)) for i in value_to_check)) :
            return value_to_check
    elif isinstance(value_to_check, int) | isinstance(value_to_check,float):
        return [value_to_check]
    else:
        raise ValueError(r'Value must be a list, an integer or a float')

def validate_get_theoretical_data_set_for_plot_input(in_channels, out_channels, kernel_size, image_size,padding,rank):
    in_channels         = check_list_or_int_float(in_channels)
    out_channels        = check_list_or_int_float(out_channels)
    kernel_size         = check_list_or_int_float(kernel_size)
    image_size          = check_list_or_int_float(image_size)
    padding             = check_list_or_int_float(padding)
    rank                = check_list_or_int_float(rank) 

    return [in_channels, out_channels, kernel_size, image_size, padding, rank]


def get_theoretical_dataset_for_plot(in_channels: list, out_channels: list, kernel_size:list, image_size:list, padding:list, rank : list):  
    [in_channels, out_channels, kernel_size, image_size, padding, rank] = validate_get_theoretical_data_set_for_plot_input(in_channels, out_channels, kernel_size, image_size,padding,rank)
    models = ["uncomp", "cp"]
    value_type = ["MAC", "RAM"]
    data = np.ndarray((len(models), len(value_type), len(in_channels), len(out_channels),len(kernel_size),len(image_size),len(padding),len(rank)))
    
    for i,in_chan in enumerate(in_channels):
        for j,out_chan in enumerate(out_channels):
            for k,kernel in enumerate(kernel_size):
                for l,img in enumerate(image_size):
                    for m,pad in enumerate(padding):
                        for n,rnk in enumerate(rank):
                            [data[0,0,i,j,k,l,m,n], data[0,1,i,j,k,l,m,n]] = MAC_and_ram_estimation_2d(in_chan,out_chan,kernel,img, 'uncomp', 1,pad,1)
                            [data[1,0,i,j,k,l,m,n], data[1,1,i,j,k,l,m,n]] = MAC_and_ram_estimation_2d(in_chan,out_chan,kernel,img, 'cp', 1,pad,1, rank=rnk)
    
    return data
    