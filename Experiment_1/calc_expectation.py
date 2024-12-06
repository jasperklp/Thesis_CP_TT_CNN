import math
import functools
import numpy as np
from tensorly import validate_tt_rank
import logging

logger = logging.getLogger(__name__)

#Defaults
#Gives the default number of bits for a memeory operation.
bits_per_element_default = 32 

@functools.lru_cache
def check_int_or_tuple_of_int(value_to_check : int | tuple, name_of_value_to_check : str = None) -> tuple: 
    """
    Checker for input which expects tuple or int

    Checks whether the input is an integer or a tuple of 2 ints. If it is an integer it returns a tuple of two ints. If it's neither of both then raises exception.

    Args:
        value_to_check: The value of the variable to be checked
        name_of_value_to_check The name of that variable
    Returns:
        A tuple containing of two ints with the size of the variable.
    """ 
    if (isinstance(value_to_check, int)):
        return (value_to_check, value_to_check)
    elif (isinstance(value_to_check, tuple)) :
          if ((len(value_to_check) == 2) & (all(isinstance(i,int) for i in value_to_check))):
            return value_to_check
    raise ValueError(f'{name_of_value_to_check} should either be a single int or a tuple of two ints')


@functools.lru_cache
def calc_output_image_dim(kernel_size : tuple, stride : tuple, padding : tuple, dilation: tuple, in_image: tuple):
    """
    Calculates a CNN's output images dimensions for 2D convolution.
    The inputs should be given a tuple of two ints or as a single int this case it will be used for both input dimensions axes.
    Args:         
        kernel_size: This is the kernels size of the CNN
        stride: This is the stride of the CNN
        padding: This is the padding of the CNN
        dilation: This is the dilation of the CNN
        in_image: This are the dimensions of the input image
    Returns:
        A list containing the output dimension sizes of the 2DConvolution    
    """
    W = math.floor((in_image[0]  + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)/stride[0]+1)
    H = math.floor((in_image[1]  + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)/stride[1]+1)
    return (W,H)

def CP_rank_from_compratio(compressionratio : float, in_channel: int, out_channel : int, kernel_size : tuple, rounding = "round"):
    """
    Calculates the CP rank from a comporession ratio

    Calculates the CP rank from a given compression ratio. This only means a reduction in parameters by original*compression ratio. Hence this ratio is often smaller than 1.

    Args:
        compressionratio: This is the compression ratio which has to be achieved
        in_channel: Amount of in_channels of the CNN
        out_channel: Number of out_channels of the CNN
        kernel_size: This is a tuple containing the dimensions of the kernel
    Returns:
        The rank which the CPD needs to have to achieve the desired compression of parameters.
    """
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
    """
    Validates if inputs have the correct type and in some cases corrects them.

    Args:
        kernel_size: kernel_size of CNN
        stride: Stride of CNN
        padding: Padding of CNN
        dilation: Dilation of CNN
        image: input image of CNN
        method: method used in the CNN
        rank: Rank of CP,Tucker or TT CNN
        in_channel: Amount of in_channels of the CNN
        out_channel: Amount of out_channels of the CNN
    
    Returns:
        A tuple containing kernel_size, stride, padding, dilation, image, rank
        
        Except for rank for all these parameters a tuple of two ints is returned
        If for some parameters a single int is given. This is converted to a tuple containing the same int twice. It is assumed that a single int is ment for two dimensions.
        If a tuple of two ints is given as an input this is returned.

    Raises
        TypeError: if for all ouput variables except rank something else than an int or tuple of two ints is given
        ValueError: if the method is cp,tucker or tt and the rank is None
        NotImplementedError: Tucker and TT rank calculations are not implemented.
    """

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
    elif method == "tt":
        # Rank will be of form 1 R1, R2, R3, 1. 
        # Thus for a network consisting of four cores one will have 5 ranks of which the outer two are 1. (They're not connected to anything)
        rank = validate_tt_rank((in_channel, kernel_size[0], kernel_size[1], out_channel),rank=rank)

    elif(method == 'tucker'):
        raise NotImplementedError(f'{method} is not implemented for this function')
    elif (method != "uncomp"):
        raise ValueError(f"Method should be uncomp, cp, tucker or tt but method is {method}")

    return kernel_size, stride, padding, dilation, image, rank



def ram_estimation_2d(in_channel : int, out_channel : int, kernel_size : int | tuple, image: int | tuple, method: int | tuple, stride = 1, padding = 1 , dilation = 1, rank=None, bits_per_element : int = bits_per_element_default,output_in_bytes : bool = False, output_total : bool = True):
    """
    Inputs certain variables of a convolutional neural layer and outputs the expected required amount of RAM necassary for the kernel and the feature images.

    Args:         
        in_channel: Number of in_channels
        out_channel: Number of out_channels
        kernel_size: Kernel size of the CNN layer, could be square or single integer
        image: Size of the image. This is assumed to be 2D. Input can be a single integer or a tuple of two ints.
        method: Method which is used inside the CNN layer. This could be uncomp (for uncompressed), cp (for canoncial polyadic), tucker (for tucker) or tt for tensor train
        stride: This is the stride used in the CNN. This could be a tuple of two ints or a single integer which is applied to both sides.
        padding: Give the padding of the CNN. This could be a tuple of two ints or one int which is applied to both sides of the input image.
        dilation: Gives the dilation of the CNN. This could be a tuple of two ints or one int which is applied to both sizes of the input image.
        rank: (defaults None) For method == tt, tucker and cp this is the rank of the decomposition for method = uncompressed this is ignored.
        bits_per_element: (default = 32) This is the amount of bits each numerical element is expected to have e.g. one torch.float32 value takes 32 bits to store
        output_in_bytes: ((default = False) Defines whether the output is converted from bits to bytes
        output_total: (default = True) Denotes whether the individual steps are summed to a total. Otherwise the intermediate steps correspond to [input images, [sizes of the kernels], [size of the intermediate filter output results if any], output image]
    Returns:
        The total amount of bits a CNN takes in memory including input image, output image, kernel parameters and if applicable in between output images.
    Raises:
        NotImplementError: When method is tucker or tt
        ValueError: When method is not uncomp, cp, tucker or tt
    """
    #Check input parameters which could ether be an int or a tuple of ints.
    [kernel_size, stride, padding, dilation, image, rank] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilation, image, method, rank, in_channel, out_channel)
    #Calculate the output image as it will always have the same shape
    image_out = calc_output_image_dim(kernel_size,stride, padding, dilation, image)
    
    kernel_storage_size = []
    filter_storage_size = []
    input_image_size    = in_channel * math.prod(image)
    output_image_size   = out_channel * math.prod(image_out)

    #Based on the mehtod, calculate the RAM required.
    if method == 'uncomp':
        kernel_storage_size.append(in_channel * out_channel * math.prod(kernel_size))
        filter_storage_size.append(0)
   
    elif method == 'cp' :
        #Add the storage size for each of the four kernels.
        kernel_storage_size.append(in_channel * rank)
        kernel_storage_size.append(kernel_size[0] * rank)
        kernel_storage_size.append(kernel_size[1] * rank)
        kernel_storage_size.append(out_channel * rank)
        
        #Insert the storage sizes of the consecutive filter images in a tuple
        filter_storage_size.append(image[0] * image[1] * rank)
        filter_storage_size.append(image[0] * image_out[1] * rank)
        filter_storage_size.append(image_out[0] * image_out[1] * rank)
          
    elif method == 'tucker':
        raise NotImplementedError
    elif method == 'tt':      
        #Add storage for each of the kernels
        # print(rank)
        kernel_storage_size.append(in_channel * rank[1])
        kernel_storage_size.append(rank[1] * kernel_size[0] * rank[2])
        kernel_storage_size.append(rank[2] * kernel_size[1] * rank[3])
        kernel_storage_size.append(rank[3]* out_channel)

        #Add filters for each of the kernels
        # print(rank)
        filter_storage_size.append(image[0] * image[1] * rank[1])
        filter_storage_size.append(image_out[0] * image[1] * rank[2])
        filter_storage_size.append(image_out[0] * image_out[1] * rank[3])



    else : 
        raise ValueError(f'Give a valid method\nValid methods are:\nuncomp, cp, tt, tucker')
    
    per_element_multiplier = bits_per_element // 8 if output_in_bytes else bits_per_element 

    if output_total :
        output_elements = input_image_size + sum(kernel_storage_size) + sum(filter_storage_size) + output_image_size
        return output_elements * per_element_multiplier
    else:
        return [input_image_size*per_element_multiplier, [i*per_element_multiplier for i in kernel_storage_size], [i*per_element_multiplier for i in filter_storage_size],output_image_size*per_element_multiplier]
    



def MAC_estimation_2d(in_channel : int, out_channel : int, kernel_size: int | tuple[int,int], image: int | tuple[int,int], method: int | tuple[int,int], stride: int | tuple[int,int], padding: int | tuple[int,int], dilation: int | tuple[int,int], rank=None, output_total : bool = True ):
    """
    Inputs certain variables of a convolutional neural layer and outputs the expected required amount of RAM necassary for the kernel and the feature images.
    Args:
        in_channel: Number of in_channels
        out_channel: Number of out_channels
        kernel_size: Kernel size of the CNN layer, could be square or single integer
        image: Size of the image. This is assumed to be 2D. Input can be a single integer or a tuple of two ints.
        method: Method which is used inside the CNN layer. This could be uncomp (for uncompressed), cp (for canoncial polyadic), tucker (for tucker) or tt for tensor train
        stride: This is the stride used in the CNN. This could be a tuple of two ints or a single integer which is applied to both sides.
        padding: Give the padding of the CNN. This could be a tuple of two ints or one int which is applied to both sides of the input image.
        dilation: Gives the dilation of the CNN. This could be a tuple of two ints or one int which is applied to both sizes of the input image.
        rank: (default is None) This is the rank (or are the ranks of) the cp, tucker and tt decompositions
        output_total = (default = True) If true sums up the intermediate results otherwise the result is a list in which each enrty represents a kernel.
    Returns:
        The amount of mulitply and add (MAC) operations which are required to calculate such CNN.
    Raises:
        NotImplementedError: For method is tucker and tt
        ValueError: When method is not uncomp, cp, tucker or tt    
    """
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
        print(image_out)
        filter_operations.append(rank[1] * in_channel * image[0] * image[1])
        filter_operations.append(rank[2] * rank[1] * kernel_size[0] * image_out[0] * image[1])
        filter_operations.append(rank[3] * rank[2] * 1 * kernel_size[1] * image_out[0] * image_out[1])
        filter_operations.append(out_channel * rank[3] * image_out[0] * image_out[1])
    else:
        raise ValueError(f'Give a valid method\nValid methods are:\nuncomp, cp, tt, tucker')
    
    if output_total == True:
        return sum(filter_operations)
    else:
        return filter_operations
    
#This function passes trough a lot of variables MAC and RAM estimation for CNN's have in common. For an explanation of the inputs see reqested functions.

def MAC_and_ram_estimation_2d(in_channel : int, out_channel : int, kernel_size, image: int | tuple[int,int], method: int | tuple[int,int], stride: int | tuple[int,int], padding: int | tuple[int,int], dilation: int | tuple[int,int], rank=None, bits_per_element : int = bits_per_element_default, output_total : bool = True ,output_in_bytes : bool = False):
    """
    This function gives on interface to get the values from MAC_estimation_2d and ram_estimation_2d

    Args:    
        in_channel: Number of in_channels
        out_channel: Number of out_channels
        kernel_size: Kernel size of the CNN layer, could be square or single integer
        image: Size of the image. This is assumed to be 2D. Input can be a single integer or a tuple of two ints.
        method: Method which is used inside the CNN layer. This could be uncomp (for uncompressed), cp (for canoncial polyadic), tucker (for tucker) or tt for tensor train
        stride: This is the stride used in the CNN. This could be a tuple of two ints or a single integer which is applied to both sides.
        padding: Give the padding of the CNN. This could be a tuple of two ints or one int which is applied to both sides of the input image.
        dilation: Gives the dilation of the CNN. This could be a tuple of two ints or one int which is applied to both sizes of the input image.
        rank: (defaults None) For method == tt, tucker and cp this is the rank of the decomposition for method = uncompressed this is ignored.
        bits_per_element: (default = 32) This is the amount of bits each numerical element is expected to have e.g. one torch.float32 value takes 32 bits to store
        output_total: (default = True) States whether the results are summed up totals or if there devided as mentioned in the two seperate functions above.
        output_in_bytes: (default = False) States whether the RAM output shoud be in bits or in bytes.
    Returns:
        A tuple containing the amount of multiply and add operatations and RAM in bits the CNN takes.    
    """
    [kernel_size, stride, padding, dilation, image,rank] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilation, image, method, rank, in_channel, out_channel)
    
    MAC = MAC_estimation_2d(in_channel, out_channel, kernel_size, image, method, stride, padding, dilation, rank=rank, output_total=output_total)
    RAM = ram_estimation_2d(in_channel, out_channel, kernel_size, image, method, stride, padding, dilation, rank=rank, bits_per_element=bits_per_element, output_total=output_total,output_in_bytes=output_in_bytes)

    return [MAC, RAM]


def check_list_or_int_float(value_to_check):
    """ 
    Checks whether the input is a list containing a list of ints or floats, an int itself or a float.

    Args:
        value_to_check
    Returns:
        The list which was inputted or a list containing the single int or float.
    Raises:
        TypeError: Value_to_check is not of the correct type
    """
    if (isinstance(value_to_check, list)) :
        if(all((isinstance(i, int) | isinstance(i, float)) for i in value_to_check)) :
            return value_to_check
    elif isinstance(value_to_check, int) | isinstance(value_to_check,float):
        return [value_to_check]
    else:
        raise TypeError('Value must be a list, an integer or a float')


def validate_get_theoretical_data_set_for_plot_input(in_channels, out_channels, kernel_size, image_size,padding,rank):
    """
    Calls check_list_or_int for certain values
    
    Args:
        in_channels:
        out_channels:
        kernel_size:
        image_size:
        padding:
        rank:
    Returns:
        A list containing the input variables"""
    in_channels         = check_list_or_int_float(in_channels)
    out_channels        = check_list_or_int_float(out_channels)
    kernel_size         = check_list_or_int_float(kernel_size)
    image_size          = check_list_or_int_float(image_size)
    padding             = check_list_or_int_float(padding)
    rank                = check_list_or_int_float(rank) 

    return [in_channels, out_channels, kernel_size, image_size, padding, rank]


def get_theoretical_dataset_for_plot(in_channels: list, out_channels: list, kernel_size:list, image_size:list, padding:list, rank : list):  
    """
    Gets an array containing of MAC and RAM calculations for multiple variables for all methods.

    The users can send ranges or single values using lists for all these parameters and a multi-dimensional array containging all the MAC and RAM results are returned.
    It is assumed stride = 1 and dilation = 1

    Input:
        in_channels: Amount of in_channels of a CNN
        out_channels: Amount of out_channels of a CNN
        kernel_size: The dimentionsizes of the kernel
        image_size: The size of the input image
        padding: The amount of padding that is used.
        rank: for CP
    Returns:
        A single numpy array containing the results if all different parameters are inserted.
    """
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
    
def get_mulitple_of_SIMD(value, SIMD_size:int  = 8):
    """
        Makes sure that all values are in the range of the SIMD size.
        
        Should act as a ceil(value SIMD_size) in which ceil(SIMD_size,SIMD_size) = SIMD_size

        Args:
            value       : Value for which the SIMD size is determined.
            SIMD_size   : Amount of data entries a processor can take for one instruction. For SSE4.2/AVX2 this is 8 for AVX512 this is 16
    """
    if (value % SIMD_size) == 0:
        return value
    else:
        return (value//SIMD_size + 1)*SIMD_size


def get_mkldnn_ram(in_channel, out_channel, kernel_size, image, method, stride, padding, dilation, rank = None, output_total:bool = True, bytes_per_float = 4):
    """
        Function gets RAM needed for an experiment when the use of MKL is forced trough an MKL input.

        To get the match for these results one needs to force intel MKL. This can be achieved by converting the input into MKL format.

    Args:
        in_channel:         #of inchannels for the undecomposed system.
        out_channel:         # of out channels for the undecomposed system.
        kernel_size:         The kernel size of the undecomposed system.
        image:              # input image size
        method:             Either "uncomp" or "cp"
        stride              Stride of the original system.
        padding             padding of the input image
        dilation            dilation of the input image
        rank:               Rank of the CP system. This can be a float or an int
        output_total:       Decides whether to give the output as a total per filter/input/model_copy or total
        bytes_per_float:    Asks for the amounts of bytes a single element taks (e.g. 4 for float, 8 for double.)
    """
    #Check input parameters which could ether be an int or a tuple of ints.
    [kernel_size, stride, padding, dilation, image, rank] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilation, image, method, rank, in_channel, out_channel)
    #Calculate the output image as it will always have the same shape
    image_out = calc_output_image_dim(kernel_size,stride, padding, dilation, image)
    
    if (method == "uncomp"):
        input_image = in_channel * math.prod(image)
        kernel = in_channel *  out_channel * math.prod(kernel_size)
        output_image = out_channel * math.prod(image_out)


        Filter_1 = []

        Input_image = sum([input_image,input_image])
        Model_size = sum([kernel])
        if in_channel > 7:
            Filter_1.append(get_mulitple_of_SIMD(in_channel)*math.prod(image)) #copy of the image
            Filter_1.append(get_mulitple_of_SIMD(out_channel)*get_mulitple_of_SIMD(in_channel)*math.prod(kernel_size))
            Filter_1.append(get_mulitple_of_SIMD(out_channel)*math.prod(image_out))
        else:
            Filter_1.append(in_channel*math.prod(image)) #Copy of the image
            Filter_1.append(get_mulitple_of_SIMD(in_channel) * out_channel * math.prod(kernel_size)) #Copy of the kernel
            Filter_1.append(out_channel*math.prod(image_out)) #Convolution
        

        output = [Input_image, Model_size, sum(Filter_1)]     

        if output_total == True:
            return sum(output)*bytes_per_float
        else:
            output = [i * bytes_per_float for i in output]
            return output

    if (method == "cp"):
        input_image = in_channel * math.prod(image)


        Filter_1 = []
        Between_filter_1_and_2 = []
        Filter_2 = []
        Between_filter_2_and_3 = []
        Filter_3 = []
        Between_filter_3_and_4 = []
        Filter_4 = []
        end = []

        Input_image = sum([input_image,input_image])
        Model_size = sum([rank*in_channel, rank*kernel_size[0], rank*kernel_size[1],rank*out_channel])


        Filter_1.append(rank*in_channel) #Copy Kernel
        Filter_1.append(get_mulitple_of_SIMD(in_channel) *  math.prod(image)) #Copy input image
        Filter_1.append(get_mulitple_of_SIMD(rank)*get_mulitple_of_SIMD(in_channel)) #Reshape Kernel
        Filter_1.append(get_mulitple_of_SIMD(rank)*math.prod(image)) #Convolution

        Between_filter_1_and_2.append(rank*math.prod(image)) #reshape

        if rank != 1:
            Filter_2.append(rank * kernel_size[0]) #Copy kernel
            Filter_2.append(get_mulitple_of_SIMD(rank)*math.prod(image)) #Reshape image
            Filter_2.append(get_mulitple_of_SIMD(rank)*kernel_size[0]) #Reshape filter 2 kernel
            Filter_2.append(get_mulitple_of_SIMD(rank) * image_out[0] * image[1])

            Between_filter_2_and_3.append(0)

            Filter_3.append(rank*kernel_size[1]) #Copy kernel
            Filter_3.append(get_mulitple_of_SIMD(rank)*kernel_size[1]) #Reshape kernel
            Filter_3.append(get_mulitple_of_SIMD(rank)*math.prod(image_out)) #Convolution
        
        else:
            Filter_2.append(get_mulitple_of_SIMD(rank) * kernel_size[1]) #Reshape kernel
            Filter_2.append(get_mulitple_of_SIMD(rank) * image_out[0] * image_out[1])#Convolution

            Between_filter_2_and_3.append(0)

            Filter_3.append(image_out[0] * image[1])
            Filter_3.append(get_mulitple_of_SIMD(rank) * kernel_size[1])
            Filter_3.append(get_mulitple_of_SIMD(rank) * math.prod(image_out))

        
        Between_filter_3_and_4.append(rank*math.prod(image_out)) #Reshape output image to non-SIMD form

        Filter_4.append(get_mulitple_of_SIMD(rank)*math.prod(image_out)) #Reshape image
        Filter_4.append(get_mulitple_of_SIMD(out_channel)*get_mulitple_of_SIMD(rank)) #Reshape kernel
        Filter_4.append(get_mulitple_of_SIMD(out_channel)*math.prod(image_out))

        end.append(out_channel*math.prod(image_out))

        output = [Input_image,Model_size, Filter_1, Between_filter_1_and_2, Filter_2, Between_filter_2_and_3, Filter_3, Between_filter_3_and_4, Filter_4,end]

        output = [sum(i)  if (isinstance(i,list)) else i for i in output]

        if output_total == True:
            return sum(output)*bytes_per_float
        else:
            return [i*bytes_per_float for i in output]
        
    

    raise ValueError("Method must be uncomp, tt or CP")


def RAM_no_mkl(out_channel, out_width, out_height):
    """
    Gives the amount of expected RAM when no intel MKL is used and groups is set to 1.

    Args:
        out_channel : Number of outchannels of the layer
        out_width   : Width of the output image
        out_height  : Height of the output image
    Returns
        The amount of memory for that layer.
    """
    return out_channel*out_width*out_height

def RAM_MKL(in_channel, out_channel, kernel_size, in_width, in_height, out_width,out_height):
    """
        Gives the amount of RAM for a layer with the use of intel MKL in the non-forced case.

        Args:
            in_channel      : Number of inchannels
            out_channel     : Number of outchannels
            kernel_size     : Kernel_size
            in_width        : Width of the input image of that layer.
            in_height       : Height of the input image of that layer.
            out_width       : Width  of the output image of that layer.
            out_height      : Height of the output image of that layer.
        Returns
            The amount of memory for that layer.
    """
    Filter = []
    if ((in_channel > 7) | (math.prod(kernel_size) == 1)):
        Filter.append(get_mulitple_of_SIMD(in_channel) * in_width * in_height) #Copy of the input image
    Filter.append(get_mulitple_of_SIMD(out_channel) * get_mulitple_of_SIMD(in_channel) * kernel_size[0] * kernel_size[1]) #Copy of the kernel
    Filter.append(get_mulitple_of_SIMD(out_channel) * out_width * out_height) #Convolution
    Filter.append(out_channel * out_width * out_height) #Reshape of the output

    return sum(Filter)

def RAM_MKL_1x1_conv(rank,kernel_size : int, in_height, in_width,out_height,out_width):
    """
        This function calculats the amount of RAM for the 1x1 layer in the non-forced case.

        Args:
            rank            : Rank of the layer
            kernel_size     : kernel_size of the layer
            in_width        : Width of the input image of that layer.
            in_height       : Height of the input image of that layer.
            out_width       : Width  of the output image of that layer.
            out_height      : Height of the output image of that layer.
        Returns:
            The amount of RAM for that layer.
    """
    #In case rank is one, the number of groups is one and you get another situation.
    if (    (rank ==  1) and 
            kernel_size <= 3 and
            1 and#It is assumed there is no batching:
            (rank * in_height * in_width <= 20480)
    ):
        # print("no_MKL")
        logger.debug("No MKL")
        return RAM_no_mkl(rank, out_height, out_width) * (kernel_size + 1)
    else :
        logger.debug("MKL")
        Filter = []
        Filter.append(get_mulitple_of_SIMD(rank) * in_width * in_height)
        Filter.append(get_mulitple_of_SIMD(rank) * kernel_size)
        Filter.append(get_mulitple_of_SIMD(rank)* out_width * out_height)
        Filter.append(rank* out_width * out_height)
        # print("MKL")
    return Filter

def     RAM_choose_MKL_or_nativePT(in_channel, out_channel, kernel_size, stride, padding, dilation, in_width, in_height, out_width,out_height, extra_kernel_copy = False):
    """
        For a non-grouped layer this function choses whether MKL or Pytorch is used for the non-forced case

        For more information on the decion paramters see: https://github.com/pytorch/pytorch/blob/a440a01832ace3a9fbff3485eaa0ff0edeb01d09/aten/src/ATen/native/Convolution.cpp

    Args:
        in_channel:     : Number of inchannels for the undecomposed system.
        out_channel:    : Number of out channels for the undecomposed system.
        kernel_size:    : The kernel size of the undecomposed system.
        image:          : Input image size
        method:         : Either "uncomp" or "cp"
        stride          : Stride of the original system.
        padding         : padding of the input image
        dilation        : dilation of the input image
        in_width        : Width of the input image of that layer.
        in_height       : Height of the input image of that layer.
        out_width       : Width  of the output image of that layer.
        out_height      : Height of the output image of that layer.
        extra_kernel_copy : (default = False) Gives an extra copy of the kernel. This is sometimes the case but only decuced empirically.
    Returns:
        Amount of memory that gets assigned in the given layer.
    """
    memory = 0
    if extra_kernel_copy == True:
        memory += in_channel * out_channel * math.prod(kernel_size)


    if (
        1 and  
        1 and
        (   (all(i != 1 for i in stride)) or
            (all(i != 1 for i in dilation)) or
            (all(i != 0 for i in padding))  or
            0 or
            (kernel_size[0] != 1) or
            (kernel_size[1] != 1) or
            1 # Get num thread does not seem to be initialized. However, more than 1 core is assumed.(torch.get_num_thread() > 1)
        ) and
        (   0 or #Only call this function when groups = 1
            (kernel_size[0] > 3 and kernel_size[1] > 3) or
            0 or #Assumed batch size is 1
            (1 * in_channel * in_width * in_height > 20480) #Assume batch size larger than 1
        )
    ):
        logger.debug("MKL")
        memory += RAM_MKL(in_channel,out_channel,kernel_size,in_width, in_height, out_width, out_height)
    
    else:
        logger.debug("No MKL")
        if math.prod(kernel_size) == 1:
            memory += RAM_no_mkl(out_channel,out_width,out_height)

        elif (kernel_size[0] == 1) != (kernel_size[1] == 1):
            #Get the largest of the two kernels
            kernel_high = max(kernel_size[0], kernel_size[1])

            #First kernel_high times the input image
            memory += kernel_high * in_channel * in_height * in_width

            #Then once the output image
            memory += RAM_no_mkl(out_channel, out_width, out_height)
        else:
            memory += RAM_no_mkl(out_channel,out_width,out_height) * (math.prod(kernel_size) + 1)
        
    return memory




def get_RAM_realistic(in_channel, out_channel, kernel_size, image, method, stride, padding, dilation, rank = None, output_total:bool = True, bytes_per_float = 4,SIMD_size : int = 8):
    """
        Function gets RAM needed for an experiment when MKL may be used, but it is up to pytorch to decide.

        To get the match for these results one needs to have intel MKLDNN activated, but not forced.
        I.e. MKLDNN.is_avaliable() should be true, but further no adaptions should be done.

    Args:
        in_channel:         #of inchannels for the undecomposed system.
        out_channel:         # of out channels for the undecomposed system.
        kernel_size:         The kernel size of the undecomposed system.
        image:              # input image size
        method:             Either "uncomp" or "cp"
        stride              Stride of the original system.
        padding             padding of the input image
        dilation            dilation of the input image
        rank:               Rank of the CP system. This can be a float or an int
        output_total:       Decides whether to give the output as a total per filter/input/model_copy or total
        bytes_per_float:    Asks for the amounts of bytes a single element taks (e.g. 4 for float, 8 for double.)
    """
    #Check input parameters which could ether be an int or a tuple of ints.
    [kernel_size, stride, padding, dilation, image, rank] = validate_MAC_or_RAM_calc_input(kernel_size, stride, padding, dilation, image, method, rank, in_channel, out_channel)
    #Calculate the output image as it will always have the same shape
    image_out = calc_output_image_dim(kernel_size,stride, padding, dilation, image)

    if(method == "uncomp"):
        Input_image = in_channel * math.prod(image)
        Model_size = in_channel *  out_channel * math.prod(kernel_size)

        Filter_1 = RAM_choose_MKL_or_nativePT(in_channel,out_channel, kernel_size, stride, padding, dilation, image[0],image[1],image_out[0],image_out[1])
        output = [Input_image, Model_size, Filter_1]     

        if output_total == True:
            return sum(output)*bytes_per_float
        else:
            output = [i * bytes_per_float for i in output]
            return output
        

    elif(method == "cp"):
        Input_image = in_channel * math.prod(image)
        Model_size = sum([rank*in_channel, rank*kernel_size[0], rank*kernel_size[1],rank*out_channel])

        Filter_1 = RAM_choose_MKL_or_nativePT(in_channel, rank, (1,1), [1,1], [0,0], [1,1], image[0], image[1], image[0], image[1], True)
        Filter_2 = RAM_MKL_1x1_conv(rank, kernel_size[0], image[0], image[1], image_out[0], image[1])
        Filter_3 = RAM_MKL_1x1_conv(rank, kernel_size[1], image_out[0], image[1], image_out[0], image_out[1])
        Filter_4 = RAM_choose_MKL_or_nativePT(rank, out_channel, (1,1), [1], [0], [1], image_out[0], image_out[1], image_out[0], image_out[1])

        output = [Input_image, Model_size, Filter_1, Filter_2, Filter_3, Filter_4]
        output = [sum(i)  if (isinstance(i,list)) else i for i in output]

        
        
    elif(method == "tt"):
        Input_image = in_channel * math.prod(image)
        Model_size = sum([rank[1] * in_channel, rank[1] * kernel_size[1] * rank[2], rank[2] * kernel_size[1] * rank[3] , rank[3] * out_channel])

        Filter_1 = RAM_choose_MKL_or_nativePT(in_channel, rank[1], (1,1), (1,1), (0,0), (1,1), image[0], image[1], image[0], image[1], True)
        Filter_2 = RAM_choose_MKL_or_nativePT(rank[1], rank[2], (kernel_size[0],1), (padding[0], 0), (stride[0],1), (dilation[0],1), image[0], image[1], image_out[0], image[1], True)
        Filter_3 = RAM_choose_MKL_or_nativePT(rank[2], rank[3], (1,kernel_size[1]), (0,padding[1]), (1,stride[0]), (1,dilation[0]), image_out[0], image[1], image_out[0], image_out[1], True)
        Filter_4 = RAM_choose_MKL_or_nativePT(rank[3], out_channel, (1,1), (1,1), (0,0), (1,1), image_out[0], image_out[1], image_out[0], image_out[1], True)

        output = [Input_image, Model_size, Filter_1, Filter_2, Filter_3, Filter_4]
        output = [sum(i)  if (isinstance(i,list)) else i for i in output]

    else:
        raise ValueError("Method must be uncomp or CP")


    if output_total == True:
        return sum(output)*bytes_per_float
    else:
        return [i*bytes_per_float for i in output]


    