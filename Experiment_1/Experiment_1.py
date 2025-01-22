import verify_model_matching_test
import verify_model_matching_test_no_MKL
from experiment_helper_functions import measurement
import logging
import inspect


def main():
    # in_channel()
    # out_channel()
    # image_size()
    # kernel_size()
    # in_channel_default_pytorch()
    # out_channel_default_pytorch()
    image_size_default_pytorch()
    kernel_size_default_pytorch()
    # mkl_switch_over_default_pytorch()

def in_channel_default_pytorch():
    #First test dependence on in_channel
    measure_data_in_channel = measurement(in_channel  = [2,4,8,16,32,64,128,256,512,1024],                                      
                                        out_channel = [128],
                                        kernel_size = [3],
                                        padding     = 1,
                                        rank        = [0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                        image_size  = 128,
                                        epochs      = 1
                                        )

    filename_in_channel = f"verify_model_matching_tt_{inspect.currentframe().f_code.co_name}"
    print(filename_in_channel)
    verify_model_matching_test_no_MKL.main(measure_data_in_channel,filename_in_channel)

def out_channel_default_pytorch():
    measure_data_out_channel  = measurement(in_channel  = [128],                                      
                                        out_channel = [2,4,8,16,32,64,128,256,512,1024],
                                        kernel_size = [3],
                                        padding     = 1,
                                        rank        = [0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                        image_size  = 128,
                                        epochs      = 1
                                        ) 
    filename_out_channel = f"verify_model_matching_tt_{inspect.currentframe().f_code.co_name}" #Gives folder name of the current out channel
    print(filename_out_channel)
    verify_model_matching_test_no_MKL.main(measure_data_out_channel,filename_out_channel)

def image_size_default_pytorch():
    measurement_data_image_size = measurement(in_channel = [4,16, 128 ,512],
                                              out_channel = [4, 16, 128 ,512],
                                              kernel_size=3,
                                              padding=1,
                                              rank=[0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                              image_size=[4,16,128,512],
                                              epochs=1
                                              )

    filename = f"verify_model_matching_ttcp_{inspect.currentframe().f_code.co_name}" #Gives folder name of the current out channel
    print(filename)
    verify_model_matching_test_no_MKL.main(measurement_data_image_size,filename, routine = "same_in_out")

def kernel_size_default_pytorch():
    measurement_data_kernel_size = measurement(in_channel = [4,16,128,512],
                                              out_channel= [4,16,128,512],
                                              kernel_size=[1,3,5,7],
                                              padding=[0,1,2,3],
                                              rank=[0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                              image_size=128,
                                              epochs=1)
    filename = f"verify_model_matching_ttcp_{inspect.currentframe().f_code.co_name}" #Gives folder name of the current out channel
    print(filename)
    verify_model_matching_test_no_MKL.main(measurement_data_kernel_size,filename, routine="same_in_out_same_kernel_pad")

def mkl_switch_over_default_pytorch():
    measurement_data = measurement( in_channel =  [7,7,7, 8,8,8, 9,9,9] ,
                                    out_channel=  [7,8,9, 7,8,9, 7,8,9],
                                    kernel_size=[3],
                                    padding=[1],
                                    rank=[0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                    image_size= 512,
                                    epochs=1)
    filename = f"verify_model_matching_{inspect.currentframe().f_code.co_name}" #Gives folder name of the current out channel
    print(filename)
    verify_model_matching_test_no_MKL.main(measurement_data,filename, routine="same_in_out")

    
def in_channel():
    #First test dependence on in_channel
    measure_data_in_channel = measurement(in_channel  = [2,4,8,16,32,64,128,256,512,1024],                                      
                                        out_channel = [128],
                                        kernel_size = [3],
                                        padding     = 1,
                                        rank        = [0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                        image_size  = 128,
                                        epochs      = 1
                                        )

    filename_in_channel = f"verify_model_matching_{inspect.currentframe().f_code.co_name}"
    print(filename_in_channel)
    verify_model_matching_test.main(measure_data_in_channel,filename_in_channel)

def out_channel():
    measure_data_out_channel  = measurement(in_channel  = [128],                                      
                                        out_channel = [2,4,8,16,32,64,128,256,512,1024],
                                        kernel_size = [3],
                                        padding     = 1,
                                        rank        = [0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                        image_size  = 128,
                                        epochs      = 1
                                        ) 
    filename_out_channel = f"verify_model_matching_{inspect.currentframe().f_code.co_name}" #Gives folder name of the current out channel
    print(filename_out_channel)
    verify_model_matching_test.main(measure_data_out_channel,filename_out_channel)

def image_size():
    measurement_data_image_size = measurement(in_channel = [4, 16, 128 ,512],
                                              out_channel = [4, 16, 128 ,512],
                                              kernel_size=3,
                                              padding=1,
                                              rank=[0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                              image_size=[4,16,128,512],
                                              epochs=1
                                              )

    filename = f"verify_model_matching_{inspect.currentframe().f_code.co_name}" #Gives folder name of the current out channel
    print(filename)
    verify_model_matching_test.main(measurement_data_image_size,filename, routine = "same_in_out")

def kernel_size():
    measurement_data_kernel_size = measurement(in_channel = [4,16,128,512],
                                              out_channel= [4,16,128,512],
                                              kernel_size=[1,3,5,7],
                                              padding=[0,1,2,3],
                                              rank=[0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                              image_size=128,
                                              epochs=1)
    filename = f"verify_model_matching_{inspect.currentframe().f_code.co_name}" #Gives folder name of the current out channel
    print(filename)
    verify_model_matching_test.main(measurement_data_kernel_size,filename, routine="same_in_out_same_kernel_pad")



if __name__ == "__main__":
    main()