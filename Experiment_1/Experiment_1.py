import verify_model_matching_test
from experiment_helper_functions import measurement
import logging
import inspect

logger = logging.getLogger(__name__)

def main():
    in_channel()

def in_channel():
    #First test dependence on in_channel
    measure_data_in_channel = measurement(in_channel  = [2,4,8,16,32,64,128,256,512,1024],                                      
                                        out_channel = [128],
                                        kernel_size = [3],
                                        padding     = 1,
                                        rank        = [0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                        image_size  = 128,
                                        epochs      = 10
                                        )

    filename_in_channel = f"verify_model_matching_{inspect.currentframe().f_code.co_name}"
    
    verify_model_matching_test.main(measure_data_in_channel,filename_in_channel)

def out_channel():
    measure_data_out_channel  = measurement(in_channel  = [128],                                      
                                        out_channel = [2,4,8,16,32,64,128,256,512,1024],
                                        kernel_size = [3],
                                        padding     = 1,
                                        rank        = [0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                        image_size  = 128,
                                        epochs      = 10
                                        ) 
    filename_out_channel = f"verify_model_matching_{inspect.currentframe().f_code.co_name}" #Gives folder name of the current out channel

    verify_model_matching_test.main(measure_data_out_channel,filename_out_channel)

if __name__ == "__main__":
    main()