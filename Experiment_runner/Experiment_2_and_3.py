import sys
import os
#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import verify_model_matching_test_no_MKL as modeler
import experiment_helper_functions as helper
import CNN_models as CNN_models
import torch
import json
import datetime
import os
import logging

logger = logging.getLogger(__name__)


def main():
    # test_time = helper.measurement(     in_channel  = [256],                                      
    #                                     out_channel = [256],
    #                                     kernel_size = [3],
    #                                     padding     = 1,
    #                                     rank        = [0.1, 0.5],
    #                                     image_size  = 128,
    #                                     epochs      = 100
    #                                     )

    # test_time = helper.measurement(in_channel = [4,16, 128 ,512],
    #                                           out_channel = [4, 16, 128 ,512],
    #                                           kernel_size=3,
    #                                           padding=1,
    #                                           rank=[0.01,0.05,0.1,0.25,0.5,0.75,1.0],
    #                                           image_size= [4,16,128,512],
    #                                           epochs= 30,
    #                                           models= ["uncomp", "tt", "cp"]
    #                                           )

    test_time = helper.measurement(in_channel = [4,8,16,32,64,96,128,192,256],
                                              out_channel = [4,8,16,32,64,96,128,192,256],
                                              kernel_size=3,
                                              padding=1,
                                              rank=[0.01,0.05,0.1,0.25,0.5,0.75,1.0],
                                              image_size= [4,8,16,32,64,96,128,192,256],
                                              epochs= 10,
                                              models= ["uncomp", "tt", "cp"]
                                              )

    filename = "experiment_test_time"
    modeler.main(test_time, filename, verbose= False, mkl_verbose="false", save_single_mem_data_per_output=True)
    print("Experiment finished.")

    




if __name__ == '__main__':
    main()

