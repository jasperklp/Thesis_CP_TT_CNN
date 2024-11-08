import experiment_runner_mkldnn as runner
import experiment_helper_functions as helper
import CNN_models
import torch
import json
import datetime
import os
import logging
import copy

torch.backends.mkldnn.VERBOSE_ON

#Here are dictionaries with the test values
#Test values are changed such that one could more easliy recoginze the kernels of the four filters
#Image size will be kept the same by chosing the right padding.


base_dict = {"in_channels" : 10, "out_channels" : 10, "kernel_size" : (3,5), "padding" : (1,1), "image_size" : 100, "rank" : 2, "epochs" : 1}
change_in_channel_dict = copy.deepcopy(base_dict)
change_in_channel_dict["in_channels"] = 100

change_out_channel_dict = copy.deepcopy(base_dict)
change_out_channel_dict["out_channels"] = 100

change_kernel_padding_dict_1 = copy.deepcopy(base_dict)
change_kernel_padding_dict_1["kernel_size"] = (3,5)
change_kernel_padding_dict_1["padding"] = (1, 2)

change_kernel_padding_dict_2 = copy.deepcopy(base_dict)
change_kernel_padding_dict_2["kernel_size"] = (5,3)
change_kernel_padding_dict_2["padding"] = (2,1)



experiment_results = {}
experiment_results["Measurements"] = [base_dict] #, change_in_channel_dict, change_out_channel_dict, change_kernel_padding_dict_1,change_kernel_padding_dict_2]

logger = logging.getLogger(__name__)


if __name__ == "__main__": 
    #Acquire name for logger and data
    start_date, start_time = helper.get_date_time(True)
    data_folder = f"{os.getcwd()}\\data"
    experiment_name = "experiment_alter_kernels_to_get_mulitpliers"
    event_name = f"{start_date}_{start_time}"
    logging.basicConfig(filename=f"{data_folder}\\log\\{experiment_name}\\{event_name}.txt",level=logging.INFO)

    #Try whether test data folder is available
    if not os.path.exists(f"{data_folder}\\data\\{experiment_name}"):
        logger.error("Output data directory folder does not exist")
        raise FileNotFoundError("Output folder does not exist")


    #Set output data
    measurement_outputs = []

    
    #Start expermint here.
    logger.info(f"Started at {start_date} {start_time}")
    
    for i in experiment_results["Measurements"]:
        in_channels     = i["in_channels"]
        out_channels    = i["out_channels"]
        kernel_size     = i["kernel_size"]
        padding         = i["padding"]
        image_size      = i["image_size"]
        rank            = i["rank"]
        epochs          = i["epochs"]

        print("Uncomp")
        measurement_outputs.append(runner.model_runner(CNN_models.uncomp_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding),epochs,image_size,verbose = False))
        print("CP")
        measurement_outputs.append(runner.model_runner(CNN_models.cp_tensorly_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,rank=rank),epochs,image_size, verbose = False))

    
 

    #End of experiment
    experiment_results["outcomes"] = measurement_outputs
    (end_date, end_time) = helper.get_date_time(True)
    logger.info(f"Finished at {end_date} {end_time}")
    outfile =  open(f"{data_folder}\\data\\{experiment_name}\\{event_name}.json", "w")
    json.dump(experiment_results,outfile, indent=6)
    outfile.close()
        
    

  
    


