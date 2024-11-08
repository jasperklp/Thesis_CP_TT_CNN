import experiment_runner_mkldnn_vtune as runner
import experiment_helper_functions as helper
import CNN_models
import torch
import json
import datetime
import os
import logging

in_channels = 1024
out_channels = 1024
kernel_size = (5,5)
epochs      = 1
image_size = 100
padding = (1,1)
c = [100]

experiment_results = {"Measured value" : "Rank", "Measured range" : c}

logger = logging.getLogger(__name__)




if __name__ == "__main__": 
    #Get name of this file
    filename = os.path.splitext(os.path.basename(__file__))[0]


    #Acquire name for logger and data
    start_date, start_time = helper.get_date_time(True)
    data_folder = f"{os.getcwd()}\\data"
    experiment_name = filename
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
    
    
    # measurement_outputs.append(runner.model_runner(CNN_models.uncomp_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding),epochs,image_size,verbose = False,measure = False))
    # measurement_outputs.append(runner.model_runner(CNN_models.uncomp_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding),epochs,image_size,verbose = False,measure = True))

    for i in c:
        measurement_outputs.append(runner.model_runner(CNN_models.cp_tensorly_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,rank=i),epochs,image_size, verbose = True,measure = False))
        measurement_outputs.append(runner.model_runner(CNN_models.cp_tensorly_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,rank=i),epochs,image_size, verbose = True,measure = True))
 

    #End of experiment
    experiment_results["outcomes"] = measurement_outputs
    (end_date, end_time) = helper.get_date_time(True)
    logger.info(f"Finished at {end_date} {end_time}")
    outfile =  open(f"{data_folder}\\data\\{experiment_name}\\{event_name}.json", "w")
    json.dump(experiment_results,outfile, indent=6)
    outfile.close()
        
    

  
    


