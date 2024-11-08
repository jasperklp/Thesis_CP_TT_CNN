import experiment_runner_mkldnn as runner
import experiment_helper_functions as helper
import CNN_models
import os
import logging
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

def main():
    measure_data = helper.measurement(in_channel  = [2,4,8,16,64,128,256,1024],                                      
                                    out_channel = [128],
                                    kernel_size = [3],
                                    padding     = 1,
                                    rank        = [0.01,0.05,0.1,0.25,0.5,0.75,1],
                                    image_size  = 128,
                                    epochs      = 1
                                    )

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

    for (in_channels, out_channels, kernel_size, stride, padding, dilation, image_size, rank, epochs) in tqdm(measure_data):
        measurement_outputs.append(runner.model_runner(CNN_models.uncomp_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,stride=stride),epochs,image_size,verbose = False)) 
        for i in rank:
            measurement_outputs.append(runner.model_runner(CNN_models.cp_tensorly_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,rank=i,stride=stride),epochs,image_size, verbose = False))




    #End of experiment
    experiment_results = {"Setup_data" : measure_data.as_dict()}
    experiment_results["outcomes"] = measurement_outputs
    (end_date, end_time) = helper.get_date_time(True)
    logger.info(f"Finished at {end_date} {end_time}")
    outfile =  open(f"{data_folder}\\data\\{experiment_name}\\{event_name}.json", "w")
    json.dump(experiment_results,outfile, indent=6)
    outfile.close()



if __name__ == "__main__":
    main()