import Experiment_runner.experiment_runner_df_pytorch as runner
import experiment_helper_functions as helper
import CNN_models
import os
import logging
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)



def main(measure_data :helper.measurement, filename, routine = None, verbose : bool = False, mkl_verbose: str = "false", save_single_mem_data_per_output : bool = False):

    #Measurement routines are given. This is a choice for an iterator of the measurement dataclass
    # See the dataclass definition in experiment_helper_functions for more info on the routines. 
    if routine == None:
        iterator = measure_data.__iter__()
    elif routine == "same_in_out":
        iterator = measure_data.iter_same_in_out()
    elif routine == "same_in_out_same_kernel_pad":
        iterator = measure_data.iter_same_in_out_same_kernel_pad()
    else:
        raise ValueError("Routine is not valid. See the main function for valid options or give no option for default routine")

    

    #Acquire name for logger and data
    start_date, start_time = helper.get_date_time(True)
    data_folder = f"{os.getcwd()}\\data"
    experiment_name = filename
    event_name = f"{start_date}_{start_time}"
    logging.basicConfig(filename=f"{data_folder}\\log\\{experiment_name}\\{event_name}.txt",level=logging.DEBUG, force=True)
    logger.info(f"The measurement routine = {routine}")
    #Try whether test data folder is available
    data_path = f"{data_folder}\\data\\{experiment_name}"
    if not os.path.exists(f"{data_path}"):
        logger.error("Output data directory folder does not exist. {}")
        raise FileNotFoundError("Output folder does not exist")
    
    logger.info("Data will be stored in the folder {data_folder}\\data\\{experiment_name}")


    #Set output data
    measurement_outputs = []

    
    #Start expermint here.
    logger.info(f"Started at {start_date} {start_time}")
    for in_channels, out_channels, kernel_size, stride, padding, dilation, image_size, rank, epochs, models in tqdm(iterator, total=measure_data.amount_of_measurements(iterator)): # or use in iterator: for no tqdm (when print out is important)
        
        #Execute uncomp model for given rank
        if mkl_verbose in {"true", "one"}:
            print(f"{in_channels=}", flush=True)
            print(f"{image_size=}", flush=True)

        if "uncomp" in models:
            if mkl_verbose in {"true", "one"}:
                print("Uncomp", flush=True)
            measurement_outputs.append(runner.model_runner(CNN_models.uncomp_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,stride=stride, dilation=dilation),epochs,image_size,verbose = verbose, mkl_verbose=mkl_verbose, save_mem_first_only=save_single_mem_data_per_output)) 
        if "cp" in models:
            for i in rank:
                #Execute CP model for given rank
                if mkl_verbose in {"true", "one"}:
                    print(f"CP with rank={i}",flush=True)
                measurement_outputs.append(runner.model_runner(CNN_models.cp_tensorly_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,rank=i,stride=stride, dilation=dilation),epochs,image_size, verbose = verbose, mkl_verbose=mkl_verbose, save_mem_first_only=save_single_mem_data_per_output))
        if "tt" in models:
            for i in rank:
                #Execute TT model for given rank
                if mkl_verbose in {"true", "one"}:
                    print(f"TT with rank={i}",flush=True)
                measurement_outputs.append(runner.model_runner(CNN_models.tt_tensorly_model(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,rank=i,stride=stride, dilation=dilation),epochs,image_size, verbose = verbose, mkl_verbose=mkl_verbose, save_mem_first_only=save_single_mem_data_per_output))




    #End of experiment
    experiment_results = {"Setup_data" : measure_data.as_dict()}
    experiment_results["outcomes"] = measurement_outputs
    (end_date, end_time) = helper.get_date_time(True)
    logger.info(f"Finished at {end_date} {end_time}")
    outfile =  open(f"{data_folder}\\data\\{experiment_name}\\{event_name}.json", "w")
    json.dump(experiment_results,outfile, indent=6)
    outfile.close()
