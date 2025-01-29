import json
import os
import statistics
import sys
import matplotlib.pyplot as plt

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Dataprocessing import dataproc_utils as utils
from Experiment_1.experiment_helper_functions import measurement







def main():
    file = "2024-12-15_11.48.47"
    folder =  "experiment_test_time"
    with open(f"{os.getcwd()}\\data\\data\\{folder}\\{file}.json") as json_file:
        data = json.load(json_file)

    measurement_parameters =  measurement.from_dict(data["Setup_data"])

    used_models = None
    used_ranks = [0.01, 0.1]

    var1 = "in_channel"
    var2 = "image_size"

    results, model_types = utils.preprocess_time_data(file, folder, var1, var2, "iter_same_in_out", used_models=used_models, used_ranks=used_ranks)




if __name__ == "__main__":
    main()