import os
import json
import sys
#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Dataprocessing import dataproc_utils as utils
from Experiment_1.experiment_helper_functions import measurement
import numpy as np
import matplotlib.pyplot as plt


def main():
    use_df_pytorch = False
    if use_df_pytorch == True:
        read_file = "2024-11-22_12.38.58"
        folder = "verify_model_matching_in_channel_default_pytorch"
    else:
        read_file  = "2024-11-29_15.52.43"
        folder = "verify_model_matching_in_channel"

    

    results, measurement_parameters, model_types = utils.preprocess_measurement_data(read_file,folder, "in_channel")

    plot_in_channel_data(results, measurement_parameters, model_types)
    plot_in_channel_ratio_expected(results, measurement_parameters, model_types)
    plot_in_channel_ratio_uncomp(results,measurement_parameters,model_types)


#Create figure
def plot_in_channel_data(results, measurement_parameters, model_types):
    fig = plt.figure()
    for i,item in enumerate(model_types):
        plt.scatter(measurement_parameters.in_channel,results[0,i,:]/(1024**2))

    plt.title("Memory for different in_channels")
    plt.ylabel("RAM MB")
    plt.xlabel("# of in_channels")
    plt.xscale("log")
    plt.legend(model_types)
    plt.show()

def plot_in_channel_ratio_expected(results, measurement_parameters, model_types):
    for i,item in enumerate(model_types):
        fig = plt.figure()
        ax = plt.gca()
        plt.scatter(measurement_parameters.in_channel,results[0,i,:]/results[1,i,:])
        plt.title("Memory for different in_channels")
        plt.ylabel("Ratio")
        plt.xlabel("In_channels")
        plt.xscale("log")
        ax.set_ylim([0.75, 1.25])
        plt.legend([item])
    plt.show()

def plot_in_channel_ratio_uncomp(results, measurement_parameters, model_types):
    fig = plt.figure()
    ax = plt.gca()
    for i,item in enumerate(model_types):
        plt.scatter(measurement_parameters.in_channel,results[0,i,:]/results[0,0,:])
    plt.title("Memory for different in_channels, img=128x128, kern=3x3, out=128")
    plt.ylabel("Ratio")
    plt.xlabel("In_channels")
    plt.xscale("log")
    ax.set_ylim([0, 10])
    ax.set_xticks(measurement_parameters.in_channel)
    ax.set_xticklabels(measurement_parameters.in_channel)
    plt.legend(model_types)
    plt.show()
    
if __name__ == "__main__":
    main()

    



