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
    df_pytorch = False
    if df_pytorch == True:
        read_file  = "2024-11-12_10.49.10"
        folder = "verify_model_matching_out_channel"
    else:
        read_file = "2024-11-29_14.54.23"
        folder = "verify_model_matching_out_channel"

    results, measurement_parameters, model_types = utils.preprocess_measurement_data(read_file,folder, "out_channel")

    # plot_out_channel_data(results, measurement_parameters, model_types)
    plot_out_channel_ratio_expected(results, measurement_parameters, model_types)
    #plot_out_channel_ratio_uncomp(results, measurement_parameters, model_types)


#Create figure
def plot_out_channel_data(results, measurement_parameters, model_types):
    fig = plt.figure()
    for i,item in enumerate(model_types):
        plt.scatter(measurement_parameters.out_channel,results[0,i,:]/(1024**2))

    plt.title("Memory for different out_channels")
    plt.ylabel("RAM MB")
    plt.xlabel("In_channels")
    plt.xscale("log")
    plt.legend(model_types)
    plt.show()

def plot_out_channel_ratio_expected(results, measurement_parameters, model_types):
    for i,item in enumerate(model_types):
        fig = plt.figure()
        ax = plt.gca()
        plt.scatter(measurement_parameters.out_channel,results[0,i,:]/results[1,i,:])
        plt.title("Memory for different out_channels")
        plt.ylabel("Ratio")
        plt.xlabel("In_channels")
        plt.xscale("log")
        ax.set_ylim([0.75, 1.25])
        plt.legend([item])
    plt.show()

def plot_out_channel_ratio_uncomp(results, measurement_parameters, model_types):
    fig = plt.figure()
    ax = plt.gca()
    for i,item in enumerate(model_types):
        plt.scatter(measurement_parameters.out_channel,results[0,i,:]/results[0,0,:])
    plt.title("Memory for different out_channels, img=128x128, kern=3x3, in=128")
    plt.ylabel("Ratio")
    plt.xlabel("# of out_channels")
    plt.xscale("log")
    # ax.set_ylim([0, 1])
    ax.set_xticks(measurement_parameters.out_channel)
    ax.set_xticklabels(measurement_parameters.out_channel)
    plt.legend(model_types)
    plt.show()
    
if __name__ == "__main__":
    main()

    



