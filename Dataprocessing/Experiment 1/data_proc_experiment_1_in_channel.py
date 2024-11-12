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
    read_file  = "2024-11-08_16.47.59"
    folder = "verify_model_matching_in_channel"

    results, measurement_parameters, model_types = utils.preprocess_measurement_data_single(read_file,folder, "in_channels")

    plot_in_channel_data(results, measurement_parameters, model_types)
    plot_in_channel_ratio_expected(results, measurement_parameters, model_types)


#Create figure
def plot_in_channel_data(results, measurement_parameters, model_types):
    fig = plt.figure()
    for i,item in enumerate(model_types):
        plt.scatter(measurement_parameters.in_channel,results[0,i,:]/(1024**2))

    plt.title("Memory for different in_channels")
    plt.ylabel("RAM MB")
    plt.xlabel("In_channels")
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
    plt.title("Memory for different in_channels")
    plt.ylabel("Ratio")
    plt.xlabel("In_channels")
    plt.xscale("log")
    ax.set_ylim([0, 8])
    plt.legend(model_types)
    plt.show()
    
if __name__ == "__main__":
    main()

    



