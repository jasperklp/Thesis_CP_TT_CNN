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
    tt = True
    if tt == True:
        read_file  = "2024-12-04_16.13.43"
        folder = "verify_model_matching_tt_kernel_size_default_pytorch"
    else:
        if df_pytorch == True:
            read_file  = "2024-11-12_10.49.10"
            folder = "verify_model_matching_kernel_size_default_pytorch"
        else:
            read_file = "2024-11-29_14.58.56"
            folder = "verify_model_matching_kernel_size"

    results, measurement_parameters, model_types = utils.preprocess_measurement_data(read_file,folder, "kernel_size", "in_channel")

    # plot_image_size_data(results, measurement_parameters, model_types)
    # plot_image_size_data_ratio(results, measurement_parameters, model_types)
    plot_image_size_expect_ratio(results, measurement_parameters, model_types)
    # plot_slope(results, measurement_parameters, model_types)


#Create figure
def plot_image_size_data(results, measurement_parameters, model_types):
    fig,ax = plt.subplots(2,2)
    # ax = ax[0]
    print(results.shape)
    for i,item in enumerate(model_types):
        ax[0][0].scatter(measurement_parameters.in_channel,results[0,i,0,:] / 1024**2)
        ax[0][1].scatter(measurement_parameters.in_channel,results[0,i,1,:] / 1024**2)
        ax[1][0].scatter(measurement_parameters.in_channel,results[0,i,2,:] / 1024**2)       
        ax[1][1].scatter(measurement_parameters.in_channel,results[0,i,3,:] / 1024**2)

    var1 = measurement_parameters.kernel_size
    var2 = measurement_parameters.in_channel
    print(var2)

    for i,kernel_size in enumerate(var1):
        ax[i//2][i%2].set_title(f"Kernel size = {kernel_size} x {kernel_size}")
        ax[i//2][i%2].set_xscale("log")
        ax[i//2][i%2].set_yscale("log")
        ax[i//2][i%2].set_ylabel("RAM MB")
        ax[i//2][i%2].set_xlabel("In_channels and out_channels")
        ax[i//2][i%2].set_xticks(measurement_parameters.in_channel)
        ax[i//2][i%2].set_xticklabels(measurement_parameters.in_channel)
   

    plt.suptitle("Memory for different kernel_sizes")
    
    
    plt.legend(model_types, loc = 'lower left', bbox_to_anchor = (1.05,1.05),borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.9,1])
    fig.subplots_adjust(hspace=0.5, right=0.8)
    plt.show()

def plot_image_size_data_ratio(results, measurement_parameters, model_types):
    fig,ax = plt.subplots(2,2)
    # ax = ax[0]
    print(results.shape)
    for i,item in enumerate(model_types):
        ax[0][0].scatter(measurement_parameters.in_channel,results[0,i,0,:] / results[0,0,0,:])
        ax[0][1].scatter(measurement_parameters.in_channel,results[0,i,1,:] / results[0,0,1,:])
        ax[1][0].scatter(measurement_parameters.in_channel,results[0,i,2,:] / results[0,0,2,:])       
        ax[1][1].scatter(measurement_parameters.in_channel,results[0,i,3,:] / results[0,0,3,:])

    var1 = measurement_parameters.kernel_size
    var2 = measurement_parameters.in_channel
    print(var2)

    for i,kernel_size in enumerate(var1):
        ax[i//2][i%2].set_title(f"Kernel size = {kernel_size} x {kernel_size}")
        ax[i//2][i%2].set_xscale("log")
        ax[i//2][i%2].set_yscale("log")
        ax[i//2][i%2].set_ylabel("Ratio")
        ax[i//2][i%2].set_xlabel("In_channels = out_channels = ")
        ax[i//2][i%2].set_ylim([0.5,50])
        ax[i//2][i%2].set_xticks(measurement_parameters.in_channel)
        ax[i//2][i%2].set_xticklabels(measurement_parameters.in_channel)
        


    plt.suptitle("Ratio between uncomp and cp for different kernel_sizes and in_channel sizes")
    
    
    plt.legend(model_types, loc = 'lower left', bbox_to_anchor = (1.05,1.05),borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.9,1])
    fig.subplots_adjust(hspace=0.5, right=0.8)
    plt.show()

def plot_image_size_expect_ratio(results, measurement_parameters, model_types):
    fig,ax = plt.subplots(2,2)
    
    for i,item in enumerate(model_types):
        ax[0][0].scatter(measurement_parameters.in_channel,results[0,i,0,:] / results[1,i,0,:])
        ax[0][1].scatter(measurement_parameters.in_channel,results[0,i,1,:] / results[1,i,1,:])
        ax[1][0].scatter(measurement_parameters.in_channel,results[0,i,2,:] / results[1,i,2,:])       
        ax[1][1].scatter(measurement_parameters.in_channel,results[0,i,3,:] / results[1,i,3,:])

    var1 = measurement_parameters.kernel_size
    var2 = measurement_parameters.in_channel
    print(var2)

    for i,kernel_size in enumerate(var1):
        ax[i//2][i%2].set_title(f"Kernel size = {kernel_size} x {kernel_size}")
        ax[i//2][i%2].set_xscale("log")
        ax[i//2][i%2].set_ylabel("Ratio")
        ax[i//2][i%2].set_xlabel("In_channels = out_channels = ")
        ax[i//2][i%2].set_ylim([0.5,1.1])
        ax[i//2][i%2].set_xticks(measurement_parameters.in_channel)
        ax[i//2][i%2].set_xticklabels(measurement_parameters.in_channel)


    plt.suptitle("Ratio between expected and measured memory for different values.")
    
    
    plt.legend(model_types, loc = 'lower left', bbox_to_anchor = (1.05,1.05),borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.9,1])
    fig.subplots_adjust(hspace=0.5, right=0.8)
    plt.show()

def plot_slope(results, measurement_parameters, model_types):
    fig,ax = plt.subplots(2,2)

    model_types_to_plot = ["uncomp", 0.05 , 0.1 , 0.25]
    kernel_size = measurement_parameters.kernel_size

    for i,item in enumerate(kernel_size):
        get_model_index = lambda x : model_types.index(model_types_to_plot[x])
        ax[0][0].scatter(measurement_parameters.in_channel,results[0,get_model_index(0),i,:] / 1024**2)
        ax[0][1].scatter(measurement_parameters.in_channel,results[0,get_model_index(1),i,:] / 1024**2)
        ax[1][0].scatter(measurement_parameters.in_channel,results[0,get_model_index(2),i,:] / 1024**2)       
        ax[1][1].scatter(measurement_parameters.in_channel,results[0,get_model_index(3),i,:] / 1024**2)
    

    for i,model_type in enumerate(model_types_to_plot):
        ax[i//2][i%2].set_title(f"Model_typ = {model_type}")
        ax[i//2][i%2].set_xscale("log")
        ax[i//2][i%2].set_yscale("log")
        ax[i//2][i%2].set_ylabel("RAM [MB]")
        ax[i//2][i%2].set_xlabel("In_channels = out_channels = ")
        ax[i//2][i%2].set_ylim([0.5,5*10**3])
        ax[i//2][i%2].set_xticks(measurement_parameters.in_channel)
        ax[i//2][i%2].set_xticklabels(measurement_parameters.in_channel)


    plt.legend(kernel_size, loc = 'lower left', bbox_to_anchor = (1.05,1.05),borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.9,1])
    fig.subplots_adjust(hspace=0.5, right=0.8)
    plt.show()

    
if __name__ == "__main__":
    main()

    



