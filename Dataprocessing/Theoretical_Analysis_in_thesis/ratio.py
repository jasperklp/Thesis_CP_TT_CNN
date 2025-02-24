import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Dataprocessing.dataproc_utils import get_theoretical_results_in_preprocess_measurement_format, uncomp_alternative_name, preprocess_measurement_data
import Dataprocessing.dataproc_utils as utils
from Experiment_runner.experiment_helper_functions import measurement


def plot_image_size_data_ratio_theo_act(results_theo, results_act, measurement_parameters : measurement, model_types):
    fig,ax = plt.subplots(2,2)
    ax : list[list[Axes]]
    # ax = ax[0]
    for i,item in enumerate(model_types):
        ax[0][0].scatter([1,2,3,4],results_act[0,i,0,:] / results_theo[0,i,0,:], c=utils.get_mathplotlib_colours(i))
        ax[0][1].scatter([1,2,3,4],results_act[0,i,1,:] / results_theo[0,i,1,:], c=utils.get_mathplotlib_colours(i))
        ax[1][0].scatter([1,2,3,4],results_act[0,i,2,:] / results_theo[0,i,2,:], c=utils.get_mathplotlib_colours(i))       
        ax[1][1].scatter([1,2,3,4],results_act[0,i,3,:] / results_theo[0,i,3,:], c=utils.get_mathplotlib_colours(i))



    var1 = measurement_parameters.image_size
    var2 = measurement_parameters.in_channel

    for i,image_size in enumerate(var1):
        ax[i//2][i%2].axhline(y=1, color='r')
        ax[i//2][i%2].set_title(f"Image size = {image_size} x {image_size}")
        # ax[i//2][i%2].set_xscale("log")
        ax[i//2][i%2].set_ylabel("Ratio")
        ax[i//2][i%2].set_xlabel("In_channels = out_channels = ")
        ax[i//2][i%2].set_xticks([1,2,3,4], labels=measurement_parameters.in_channel)

    ax[0][0].set_ylim([0,6]) 
    ax[0][1].set_ylim([0,6])
    ax[1][0].set_ylim([0,6]) 
    ax[1][1].set_ylim([0,6])

    plt.suptitle("Ratio between actual over theoretical amount of memory")
    
    plt.legend(model_types + ["ratio=1"], loc = 'lower left', bbox_to_anchor = (1.05,0.5),borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.9,1])
    fig.subplots_adjust(hspace=0.8, right=0.8)
    plt.show()



def system_size():

    folder = "verify_model_matching_ttcp_image_size_default_pytorch"
    read_file = "2025-01-21_16.26.18"

    results_theory, _, model_types = get_theoretical_results_in_preprocess_measurement_format(read_file, folder, "same_in_out", "image_size", "in_channel", used_models = ["uncomp" , "cp" , "tt"],used_ranks=[0.01,0.05,0.1,0.25,1.0] )    
    results_actual, measument_parameters, model_types =  preprocess_measurement_data(read_file,folder, "image_size", "in_channel", used_models = ["uncomp" , "cp" , "tt"],used_ranks=[0.01,0.05,0.1,0.25,1.0])

    # print(measument_parameters)
    # print(model_types)
    # print(results)
    model_types[0] = uncomp_alternative_name()
    plot_image_size_data_ratio_theo_act(results_theory, results_actual, measument_parameters, model_types )

# def kernel_size():

#     folder = "verify_model_matching_ttcp_kernel_size_default_pytorch"
#     read_file = "2025-01-21_16.27.13"

#     results, measument_parameters, model_types = get_theoretical_results_in_preprocess_measurement_format(read_file, folder, "same_in_out_same_kernel_pad", "kernel_size", "in_channel", used_ranks=[0.01,0.05,0.1,0.25,1.0] )    

#     # print(measument_parameters)
#     # print(model_types)
#     # print(results)
#     print(model_types)
#     model_types[0] = uncomp_alternative_name()
#     print(model_types)
#     plot_kernel_size_data_ratio(results, measument_parameters, model_types )


if __name__ == "__main__":
    system_size()
    # kernel_size()

