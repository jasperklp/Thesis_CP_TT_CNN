import os
import sys
import json

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Dataprocessing.dataproc_utils import get_theoretical_results_in_preprocess_measurement_format, uncomp_alternative_name
from Dataprocessing.Experiment_1.data_proc_experiment_1_system_size import plot_image_size_data_ratio
from Dataprocessing.Experiment_1.data_proc_experiment_1_kernel_size import plot_image_size_data_ratio as plot_kernel_size_data_ratio







def system_size():

    folder = "verify_model_matching_ttcp_image_size_default_pytorch"
    read_file = "2025-01-21_16.26.18"

    results, measument_parameters, model_types = get_theoretical_results_in_preprocess_measurement_format(read_file, folder, "same_in_out", "image_size", "in_channel", used_ranks=[0.01,0.05,0.1,0.25,1.0] )    

    # print(measument_parameters)
    # print(model_types)
    # print(results)
    print(model_types)
    model_types[0] = uncomp_alternative_name()
    print(model_types)
    plot_image_size_data_ratio(results, measument_parameters, model_types )

def kernel_size():

    folder = "verify_model_matching_ttcp_kernel_size_default_pytorch"
    read_file = "2025-01-21_16.27.13"

    results, measument_parameters, model_types = get_theoretical_results_in_preprocess_measurement_format(read_file, folder, "same_in_out_same_kernel_pad", "kernel_size", "in_channel", used_ranks=[0.01,0.05,0.1,0.25,1.0] )    

    # print(measument_parameters)
    # print(model_types)
    # print(results)
    print(model_types)
    model_types[0] = uncomp_alternative_name()
    print(model_types)
    plot_kernel_size_data_ratio(results, measument_parameters, model_types )


if __name__ == "__main__":
    system_size()
    # kernel_size()