import json
import os
import statistics
import sys
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Dataprocessing import dataproc_utils as utils
from Experiment_runner.experiment_helper_functions import measurement, name_number







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

    
    fig,ax = plt.subplots(4,4)
    ax : list[list[Axes]]
    used_inchannel = [4,16,64,256]
    used_image_size = [4,16,64,256]
    used_image_size_indices = [measurement_parameters.image_size.index(i) for i in used_image_size]
    # used_inchannel = measurement_parameters.in_channel

    handles = []

    for i,_ in enumerate(model_types):
        for j,in_ch in enumerate(used_inchannel):
            for k,size in enumerate(used_image_size):
                # print(k)
                handle = ax[k][j].scatter(0.995+i*0.0025,results[0,i,measurement_parameters.image_size.index(in_ch),measurement_parameters.image_size.index(size)]/results[0,0,measurement_parameters.image_size.index(in_ch),measurement_parameters.image_size.index(size)] , c=utils.get_mathplotlib_colours(i))
                ax[k][j].scatter(1.095+i*0.0025,results[1,i,measurement_parameters.image_size.index(in_ch),measurement_parameters.image_size.index(size)]/results[1,0,measurement_parameters.image_size.index(in_ch),measurement_parameters.image_size.index(size)] , c=utils.get_mathplotlib_colours(i))
                
                if k == 0 and j==0:
                    handles.append(handle)

                # ax[j][2*k+1].scatter(used_inchannel[j],results[1,i,measurement_parameters.image_size.index(in_ch),measurement_parameters.image_size.index(size)] , c=utils.get_mathplotlib_colours(i))


    for i in range(len(ax) * len(ax[0])):
        if i//4  <= 1:
            ax[i//4][i%4].set_ylim([0, 5])
        else:
            ax[i//4][i%4].set_ylim([0, 2.5])
    
        ax[i//4][i%4].set_xlim([0.9, 1.2])
        ax[i//4][i%4].set_xticks([1, 1.1], ["t", "mem"], rotation=0)

        if i%4 == 0:
            ax[i//4][i%4].set_ylabel(f"{used_image_size[i//4]} x {used_image_size[i//4]}")

        if i//4 == 3:
            ax[i//4][i%4].set_xlabel(f"{used_inchannel[i%4]}")

        
            

    fig.supylabel("Image size")
    fig.supxlabel("Input channels = Output channels = ")
    fig.suptitle("Different inference time and memory ratios\n with the regular CNN as the baseline\nt indicates the time ratio, mem indicates the memory ratio")
    model_types[0] = utils.uncomp_alternative_name()
    plt.legend(handles=handles, labels = model_types, loc = 'lower left', bbox_to_anchor = (1.05,0.5),borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.9,1])
    fig.subplots_adjust(hspace=0.5, wspace=0.4, right=0.8)
    fig.set_size_inches(6.4,6.4)
    plt.show()

if __name__ == "__main__":
    main()