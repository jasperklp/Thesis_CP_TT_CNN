import json
import os
import statistics
import sys
import matplotlib.pyplot as plt

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

    
    fig,ax = plt.subplots(3,3)
    used_inchannel = [8,32,256]
    used_image_size = [8,32,256]
    used_image_size_indices = [measurement_parameters.image_size.index(i) for i in used_image_size]
    # used_inchannel = measurement_parameters.in_channel


    for i,_ in enumerate(model_types):
        for j,in_ch in enumerate(used_inchannel):
            for k,size in enumerate(used_image_size):
                ha_list = ['right' , 'right', 'left' , 'right', 'left']
                va_list = ['center' , 'center', 'center', 'center', 'center']

                if k==2 or j==2:
                    ha_list = ['right' , 'right', 'right' , 'left', 'left']

                ax[k][j].scatter(used_inchannel[j],results[0,i,measurement_parameters.image_size.index(in_ch),measurement_parameters.image_size.index(size)] , c=utils.get_mathplotlib_colours(i))
                ax[k][j].annotate("  " + name_number(int(results[1,i,measurement_parameters.image_size.index(in_ch),measurement_parameters.image_size.index(size)]), always_three_digits=True) + "  " 
                                  , (used_inchannel[j],results[0,i,measurement_parameters.image_size.index(in_ch),measurement_parameters.image_size.index(size)])
                                  , ha=ha_list[i], va = va_list[i]
                )
                

                # ax[j][2*k+1].scatter(used_inchannel[j],results[1,i,measurement_parameters.image_size.index(in_ch),measurement_parameters.image_size.index(size)] , c=utils.get_mathplotlib_colours(i))


    for i in range(len(ax) * len(ax[0])):
        _, top = ax[i//3][i%3].get_ylim()
        ax[i//3][i%3].set_ylim([0, float(top*1.1)])

    for i in range(3):
        ax[0][i].set_ylim([0, 0.0017])

    for i in range(3):
        ax[1][i].set_ylim([0, 0.0032])


    plt.legend(model_types, loc = 'lower left', bbox_to_anchor = (1.05,0.5),borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.9,1])
    fig.subplots_adjust(hspace=0.5, right=0.8)
    plt.show()

if __name__ == "__main__":
    main()