import json
import os
import statistics
import sys
import matplotlib.pyplot as plt

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Dataprocessing import dataproc_utils as utils
from Experiment_runner.experiment_helper_functions import measurement


def main():
    file = "2024-12-15_11.48.47"
    folder =  "experiment_test_time"
    with open(f"{os.getcwd()}\\data\\data\\{folder}\\{file}.json") as json_file:
        data = json.load(json_file)

    measurement_parameters =  measurement.from_dict(data["Setup_data"])

    used_models = None
    used_ranks = [0.01, 0.05, 0.1 , 0.25]
    # plot_time_and_mem(data)
    # plot_time_per_system(file,  folder, measurement_parameters, used_models=["uncomp", "cp"])
    # plot_time_per_system(file,  folder, measurement_parameters, used_models=["uncomp", "tt"])
    plot_time_per_mem(file, folder, measurement_parameters, used_models=used_models, used_ranks=used_ranks)
    plot_time_per_MAC(file, folder, measurement_parameters, used_models=used_models, used_ranks=used_ranks)
    # plot_time_per_mem(file, folder, measurement_parameters, used_models=used_models, used_ranks=used_ranks, zoomed="super")
    # plot_time_per_MAC(file, folder, measurement_parameters, used_models=used_models, used_ranks=used_ranks, zoomed="super")
    plot_time_per_mem(file, folder, measurement_parameters, used_models=used_models, used_ranks=used_ranks, zoomed=True)
    plot_time_per_MAC(file, folder, measurement_parameters, used_models=used_models, used_ranks=used_ranks, zoomed=True)
    # plot_time_over_mem_four(file, folder, measurement_parameters)
    plt.show()


def plot_time_per_system(file, folder, measurement_parameters : measurement, used_models = None, used_ranks = None):
    fig, ax = plt.subplots(2,2)

    var1 = "in_channel"
    var2 = "image_size"

    results, model_types = utils.preprocess_time_data(file, folder, var1, var2, "iter_same_in_out", used_models=used_models, used_ranks=used_ranks)
    # print(measurement_parameters)
    for i,model in enumerate(model_types):
        ax[0][0].scatter(getattr(measurement_parameters,var1), results[0,i,0,:])
        ax[0][1].scatter(getattr(measurement_parameters,var1), results[0,i,1,:])
        ax[1][0].scatter(getattr(measurement_parameters,var1), results[0,i,2,:])
        ax[1][1].scatter(getattr(measurement_parameters,var1), results[0,i,3,:])

    for i,image_size in enumerate(getattr(measurement_parameters, var2)):
        ax[i//2][i%2].set_title(f"Image size = {image_size} x {image_size}")
        ax[i//2][i%2].set_xscale("log")
        ax[i//2][i%2].set_yscale("log")
        ax[i//2][i%2].set_ylabel("Time in seconds")
        ax[i//2][i%2].set_xlabel("In_channels = out_channels = ")
        ax[i//2][i%2].set_xticks(measurement_parameters.in_channel)
        ax[i//2][i%2].set_xticklabels(measurement_parameters.in_channel)

    model_types[0] = utils.uncomp_alternative_name()
    plt.legend(model_types, loc = 'lower left', bbox_to_anchor = (1.05,1.05),borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.9,1])
    fig.subplots_adjust(hspace=0.5, right=0.8)
    #plt.show()

def plot_time_per_mem(file, folder, measurement_parameters : measurement, used_models = None, used_ranks = None, zoomed : bool|str = False):
    fig = plt.figure()
    var1 = "in_channel"
    var2 = "image_size"
    results, model_types, nr_of_tests = utils.preprocess_time_all_combinations(file, folder, used_models=used_models, used_ranks=used_ranks)
    markers = ["o", "v", "^", "<", ">", "1","2","3","4","8","s","p","P","*","h","H","D","d"]

    for i in range(nr_of_tests):
        for j,model in enumerate(model_types):
            plt.scatter(results[1,j,i] / 1024**2,results[0,j,i],c=utils.get_mathplotlib_colours(j))

    # plt.xscale("log")
    # plt.yscale("log")

    plt.xlabel("Memory [MB]")
    plt.ylabel("Time [s]")
    plt.title("in_ch, out_ch, img_size in range [4,8,16,32,64,96,128,192,256] all comb.")

    if zoomed == True:
        plt.suptitle("Memory operations vs time (Zoomed in)")
        plt.axis([0, 100, 0, 0.02])
    elif zoomed == "super":
        plt.suptitle("Memory operations vs time (Super zoomed in)")
        plt.axis([0, 20, 0, 0.0025])
    else:
        plt.suptitle("Memory operations vs time")


    model_types[0] = utils.uncomp_alternative_name()
    plt.legend(model_types)
    plt.tight_layout(rect=[0,0,0.9,1])
    # fig.subplots_adjust(hspace=0.5, right=0.8)
    #plt.show()

def plot_time_per_MAC(file, folder, measurement_parameters : measurement, used_models = None, used_ranks = None, zoomed : bool = False):
    fig = plt.figure()

    results, model_types, nr_of_tests = utils.preprocess_time_all_combinations(file, folder, used_models=used_models, used_ranks=used_ranks)
    markers = ["o", "v", "^", "<", ">", "1","2","3","4","8","s","p","P","*","h","H","D","d"]

    for i in range(nr_of_tests):
        for j,model in enumerate(model_types):
            plt.scatter(results[2,j,i],results[0,j,i], c=utils.get_mathplotlib_colours(j))

    # plt.xscale("log")
    # plt.yscale("log")
    
    plt.xlabel("MAC operations")
    plt.ylabel("Time [s]")

    plt.title("in_ch, out_ch, img_size in range [4,8,16,32,64,96,128,192,256] all comb.")

    if zoomed == True:
        plt.suptitle("MAC operations vs time (Zoomed in)")
        plt.axis([0, 0.5*10**10, 0, 0.02])
    elif zoomed == "super":
        plt.suptitle("MAC operations vs time (Super zoomed in)")
        plt.axis([0, 1*10**9, 0, 0.0025])
    else:
        plt.suptitle("MAC operations vs time")
    # plt.axvline(24)
    model_types[0] = utils.uncomp_alternative_name()
    plt.legend(model_types)
    plt.tight_layout(rect=[0,0,0.9,1])
    # fig.subplots_adjust(hspace=0.5, right=0.8)
    #plt.show()
    

def plot_time_over_mem_four(file, folder, measurement_parameters : measurement, used_models = None, used_ranks = None):
    fig,ax = plt.subplots(2,2)
    matplotlib_colours = utils.get_mathplotlib_colours()
    var1 = "image_size"
    var2 = "in_channel"
    results, measurement_parameters, model_types = utils.preprocess_measurement_data(file, folder, var1, used_models=["uncomp", "tt","cp"], used_ranks=[0.01, 0.05, 0.1])
    markers = ["o", "v", "^", "<", ">", "1","2","3","4","8","s","p","P","*","h","H","D","d"]

    for i, var in enumerate(getattr(measurement_parameters,var2)):
        for j,model in enumerate(model_types):
            ax[0][0].scatter(results[1,j,0,i] ,results[0,j,0,i], marker = markers[i], c=matplotlib_colours[j])
            ax[0][1].scatter(results[1,j,1,i] ,results[0,j,1,i], marker = markers[i], c=matplotlib_colours[j])
            ax[1][0].scatter(results[1,j,2,i] ,results[0,j,2,i], marker = markers[i], c=matplotlib_colours[j])
            ax[1][1].scatter(results[1,j,3,i] ,results[0,j,3,i], marker = markers[i], c=matplotlib_colours[j])

    for i, var in enumerate(getattr(measurement_parameters,var2)):
            if i >= 4:
                continue
            print(i)
            print(i//2)
            ax[i//2][i%2].set_xlabel("Memory [B]")
            ax[i//2][i%2].set_ylabel("Time [s]")
            # ax[i//2][i%2].set_xscale("log")
            # ax[i//2][i%2].set_yscale("log")
            ax[i//2][i%2].set_title(f"{var2} = {var} x {var}")
            
    # ax[1][1].set(xlim=(900,2200), ylim =(0,0.4))


    # plt.axvline(24)

    plt.legend(model_types, loc = 'lower left', bbox_to_anchor = (1.05,1.05),borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.9,1])
    fig.subplots_adjust(hspace=0.5, right=0.8)
    #plt.show()


def plot_time_and_mem(data):
    for i in data["outcomes"]:
        print(f"{i["model_name"]=}")
        print(f"{i["in_channel"]=}")
        print(f"{i["out_channel"]=}")
        print(f"{i["image_size"]=}")
        if i.get("rank") is not None:
            print(f"{i["rank"]=}")
        print(f"Mean time = {statistics.mean(i["Inference duration"]):.6f}")
        if len(i["Inference duration"]) > 1:
            print(f"Standard deviation = {statistics.stdev(i["Inference duration"]):.6f}")
        print(f"Memory usage is {i["measurements"][0]["Total allocated RAM"]/(1024**2):.2f} MB")
        print("\n")


# def plot_system_time():

if __name__ == '__main__':
    main()