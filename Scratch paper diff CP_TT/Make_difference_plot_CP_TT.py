import sys
import os
#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Experiment_runner import calc_expectation as calc_expextation
from Experiment_runner import experiment_helper_functions as helper
import matplotlib.pyplot as plt


def main():
    m_in_channel    = [4,16,128,512]
    m_out_channel   = m_in_channel
    m_kernel_size   = [3]
    m_padding       = [i//2 for i in m_kernel_size]
    m_image_size    = [4,16,128,512]
    m_rank          = [0.001,0.01, 0.1, 0.25]

    measurement = helper.measurement(m_in_channel, m_out_channel, m_kernel_size, m_image_size, m_rank, m_padding)

    results  = np.zeros((3, len(m_rank), len(m_in_channel), len(m_image_size))) #model type , #rank, #in_out_channel #image_size
    # print(f"{results.shape=}")

    for in_channel, out_channel, kernel_size, stride, padding, dilation, image_size, rank, _ in measurement.iter_same_in_out():
        for r in rank:
            results[0][m_rank.index(r)][m_in_channel.index(in_channel)][m_image_size.index(image_size)] = calc_expextation.ram_estimation_2d(in_channel,out_channel,kernel_size,image_size,'cp',stride,padding,dilation,r)
            results[1][m_rank.index(r)][m_in_channel.index(in_channel)][m_image_size.index(image_size)] = calc_expextation.ram_estimation_2d(in_channel,out_channel,kernel_size,image_size,'tt',stride,padding,dilation,r)
        results[2][0][m_in_channel.index(in_channel)][m_image_size.index(image_size)] = calc_expextation.ram_estimation_2d(in_channel,out_channel, kernel_size,image_size,'uncomp',stride,padding,dilation)
    plot_elements_tt_vs_cp(results,measurement)
    plot_ratio_elements_tt_vs_cp(results, measurement)

        

def plot_elements_tt_vs_cp(results, measurements):
    fig,ax = plt.subplots(2,2)
    models = ["cp", "tt", "uncomp"]
    ranks = measurements.rank
    model_types = [f"{model} {rank}" for model in models for rank in ranks if ((model != "uncomp"))]
    model_types.append("uncomp")
    for i,item in enumerate(model_types):
        ax[0][0].scatter(measurements.in_channel,results[i//len(ranks), i%len(ranks), :, 0])
        ax[0][1].scatter(measurements.in_channel,results[i//len(ranks), i%len(ranks), :, 1])
        ax[1][0].scatter(measurements.in_channel,results[i//len(ranks), i%len(ranks), :, 2])
        ax[1][1].scatter(measurements.in_channel,results[i//len(ranks), i%len(ranks), :, 3])

    for i, image_size in enumerate(measurements.image_size):
        ax[i//2][i%2].set_title(f"Image size = {image_size} x {image_size}")
        ax[i//2][i%2].set_xscale("log")
        ax[i//2][i%2].set_yscale("log")
        ax[i//2][i%2].set_xlabel("In_channels and out_channels")
        ax[i//2][i%2].set_xticks(measurements.in_channel)
        ax[i//2][i%2].set_xticklabels(measurements.in_channel)
    
    plt.suptitle("Memory elements for different CNN layer types.")
    plt.legend(model_types, loc = 'lower left', bbox_to_anchor = (1.05,1.05),borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.9,1])
    fig.subplots_adjust(hspace=0.5, right=0.8)
    plt.show()

def plot_ratio_elements_tt_vs_cp(results, measurements):
    fig,ax = plt.subplots(2,2)
    plt.suptitle("Ratio (TT/CP) of amount of elements")
    for i,r in enumerate(measurements.rank):
        ax[0][0].scatter(measurements.in_channel,results[1, i, :, 0]/results[0, i, :, 0])
        ax[0][1].scatter(measurements.in_channel,results[1, i, :, 1]/results[0, i, :, 1])
        ax[1][0].scatter(measurements.in_channel,results[1, i, :, 2]/results[0, i, :, 2])
        ax[1][1].scatter(measurements.in_channel,results[1, i, :, 3]/results[0, i, :, 3])

    for i, image_size in enumerate(measurements.image_size):
        ax[i//2][i%2].set_title(f"Image size = {image_size} x {image_size}")
        ax[i//2][i%2].set_xscale("log")
        # ax[i//2][i%2].set_yscale("log")
        ax[i//2][i%2].set_xlabel("In_channels and out_channels")
        ax[i//2][i%2].set_xticks(measurements.in_channel)
        ax[i//2][i%2].set_xticklabels(measurements.in_channel)

    plt.legend(measurements.rank, loc = 'lower left', bbox_to_anchor = (1.05,1.05),borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.9,1])
    fig.subplots_adjust(hspace=0.5, right=0.8)  

    plt.show()
    




if __name__ == "__main__":
    main()
