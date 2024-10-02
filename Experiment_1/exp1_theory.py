import calc_expectation
import numpy as np
import matplotlib.pyplot as plt    


plot_rank_relation                  : bool  = False
plot_image_size_relation            : bool  = False
plot_in_channel_relation            : bool  = True
plot_out_channel_relation           : bool  = False
plot_kernel_relation                : bool  = False
plot_in_and_out_channel_relation    : bool  = False
plot_rank_over_in_channel           : bool  = False

matplotlib_colours = ['tab:blue', 'tab:orange','tab:green', 'tab:red','tab:purple', 'tab:brown', 'tab:pink','tab:grey', 'tab:olive','tab:cyan']


def plot_dataset(in_channels,out_channels, kernel, image_size, padding, rank, log : bool = True,different_compared_to_baseline = False):
    in_channels     = calc_expectation.check_list_or_int_float(in_channels)
    out_channels    = calc_expectation.check_list_or_int_float(out_channels)
    kernel          = calc_expectation.check_list_or_int_float(kernel)
    image_size      = calc_expectation.check_list_or_int_float(image_size)
    padding         = calc_expectation.check_list_or_int_float(padding)
    measurement_value =  max([in_channels, out_channels, kernel, image_size, padding], key=len)

    data = calc_expectation.get_theoretical_dataset_for_plot(in_channels,out_channels,kernel,image_size,padding, rank)
    data = np.squeeze(data)

    if different_compared_to_baseline == True:
        data[1,0,:,:] = data[1,0,:,:] / data[0,0,:,:]
        data[1,1,:,:] = data[1,1,:,:] / data[0,1,:,:]
        data[0,0,:,:] = data[0,0,:,:] / data[0,0,:,:]
        data[0,1,:,:] = data[0,1,:,:] / data[0,1,:,:]


    if log == True:
        data = np.log10(data)

    datashape = np.shape(data)

    macfigure = plt.figure()
    for i in range(datashape[2]):
        plt.scatter(0,data[0,0,i,0], c=matplotlib_colours[i])
        ax = plt.scatter(c* np.ones((1,len(c))), data[1,0,i,:], c=matplotlib_colours[i])
        ax.set_label(measurement_value[i])


    ramfigure = plt.figure()
    for i in range(datashape[2]):
        plt.scatter(0,data[0,1,i,0], c=matplotlib_colours[i])
        ax = plt.scatter(c* np.ones((1,len(c))), (data[1,1,i,:]), c=matplotlib_colours[i])
        ax.set_label(measurement_value[i])

    return [macfigure, ramfigure]


if __name__ == "__main__":
    if plot_rank_relation == True:
        c = [ 0.1, 0.25, 0.5, 0.75, 1.0]
        in_channels = 1024
        out_channels = 1024
        image_size = 400
        padding = 1
        kernel =  3
        data = calc_expectation.get_theoretical_dataset_for_plot(in_channels,out_channels,kernel,image_size,padding, c)

        data = np.squeeze(data)
        data_uncomp = data[0,1,0]
        data_cp = data[1,1,:]
        print(data_cp.shape)
        plt.scatter(0,data_uncomp)
        plt.scatter(c,data_cp)
        plt.legend("Uncomporessed" , "CP")
        plt.xlabel("Rank (as a part of the uncompressed case)")
        plt.ylabel("Expected memory")
        
    if plot_image_size_relation == True:
        c = [ 0.1, 0.25, 0.5, 0.75, 1.0]
        in_channels = 1024
        out_channels = 1024
        image_size = [4, 40, 400, 4000, 40000 ]
        padding = 1
        kernel =  3
        print_in_log = False
        show_difference = True
        
        [macfigure, ramfigure] = plot_dataset(in_channels,out_channels,kernel,image_size,padding,c,print_in_log,show_difference)
        plt.figure(macfigure)
        plt.title(f"MAC operations for different image sizes")
        plt.legend()

        plt.figure(ramfigure)
        plt.title("RAM for different image sizes")
        plt.legend()

    if  plot_in_channel_relation == True:
        c = [ 0.1, 0.25, 0.5, 0.75, 1.0]
        in_channels = [16, 32, 64, 128,256, 1024,2048,4096]
        out_channels = 256
        image_size = 40
        padding = 1
        kernel =  3
        print_in_log = False
        show_difference = True
        
        [macfigure, ramfigure] = plot_dataset(in_channels,out_channels,kernel,image_size,padding,c,log=print_in_log, different_compared_to_baseline=show_difference)
        plt.figure(macfigure)
        plt.title("MAC operations for different in_channel values")
        plt.suptitle(f"Out_channels = {out_channels}")
        plt.legend()

        plt.figure(ramfigure)
        plt.title("RAM for different in_channel values")
        plt.suptitle(f"Out_channels = {out_channels}")
        plt.legend()

    if(plot_out_channel_relation) == True:
        c = [ 0.1, 0.25, 0.5, 0.75, 1.0]
        in_channels = 256
        out_channels = [16, 32, 64, 128,256, 1024,2048,4096]
        image_size = 40
        padding     = 1
        kernel = 3
        print_in_log = False
        show_difference = True

        [macfigure, ramfigure] = plot_dataset(in_channels,out_channels,kernel,image_size,padding,c,log=print_in_log, different_compared_to_baseline=show_difference)
        plt.figure(macfigure)
        plt.title("MAC operations for different out_channel values")
        plt.suptitle(f"In_channels = {in_channels}")
        plt.legend()

        plt.figure(ramfigure)
        plt.title("RAM for different out_channel values")
        plt.suptitle(f"In_channels = {in_channels}")
        plt.legend()

    if (plot_kernel_relation == True):
        c = [ 0.1, 0.25, 0.5, 0.75, 1.0]
        in_channels = 1024
        out_channels = 1024
        image_size = 40
        padding = 0
        kernel =  [1,3,5,7,9]
        print_in_log = False
        show_difference = False
        
        [macfigure, ramfigure] = plot_dataset(in_channels,out_channels,kernel,image_size,padding,c,log=print_in_log,different_compared_to_baseline=show_difference)
        plt.figure(macfigure)
        plt.title(f"MAC operations for different kernel values")
        plt.legend()

        plt.figure(ramfigure)
        plt.title("RAM for different kernel values")
        plt.legend()
    

    if plot_rank_over_in_channel == True:
        c = 0.5
        in_channels = [2, 8, 32, 64, 256, 1024]
        out_channels = 256
        image_size = 40
        padding = 0
        kernel = 3
        print_in_log = False
        show_difference = False

        channel_ratio     = [i for i in in_channels]
        rank    = [calc_expectation.CP_rank_from_compratio(c, i ,out_channels,(kernel, kernel)) for i in in_channels]

        plt.figure()
        plt.scatter(channel_ratio,rank)
        plt.xlabel("in_channels")
        plt.ylabel("rank")



plt.show()




