import calc_expectation as calc
import numpy as np
import matplotlib.pyplot as plt    


plot_rank_relation          : bool  = False
plot_image_size_relation    : bool  = False
plot_in_channel_relation    : bool  = False
plot_kernel_relation        : bool  = True

matplotlib_colours = ['tab:blue', 'tab:orange','tab:green', 'tab:red','tab:purple', 'tab:brown', 'tab:pink','tab:grey', 'tab:olive','tab:cyan']


def plot_dataset(in_channels,out_channels, kernel, image_size, padding, rank, log : bool = True,different_compared_to_baseline = False):
    in_channels     = calc.check_list_or_int_float(in_channels)
    out_channels    = calc.check_list_or_int_float(out_channels)
    kernel          = calc.check_list_or_int_float(kernel)
    image_size      = calc.check_list_or_int_float(image_size)
    padding         = calc.check_list_or_int_float(padding)
    measurement_value =  max([in_channels, out_channels, kernel, image_size, padding], key=len)

    data = calc.get_theoretical_dataset_for_plot(in_channels,out_channels,kernel,image_size,padding, rank)
    data = np.squeeze(data)


    if log == True:
        data = np.log10(data)

    if different_compared_to_baseline == True:
        data[1,0,:,:] = data[1,0,:,:] - data[0,0,:,:]
        data[1,1,:,:] = data[1,1,:,:] - data[0,1,:,:]
        data[0,0,:,:] = data[0,0,:,:] - data[0,0,:,:]
        data[0,1,:,:] = data[0,1,:,:] - data[0,1,:,:]

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
        data = calc.get_theoretical_dataset_for_plot(in_channels,out_channels,kernel,image_size,padding, c)

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
        image_size = [4, 40, 400, 4000 ]
        padding = 1
        kernel =  3
        
        [macfigure, ramfigure] = plot_dataset(in_channels,out_channels,kernel,image_size,padding,c)
        plt.figure(macfigure)
        plt.title("MAC operations for different image sizes")
        plt.legend()

        plt.figure(ramfigure)
        plt.title("RAM for different image sizes")
        plt.legend()

    if  plot_in_channel_relation == True:
        c = [ 0.1, 0.25, 0.5, 0.75, 1.0]
        in_channels = [2, 8, 32, 64, 256, 1024]
        out_channels = 1024
        image_size = 40
        padding = 1
        kernel =  3
        
        [macfigure, ramfigure] = plot_dataset(in_channels,out_channels,kernel,image_size,padding,c)
        plt.figure(macfigure)
        plt.title("MAC operations for different in_channel values")
        plt.legend()

        plt.figure(ramfigure)
        plt.title("RAM for different in_channel values")
        plt.legend()

    if (plot_kernel_relation == True):
        c = [ 0.1, 0.25, 0.5, 0.75, 1.0]
        in_channels = 1024
        out_channels = 1024
        image_size = 40
        padding = 0
        kernel =  [1,3,5,7,9]
        
        [macfigure, ramfigure] = plot_dataset(in_channels,out_channels,kernel,image_size,padding,c,log=True,different_compared_to_baseline=True)
        plt.figure(macfigure)
        plt.title("MAC operations for different kernel values")
        plt.legend()

        plt.figure(ramfigure)
        plt.title("RAM for different kernel values")
        plt.legend()

plt.show()




