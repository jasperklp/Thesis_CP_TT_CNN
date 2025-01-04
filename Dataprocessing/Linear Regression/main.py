import os
import sys
import json

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Dataprocessing.dataproc_utils as utils

file_folder = "experiment_test_time"
filename = "2024-12-15_11.48.47"


def main(): 
    with open(f"{os.getcwd()}\\data\\data\\{file_folder}\\{filename}.json") as json_file:
            data = json.load(json_file)

    
    ## Create pandas data set of all data which needs to be incorporated into the model
        #For all measurements

    df = utils.get_pandas_infernce_memory_pairs(data, used_models=["tt","cp"], used_ranks=[0.01, 0.05,0.1, 0.25])

    plt.scatter(df["memory"], df["duration"])
    

    ## Use polyfit to make a fit for each data point.

    fit = np.polynomial.polynomial.Polynomial.fit(df["memory"], df["duration"],3)
    print(fit)
    x = np.linspace(df["memory"].min(), df["memory"].max())
    print(x)
    plt.plot(x,fit(x), c = "red")


    # Do a polyfit using only parameters for uncompressed model
    plt.figure()
    plt.scatter(df["MAC"], df["duration"])
    fit2 = np.polynomial.polynomial.Polynomial.fit(df["MAC"], df["duration"], 1)
    print(fit2)
    x = np.linspace(df["MAC"].min(), df["MAC"].max())

    plt.plot(x,fit2(x),c="red")

    ## Verification of the data



    plt.show()















if __name__ == "__main__" :
    main()
