import os
import sys
import json

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_folder = "experiment_test_time"
filename = "2024-12-15_11.48.47"


def main(): 
    with open(f"{os.getcwd()}\\data\\data\\{file_folder}\\{filename}.json") as json_file:
            data = json.load(json_file)

    
    ## Create pandas data set of all data which needs to be incorporated into the model
        #For all measurements

    headers = { 
         "model"    : [],
         "memory"   : [],
         "duration" : [] 
    }

    df = pd.DataFrame(headers)

    print(df)



    ## Use polyfit to make a fit for each data point.


    ## Verification of the data















if __name__ == "__main__" :
    main()
