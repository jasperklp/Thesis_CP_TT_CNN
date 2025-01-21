import os
import sys
import json

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Dataprocessing.dataproc_utils as utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from permetrics.regression import RegressionMetric

file_folder = "experiment_test_time"
filename = "2024-12-15_11.48.47"


def main(): 
    with open(f"{os.getcwd()}\\data\\data\\{file_folder}\\{filename}.json") as json_file:
            data = json.load(json_file)

    
    ## Create pandas data set of all data which needs to be incorporated into the model
        #For all measurements


    # Get results for TT CP
    df = utils.get_pandas_infernce_memory_pairs(data, used_models=["tt","cp"], used_ranks=[0.01, 0.05,0.1, 0.25])
    print(df)
    input = df[["memory", "MAC"]]
    output = df[["duration"]]

    print("TT/CP degree 1")
    _,_,_, output_test, output_pred_test = polynomial_fit(input, output, 1)
    plot_predicted_vs_actual(output_test, output_pred_test, "TT/CP CNN mem + ops degree 1")

    print("TT/CP degree 2")
    _,_,_, output_test, output_pred_test = polynomial_fit(input, output, 2)

    print("TT/CP degree 1 MAC only")
    polynomial_fit(df[["MAC"]], output, 1)

    print("TT/CP degree 1 memonly")
    polynomial_fit(df[["memory"]], output, 1)
    

    df = utils.get_pandas_infernce_memory_pairs(data, used_models=["uncomp"])
    input = df[["memory", "MAC"]]
    output = df[["duration"]]

    print("Uncomp degree 1")
    _,_,_, output_test, output_pred_test = polynomial_fit(input, output, 1)
    plot_predicted_vs_actual(output_test, output_pred_test, "Regular CNN mem + ops degree 1")

    print("Uncomp degree 2")
    polynomial_fit(input, output, 2)

    print("Uncomp degree 1 MAC only")
    _,_,fit,_,_ =   polynomial_fit(df[["MAC"]], output, 1)

    print("Uncomp degree 1 memonly")
    polynomial_fit(df[["memory"]], output, 1)

    print(fit.coef_)
    print(fit.intercept_)

    




def polynomial_fit(input, output, polynomial_degree):
    #Splits test and train data 80%-20%, rand_state is not None gives that the split is deterministic
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=0)
    poly_features2d = PolynomialFeatures(degree = polynomial_degree)

    input_train_poly2d = poly_features2d.fit_transform(input_train)
    input_test_poly2d = poly_features2d.transform(input_test)

    lr = LinearRegression()

    prediction_model = lr.fit(input_train_poly2d, output_train)

    output_pred_train = prediction_model.predict(input_train_poly2d)
    output_pred_test = prediction_model.predict(input_test_poly2d)

    evaluatortrain = RegressionMetric(output_train.to_numpy(), output_pred_train,  X_shape = input_train.shape)
    evaluatortest = RegressionMetric(output_test.to_numpy(), output_pred_test, X_shape = input_train.shape)

    print("Evaluator on train data")
    print(evaluatortrain.RMSE())
    print(evaluatortrain.VAF())
    print(evaluatortrain.R2())

    print("Evaluator on test data")
    print(evaluatortest.RMSE())
    print(evaluatortest.VAF())
    print(evaluatortest.R2())
    print("\n")

    return evaluatortest, evaluatortrain, prediction_model, output_test, output_pred_test


def plot_predicted_vs_actual(y_actual, y_pred, title:str):
    plt.scatter(y_actual,y_pred)
    plt.plot(plt.xlim(), plt.xlim(), c='red')
    plt.xlabel("Predicted time [s]")
    plt.ylabel("Actual time[s]")
    plt.title(title)
    plt.legend(["Actual prediction", "Perfect prediction"])
    plt.show()


if __name__ == "__main__" :
    main()



