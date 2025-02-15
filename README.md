# Thesis_CP_CNN

tltorch folder has been taken from the PyPI version 0.5.0.
https://github.com/tensorly/torch

GIL folder has been taken from https://github.com/jacobgil/pytorch-tensor-decompositions

The experiments used in the report can be found in the Experiment runner folder. One file will gather the data for the first experiment and the other for the second experiment.

#######

Run an experiment. (file experiment 1 and experiment 2&3)

#######

In order to obtain the data for the experiments shown in this thesis the file should just be run. In the main file all sorts of tests are stated. The tests, which are uncommend are part of the exctual thesis. 

But a single measurement can be setup as follows:

First one creates a measurement dataclass containing all possible data combinations. If desired one can chose an iterator. By default it tests for all possible data combinations. Then one needs to select a runner. This can be the verify_model_matching_MKL which runs forced on the MKL (not used in the final result of the theis) or the verify_model_matching_df_pytorch, which runs the experiment with default pytorch settings.

The results will be put in the ./data/data/{function_name} folder. Additionaly a log will be put in ./data/log/{function name} folder.

#######

Process an experiment

#######

To process an experiment, one should chose the correct processor for each experiment. To obtain the results as given in the thesis, one should change the file name to the time which run the experiment.


#######

Theoretical analsysis in thesis

######

In the thesis a theoretical analysis is done. The results of the theoretical analysis can be found in the ./Dataprocessing/Thoertical_Analysis_in_thesis folder.
