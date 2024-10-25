import os
import sys

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import statistics
from itertools import chain
from Dataprocessing import dataproc_utils as utils

file = "2024-10-25_14.57.21"


with open(f"{os.getcwd()}\\data\\data\\experiment_alter_kernels_to_get_mulitpliers\\{file}.json") as json_file:
    data = json.load(json_file)


print(utils.verify_if_measurements_are_detministic(data["outcomes"], False))