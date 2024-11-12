import os
import os.path
import sys
import pandas as pd
import openpyxl
import fnmatch



#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import statistics
from itertools import chain
from Dataprocessing import dataproc_utils as utils
from Dataprocessing import openpyxl_proc_utils as xl_utils

read_file = "2024-11-08_12.15.27"
save_file = f"{os.path.dirname(os.path.abspath(__file__))}\\files\\multiplier\\{read_file}.xlsx"


with open(f"{os.getcwd()}\\data\\data\\experiment_alter_kernels_to_get_mulitpliers\\{read_file}.json") as json_file:
    data = json.load(json_file)


print(f"Data is deterministic is {utils.verify_if_measurements_are_detministic(data["outcomes"], False)}")

#As all data points are deterministic and do take the same values. We can just take the first measurement
for i in data["outcomes"]: 
    i["measurements"] = i["measurements"][0]

wb = openpyxl.Workbook()
dicts = []
for i in ["uncomp","CP_tensorly"]:
    dicts.append({"name": i,
     "measurements": [k for k in data["outcomes"] if k["model_name"] == f"{i}"],
     "sheet" : wb.create_sheet(f"{i}")})



for j in dicts:
    sheet = j["sheet"]
    for number,item in enumerate(j["measurements"]):
        offset = 1 #Python starts counting at zero but excel at 1
        
        #First insert parameters for each measurment
        for key_nr,key in enumerate(item):
            if key == "measurements":
                offset -= 1 #Decrease one to keep rows with data adjacent.
                continue
            if fnmatch.fnmatch(key,"*Expected*"):
                offset -= 1 #Decrease one to keep rows with data adjacent.
                continue
            
            xl_utils.print_in_first_two_cells_from_dict(sheet,5, number, key_nr+offset, key,item[key])

        #Also add total allocated RAM and peak allocated RAM
        offset = key_nr + offset
        xl_utils.print_in_first_two_cells_from_dict(sheet,5,number,offset  ,"Peak allocated RAM",item["measurements"]["Peak allocated RAM"])
        xl_utils.print_in_first_two_cells_from_dict(sheet,5,number,offset+1,"Total allocated RAM",item["measurements"]["Total allocated RAM"])
        offset += 2

        #Add expectation values
        xl_utils.print_in_first_two_cells_from_dict(sheet,5,number,offset,"Expected RAM", "")
        offset += 1
        for i,value in enumerate(item["Expected RAM"]):
            xl_utils.print_in_first_two_cells_from_dict(sheet,5,number,offset+i,"",value)
        offset += len(item["Expected RAM"])

        #Print measurement results
        xl_utils.print_in_first_two_cells_from_dict(sheet,5,number,offset,"Measurement results","")
        offset += 1
        for i in item["measurements"]["Filter per model"]:
            xl_utils.print_line_in_minor_row(sheet, 5 , number, offset,[i["name"],"Bytes","name"])
            offset += 1
            for event_nr,event in enumerate(i["Events"]):
                xl_utils.print_line_in_minor_row(sheet,5,number,offset, ["",event["Bytes"],event["Operation name"]])
                offset +=1
            offset += 1


    
    
try:
    sheet = wb["Sheet"]
    wb.remove(sheet)
except:
    print("Default sheet could not be removed.")

wb.save(save_file)




