import os
import json
import matplotlib.pyplot as plt 
file = "2024-10-22_16.06.40"
import utils

with open(f"{os.getcwd()}\\data\\data\\test_experiment_2\\{file}.json") as json_file:
    data = json.load(json_file)

uncomp_data = []
cp_tensorly_data = []
cp_gil_data = []


for i in data["outcomes"]:
    if i["model_name"] == "uncomp":
        uncomp_data.append(i)
        continue

    if i["model_name"] == "CP_tensorly":
        cp_tensorly_data.append(i)
        continue

    if i["model_name"] == "CP_GIL":
        cp_gil_data.append(i)
        continue

    raise ValueError(f"Name not in list, name is {i.get("model_name")}")


expected_memory_uncomp = []
measured_memory_uncomp = []
expected_memory_cp = []
measured_memory_cp = []
c = data["Measured range"]

for i in uncomp_data:
    expected_memory_uncomp.append(i["Expected RAM total"]/ (1024**2))
    measured_event_memory = 0
    for j in i["measurements"]:
        measured_event_memory += j["Total allocated RAM"] / (1024**2)
    measured_memory_uncomp.append(measured_event_memory/i["nr of measurements"])

for i in cp_tensorly_data:
    expected_memory_cp.append(i["Expected RAM total"] / (1024**2))
    measured_event_memory = 0
    for j in i["measurements"]:
        measured_event_memory += j["Total allocated RAM"] / (1024**2)
    measured_memory_cp.append(measured_event_memory/i["nr of measurements"])

plt.figure()
plt.scatter([-0.05], expected_memory_uncomp)
plt.scatter(c, expected_memory_cp)
plt.scatter([-0.05], measured_memory_uncomp)
plt.scatter(c,measured_memory_cp)
plt.suptitle("Expected amount of memory vs total memory allocation")
plt.title(f"in_chan = out_chan = {1024}, kernel_sz = {3,3}, image_sz = {100,100}")
plt.ylabel("Expected RAM or Allocated RAM [MB]")
plt.xlabel("Compression ratio for CP")
plt.xticks([-0.05] + c, ["Uncomp."] + c,rotation = 315)
plt.legend(["Expected uncomporessed","Expected CP", "Measured uncompressed", "Measured CP"])

ratio_uncomp = [j/i for i,j in zip(expected_memory_uncomp,measured_memory_uncomp)] 
ratio_cp = [j/i for i,j in zip(expected_memory_cp,measured_memory_cp)]

plt.figure()
plt.scatter([-0.05], ratio_uncomp)
plt.scatter(c,ratio_cp)
plt.suptitle("Ratio between expected amount of RAM and true amount of RAM")
plt.title(f"in_chan = out_chan = {1024}, kernel_sz = {3,3}, image_sz = {100,100}")
plt.ylabel("Measured RAM/expected RAM")
plt.xlabel("Compression ratio for CP")
plt.xticks([-0.05] + c, ["Uncomp."] + c,rotation = 315)
plt.legend(["Ratio uncompressed", "Ratio CP"])
plt.show()
