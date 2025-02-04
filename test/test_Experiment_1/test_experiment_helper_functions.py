from ...Experiment_runner import experiment_helper_functions as helper
import pytest
import json

def test_name_number_type():
    with pytest.raises(TypeError,match=r"Number should have type int, but it has type .*"):
        helper.name_number(1.1)

def test_name_number_val_size():
    assert helper.name_number(1395864371) == "1.30Gb"

def test_name_number_size():
    assert helper.name_number(1395864371,False, True) == "Gb"

def test_name_number_val():
    assert helper.name_number(1395864371,True, False) == "1.30"

def test_name_numer_negative():
    assert helper.name_number(-1395864371) == "-1.30Gb"

def test_name_number_no_output():
    with pytest.raises(ValueError):
        helper.name_number(1395864371, False, False)


def test_json_mem():
    with open(".//test//test_Experiment_1//test_json_1.json") as json_file:
        data = json.load(json_file)
        with pytest.raises(ValueError):
            helper.json_get_memory_changes_per_model_ref(data)

#(Depricated) Should not emit as there will always be made a start end and inbetween record functions.
# @pytest.mark.filterwarnings("ignore")
# def test_json_mem_event_misses():
#     with open(".//test//test_Experiment_1//test_json_2.json") as json_file:
#         data = json.load(json_file)
#         with pytest.warns(UserWarning, match="Not all memory events are added to a memory record"):
#             helper.json_get_memory_changes_per_model_ref(data,True)


# Turned of did not prove to be very usefull.
# def test_json_memory_events_without_assigned_operation():
#     with open(".//test//test_Experiment_1//test_json_2.json") as json_file:
#         data = json.load(json_file)
#         events = data["traceEvents"]
#         with pytest.warns(UserWarning, match="There are memory events without an assigned cpu operation"):
#             helper.get_function_call_for_mem_ref(events)

def test_json_memory_events_succesful(capsys):
    with open(".//test//test_Experiment_1//test_json_3.json") as json_file:
        data = json.load(json_file)
        events = data["traceEvents"]
        helper.json_get_memory_changes_per_model_ref(data,True)
    captured = capsys.readouterr()
    assert captured.out == f"Printing events\n\tStart\n\t{events[0]["name"]}\n\t\t{helper.name_number(events[3]["args"]["Bytes"])}\tfor operation {events[1]["name"]}\n\tEnd\n"

def test_get_total_alloc_and_bytes():
    with open(".//test//test_Experiment_1//test_json_3.json") as json_file:
        data = json.load(json_file)
        events = data["traceEvents"]
        (peak_memory, total_alloc) = helper.get_peak_and_total_alloc_memory(events)
    
    assert total_alloc == 36416
    assert peak_memory == 36416