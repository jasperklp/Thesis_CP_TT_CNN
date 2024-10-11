import experiment_runner as runner
import experiment_helper_functions as helper
import CNN_models

model_1 = CNN_models.uncomp_model(256,256,3)
model_2 = CNN_models.cp_tensorly_model(256,256,3,rank=0.1)

image_size_uncomp   = 40
image_size_cp       = 40


    

if __name__ == "__main__":
    time = runner.model_runner(model_1, 1 ,40, verbose=True)
    time = runner.model_runner(model_2, 1, 40, verbose=True)

    [MAC, RAM] = model_1.MAC_and_RAM(40,output_in_bytes=True, output_total=False)
    print(f"RAM uncomp_model = {helper.print_RAM(RAM)}")

    [MAC, RAM] = model_2.MAC_and_RAM(40,output_in_bytes=True, output_total=False)
    print(f"RAM CP_model = {helper.print_RAM(RAM)}")