import experiment_runner as runner
import CNN_models

model_1 = CNN_models.uncomp_model(256,256,3)
model_2 = CNN_models.cp_tensorly_model(256,256,3,rank=0.1)
if __name__ == "__main__":
    time = runner.model_runner(model_1, 2000 ,4, verbose=True)
    time = runner.model_runner(model_2, 2000, 4, verbose=True)
    print(time)

    [MAC, RAM] = model_1.MAC_and_RAM(4)
    print(f"RAM uncomp_model = {RAM}")

    [MAC, RAM] = model_2.MAC_and_RAM(4)
    print(f"RAM CP_model = {RAM}")