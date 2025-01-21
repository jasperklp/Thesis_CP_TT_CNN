import sys
import os

#Adds root of thesis folder to sys path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Experiment_1.calc_expectation import calc_output_image_dim, CP_rank_from_compratio
from tensorly import validate_tt_rank


def example_1():
    print("Example 1")
    T = 16
    S = 16
    W = 16
    H = 16
    d1 = 3
    d2 = 3
    padding = 1
    stride = 1
    dilation = 1
    c = 0.1

    run_example(T, S,  W, H, d1, d2, padding, stride, dilation, c)

def example_2():
    print("Example 2")
    T = 256
    S = 256
    W = 16
    H = 16
    d1 = 3
    d2 = 3
    padding = 1
    stride = 1
    dilation = 1
    c = 0.1

    run_example(T, S,  W, H, d1, d2, padding, stride, dilation, c)

def example_3():
    print("Example 3")
    T = 16
    S = 16
    W = 256
    H = 256
    d1 = 3
    d2 = 3
    padding = 1
    stride = 1
    dilation = 1
    c = 0.1

    run_example(T, S,  W, H, d1, d2, padding, stride, dilation, c)

def example_4():
    print("Example 4")
    T = 256
    S = 256
    W = 256
    H = 256
    d1 = 3
    d2 = 3
    padding = 1
    stride = 1
    dilation = 1
    c = 0.1

    run_example(T, S,  W, H, d1, d2, padding, stride, dilation, c)
    
    


def run_example(T, S, W, H, d1, d2, padding, stride, dilation, compression_ratio):
    [H_out, W_out] = calc_output_image_dim((d1,d2), (stride, stride), (padding, padding), (dilation, dilation), (H,W))

    CP_rank = CP_rank_from_compratio(compression_ratio, S, T, (d1,d2))

    TT_rank = validate_tt_rank((S,d1,d2,T), compression_ratio)[1:-1]

    input_image = S * H * W
    output_image = T * H_out * W_out

    #Kernel specific
    regular_kernel = S * T * d1 * d2

    #CP specific
    CP_kernel = CP_rank * (S + T + d1 + d2)
    CP_in_between = CP_rank * (H*W + H_out*W + H_out*W_out)

    #TT specific 
    TT_kernel = TT_rank[0] * S + TT_rank[1] * (TT_rank[0] * d1 + TT_rank[2] * d2) + T * TT_rank[2]
    TT_in_between = TT_rank[0] * H * W + TT_rank[1] * H_out * W + TT_rank[2] * H_out * W_out

    print(f"The output image has size {H_out} x {W_out}")
    print(f"The CP-rank = {CP_rank}")
    print(f"The TT_rank = {TT_rank}\n")

    print("All CNN's")
    print(f"{input_image =}")
    print(f"{output_image =}")
    print(f"Sum input and output image = {input_image + output_image}\n")


    
    print("Regular CNN")
    print(f"Kernel el = {regular_kernel}")
    print(f"Regular Total = {input_image + regular_kernel + output_image}\n")

    print("CP CNN")
    print(f"Kernel el = {CP_kernel}")
    print(f"In between mem el = {CP_in_between}")
    print(f"CP Total = {input_image + CP_kernel + CP_in_between + output_image}\n")

    print("TT CNN")
    print(f"Kernel elements = {TT_kernel}")
    print(f"In_beteen mem el = {TT_in_between}")
    print(f"TT Total = {input_image + TT_kernel + TT_in_between + output_image}\n\n")








if __name__ == '__main__' :
    example_1()
    example_2()
    example_3()
    example_4()