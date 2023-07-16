
import numpy as np
from utilities import (
    save_npy,
    load_optimum,
    load_state,
    loadPathAndPlot,
    loadGapAndPlot,
    plot_figure_path,
    spectral_norm,
    print_matrix,
    convert_to_doubly_stochastic,
    is_primitive,
    try_geo,
    init_comm_matrix,
    is_doubly_stochastic,
    fix_lambda_transformation,
)
from Problems.logistic_regression import LR_L2
from graph import Weight_matrix, Geometric_graph, Exponential_graph, Grid_graph
import math

# comm_matrix = init_comm_matrix(8, "fully_connected")
# eigenvalues, eigenvectors = np.linalg.eig(comm_matrix)
# print("eigenvalues: ", eigenvalues)
# print("eigenvectors: ", eigenvectors)
# exit(0)

np.random.seed(0)
for node_num in [8, 16, 20, 24, 25]:
    for i in range(2):
        print("node: ", node_num, "i: ", i)
        comm_matrix = init_comm_matrix(node_num, "erdos_renyi")
        found, matrix = fix_lambda_transformation(comm_matrix, 0.5)
        if found:
            # store the matrix
            # np.save(
            #     f"/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/comm_matrix/comm_matrix_{node_num}.npy",
            #     matrix,
            # )
            break

exit(0)

exp_path = "/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/opt_CRR_reg01"
line_formats = [
    "-vb",
    "-^m",
    "-dy",
    "-sr",
    "-1k",
    "-2g",
    "-3r",
    "-.kp",
    "-+c",
    "-xm",
    "-|y",
    "-_r",
]
plot_every = 500

L = 0.12005882830300582
lrs = [0.01]
batch_size = [12000]
update_round = [1]
exp_names = []
legends = []
for idx, bz in enumerate(batch_size):
    exp_names = []
    legends = []

    for lr in lrs:
        for ur in update_round:
            # exp_names.append(f"central_SGD/CSGD_gap_bz{bz}_lr{lr:.6f}.npy")
            exp_names.append(f"central_CRR/CRR_gap_bz{bz}_lr{lr:.6f}.npy")
            # exp_names.append(f"DSGD/DSGD_gap_bz{bz}_lr{lr:.6f}_ur{ur}.npy")
            # exp_names.append(f"DRR/DRR_gap_bz{bz}_lr{lr:.6f}_ur{ur}.npy")

            # legends.append(f"SGD, bz={bz}, lr={lr:.6f}")
            legends.append(f"CRR, bz={bz}, lr={lr:.6f}")
            # legends.append(f"DSGD, bz={bz}, lr={lr:.6f}, ur={ur}")
            # legends.append(f"DRR, bz={bz}, lr={lr:.6f}, ur={ur}")

    plot_figure_path(
        exp_path,
        exp_names,
        line_formats,
        legends,
        f"{exp_path}/convergence_CRR_bz{bz}.pdf",
        plot_every,
    )