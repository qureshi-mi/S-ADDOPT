
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
    fix_lambda_transformation,
    convert_to_doubly_stochastic,
    is_primitive,
    try_geo,
)
from Problems.logistic_regression import LR_L2
from graph import Weight_matrix, Geometric_graph, Exponential_graph, Grid_graph
import math


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