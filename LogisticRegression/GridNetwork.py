## Generates all the plots to compare different algorithms over Exponential directed graphs using logistic regression.

import os
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from graph import Weight_matrix, Geometric_graph, Exponential_graph, Grid_graph
from analysis import error
from Problems.logistic_regression import LR_L2
from Problems.log_reg_cifar import LR_L4
from Optimizers import COPTIMIZER as copt
from Optimizers import DOPTIMIZER as dopt
from utilities import initDir, save_npy, plot_figure_path, load_state
import copy as cp

"""
Data processing for MNIST
"""
node_num = 16  ## number of nodes
side_length = 4  ## side length of the square
logis_model = LR_L2(
    node_num, limited_labels=False, balanced=True
)  ## instantiate the problem class
dim = logis_model.p  ## dimension of the model
L = logis_model.L  ## L-smooth constant
total_train_sample = logis_model.N  ## total number of training samples
avg_local_sample = logis_model.b  ## average number of local samples
step_size = 1 / L / 2  ## selecting an appropriate step-size

"""
Initializing variables
"""
CEPOCH_base = 2000
DEPOCH_base = 2000

model_para_central = np.random.normal(0, 1, dim)
# model_para_dis = np.random.normal(0, 1, (node_num, dim))
model_para_dis = [cp.deepcopy(model_para_central) for i in range(node_num)]
undir_graph = Grid_graph(side_length).undirected()
communication_matrix = Weight_matrix(undir_graph).column_stochastic()
communication_rounds = [1, 5, 10, 20]

C_lr = [1/8000]
D_lr = [1/8000]
C_lr_dec = False
D_lr_dec = False
C_batch_size = [1]
D_batch_size = [1]

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
exp_log_path = "/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/grid_ur_reg01"
ckp_load_path = "/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/optimum"
plot_every = 100
save_every = 5000

"""
Optimum solution
"""
if ckp_load_path is not None:
    theta_CSGD_0, theta_opt = load_state(ckp_load_path, "optimum")
theta_CSGD_0 = None
error_lr_0 = error(logis_model, theta_opt, logis_model.F_val(theta_opt))


def CSGD_check(
    logis_model,
    model_para,
    C_lr,
    C_lr_dec,
    C_batch_size,
    CEPOCH_base,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
):
    exp_save_path = f"{exp_log_path}/central_SGD"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    params = []
    for bz in C_batch_size:
        for lr in C_lr:
            params.append((bz, lr))

    for idx, (bz, lr) in enumerate(params):
        if os.path.exists(f"{exp_save_path}/CSGD_gap_bz{bz}_lr{lr:.6f}.npy"):
            continue

        theta_SGD, theta_opt, F_opt = copt.SGD(
            logis_model,
            lr,
            CEPOCH_base,
            model_para,
            bz,
            C_lr_dec,
            train_log_path,
            f"CSGD_bz{bz}_lr{lr:.3f}_check",
            save_every,
        )
        np.save(f"{exp_save_path}/CSGD_opt_theta_bz{bz}_lr{lr:.6f}.npy", theta_opt)

        res_F_SGD = error_lr.cost_gap_path(theta_SGD, gap_type="theta")
        np.save(f"{exp_save_path}/CSGD_gap_bz{bz}_lr{lr:.6f}_theta.npy", res_F_SGD)
        res_F_SGD_F = error_lr.cost_gap_path(theta_SGD, gap_type="F")
        np.save(f"{exp_save_path}/CSGD_gap_bz{bz}_lr{lr:.6f}_F.npy", res_F_SGD_F)

    plot_figure_path(
        exp_save_path,
        [f"CSGD_gap_bz{bz}_lr{lr:.6f}_theta.npy" for idx, (bz, lr) in enumerate(params)],
        line_formats,
        [f"bz = {bz}, lr = {lr}" for idx, (bz, lr) in enumerate(params)],
        f"{exp_save_path}/convergence_SGD_theta.pdf",
        plot_every,
    )
    plot_figure_path(
        exp_save_path,
        [f"CSGD_gap_bz{bz}_lr{lr:.6f}_F.npy" for idx, (bz, lr) in enumerate(params)],
        line_formats,
        [f"bz = {bz}, lr = {lr}" for idx, (bz, lr) in enumerate(params)],
        f"{exp_save_path}/convergence_SGD_F.pdf",
        plot_every,
    )


def CRR_check(
    logis_model,
    model_para,
    C_lr,
    C_lr_dec,
    C_batch_size,
    CEPOCH_base,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
):
    exp_save_path = f"{exp_log_path}/central_CRR"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    params = []
    for bz in C_batch_size:
        for lr in C_lr:
            params.append((bz, lr))

    for idx, (bz, lr) in enumerate(params):
        if os.path.exists(f"{exp_save_path}/CRR_gap_bz{bz}_lr{lr:.6f}.npy"):
            continue

        theta_CRR, theta_opt, F_opt = copt.C_RR(
            logis_model,
            lr,
            CEPOCH_base,
            model_para,
            bz,
            C_lr_dec,
            train_log_path,
            f"CRR_bz{bz}_lr{lr}_check",
            save_every,
        )
        np.save(f"{exp_save_path}/CRR_opt_theta_bz{bz}_lr{lr:.6f}.npy", theta_opt)

        res_F_CRR = error_lr.cost_gap_path(theta_CRR, gap_type="theta")
        np.save(f"{exp_save_path}/CRR_gap_bz{bz}_lr{lr:.6f}_theta.npy", res_F_CRR)
        res_F_CRR_F = error_lr.cost_gap_path(theta_CRR, gap_type="F")
        np.save(f"{exp_save_path}/CRR_gap_bz{bz}_lr{lr:.6f}_F.npy", res_F_CRR_F)

    plot_figure_path(
        exp_save_path,
        [f"CRR_gap_bz{bz}_lr{lr:.6f}_theta.npy" for idx, (bz, lr) in enumerate(params)],
        line_formats,
        [f"bz = {bz}, lr = {lr}" for idx, (bz, lr) in enumerate(params)],
        f"{exp_save_path}/convergence_CRR_theta.pdf",
        plot_every,
    )
    plot_figure_path(
        exp_save_path,
        [f"CRR_gap_bz{bz}_lr{lr:.6f}_F.npy" for idx, (bz, lr) in enumerate(params)],
        line_formats,
        [f"bz = {bz}, lr = {lr}" for idx, (bz, lr) in enumerate(params)],
        f"{exp_save_path}/convergence_CRR_F.pdf",
        plot_every,
    )


"""
Decentralized Algorithms
"""

def DSGD_check(
    logis_model,
    model_para,
    D_lr,
    D_lr_dec,
    D_batch_size,
    DEPOCH_base,
    communication_matrix,
    communication_rounds,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
):
    exp_save_path = f"{exp_log_path}/DSGD"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    exp_names = []
    legends = []
    params = []
    for bz in D_batch_size:
        for lr in D_lr:
            for cr in communication_rounds:
                params.append((bz, lr, cr))

    for idx, (bz, lr, cr) in enumerate(params):
        if os.path.exists(f"{exp_save_path}/DSGD_gap_bz{bz}_lr{lr:.6f}_ur{cr}.npy"):
            continue

        theta_D_SGD = dopt.D_SGD(
            logis_model,
            communication_matrix,
            lr,
            int(DEPOCH_base),
            model_para,
            bz,
            cr,
            D_lr_dec,
            train_log_path,
            f"DSGD_bz{bz}_ur{cr}_lr{lr}",
            save_every,
        )

        exp_names.append(f"bz{bz}_ur{cr}_lr{lr}")
        legends.append(f"bz = {bz}, ur = {cr}, lr = {lr}")
        np.save(
            f"{exp_save_path}/DSGD_opt_theta_bz{bz}_lr{lr:.6f}_ur{cr}.npy", theta_opt
        )

        res_F_D_SGD = error_lr.cost_gap_path(theta_D_SGD, gap_type="theta")
        np.save(f"{exp_save_path}/DSGD_gap_bz{bz}_lr{lr:.6f}_ur{cr}_theta.npy", res_F_D_SGD)
        res_F_D_SGD_F = error_lr.cost_gap_path(np.sum(theta_D_SGD, axis=1) / node_num, gap_type="F")
        np.save(f"{exp_save_path}/DSGD_gap_bz{bz}_lr{lr:.6f}_ur{cr}_F.npy", res_F_D_SGD_F)

    plot_figure_path(
        exp_save_path,
        [
            f"DSGD_gap_bz{bz}_lr{lr:.6f}_ur{cr}_theta.npy"
            for idx, (bz, lr, cr) in enumerate(params)
        ],
        line_formats,
        legends,
        f"{exp_save_path}/convergence_DSGD_theta.pdf",
        plot_every,
    )
    plot_figure_path(
        exp_save_path,
        [
            f"DSGD_gap_bz{bz}_lr{lr:.6f}_ur{cr}_F.npy"
            for idx, (bz, lr, cr) in enumerate(params)
        ],
        line_formats,
        legends,
        f"{exp_save_path}/convergence_DSGD_F.pdf",
        plot_every,
    )


def DRR_check(
    logis_model,
    model_para,
    D_lr,
    D_lr_dec,
    D_batch_size,
    DEPOCH_base,
    communication_matrix,
    communication_rounds,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
):
    exp_save_path = f"{exp_log_path}/DRR"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    exp_names = []
    legends = []
    params = []
    for bz in D_batch_size:
        for lr in D_lr:
            for cr in communication_rounds:
                params.append((bz, lr, cr))

    for idx, (bz, lr, cr) in enumerate(params):
        if os.path.exists(f"{exp_save_path}/DRR_gap_bz{bz}_lr{lr:.6f}_ur{cr}.npy"):
            continue

        theta_D_RR = dopt.D_RR(
            logis_model,
            communication_matrix,
            lr,
            int(DEPOCH_base),
            model_para,
            bz,
            cr,
            D_lr_dec,
            train_log_path,
            f"DRR_bz{bz}_ur{cr}_lr{lr}",
            save_every,
        )

        exp_names.append(f"bz{bz}_ur{cr}_lr{lr}")
        legends.append(f"bz = {bz}, ur = {cr}, lr = {lr}")
        np.save(
            f"{exp_save_path}/DRR_opt_theta_bz{bz}_lr{lr:.6f}_ur{cr}.npy", theta_opt
        )

        res_F_D_RR = error_lr.cost_gap_path(theta_D_RR, gap_type="theta")
        np.save(f"{exp_save_path}/DRR_gap_bz{bz}_lr{lr:.6f}_ur{cr}_theta.npy", res_F_D_RR)
        res_F_D_RR_F = error_lr.cost_gap_path(np.sum(theta_D_RR, axis=1) / node_num, gap_type="F")
        np.save(f"{exp_save_path}/DRR_gap_bz{bz}_lr{lr:.6f}_ur{cr}_F.npy", res_F_D_RR_F)

    plot_figure_path(
        exp_save_path,
        [
            f"DRR_gap_bz{bz}_lr{lr:.6f}_ur{cr}_theta.npy"
            for idx, (bz, lr, cr) in enumerate(params)
        ],
        line_formats,
        legends,
        f"{exp_save_path}/convergence_DRR_theta.pdf",
        plot_every,
    )
    plot_figure_path(
        exp_save_path,
        [
            f"DRR_gap_bz{bz}_lr{lr:.6f}_ur{cr}_F.npy"
            for idx, (bz, lr, cr) in enumerate(params)
        ],
        line_formats,
        legends,
        f"{exp_save_path}/convergence_DRR_F.pdf",
        plot_every,
    )


print(f"{'-'*50}")
print("Grid Graph")
print(f"node num = {node_num}")
print(f"dim = {dim}")
print(f"L = {L}")
print(f"total train sample = {total_train_sample}")
print(f"avg local sample {avg_local_sample}")
print(f"CEPOCH base = {CEPOCH_base}")
print(f"DEPOCH base {DEPOCH_base}")
print(f"communication rounds = {communication_rounds}")
print(f"C lr = {C_lr}")
print(f"D lr = {D_lr}")
print(f"C batch size = {C_batch_size}")
print(f"D batch size = {D_batch_size}")
print(f"{'-'*50}")

start = time.time()

print()
print("CSGD")
CSGD_check(
    logis_model,
    model_para_central,
    C_lr,
    C_lr_dec,
    C_batch_size,
    CEPOCH_base,
    exp_log_path,
    save_every,
    error_lr_0,
    line_formats,
    plot_every,
)

print()
print("CRR")
CRR_check(
    logis_model,
    model_para_central,
    C_lr,
    C_lr_dec,
    C_batch_size,
    CEPOCH_base,
    exp_log_path,
    save_every,
    error_lr_0,
    line_formats,
    plot_every,
)

print()
print("DSGD")
DSGD_check(
    logis_model,
    model_para_dis,
    D_lr,
    D_lr_dec,
    D_batch_size,
    DEPOCH_base,
    communication_matrix,
    communication_rounds,
    exp_log_path,
    save_every,
    error_lr_0,
    line_formats,
    plot_every,
)

print()
print("DRR")
DRR_check(
    logis_model,
    model_para_dis,
    D_lr,
    D_lr_dec,
    D_batch_size,
    DEPOCH_base,
    communication_matrix,
    communication_rounds,
    exp_log_path,
    save_every,
    error_lr_0,
    line_formats,
    plot_every,
)

end = time.time()
print(f"{'-'*50}")
print(f"Total time: {end-start:.2f} seconds")
print(f"{'-'*50}")
